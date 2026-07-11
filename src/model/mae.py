from typing import Optional
from einops import rearrange

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import transformers

import numpy as np

from timm.models.vision_transformer import Block, Mlp
from timm.layers.config import set_fused_attn, use_fused_attn

from ..patching import PatchStrategy
from .pos_embed import get_2d_sincos_pos_embed
from .utils import plot_fbank
from .ambisonic_feature_extractor import FeatureExtractor
# new modules
from .spatial_targets import RouteATarget, multiscale_diffuseness
from ..masking import MixedSpatialMaskMaker
from .foa_rotation import random_rotations, rotate_foa_waveform


import io
from PIL import Image

set_fused_attn(True)
use_fused_attn(True)


# =============================================================================
# Loss helpers
# =============================================================================

def masked_mse(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """Unweighted MSE over masked patches; pred/target (B, P, C, fs, ts)."""
    m = mask.float().view(*mask.shape, 1, 1, 1)
    diff = (pred - target).pow(2) * m
    denom = m.sum() * pred.shape[-3] * pred.shape[-2] * pred.shape[-1]
    return diff.sum() / denom.clamp_min(1e-6)


def masked_l1(pred: torch.Tensor, target: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    """Unweighted L1 over masked patches; pred/target (B, P, C, fs, ts)."""
    m = mask.float().view(*mask.shape, 1, 1, 1)
    diff = (pred - target).abs() * m
    denom = m.sum() * pred.shape[-3] * pred.shape[-2] * pred.shape[-1]
    return diff.sum() / denom.clamp_min(1e-6)


def masked_mse_vec(pred: torch.Tensor, target: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """MSE over masked patches for per-patch vector predictions (B, P, C)."""
    m = mask.float().unsqueeze(-1)
    diff = (pred - target).pow(2) * m
    return diff.sum() / (m.sum() * pred.shape[-1]).clamp_min(1e-6)


def q_ce_loss(logits: torch.Tensor, q_tgt: torch.Tensor,
              weight: torch.Tensor):
    """Weighted cross-entropy of predicted angular distribution vs RouteA q.

    logits (B,P,G), q_tgt (B,P,G) rows sum to 1 (or 0 for silent patches),
    weight (B,P) -- typically log1p(Wp_n) * masked, with Wp_n the
    reference-normalized directional confidence.
    """
    logp = F.log_softmax(logits.float(), dim=-1)
    ce = -(q_tgt * logp).sum(dim=-1)                       # (B,P)
    denom = weight.sum().clamp_min(1e-6)
    loss = (ce * weight).sum() / denom
    with torch.no_grad():
        H = -(q_tgt * q_tgt.clamp_min(1e-9).log()).sum(dim=-1)
        kl = ((ce - H) * weight).sum() / denom
    return loss, kl


# =============================================================================
# NEW plain PatchEmbed (replaces InterChannelBlock + SpatialFrontEnd)
# =============================================================================

class PatchEmbed(nn.Module):
    """Plain per-patch Conv2D tokenizer, 7ch -> D. Non-overlapping (kernel==stride)
    so no token sees a neighbour's input.  Channels are mixed from the start."""
    def __init__(self, in_ch: int, dim: int, fshape: int, tshape: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=(fshape, tshape),
                              stride=(fshape, tshape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 7, F, T)   (frequency-major, as used in _encode_visible)
        x = self.proj(x)                      # (B, D, Nf, Nt)
        B, D, Nf, Nt = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        return x.reshape(B, Nf * Nt, D)       # freq-major flat


# =============================================================================
# Decoder (unchanged)
# =============================================================================

class ConditionedDecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=nn.GELU)
        self.last_cross_attn_norm = 0.0

    def forward(self, x: torch.Tensor,
                context: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        if context is not None:
            q = self.norm_q(x)
            kv = self.norm_kv(context)
            delta = self.cross_attn(q, kv, kv, need_weights=False)[0]
            self.last_cross_attn_norm = delta.norm(p=2, dim=-1).mean().item()
            x = x + delta
        else:
            self.last_cross_attn_norm = 0.0
        x = x + self.mlp(self.norm2(x))
        return x


class SemanticDecoder(nn.Module):
    """Decoder with the four SphereV5 heads."""

    def __init__(self, encoder_embed_dim: int, decoder_embed_dim: int,
                 depth: int, num_heads: int, patch_shape: tuple,
                 num_patches: int, grid_size: tuple, n_grid: int,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.num_patches = num_patches
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_shape = patch_shape
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False
        )
        pe = get_2d_sincos_pos_embed(decoder_embed_dim, grid_size, cls_token_num=0)
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))
        self.blocks = nn.ModuleList([
            ConditionedDecoderBlock(decoder_embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_embed_dim)

        fs, ts = patch_shape
        self.q_head = nn.Linear(decoder_embed_dim, n_grid)          # per patch
        self.diff_head = nn.Linear(decoder_embed_dim, 2 * fs * ts)  # psi short+long
        self.leveldiff_head = nn.Linear(decoder_embed_dim, 3 * fs * ts)
        self.level_head = nn.Linear(decoder_embed_dim, 2)           # per patch

    def forward(self, z_vis: torch.Tensor, visible_mask: torch.Tensor,
                context: Optional[torch.Tensor]):
        N, P, D = z_vis.shape[0], self.num_patches, self.decoder_embed_dim
        z_vis_proj = self.decoder_embed(z_vis)
        x = self.mask_token.type_as(z_vis_proj).expand(N, P, D).clone()
        x[visible_mask] = z_vis_proj.reshape(-1, D)
        x = x + self.pos_embed

        cross_attn_norms = []
        for blk in self.blocks:
            x = blk(x, context)
            cross_attn_norms.append(blk.last_cross_attn_norm)

        x = self.norm(x)
        fs, ts = self.patch_shape
        preds = {
            "q_logits": self.q_head(x).float(),                        # (N,P,G)
            "diff": self.diff_head(x).view(N, P, 2, fs, ts).float(),
            "leveldiff": self.leveldiff_head(x).view(N, P, 3, fs, ts).float(),
            "level": self.level_head(x).float(),                       # (N,P,2)
        }
        return preds, cross_attn_norms


# =============================================================================
# Lightning module
# =============================================================================

class SphereV5(pl.LightningModule):
    """Spatial MAE with RouteA q, multi-scale psi, level-difference and level
    targets; mixed span/random/censor masking; on-GPU SO(3) rotation aug."""

    def __init__(
            self,
            encoder_embedding_dim: int = 384,
            encoder_depth: int = 6,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,

            decoder_depth: int = 4,
            decoder_num_heads: int = 8,
            decoder_embedding_dim: int = 256,

            # ---- loss weights ----
            q_loss_weight: float = 1.0,
            diffuseness_loss_weight: float = 0.5,
            leveldiff_loss_weight: float = 0.5,
            level_loss_weight: float = 0.25,

            lr: float = 2e-4,
            b1: float = 0.9, b2: float = 0.95,
            weight_decay: float = 0.01,
            warmup_steps: int = 10000,

            patch_strategy: PatchStrategy = None,
            in_channels: int = 7,
            sr: int = 32000,
            num_mel_bins: int = 128,
            input_length: int = 500,
            target_length: int = 200,     # 2 s crops @ 100 fps

            inter_channel_heads: int = 4,          # (kept for compatibility, unused now)
            inter_channel_layers: int = 1,         # (unused)

            # ---- GRAM-T mono conditioning ----
            gramt_model_id: Optional[str] = "labhamlet/gramt-mono",
            gramt_mask_context: bool = True,   # ABLATION: False = decoder sees
                                               # full-clip mono context at all
                                               # positions (no null token)

            # ---- RouteA direction target ----
            n_grid: int = 256,
            vmf_kappa: float = 40.0,
            grid_chunk: int = 64,

            # ---- coherence windows (frames, mel) ----
            coh_tile_t: int = 5,  coh_tile_f: int = 3,    # ~50 ms
            coh_long_t: int = 35, coh_long_f: int = 3,    # ~350 ms
            diffuseness_c: float = 1.0,   # SN3D: E=|W|^2+sum|YZX|^2 -> c=1 exact

            # ---- masking ----
            mask_ratio: float = 0.8,
            mask_p_span: float = 0.6,
            mask_p_random: float = 0.2,
            mask_p_censor: float = 0.2,
            span_min_tokens: int = 2,
            span_max_tokens: Optional[int] = None,

            # ---- rotation augmentation ----
            rotation_mode: Optional[str] = "so3",
            rotation_prob: float = 1.0,

            # ---- mel front end ----
            f_max: Optional[float] = None,

            log_every_n_steps: int = 500,

            samples_per_clip: int = 4,
            native_sr: Optional[int] = None,   # native sr of shards; None = already self.sr

            **kwargs,
        ):
        super().__init__()
        self.save_hyperparameters(ignore=['patch_strategy'])

        assert in_channels == 7, \
            "Ambisonics expects 7 channels: W, Y, Z, X log-mel + IVy, IVz, IVx"

        self.sr = sr
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.input_length = input_length
        self.in_channels = in_channels
        self.lr = lr; self.b1 = b1; self.b2 = b2
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.q_loss_weight = q_loss_weight
        self.diffuseness_loss_weight = diffuseness_loss_weight
        self.leveldiff_loss_weight = leveldiff_loss_weight
        self.level_loss_weight = level_loss_weight
        self.coh_tile_t, self.coh_tile_f = coh_tile_t, coh_tile_f
        self.coh_long_t, self.coh_long_f = coh_long_t, coh_long_f
        self.diffuseness_c = diffuseness_c
        self.rotation_mode = rotation_mode
        self.rotation_prob = rotation_prob
        self.gramt_mask_context = gramt_mask_context
        self.log_every_n_steps = log_every_n_steps
        self.encoder_embedding_dim = encoder_embedding_dim

        self.patch_strategy = patch_strategy
        self.p_f_dim, self.p_t_dim = self.patch_strategy.get_patch_size()
        self.num_patches = self.p_f_dim * self.p_t_dim
        self.grid_size = (self.p_f_dim, self.p_t_dim)
        self.fshape, self.tshape = self.patch_strategy.fshape, self.patch_strategy.tshape
        self.patch_shape = (self.fshape, self.tshape)

        assert self.patch_strategy.fstride == self.fshape \
            and self.patch_strategy.tstride == self.tshape, \
            "SphereV5 requires non-overlapping patches (stride == shape)"
        assert self.num_mel_bins == self.p_f_dim * self.fshape, \
            f"num_mel_bins {num_mel_bins} != p_f_dim*fshape {self.p_f_dim}*{self.fshape}"
        assert self.target_length == self.p_t_dim * self.tshape, \
            f"target_length {target_length} != p_t_dim*tshape {self.p_t_dim}*{self.tshape}"

        # ----- Patch embed: plain Conv2D (replaces SpatialFrontEnd) -----
        self.patch_embed = PatchEmbed(in_ch=self.in_channels,
                                      dim=encoder_embedding_dim,
                                      fshape=self.fshape,
                                      tshape=self.tshape)

        # ----- Pos embed + CLS -----------------------------------------
        self.cls_token = nn.Parameter(
            nn.init.normal_(torch.empty(1, 1, self.encoder_embedding_dim), std=0.02)
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.encoder_embedding_dim),
            requires_grad=False,
        )
        pe = get_2d_sincos_pos_embed(
            self.encoder_embedding_dim, self.grid_size, cls_token_num=1,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

        # ----- Plain ViT encoder ----------------------------------------
        self.encoder_dropout = nn.Dropout(0.0)
        self.encoder_blocks = nn.ModuleList([
            Block(
                dim=self.encoder_embedding_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
            )
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(self.encoder_embedding_dim)

        # ----- Semantic conditioning (GRAM-T) ---------------------------
        self.gram = None
        self.gramt_proj = None
        self.gramt_pos_embed = None

        if gramt_model_id is not None:
            self.gram = transformers.AutoModel.from_pretrained(
                gramt_model_id, trust_remote_code=True
            )
            for p in self.gram.parameters():
                p.requires_grad_(False)
            self.gram.eval()

            with torch.no_grad():
                dummy_w = torch.zeros(1, 1, target_length, num_mel_bins)
                dummy_out = self.gram(dummy_w, strategy="raw")
            T_gram, flat_dim = dummy_out.shape[1], dummy_out.shape[2]
            self.gramt_n_freq = self.p_f_dim
            assert flat_dim % self.gramt_n_freq == 0, \
                f"GRAMT raw dim {flat_dim} not divisible by p_f_dim={self.gramt_n_freq}"
            self.gramt_native_dim = flat_dim // self.gramt_n_freq
            self.gramt_t_dim = T_gram
            assert T_gram * self.gramt_n_freq == self.num_patches, (
                f"GRAMT token grid {T_gram}x{self.gramt_n_freq} != "
                f"spatial patches {self.num_patches}"
            )

            self.gramt_proj = nn.Linear(
                self.gramt_native_dim, decoder_embedding_dim
            )
            self.gramt_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, decoder_embedding_dim),
                requires_grad=False,
            )
            pe_ctx = get_2d_sincos_pos_embed(
                decoder_embedding_dim, self.grid_size, cls_token_num=0
            )
            self.gramt_pos_embed.data.copy_(
                torch.from_numpy(pe_ctx).float().unsqueeze(0)
            )

            # Null token only exists in the mask-aligned arm.  Created
            # conditionally (not created-and-ignored) so DDP with
            # find_unused_parameters=False stays happy and the ablation arm
            # is honest about its parameter set.  Cross-arm checkpoint
            # loading: strict=False (state_dicts differ by this one key).
            if gramt_mask_context:
                self.gramt_null_token = nn.Parameter(
                    torch.zeros(1, 1, decoder_embedding_dim))
                nn.init.normal_(self.gramt_null_token, std=0.02)

        # ----- Decoder ----------------------------------------------------
        self.decoder = SemanticDecoder(
            encoder_embed_dim=self.encoder_embedding_dim,
            decoder_embed_dim=decoder_embedding_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            patch_shape=self.patch_shape,
            num_patches=self.num_patches,
            grid_size=self.grid_size,
            n_grid=n_grid,
            mlp_ratio=4.0,
        )

        # ----- Spatial targets --------------------------------------------
        self.route_a = RouteATarget(
            n_grid=n_grid, kappa=vmf_kappa,
            fshape=self.fshape, tshape=self.tshape,
            tile_f=coh_tile_f, tile_t=coh_tile_t,
            grid_chunk=grid_chunk,
        )

        # ----- Mask generator ----------------------------------------------
        self.mask_maker = MixedSpatialMaskMaker(
            n_freq_patches=self.p_f_dim, n_time_patches=self.p_t_dim,
            mask_ratio=mask_ratio,
            p_span=mask_p_span, p_random=mask_p_random, p_censor=mask_p_censor,
            span_min=span_min_tokens, span_max=span_max_tokens,
        )

        # ----- Mel feature extractor ---------------------------------------
        self.melspec = FeatureExtractor(
            sample_rate=self.sr, n_fft=1024, win_length=1024,
            hop_length=self.sr // 100, f_min=50,
            f_max=(f_max if f_max is not None else self.sr // 2),
            n_mels=self.num_mel_bins, power=2.0,
        ).float()
        assert self.melspec.power == 2.0, \
            "FeatureExtractor.power must be 2.0 for intensity/energy consistency"

        # ----- Running standardization stats (global affine == safe) --------
        self.register_buffer("leveldiff_running_mean", torch.zeros(1))
        self.register_buffer("leveldiff_running_var",  torch.ones(1))
        self.register_buffer("leveldiff_stats_inited", torch.zeros(1, dtype=torch.bool))
        self.register_buffer("level_running_mean", torch.zeros(2))
        self.register_buffer("level_running_var",  torch.ones(2))
        self.register_buffer("level_stats_inited", torch.zeros(1, dtype=torch.bool))
        self.stats_momentum = 0.01

        # ----- Wp reference scale (global, EMA in log-domain) ---------------
        # Single global constant: NOT per-clip (would destroy absolute-level
        # info in the level target) and NOT per-patch (would destroy the
        # contrast the CE weight needs).  Tracked as log(mean(Wp)) so heavy
        # tails don't destabilize the EMA; arithmetic mean on purpose (the
        # patch population is bimodal, geometric mean is dragged to the
        # silent floor).  Before initialization exp(0)=1, i.e. no rescale;
        # the first training batch initializes it.
        self.register_buffer("wp_log_ref", torch.zeros(1))
        self.register_buffer("wp_ref_inited", torch.zeros(1, dtype=torch.bool))

        self._init_our_weights()

        self.samples_per_clip = samples_per_clip
        self.native_sr = native_sr if native_sr is not None else sr
        self._gpu_resampler = None       # lazy; built on first batch, on-device

    def _init_our_weights(self):
        def init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Apply to encoder and decoder parts (PatchEmbed is already inited by Conv)
        for mod in [self.encoder_blocks, self.encoder_norm, self.decoder]:
            if isinstance(mod, nn.Module):
                mod.apply(init_fn)
        if self.gramt_proj is not None:
            self.gramt_proj.apply(init_fn)

    def _maybe_resample(self, audio: torch.Tensor) -> torch.Tensor:
        """GPU resample native-sr waveforms to self.sr. Kernel is built once
        and cached; conv1d on GPU, negligible cost vs the mel front end."""
        if self.native_sr == self.sr:
            return audio
        if self._gpu_resampler is None:
            import torchaudio
            self._gpu_resampler = torchaudio.transforms.Resample(
                self.native_sr, self.sr, lowpass_filter_width=64,
            ).to(audio.device).float()
        return self._gpu_resampler(audio.float())

    def on_fit_start(self):
        if self.gram is not None:
            self.gram.eval()

    # ----- Patchification (unchanged) -----
    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        fs, ts = self.fshape, self.tshape
        pf, pt = self.p_f_dim, self.p_t_dim
        B, C, T, Fm = x.shape
        x = x.transpose(2, 3)
        x = x.reshape(B, C, pf, fs, pt, ts)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.reshape(B, pf * pt, C, fs, ts)

    # ----- Running standardization helpers -----
    def _ema_standardize(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        mean_buf = getattr(self, f"{prefix}_running_mean")
        var_buf = getattr(self, f"{prefix}_running_var")
        inited = getattr(self, f"{prefix}_stats_inited")
        if self.training:
            with torch.no_grad():
                if mean_buf.numel() == 1:
                    bm = x.mean().reshape(1)
                    bv = x.var(unbiased=False).reshape(1)
                else:
                    dims = tuple(range(x.dim() - 1))
                    bm = x.mean(dim=dims)
                    bv = x.var(dim=dims, unbiased=False)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(bm, op=torch.distributed.ReduceOp.AVG)
                    torch.distributed.all_reduce(bv, op=torch.distributed.ReduceOp.AVG)
                if not bool(inited.any()):
                    mean_buf.copy_(bm); var_buf.copy_(bv); inited.fill_(True)
                else:
                    m = self.stats_momentum
                    mean_buf.mul_(1 - m).add_(m * bm)
                    var_buf.mul_(1 - m).add_(m * bv)
        sd = var_buf.clamp_min(1e-10).sqrt().clamp_min(1e-5)
        return (x - mean_buf) / sd

    # ----- Wp reference normalization -----
    def _normalize_wp(self, Wp: torch.Tensor) -> torch.Tensor:
        """Divide raw Wp (B,P) by a global running reference scale.

        The reference is the EMA of log(batch arithmetic mean of Wp), so the
        rescale is a single global constant shared by all clips and patches.
        Updated only in training; frozen (buffer) at eval/inference.
        """
        if self.training:
            with torch.no_grad():
                b = torch.log(Wp.float().mean().clamp_min(1e-6)).reshape(1)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(b, op=torch.distributed.ReduceOp.AVG)
                if not bool(self.wp_ref_inited.any()):
                    self.wp_log_ref.copy_(b)
                    self.wp_ref_inited.fill_(True)
                else:
                    m = self.stats_momentum
                    self.wp_log_ref.mul_(1 - m).add_(m * b)
        return Wp / self.wp_log_ref.exp()

    def _extract_features(self, waveform: torch.Tensor) -> dict:
        with torch.amp.autocast('cuda', enabled=False):
            waveform = waveform.float()
            rms = torch.sqrt(waveform[:, :1].pow(2).mean(dim=-1, keepdim=True) + 1e-8)
            waveform = waveform / rms
            return self.melspec.extract_all(waveform)

    # ----- Semantic context (GRAM-T) -----
    def _mono_context(self, w_logmel: torch.Tensor,
                    visible_mask: torch.Tensor) -> Optional[torch.Tensor]:
        if self.gram is None:
            return None
        with torch.no_grad():
            self.gram.eval()
            tokens = self.gram(w_logmel, strategy="raw")   # (B, T_gram, F_gram*D)
        B, T_gram, FD = tokens.shape
        F_gram, D = self.gramt_n_freq, self.gramt_native_dim
        tokens = tokens.view(B, T_gram, F_gram, D).permute(0, 2, 1, 3).contiguous()
        tokens = tokens.reshape(B, F_gram * T_gram, D)     # freq-major, aligns with visible_mask
        tokens = self.gramt_proj(tokens)

        if self.gramt_mask_context:
            # Mask-aligned arm: replace context at masked positions with the
            # null token, BEFORE adding pos embed so null positions still
            # carry position.  Ablation arm (False): tokens pass through
            # unmasked -- decoder sees full-clip mono context everywhere.
            vis = visible_mask.unsqueeze(-1).to(tokens.dtype)      # (B, P, 1)
            null = self.gramt_null_token.type_as(tokens)           # (1, 1, D)
            tokens = tokens * vis + null * (1.0 - vis)

        tokens = tokens + self.gramt_pos_embed
        return tokens.float()

    # ----- Encoder (adapted to new PatchEmbed) -----
    def _encode_visible(self, x_7ch: torch.Tensor,
                        visible_mask: torch.Tensor) -> torch.Tensor:
        B = x_7ch.shape[0]
        # x_7ch: (B, 7, T, F) -> (B, 7, F, T) for freq-major patch embedding
        x = x_7ch.transpose(2, 3)
        # Directly call patch_embed (returns (B, P, D))
        embedded = self.patch_embed(x)
        embedded = embedded + self.pos_embed[:, 1:, :]

        L_vis = visible_mask[0].sum().item()
        assert (visible_mask.sum(dim=1) == L_vis).all(), \
            "All samples in batch must have the same number of visible patches"
        z_vis = embedded[visible_mask].view(B, L_vis, -1)

        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        x = torch.cat([cls, z_vis], dim=1)
        x = self.encoder_dropout(x)

        for block in self.encoder_blocks:
            x = block(x)

        return self.encoder_norm(x)

    @torch.no_grad()
    def on_after_batch_transfer(self, batch, dataloader_idx):
        audio = batch[0] if isinstance(batch, (tuple, list)) else batch

        # 0. GPU resample (was CPU, per-sample, in the datamodule)
        audio = self._maybe_resample(audio)
        N = audio.shape[0]

        # 1. Rotation aug — per clip, on the waveform (must precede features)
        if self.training and self.rotation_mode is not None:
            R = random_rotations(N, self.rotation_mode, device=audio.device)
            if self.rotation_prob < 1.0:
                keep = (torch.rand(N, device=audio.device)
                        < self.rotation_prob).view(N, 1, 1)
                eye = torch.eye(3, device=audio.device).expand(N, 3, 3)
                R = torch.where(keep, R, eye)
            audio = rotate_foa_waveform(audio, R)

        # 2. Features ONCE per clip (per-clip W-RMS norm lives inside)
        feats = self._extract_features(audio)
        stacked = torch.cat([feats["logmel"], feats["aiv"],
                            feats["intensity_mel"], feats["energy_mel"]],
                            dim=1)                          # (N, 11, T_total, F)

        # 3. Multi-crop: S random 2 s windows per clip, gathered in one shot
        S = self.samples_per_clip if self.training else 1
        N_, C, T_total, n_freq = stacked.shape
        T_crop = self.target_length
        assert T_total >= T_crop, f"clip too short: {T_total} < {T_crop}"

        starts = torch.randint(0, T_total - T_crop + 1, (N, S),
                            device=self.device)          # (N, S)
        t_range = torch.arange(T_crop, device=self.device)
        # (N, S, C, T_crop, F) gather indices along the time dim
        time_idx = (starts.view(N, S, 1, 1, 1) + t_range.view(1, 1, 1, T_crop, 1)) \
            .expand(N, S, C, T_crop, n_freq)
        stacked = torch.gather(
            stacked.unsqueeze(1).expand(N, S, C, T_total, n_freq),
            dim=3, index=time_idx,
        ).reshape(N * S, C, T_crop, n_freq)                 # (N*S, 11, T_crop, F)

        # 4. Split channels + targets (unchanged, just on N*S rows)
        fb7 = stacked[:, 0:7]
        intensity_mel = stacked[:, 7:10]
        energy_mel = stacked[:, 10:11]
        leveldiff = (stacked[:, 1:4] - stacked[:, 0:1]).float()

        visible_mask = self.mask_maker(N * S).to(self.device)

        return (fb7.to(torch.bfloat16),
                intensity_mel.float(),
                energy_mel.float(),
                leveldiff,
                visible_mask)

    # ----- Forward -----
    def forward(self, fb7, intensity_mel, energy_mel, leveldiff, visible_mask):
        B = fb7.shape[0]
        w_logmel = fb7[:, 0:1]

        z = self._encode_visible(fb7, visible_mask)
        z_vis = z[:, 1:, :]

        mono_ctx = self._mono_context(w_logmel, visible_mask)
        preds, cross_attn_norms = self.decoder(z_vis, visible_mask, mono_ctx)

        with torch.no_grad():
            q_tgt, Wp = self.route_a(
                intensity_mel.transpose(2, 3).contiguous())

            # Global-reference normalization of Wp BEFORE log1p: restores
            # inter-patch contrast in the CE weight and gives the level
            # target a well-scaled log1p component.
            Wp_n = self._normalize_wp(Wp)

            psi = multiscale_diffuseness(
                intensity_mel, energy_mel,
                windows=((self.coh_tile_t, self.coh_tile_f),
                         (self.coh_long_t, self.coh_long_f)),
                c=self.diffuseness_c,
            )
            diff_tgt = self._patchify(psi)

            lvld_tgt = self._ema_standardize(
                self._patchify(leveldiff).float(), "leveldiff")

            Ep = self._patchify(energy_mel).float().mean(dim=(-2, -1))
            lvl_raw = torch.cat([torch.log(Ep + 1e-6),
                                 torch.log1p(Wp_n).unsqueeze(-1)], dim=-1)
            lvl_tgt = self._ema_standardize(lvl_raw, "level")

            masked = ~visible_mask
            w_q = torch.log1p(Wp_n) * masked.float()

        l_q, kl_q = q_ce_loss(preds["q_logits"], q_tgt, w_q)
        l_diff = masked_l1(preds["diff"], diff_tgt, masked)
        l_leveldiff = masked_mse(preds["leveldiff"], lvld_tgt, masked)
        l_level = masked_mse_vec(preds["level"], lvl_tgt, masked)

        total = (self.q_loss_weight * l_q
                 + self.diffuseness_loss_weight * l_diff
                 + self.leveldiff_loss_weight * l_leveldiff
                 + self.level_loss_weight * l_level)

        return {
            "loss": total,
            "l_q": l_q, "kl_q": kl_q,
            "l_diff": l_diff,
            "l_leveldiff": l_leveldiff,
            "l_level": l_level,
            "q_logits": preds["q_logits"], "q_tgt": q_tgt,
            "Wp": Wp, "Wp_n": Wp_n, "w_q": w_q,
            "diff_pred": preds["diff"], "diff_target": diff_tgt,
            "leveldiff_pred": preds["leveldiff"], "leveldiff_target": lvld_tgt,
            "level_pred": preds["level"], "level_target": lvl_tgt,
            "masked": masked,
            "z_vis": z_vis,
            "cross_attn_norms": cross_attn_norms,
        }

    # ----- Metrics (unchanged) -----
    @torch.no_grad()
    def _mean_dirs(self, q: torch.Tensor) -> torch.Tensor:
        v = q @ self.route_a.G
        return F.normalize(v, dim=-1, eps=1e-6)

    @torch.no_grad()
    def _q_angular_error_deg(self, q_logits, q_tgt) -> torch.Tensor:
        d_pred = self._mean_dirs(F.softmax(q_logits.float(), dim=-1))
        d_tgt = self._mean_dirs(q_tgt)
        cos = (d_pred * d_tgt).sum(dim=-1).clamp(-1.0, 1.0)
        return torch.acos(cos) * (180.0 / np.pi)

    @torch.no_grad()
    def _wq_contrast(self, w_q: torch.Tensor) -> torch.Tensor:
        """p90/p10 ratio of the positive CE weights -- direct readout of
        whether the weighting has inter-patch contrast.  Pre-fix this was
        ~1.5-2; post-fix it should sit around 1e2-1e3."""
        vals = w_q[w_q > 0].float()
        if vals.numel() < 10:
            return torch.tensor(1.0, device=w_q.device)
        qs = torch.quantile(vals, torch.tensor([0.1, 0.9], device=vals.device))
        return qs[1] / qs[0].clamp_min(1e-9)

    def _log_figure(self, fig, title: str, step: Optional[int] = None):
        """Log a matplotlib figure as an image – compatible with WandB and TensorBoard."""
        if step is None:
            step = self.global_step

        # Save to an in-memory PNG buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        # Read back as RGB numpy array
        img = Image.open(buf).convert('RGB')
        img_array = np.array(img)

        # Use Lightning's unified log_image (works for both WandbLogger and TensorBoardLogger)
        if hasattr(self.logger, "log_image"):
            self.logger.log_image(title, [img_array], step=step)
        else:
            # Fallback for older loggers
            if hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "add_figure"):
                self.logger.experiment.add_figure(title, fig, global_step=step)

        plt.close(fig)

    def _fbank_to_patches(self, fbank: torch.Tensor) -> torch.Tensor:
        return self._patchify(fbank.float())

    def _patches_to_img(self, patches: torch.Tensor) -> np.ndarray:
        fs, ts = self.fshape, self.tshape
        pf, pt = self.p_f_dim, self.p_t_dim
        B, P, C, _, _ = patches.shape
        x = patches.reshape(B, pf, pt, C, fs, ts)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.reshape(B, C, pf * fs, pt * ts)
        x = x.transpose(2, 3)
        return x.cpu().float().numpy()

    def _log_spectrogram(self, fbank: torch.Tensor, title: str, loss=None):
        with torch.no_grad():
            patches = self._fbank_to_patches(fbank[:1].float())
        img = self._patches_to_img(patches)[0]
        caption = f"Loss: {loss:.4f}" if loss is not None else title
        fig = plot_fbank(img, vmin=img.min(), vmax=img.max(), title=caption)
        self._log_figure(fig, title)
        plt.close(fig)

    def _add_mask_overlay(self, ax, visible_mask, scale_f=1, scale_t=1):
        pf, pt = self.p_f_dim, self.p_t_dim
        vis = visible_mask[0].cpu().view(pf, pt).numpy()
        for r in range(pf):
            for c in range(pt):
                if not vis[r, c]:
                    ax.add_patch(plt.Rectangle(
                        (c * scale_t - 0.5, r * scale_f - 0.5), scale_t, scale_f,
                        linewidth=0.6, edgecolor="cyan", facecolor="none"))

    def _log_q_angular_error(self, q_logits, q_tgt, visible_mask, title):
        with torch.no_grad():
            ang = self._q_angular_error_deg(q_logits[:1], q_tgt[:1]).squeeze(0)
        pf, pt = self.p_f_dim, self.p_t_dim
        grid = ang.view(pf, pt).cpu().float().numpy()
        fig, ax = plt.subplots(figsize=(max(4, pt // 2), max(3, pf // 2)))
        im = ax.imshow(grid, aspect="auto", origin="lower", vmin=0, vmax=90,
                       cmap="plasma")
        fig.colorbar(im, ax=ax, label="Angular error (deg)")
        self._add_mask_overlay(ax, visible_mask)
        ax.set_title(f"{title}  (cyan = masked)")
        ax.set_xlabel("Time patch"); ax.set_ylabel("Freq patch")
        self._log_figure(fig, title)
        plt.close(fig)

    def _log_diffuseness_heatmap(self, diff_pred, diff_target, visible_mask,
                                 title, channel: int = 0):
        with torch.no_grad():
            error = (diff_pred[:1, :, channel] - diff_target[:1, :, channel]).abs()
            error_per_patch = error.mean(dim=(-2, -1)).squeeze(0)
            pf, pt = self.p_f_dim, self.p_t_dim
            grid = error_per_patch.view(pf, pt).cpu().float().numpy()
        fig, ax = plt.subplots(figsize=(max(4, pt // 2), max(3, pf // 2)))
        im = ax.imshow(grid, aspect="auto", origin="lower", vmin=0, cmap="hot")
        fig.colorbar(im, ax=ax, label=f"L1 error (psi ch{channel})")
        self._add_mask_overlay(ax, visible_mask)
        ax.set_title(f"{title}  (cyan = masked)")
        ax.set_xlabel("Time patch"); ax.set_ylabel("Freq patch")
        self._log_figure(fig, title)
        plt.close(fig)

    def _log_leveldiff_reconstruction(self, lvld_pred, lvld_target,
                                      visible_mask, title):
        with torch.no_grad():
            pred_ch0 = lvld_pred[:1, :, 0:1]
            tgt_ch0 = lvld_target[:1, :, 0:1]
            pf, pt = self.p_f_dim, self.p_t_dim
            fs, ts = self.patch_shape

            def patches_to_image(patches):
                x = patches.view(1, pf, pt, 1, fs, ts) \
                    .permute(0, 3, 1, 4, 2, 5).contiguous()
                return x.view(pf * fs, pt * ts).cpu().float().numpy()

            img_pred = patches_to_image(pred_ch0)
            img_tgt = patches_to_image(tgt_ch0)
            error = np.abs(img_pred - img_tgt)

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        vmin = min(img_pred.min(), img_tgt.min())
        vmax = max(img_pred.max(), img_tgt.max())
        for ax, img, name, kw in (
                (axs[0], img_pred, "Predicted leveldiff (Y-W)",
                 dict(vmin=vmin, vmax=vmax, cmap="viridis")),
                (axs[1], img_tgt, "Target leveldiff (Y-W)",
                 dict(vmin=vmin, vmax=vmax, cmap="viridis")),
                (axs[2], error, "Absolute error", dict(cmap="hot"))):
            im = ax.imshow(img, aspect="auto", origin="lower", **kw)
            ax.set_title(name)
            ax.set_xlabel("Time patch"); ax.set_ylabel("Freq patch")
            fig.colorbar(im, ax=ax)
        self._add_mask_overlay(axs[2], visible_mask, scale_f=fs, scale_t=ts)
        plt.suptitle(title)
        self._log_figure(fig, title)
        plt.close(fig)

    def _log_cross_attn_norms(self, cross_attn_norms, title="cross_attn_norm"):
        layers = list(range(len(cross_attn_norms)))
        fig, ax = plt.subplots()
        ax.bar(layers, cross_attn_norms, color='steelblue')
        ax.set_xlabel("Decoder layer"); ax.set_ylabel("Mean L2 norm")
        ax.set_title(title)
        ax.set_xticks(layers)
        self._log_figure(fig, title)
        plt.close(fig)

    # ----- Training step -----
    def training_step(self, batch, batch_idx):
        fb7, intensity_mel, energy_mel, leveldiff, visible_mask = batch

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = self.forward(fb7, intensity_mel, energy_mel,
                               leveldiff, visible_mask)

        loss = out["loss"]

        with torch.no_grad():
            ang = self._q_angular_error_deg(out["q_logits"], out["q_tgt"])
            # Metric weighting now matches the loss weighting (log1p(Wp_n)
            # on masked patches) instead of raw linear Wp, which was
            # dominated by the loudest few patches.  Expect a level shift
            # vs pre-fix runs: metric redefinition, not a regression.
            w = out["w_q"]
            ang_err_masked = (ang * w).sum() / w.sum().clamp_min(1e-6)
            psi_means = out["diff_target"].mean(dim=(0, 1, 3, 4))
            wq_contrast = self._wq_contrast(out["w_q"])

        self.log_dict({
            "loss": loss,
            "l_q": out["l_q"],
            "kl_q": out["kl_q"],
            "l_diff": out["l_diff"],
            "l_leveldiff": out["l_leveldiff"],
            "l_level": out["l_level"],
            "ang_err_masked_deg": ang_err_masked,
        }, prog_bar=True)
        self.log("targets/wp_mean", out["Wp"].mean())
        self.log("targets/wp_n_mean", out["Wp_n"].mean())
        self.log("targets/wp_log_ref", self.wp_log_ref.squeeze())
        self.log("targets/wq_contrast_p90_p10", wq_contrast)
        self.log("targets/psi_short_mean", psi_means[0])
        self.log("targets/psi_long_mean", psi_means[1])
        for i, norm_val in enumerate(out["cross_attn_norms"]):
            self.log(f"cross_attn_norm/layer_{i}", norm_val)

        if self.global_step % self.log_every_n_steps == 0:
            self._log_spectrogram(fb7[:, 1:2], title="spec/y_input",
                                  loss=loss.item())
            self._log_q_angular_error(
                out["q_logits"], out["q_tgt"], visible_mask,
                title="q/angular_error_heatmap")
            self._log_diffuseness_heatmap(
                out["diff_pred"], out["diff_target"], visible_mask,
                title="diffuseness_error_heatmap_short", channel=0)
            self._log_leveldiff_reconstruction(
                out["leveldiff_pred"], out["leveldiff_target"], visible_mask,
                title="leveldiff_reconstruction")
            self._log_cross_attn_norms(
                out["cross_attn_norms"], title="cross_attn_norms_per_layer")

        return loss

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, self.lr, weight_decay=self.weight_decay,
            betas=(self.b1, self.b2),
        )
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.max_steps,
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    # ----- Inference / fine-tuning interface (unchanged) -----
    def pass_through_encoder(self, x_logmel4: torch.Tensor,
                             x_iv: torch.Tensor) -> torch.Tensor:
        B = x_logmel4.shape[0]
        x_7ch = torch.cat([x_logmel4, x_iv], dim=1)
        x = x_7ch.transpose(2, 3)
        embedded = self.patch_embed(x)                              # (B, P, D)
        embedded = embedded + self.pos_embed[:, 1:, :]
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        x = torch.cat([cls, embedded], dim=1)
        x = self.encoder_dropout(x)
        for block in self.encoder_blocks:
            x = block(x)
        return self.encoder_norm(x)

    def get_audio_representation(self, x_logmel4, x_iv, strategy: str = "mean"):
        z = self.pass_through_encoder(x_logmel4, x_iv)
        if strategy == "mean":
            return z[:, 1:, :].mean(axis=1)
        if strategy == "cls":
            return z[:, 0, :]
        if strategy == "raw":
            f = self.grid_size[0]
            return rearrange(z[:, 1:, :], "b (f t) d -> b t (f d)",
                             f=f, d=self.encoder_embedding_dim)

        raise ValueError(f"Unknown strategy '{strategy}'")