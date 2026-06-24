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
from ..data_modules.dataset_functions import pad_or_truncate_batch
from .ambisonic_feature_extractor import FeatureExtractor

set_fused_attn(True)
use_fused_attn(True)

pad_or_truncate_batch = torch.compile(pad_or_truncate_batch)


def masked_mse(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """Unweighted MSE over masked patches."""
    m = mask.float().view(*mask.shape, 1, 1, 1)
    diff = (pred - target).pow(2) * m
    denom = m.sum() * pred.shape[-3] * pred.shape[-2] * pred.shape[-1]
    return diff.sum() / denom.clamp_min(1e-6)


def masked_l1(pred: torch.Tensor, target: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    """Unweighted L1 over masked patches."""
    m = mask.float().view(*mask.shape, 1, 1, 1)
    diff = (pred - target).abs() * m
    denom = m.sum() * pred.shape[-3] * pred.shape[-2] * pred.shape[-1]
    return diff.sum() / denom.clamp_min(1e-6)


def cosine_dir_loss(pred, target_unit, weight):
    """Magnitude-weighted angular loss (1 - cos) on unit-vector predictions."""
    pred = pred.float()
    target_unit = target_unit.float()
    weight = weight.float()
    cos = (pred * target_unit).sum(dim=-3, keepdim=True).clamp(-1.0, 1.0)
    per_bin = (1.0 - cos) * weight
    return per_bin.sum() / weight.sum().clamp_min(1e-6)


# =============================================================================
# Building blocks (unchanged)
# =============================================================================

class InterChannelBlock(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int,
                 num_heads: int = 4, num_layers: int = 1,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.channel_emb = nn.Parameter(torch.zeros(1, 1, in_channels, embed_dim))
        nn.init.trunc_normal_(self.channel_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(mlp_ratio * embed_dim),
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.attn = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, P, C, D = x.shape
        x = x + self.channel_emb
        x = x.view(B * P, C, D)
        x = self.attn(x)
        x = x.view(B, P, C, D).mean(dim=2)
        return x


class SpatialFrontEnd(nn.Module):
    # ... (identical to original) ...
    def __init__(self, in_channels: int, embed_dim: int,
                 fshape: int, tshape: int, fstride: int, tstride: int,
                 inter_channel_heads: int = 4,
                 inter_channel_layers: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_channels, in_channels * embed_dim,
            kernel_size=(fshape, tshape),
            stride=(fstride, tstride),
            groups=in_channels,
        )
        self.inter_channel = InterChannelBlock(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_heads=inter_channel_heads,
            num_layers=inter_channel_layers,
        )
        self.num_patch = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.proj(x)
        Nf, Nt = x.size(-2), x.size(-1)
        x = x.view(B, self.in_channels, self.embed_dim, Nf, Nt)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        x = x.view(B, Nf * Nt, self.in_channels, self.embed_dim)
        x = self.inter_channel(x)
        return x


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
        # store last norm for logging
        self.last_cross_attn_norm = 0.0

    def forward(self, x: torch.Tensor,
                context: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        if context is not None:
            q = self.norm_q(x)
            kv = self.norm_kv(context)
            delta = self.cross_attn(q, kv, kv, need_weights=False)[0]
            # mean L2 norm over all tokens and batch
            self.last_cross_attn_norm = delta.norm(p=2, dim=-1).mean().item()
            x = x + delta
        else:
            self.last_cross_attn_norm = 0.0
        x = x + self.mlp(self.norm2(x))
        return x

class SemanticDecoder(nn.Module):
    # ... (identical to original) ...
    def __init__(self, encoder_embed_dim: int, decoder_embed_dim: int,
                 depth: int, num_heads: int, patch_shape: tuple,
                 num_patches: int, grid_size: tuple, mlp_ratio: float = 4.0):
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
        self.dir_head  = nn.Linear(decoder_embed_dim, 3 * fs * ts)
        self.diff_head = nn.Linear(decoder_embed_dim, 1 * fs * ts)
        self.energy_head = nn.Linear(decoder_embed_dim, 3 * fs * ts)

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
        dir_raw = self.dir_head(x).view(N, P, 3, fs, ts).float()
        dir_pred = F.normalize(dir_raw, p=2, dim=2, eps=1e-6)
        diff_pred = self.diff_head(x).view(N, P, 1, fs, ts)
        energy_pred = self.energy_head(x).view(N, P, 3, fs, ts)
        return dir_pred, diff_pred, energy_pred, cross_attn_norms


# =============================================================================
# Lightning module – now with integrated GRAM‑T
# =============================================================================

class SphereV4(pl.LightningModule):
    """Semantically-conditioned spatial MAE for first-order Ambisonics.

    Encoder input: 7 channels [W, Y, Z, X] log-mel + AIV.
    Decoder cross-attends on frozen mono tokens from either:
      - an integrated GRAM‑T model (specified by `gramt_model_id`), OR
      - an externally provided `mono_encoder` (for backward compatibility).
    Targets at masked positions: direction (unit AIV), diffuseness, std. YZX.
    """
    def __init__(
            self,
            encoder_embedding_dim: int = 384,
            encoder_depth: int = 6,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,

            decoder_depth: int = 4,
            decoder_num_heads: int = 8,
            decoder_embedding_dim: int = 256,

            # Loss weights (paper): L = L_dir + 0.5 L_Psi + 0.5 L_E
            dir_loss_weight: float = 1.0,
            diffuseness_loss_weight: float = 0.5,
            energy_loss_weight: float = 0.5,

            lr: float = 2e-4,
            b1: float = 0.9, b2: float = 0.95,
            weight_decay: float = 0.01,
            warmup_steps: int = 10000,

            patch_strategy: PatchStrategy = None,
            in_channels: int = 7,
            sr: int = 32000,
            num_mel_bins: int = 128,
            input_length: int = 500,
            target_length: int = 200,

            inter_channel_heads: int = 4,
            inter_channel_layers: int = 1,

            # ---- GRAM‑T / generic mono conditioning ----
            gramt_model_id: Optional[str] = "labhamlet/gramt-mono",   # e.g. "labhamlet/gramt-mono"

            # ---- diffuseness ----
            diffuseness_tau: int = 11,
            diffuseness_c: float = 1.0,

            log_every_n_steps: int = 500,

            **kwargs,
        ):
        super().__init__()
        self.save_hyperparameters(ignore=['patch_strategy'])

        assert in_channels == 7, \
            "Embisonics expects 7 channels: W, Y, Z, X log-mel + IVx, IVy, IVz"
        assert diffuseness_tau % 2 == 1, "diffuseness_tau must be odd"

        self.sr = sr
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.input_length = input_length
        self.in_channels = in_channels
        self.lr = lr; self.b1 = b1; self.b2 = b2
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.dir_loss_weight = dir_loss_weight
        self.diffuseness_loss_weight = diffuseness_loss_weight
        self.energy_loss_weight = energy_loss_weight
        self.diffuseness_tau = diffuseness_tau
        self.diffuseness_c = diffuseness_c
        self.log_every_n_steps = log_every_n_steps
        self.encoder_embedding_dim = encoder_embedding_dim

        self.patch_strategy = patch_strategy
        self.p_f_dim, self.p_t_dim = self.patch_strategy.get_patch_size()
        self.num_patches = self.p_f_dim * self.p_t_dim
        self.grid_size = (self.p_f_dim, self.p_t_dim)
        self.fshape, self.tshape = self.patch_strategy.fshape, self.patch_strategy.tshape
        self.patch_shape = (self.fshape, self.tshape)

        # ----- Patch embed + inter-channel fusion (7 channels) ----------
        self.patch_embed = SpatialFrontEnd(
            in_channels=self.in_channels,
            embed_dim=self.encoder_embedding_dim,
            fshape=self.fshape, tshape=self.tshape,
            fstride=self.patch_strategy.fstride,
            tstride=self.patch_strategy.tstride,
            inter_channel_heads=inter_channel_heads,
            inter_channel_layers=inter_channel_layers,
        )
        self.patch_embed.num_patch = self.num_patches

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

        # ----- Plain ViT encoder (self-attention only) -----------------
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

        # ----- Semantic conditioning (GRAM‑T or external) --------------
        # --- GRAM‑T integration (replaces generic mono_encoder) ---
        self.gram = None
        self.gramt_proj = None
        self.gramt_pos_embed = None

        if gramt_model_id is not None:
            # Load frozen GRAM‑T
            self.gram = transformers.AutoModel.from_pretrained(
                gramt_model_id, trust_remote_code=True
            )
            for p in self.gram.parameters():
                p.requires_grad_(False)
            self.gram.eval()

            # Determine native dim and verify grid compatibility
            with torch.no_grad():
                dummy_w = torch.zeros(1, 1, target_length, num_mel_bins)
                dummy_out = self.gram(dummy_w, strategy="raw")
            T_gram, flat_dim = dummy_out.shape[1], dummy_out.shape[2]
            self.gramt_n_freq = self.p_f_dim   # frequency patches
            assert flat_dim % self.gramt_n_freq == 0, \
                f"GRAMT raw dim {flat_dim} not divisible by p_f_dim={self.gramt_n_freq}"
            self.gramt_native_dim = flat_dim // self.gramt_n_freq
            self.gramt_t_dim = T_gram
            assert T_gram * self.gramt_n_freq == self.num_patches, (
                f"GRAMT token grid {T_gram}x{self.gramt_n_freq} != "
                f"spatial patches {self.num_patches}"
            )

            # Project GRAM‑T tokens into decoder dimension
            self.gramt_proj = nn.Linear(
                self.gramt_native_dim, decoder_embedding_dim
            )

            # Dedicated 2D position embedding for the context sequence
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

        # ----- Decoder --------------------------------------------------
        self.decoder = SemanticDecoder(
            encoder_embed_dim=self.encoder_embedding_dim,
            decoder_embed_dim=decoder_embedding_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            patch_shape=self.patch_shape,
            num_patches=self.num_patches,
            grid_size=self.grid_size,
            mlp_ratio=4.0,
        )

        # ----- Mel feature extractor -----------------------------------
        self.melspec = FeatureExtractor(
            sample_rate=self.sr, n_fft=1024, win_length=1024,
            hop_length=self.sr // 100, f_min=50, f_max=self.sr // 2,
            n_mels=self.num_mel_bins, power=2.0,
        ).float()

        self._init_our_weights()

    def _init_our_weights(self):
        def init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Exclude frozen gram/mono_encoder
        for mod in [self.patch_embed, self.encoder_blocks,
                    self.encoder_norm, self.decoder]:
            if isinstance(mod, nn.Module):
                mod.apply(init_fn)
        if self.gram is not None:
            if isinstance(self.gramt_proj, nn.Module):
                self.gramt_proj.apply(init_fn)
        else:
            if isinstance(self.mono_proj, nn.Module):
                self.mono_proj.apply(init_fn)

        self.register_buffer("energy_running_mean", torch.zeros(1))
        self.register_buffer("energy_running_var",  torch.ones(1))
        self.register_buffer("energy_stats_inited", torch.zeros(1, dtype=torch.bool))
        self.energy_norm_momentum = 0.01

    def on_fit_start(self):
        if self.gram is not None:
            self.gram.eval()

    # ----- Patchification helpers (unchanged) -----------------------
    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        fs, ts = self.fshape, self.tshape
        pf, pt = self.p_f_dim, self.p_t_dim
        B, C, T, Fm = x.shape
        x = x.transpose(2, 3)
        x = x.reshape(B, C, pf, fs, pt, ts)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.reshape(B, pf * pt, C, fs, ts)

    def _dir_target(self, aiv: torch.Tensor):
        mag = torch.linalg.norm(aiv, dim=1, keepdim=True)
        unit = aiv / mag.clamp_min(1e-6)
        weight = mag.clamp(0.0, 1.0)
        unit_p = self._patchify(unit)
        weight_p = self._patchify(weight)
        return unit_p, weight_p

    def _energy_target(self, yzx_logmel: torch.Tensor) -> torch.Tensor:
        e = self._patchify(yzx_logmel).float()           # (N, P, 3, fs, ts)
        if self.training:
            with torch.no_grad():
                batch_mean = e.mean()
                batch_var  = e.var(unbiased=False)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(batch_mean, op=torch.distributed.ReduceOp.AVG)
                    torch.distributed.all_reduce(batch_var,  op=torch.distributed.ReduceOp.AVG)
                if not bool(self.energy_stats_inited):
                    self.energy_running_mean.fill_(batch_mean.item())
                    self.energy_running_var.fill_(batch_var.item())
                    self.energy_stats_inited.fill_(True)
                else:
                    m = self.energy_norm_momentum
                    self.energy_running_mean.mul_(1 - m).add_(m * batch_mean)
                    self.energy_running_var.mul_(1 - m).add_(m * batch_var)
        mu = self.energy_running_mean
        sd = self.energy_running_var.clamp_min(1e-10).sqrt().clamp_min(1e-5)
        return (e - mu) / sd

    def _temporal_average(self, x: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 1:
            return x
        B, C, T, Fm = x.shape
        xx = x.permute(0, 1, 3, 2).reshape(B * C * Fm, 1, T)
        xx = F.avg_pool1d(xx, kernel_size=k, stride=1, padding=k // 2,
                          count_include_pad=False)
        xx = xx[..., :T]
        return xx.reshape(B, C, Fm, T).permute(0, 1, 3, 2)

    def _diffuseness_target(self, intensity_mel: torch.Tensor,
                            energy_mel: torch.Tensor) -> torch.Tensor:
        I = intensity_mel.float()
        E = energy_mel.float()
        I_avg = self._temporal_average(I, self.diffuseness_tau)
        E_avg = self._temporal_average(E, self.diffuseness_tau)
        num = torch.linalg.norm(I_avg, dim=1, keepdim=True)
        psi = 1.0 - num / (self.diffuseness_c * E_avg + 1e-6)
        return psi.clamp(0.0, 1.0)

    def _extract_features(self, waveform: torch.Tensor) -> dict:
        with torch.amp.autocast('cuda', enabled=False):
            waveform = waveform.float()
            rms = torch.sqrt(waveform[:, :1].pow(2).mean(dim=-1, keepdim=True) + 1e-8)
            waveform = waveform / rms
            return self.melspec.extract_all(waveform)

    # ----- Semantic context (GRAM‑T or external) -------------------
    def _mono_context(self, w_logmel: torch.Tensor) -> Optional[torch.Tensor]:
        """
        w_logmel: (B, 1, T, F)
        Returns context sequence (B, N_ctx, decoder_embed_dim) or None.
        """
        if self.gram is not None:
            # --- GRAM‑T path ---
            with torch.no_grad():
                self.gram.eval()
                # GRAM‑T expects (B,C, T, F)
                tokens = self.gram(w_logmel, strategy="raw") # (B, T_gram, F_gram*D)
            B, T_gram, FD = tokens.shape
            F_gram, D = self.gramt_n_freq, self.gramt_native_dim
            # Reshape to (B, T_gram, F_gram, D) -> (B, F_gram, T_gram, D)
            tokens = tokens.view(B, T_gram, F_gram, D).permute(0, 2, 1, 3).contiguous()
            tokens = tokens.reshape(B, F_gram * T_gram, D)  # (B, num_patches, D)
            tokens = self.gramt_proj(tokens)                 # (B, num_patches, decoder_dim)
            tokens = tokens + self.gramt_pos_embed
            return tokens.float()
        else:
            return None

    # ----- Encoder (unchanged) ------------------------------------
    def _encode_visible(self, x_7ch: torch.Tensor,
                        visible_mask: torch.Tensor) -> torch.Tensor:
        B = x_7ch.shape[0]
        x = x_7ch.transpose(2, 3)
        embedded = self.patch_strategy.embed(x, self.patch_embed)
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
    def _prepare_batch(self, batch):
        audio, context_idx, _ = batch   # R_mats ignored
        audio = audio.to(self.device, non_blocking=True)
        context_idx = context_idx.to(self.device, non_blocking=True)

        feats = self._extract_features(audio)
        stacked = torch.cat([feats["logmel"], feats["aiv"],
                             feats["intensity_mel"], feats["energy_mel"]],
                            dim=1)

        N, C, T_total, n_freq = stacked.shape
        T_crop = self.target_length
        max_start = T_total - T_crop
        offsets = torch.randint(0, max_start + 1, (N,), device=self.device)
        t_range = torch.arange(T_crop, device=self.device)
        time_idx = (offsets.view(N, 1, 1, 1) + t_range.view(1, 1, T_crop, 1)) \
            .expand(N, C, T_crop, n_freq)
        stacked = torch.gather(stacked, dim=2, index=time_idx)

        fb7 = stacked[:, 0:7]
        intensity_mel = stacked[:, 7:10]
        energy_mel = stacked[:, 10:11]

        visible_mask = torch.zeros(N, self.num_patches, dtype=torch.bool, device=self.device)
        visible_mask.scatter_(1, context_idx.long(), True)
        assert visible_mask[0].float().mean() <= 0.2 + 1e-6, \
            "Masking ratio rho must be >= 0.8 (visible patches <= 20%)"

        return (fb7.to(torch.bfloat16),
                intensity_mel.float(),
                energy_mel.float(),
                visible_mask)

    # ----- Forward (unchanged) ------------------------------------
    def forward(self, fb7, intensity_mel, energy_mel, visible_mask):
        B = fb7.shape[0]
        w_logmel = fb7[:, 0:1]
        yzx_logmel = fb7[:, 1:4]
        aiv = fb7[:, 4:7]

        z = self._encode_visible(fb7, visible_mask)
        z_vis = z[:, 1:, :]

        mono_ctx = self._mono_context(w_logmel)

        dir_pred, diff_pred, energy_pred, cross_attn_norms = self.decoder(
            z_vis, visible_mask, mono_ctx
        )
        dir_unit_tgt, dir_weight = self._dir_target(aiv)
        diff_tgt = self._patchify(self._diffuseness_target(intensity_mel, energy_mel))
        energy_tgt = self._energy_target(yzx_logmel)

        masked = ~visible_mask
        masked_b = masked.float().view(B, self.num_patches, 1, 1, 1)
        dir_weight_m = dir_weight * masked_b

        l_dir = cosine_dir_loss(dir_pred, dir_unit_tgt, dir_weight_m)
        l_diff = masked_l1(diff_pred, diff_tgt, masked)
        l_energy = masked_mse(energy_pred, energy_tgt, masked)

        total = (self.dir_loss_weight * l_dir
                 + self.diffuseness_loss_weight * l_diff
                 + self.energy_loss_weight * l_energy)

        return {
            "loss": total,
            "l_dir": l_dir,
            "l_diff": l_diff,
            "l_energy": l_energy,
            "dir_pred": dir_pred,
            "dir_target": dir_unit_tgt,
            "dir_weight_target": dir_weight,
            "diff_pred": diff_pred,
            "diff_target": diff_tgt,
            "energy_pred": energy_pred,
            "energy_target": energy_tgt,
            "masked": masked,
            "z_vis": z_vis,
            "cross_attn_norms": cross_attn_norms,   # list of floats, one per layer
        }


    # ----- Logging helpers (unchanged) ----------------------------
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
        self.logger.experiment.add_figure(title, fig, global_step=self.global_step)
        plt.close(fig)

    def _log_iv_angular_error(self, dir_pred: torch.Tensor,
                              dir_target: torch.Tensor,
                              visible_mask: torch.Tensor, title: str):
        with torch.no_grad():
            cos = (dir_pred[:1] * dir_target[:1]).sum(dim=2).clamp(-1.0, 1.0)
            ang = torch.acos(cos) * (180.0 / np.pi)
            ang_per_patch = ang.mean(dim=(-2, -1)).squeeze(0).cpu().float()
        pf, pt = self.p_f_dim, self.p_t_dim
        grid = ang_per_patch.view(pf, pt).numpy()
        fig, ax = plt.subplots(figsize=(max(4, pt // 2), max(3, pf // 2)))
        im = ax.imshow(grid, aspect="auto", origin="lower", vmin=0, vmax=90, cmap="plasma")
        fig.colorbar(im, ax=ax, label="Angular error (deg)")
        vis = visible_mask[0].cpu().view(pf, pt).numpy()
        for r in range(pf):
            for c in range(pt):
                if not vis[r, c]:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                 linewidth=0.6, edgecolor="cyan", facecolor="none"))
        ax.set_title(f"{title}  (cyan = masked)")
        ax.set_xlabel("Time patch"); ax.set_ylabel("Freq patch")
        self.logger.experiment.add_figure(title, fig, global_step=self.global_step)
        plt.close(fig)

    def _log_diffuseness_heatmap(self, diff_pred, diff_target, visible_mask, title):
        """L1 error heatmap for diffuseness, masked patches highlighted."""
        with torch.no_grad():
            diff_pred = diff_pred[:1]   # take first sample
            diff_target = diff_target[:1]
            error = (diff_pred - diff_target).abs()
            # average over patch pixels (fs*ts)
            error_per_patch = error.mean(dim=(-2, -1)).squeeze(0).squeeze(0)  # (num_patches,)
            pf, pt = self.p_f_dim, self.p_t_dim
            grid = error_per_patch.view(pf, pt).cpu().float().numpy()
        fig, ax = plt.subplots(figsize=(max(4, pt // 2), max(3, pf // 2)))
        im = ax.imshow(grid, aspect="auto", origin="lower", vmin=0, cmap="hot")
        fig.colorbar(im, ax=ax, label="L1 error (diffuseness)")
        vis = visible_mask[0].cpu().view(pf, pt).numpy()
        for r in range(pf):
            for c in range(pt):
                if not vis[r, c]:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                 linewidth=0.6, edgecolor="cyan", facecolor="none"))
        ax.set_title(f"{title}  (cyan = masked)")
        ax.set_xlabel("Time patch"); ax.set_ylabel("Freq patch")
        self.logger.experiment.add_figure(title, fig, global_step=self.global_step)
        plt.close(fig)

    def _log_energy_reconstruction(self, energy_pred, energy_target, visible_mask, title):
        """Show target vs predicted energy (first channel) with error."""
        with torch.no_grad():
            # pick first channel (e.g. Y) and first sample
            pred_ch0 = energy_pred[:1, :, 0:1]      # (1, P, 1, fs, ts)
            tgt_ch0  = energy_target[:1, :, 0:1]
            # reshape to image: (C=1, H=pf*fs, W=pt*ts)
            pf, pt = self.p_f_dim, self.p_t_dim
            fs, ts = self.patch_shape
            def patches_to_image(patches):   # patches shape (1, P, 1, fs, ts)
                x = patches.view(1, pf, pt, 1, fs, ts).permute(0, 3, 1, 4, 2, 5).contiguous()
                x = x.view(1, 1, pf * fs, pt * ts).squeeze(0).squeeze(0)  # (H, W)
                return x.cpu().float().numpy()
            img_pred = patches_to_image(pred_ch0)
            img_tgt  = patches_to_image(tgt_ch0)
            error = np.abs(img_pred - img_tgt)

        # create side-by-side plot
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        vmin = min(img_pred.min(), img_tgt.min())
        vmax = max(img_pred.max(), img_tgt.max())
        im0 = axs[0].imshow(img_pred, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        axs[0].set_title("Predicted energy (ch 0)")
        axs[0].set_xlabel("Time patch"); axs[0].set_ylabel("Freq patch")
        fig.colorbar(im0, ax=axs[0])
        im1 = axs[1].imshow(img_tgt, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        axs[1].set_title("Target energy (ch 0)")
        axs[1].set_xlabel("Time patch"); axs[1].set_ylabel("Freq patch")
        fig.colorbar(im1, ax=axs[1])
        im2 = axs[2].imshow(error, aspect="auto", origin="lower", cmap="hot")
        axs[2].set_title("Absolute error")
        axs[2].set_xlabel("Time patch"); axs[2].set_ylabel("Freq patch")
        fig.colorbar(im2, ax=axs[2])
        # overlay masked regions on the error map
        vis = visible_mask[0].cpu().view(pf, pt).numpy()
        for r in range(pf):
            for c in range(pt):
                if not vis[r, c]:
                    axs[2].add_patch(plt.Rectangle((c * ts - 0.5, r * fs - 0.5), ts, fs,
                                     linewidth=0.6, edgecolor="cyan", facecolor="none"))
        plt.suptitle(title)
        self.logger.experiment.add_figure(title, fig, global_step=self.global_step)
        plt.close(fig)

    def _log_cross_attn_norms(self, cross_attn_norms, title="cross_attn_norm"):
        """Bar chart of cross-attention output norms per decoder layer."""
        layers = list(range(len(cross_attn_norms)))
        fig, ax = plt.subplots()
        ax.bar(layers, cross_attn_norms, color='steelblue')
        ax.set_xlabel("Decoder layer"); ax.set_ylabel("Mean L2 norm")
        ax.set_title(title)
        ax.set_xticks(layers)
        self.logger.experiment.add_figure(title, fig, global_step=self.global_step)
        plt.close(fig)
    # ----- Training step (unchanged) -----------------------------
    def training_step(self, batch, batch_idx):
        fb7, intensity_mel, energy_mel, visible_mask = self._prepare_batch(batch)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = self.forward(fb7, intensity_mel, energy_mel, visible_mask)

        loss = out["loss"]

        with torch.no_grad():
            w = out["dir_weight_target"] * out["masked"].view(
                *out["masked"].shape, 1, 1, 1).float()
            cos = (out["dir_pred"] * out["dir_target"]).sum(dim=2, keepdim=True).clamp(-1.0, 1.0)
            ang = torch.acos(cos) * (180.0 / np.pi)
            ang_err_masked = (ang * w).sum() / w.sum().clamp_min(1e-6)

        self.log_dict({
            "loss": loss,
            "l_dir": out["l_dir"],
            "l_diff": out["l_diff"],
            "l_energy": out["l_energy"],
            "ang_err_masked_deg": ang_err_masked,
        }, prog_bar=True)
        for i, norm_val in enumerate(out["cross_attn_norms"]):
            self.log(f"cross_attn_norm/layer_{i}", norm_val)

        if self.global_step % self.log_every_n_steps == 0:
            # existing spectrogram
            self._log_spectrogram(fb7[:, 1:2], title="spec/y_input", loss=loss.item())
            # existing angular error heatmap
            self._log_iv_angular_error(
                out["dir_pred"], out["dir_target"], visible_mask,
                title="iv/angular_error_heatmap",
            )
            # NEW: diffuseness error
            self._log_diffuseness_heatmap(
                out["diff_pred"], out["diff_target"], visible_mask,
                title="diffuseness_error_heatmap",
            )
            # NEW: energy reconstruction
            self._log_energy_reconstruction(
                out["energy_pred"], out["energy_target"], visible_mask,
                title="energy_reconstruction",
            )
            # NEW: cross-attention norm bar chart
            self._log_cross_attn_norms(
                out["cross_attn_norms"], title="cross_attn_norms_per_layer"
            )

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

    # ----- Inference / fine-tuning interface (unchanged) ---------
    def pass_through_encoder(self, x_logmel4: torch.Tensor,
                             x_iv: torch.Tensor) -> torch.Tensor:
        B = x_logmel4.shape[0]
        x_7ch = torch.cat([x_logmel4, x_iv], dim=1)
        x = x_7ch.transpose(2, 3)
        embedded = self.patch_strategy.embed(x, self.patch_embed)
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

