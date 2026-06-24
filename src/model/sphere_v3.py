"""
Spatial tokenizer pretraining: standalone self-supervised spatial encoder.

Key differences from SphereV3:
  - 6-channel input (YZX + IV); W is NOT used at all (no input, no target,
    no loss weight).
  - GRAM-T removed entirely (no semantic model in the loop).
  - Plain ViT encoder blocks (self-attention only, no cross-attention).
  - Reconstruction targets at masked positions:
      * YZX log-mel magnitudes (3 ch), unweighted MSE.
      * IV unit vectors (3 ch), cosine loss, magnitude-weighted by |IV|.
  - Inference interface takes (YZX, IV) — downstream fusion with any
    semantic backbone happens externally via late fusion at the task head.

Rationale: produces a semantic-backbone-agnostic spatial tokenizer whose
outputs can be combined with GRAM-T, BEATs, ATST, etc. via a lightweight
per-backbone adapter. YZX reconstruction preserves room-acoustic /
reverberation / diffuse-field information that pure IV reconstruction
would miss; predicting raw YZX (rather than W-normalized) and using
unweighted MSE (rather than |W|-weighted) keeps that signal intact at
the cost of mild semantic leakage. The IV cosine loss keeps its
|IV|-weighting because IV direction is degenerate at low |IV|; the YZX
MSE is unweighted because the YZX target is well-defined at all energies
and we explicitly want the encoder to model low-energy regions where
reverberation tails live.
"""

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

from timm.models.vision_transformer import Block
from timm.layers.config import set_fused_attn, use_fused_attn

from ..patching import PatchStrategy
from .pos_embed import get_2d_sincos_pos_embed
from .utils import plot_fbank
from ..data_modules.dataset_functions import pad_or_truncate_batch
from .ambisonic_feature_extractor import FeatureExtractor

set_fused_attn(True)
use_fused_attn(True)

pad_or_truncate_batch = torch.compile(pad_or_truncate_batch)


# =============================================================================
# Geometry / loss helpers
# =============================================================================

def rotate_foa_waveform(wav: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Rotate an ACN/SN3D-ordered FOA waveform.

    wav: (B, 4, T) with channels (W, Y, Z, X).
    R:   (B, 3, 3) rotation matrices in standard XYZ axes.
    """
    W, YZX = wav[:, :1], wav[:, 1:4]
    XYZ = YZX[:, [2, 0, 1]]
    XYZ_rot = torch.einsum("nij,njt->nit", R, XYZ)
    YZX_rot = XYZ_rot[:, [1, 2, 0]]
    return torch.cat([W, YZX_rot], dim=1)


def masked_yzx_mse(pred: torch.Tensor, target: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """Plain (unweighted) MSE on the 3-channel YZX reconstruction at masked
    patches only.

    pred, target: (B, P, 3, fs, ts) — log-mel magnitudes per channel.
    mask:         (B, P) bool — True where MASKED (loss is computed here).

    Unweighted by design: we want the encoder to model raw YZX structure
    everywhere, including low-energy regions where reverberation tails and
    diffuse-field information live. Weighting by |W| would push the encoder
    to focus on direct sound and ignore room acoustics — the opposite of
    why we chose raw YZX targets in the first place.
    """
    m = mask.float().view(*mask.shape, 1, 1, 1)               # (B, P, 1, 1, 1)
    diff = (pred - target).pow(2) * m
    # Normalize: number of masked patches * channels * patch elements
    denom = m.sum() * pred.shape[-3] * pred.shape[-2] * pred.shape[-1]
    return diff.sum() / denom.clamp_min(1e-6)


def cosine_iv_loss(pred, target_unit, weight):
    # Force fp32 for the cosine computation - bf16 precision is too low
    # for unit-vector inner products near 1.
    pred = pred.float()
    target_unit = target_unit.float()
    weight = weight.float()
    cos = (pred * target_unit).sum(dim=-3, keepdim=True).clamp(-1.0, 1.0)
    per_bin = (1.0 - cos) * weight
    return per_bin.sum() / weight.sum().clamp_min(1e-6)

# =============================================================================
# Building blocks
# =============================================================================

class InterChannelBlock(nn.Module):
    """At each TF location, fuse the C channel tokens via self-attention.

    Input:  (B, P, C, D)
    Output: (B, P, D)   (mean-pool over channels after SA)
    """
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
        # x: (B, P, C, D)
        B, P, C, D = x.shape
        x = x + self.channel_emb                            # broadcast over P
        x = x.view(B * P, C, D)
        x = self.attn(x)                                    # SA over channels
        x = x.view(B, P, C, D).mean(dim=2)                  # fuse: mean-pool
        return x


class SpatialFrontEnd(nn.Module):
    """Per-channel patch embed + inter-channel SA.

    Same shape as FOAFrontEndV3 but defaults to 6 channels (YZX + IV).

    Input:  (B, 6, F, T)
    Output: (B, num_patches, embed_dim)
    """
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
        self.num_patch = None  # filled in by the parent module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        B = x.size(0)
        x = self.proj(x)                                     # (B, C*D, Nf, Nt)
        Nf, Nt = x.size(-2), x.size(-1)
        x = x.view(B, self.in_channels, self.embed_dim, Nf, Nt)
        x = x.permute(0, 3, 4, 1, 2).contiguous()            # (B, Nf, Nt, C, D)
        x = x.view(B, Nf * Nt, self.in_channels, self.embed_dim)
        print(x.shape)
        x = self.inter_channel(x)                            # (B, P, D)
        return x


class SpatialReconstructionDecoder(nn.Module):
    """Two-head MAE decoder for the spatial tokenizer.

    Reconstructs:
      - YZX log-mel magnitudes (3 channels) at masked patches.
      - IV unit vectors (3 channels) at masked patches.
    """
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
            Block(decoder_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_embed_dim)

        fs, ts = patch_shape
        self.yzx_head = nn.Linear(decoder_embed_dim, 3 * fs * ts)
        self.iv_head  = nn.Linear(decoder_embed_dim, 3 * fs * ts)

    def forward(self, z_vis: torch.Tensor, visible_mask: torch.Tensor):
        N, P, D = z_vis.shape[0], self.num_patches, self.decoder_embed_dim
        z_vis_proj = self.decoder_embed(z_vis)
        x = self.mask_token.type_as(z_vis_proj).expand(N, P, D).clone()
        x[visible_mask] = z_vis_proj.reshape(-1, D)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        fs, ts = self.patch_shape
        yzx_pred = self.yzx_head(x).view(N, P, 3, fs, ts)
        iv_pred_raw = self.iv_head(x).view(N, P, 3, fs, ts).float()
        iv_pred = F.normalize(iv_pred_raw, p=2, dim=2, eps=1e-6)
        return yzx_pred, iv_pred


# =============================================================================
# Lightning module
# =============================================================================

class SphereV3(pl.LightningModule):
    """Self-supervised spatial tokenizer for first-order Ambisonics.

    Encoder input: 6 channels (YZX log-mel + IV).
    Pretraining targets at masked positions: YZX log-mel + IV unit vectors.
    No semantic-model context anywhere. W is used only as the loss weight
    for the YZX reconstruction (low-energy bins contribute less).
    """
    def __init__(
            self,
            encoder_embedding_dim: int = 384,
            encoder_depth: int = 6,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,

            decoder_depth: int = 8,
            decoder_num_heads: int = 6,
            decoder_embedding_dim: int = 384,

            yzx_loss_weight: float = 0.25,
            iv_loss_weight: float = 1.0,

            lr: float = 2e-4,
            b1: float = 0.9, b2: float = 0.95,
            weight_decay: float = 0.01,
            warmup_steps: int = 10000,

            patch_strategy: PatchStrategy = None,
            in_channels: int = 6,
            sr: int = 32000,
            num_mel_bins: int = 128,
            input_length: int = 500,
            target_length: int = 200,

            inter_channel_heads: int = 4,
            inter_channel_layers: int = 1,
            l_iv_weight=10,

            log_every_n_steps: int = 500,

            **kwargs,
        ):
        super().__init__()
        self.save_hyperparameters(ignore=['patch_strategy'])

        assert in_channels == 6, \
            "SphereTokenizer expects 6 channels: Y, Z, X, IVx, IVy, IVz"

        self.sr = sr
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.input_length = input_length
        self.in_channels = in_channels
        self.lr = lr; self.b1 = b1; self.b2 = b2
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.yzx_loss_weight = yzx_loss_weight
        self.iv_loss_weight = iv_loss_weight
        self.log_every_n_steps = log_every_n_steps
        self.encoder_embedding_dim = encoder_embedding_dim

        self.patch_strategy = patch_strategy
        self.p_f_dim, self.p_t_dim = self.patch_strategy.get_patch_size()
        self.num_patches = self.p_f_dim * self.p_t_dim
        self.grid_size = (self.p_f_dim, self.p_t_dim)
        self.fshape, self.tshape = self.patch_strategy.fshape, self.patch_strategy.tshape
        self.patch_shape = (self.fshape, self.tshape)

        # ----- Patch embed + inter-channel fusion (6 channels) ----------
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

        # ----- Plain ViT encoder (no cross-attention) -------------------
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

        # ----- Decoder --------------------------------------------------
        self.decoder = SpatialReconstructionDecoder(
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
        for mod in [self.patch_embed, self.encoder_blocks,
                    self.encoder_norm, self.decoder]:
            if isinstance(mod, nn.Module):
                mod.apply(init_fn)

    # ----- Patchification helpers ----------------------------------
    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T, F) -> (B, P, C, fs, ts).

        Freq-first patch ordering to match the patch_embed Conv2d / pos_embed
        layout. Identical to SphereV3's _patchify.
        """
        fs, ts = self.fshape, self.tshape
        pf, pt = self.p_f_dim, self.p_t_dim
        B, C, T, Fm = x.shape
        x = x.transpose(2, 3)                                # (B, C, F, T)
        x = x.reshape(B, C, pf, fs, pt, ts)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()         # (B, pf, pt, C, fs, ts)
        return x.reshape(B, pf * pt, C, fs, ts)

    def _yzx_target(self, yzx_fb: torch.Tensor) -> torch.Tensor:
        """(B, 3, T, F) -> (B, P, 3, fs, ts).  Raw log-mel patches."""
        return self._patchify(yzx_fb)

    def _iv_target(self, iv: torch.Tensor):
        """IV unit vectors + |IV|-derived weight, both patchified."""
        mag = torch.linalg.norm(iv, dim=1, keepdim=True)
        iv_unit = iv / mag.clamp_min(1e-6)
        weight = mag.clamp(0.0, 1.0)
        iv_unit_p = self._patchify(iv_unit)                  # (B, P, 3, fs, ts)
        weight_p = self._patchify(weight)                    # (B, P, 1, fs, ts)
        return iv_unit_p, weight_p

    # ----- waveform -> mel ----------------------------------------
    def _wav2fbank(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            waveform = waveform.float()
            rms = torch.sqrt(waveform[:, :1].pow(2).mean(dim=-1, keepdim=True) + 1e-8)
            waveform = waveform / rms
            mel = self.melspec(waveform)
            return mel.transpose(3, 2)                        # (B, C, T, F)

    # ----- Encoder ------------------------------------------------
    def _encode_visible(self, x_6ch: torch.Tensor,
                        visible_mask: torch.Tensor) -> torch.Tensor:
        """x_6ch: (B, 6, T, F).  visible_mask: (B, P) bool."""
        B = x_6ch.shape[0]
        x = x_6ch.transpose(2, 3)                              # (B, 6, F, T)
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

    # ----- Batch prep (rotation aug + crop + mask) ----------------
    @torch.no_grad()
    def _prepare_batch(self, batch):
        """Same waveform-level rotation + crop as SphereV3.

        Returns: (yzx_fb, iv_fb, visible_mask).
        W is dropped entirely — the tokenizer doesn't use it as input,
        target, or loss weight.
        """
        audio, context_idx, R_mats = batch
        audio = audio.to(self.device, non_blocking=True)         # (B, 4, T_samples)
        context_idx = context_idx.to(self.device, non_blocking=True)
        R_mats = R_mats.to(self.device, non_blocking=True)       # (B, Rn, 3, 3)

        B = audio.shape[0]
        Rn = R_mats.shape[1]

        # One random rotation per sample, applied at the waveform level.
        r_idx = torch.randint(0, Rn, (B,), device=self.device)
        R_pick = R_mats[torch.arange(B, device=self.device), r_idx].to(audio.dtype)
        rot_audio = rotate_foa_waveform(audio, R_pick)
        print(rot_audio.shape)


        fb = self._wav2fbank(rot_audio)

        # Temporal crop.
        N, C, T_total, n_freq = fb.shape
        T_crop = self.target_length
        max_start = T_total - T_crop
        offsets = torch.randint(0, max_start + 1, (N,), device=self.device)
        t_range = torch.arange(T_crop, device=self.device)
        time_idx = (offsets.view(N, 1, 1, 1) + t_range.view(1, 1, T_crop, 1)) \
            .expand(N, C, T_crop, n_freq)
        fb = torch.gather(fb, dim=2, index=time_idx)

        # We extract only YZX and IV; W is dropped here because the
        # tokenizer never uses it (no input, no target, no loss weight).
        yzx_fb = fb[:, 1:4]
        iv_fb  = fb[:, 4:7]

        visible_mask = torch.zeros(B, self.num_patches, dtype=torch.bool,
                                   device=self.device)
        visible_mask.scatter_(1, context_idx.long(), True)


        return (yzx_fb.to(torch.bfloat16),
                iv_fb.to(torch.bfloat16),
                visible_mask)

    # ----- Forward -------------------------------------------------
    def forward(self, yzx_fb, iv_fb, visible_mask):
        B = yzx_fb.shape[0]

        # Encoder sees only YZX + IV.
        x_6ch = torch.cat([yzx_fb, iv_fb], dim=1)                  # (B, 6, T, F)
        z = self._encode_visible(x_6ch, visible_mask)
        z_vis = z[:, 1:, :]

        yzx_pred, iv_pred = self.decoder(z_vis, visible_mask)

        # ----- Targets -------------------------------------------------
        yzx_tgt = self._yzx_target(yzx_fb)                          # (B, P, 3, fs, ts)
        iv_unit_tgt, iv_weight = self._iv_target(iv_fb)             # (B, P, 3/1, fs, ts)

        masked = ~visible_mask                                       # (B, P)
        masked_b = masked.float().view(B, self.num_patches, 1, 1, 1)
        iv_weight_m = iv_weight * masked_b

        # YZX: plain unweighted MSE on masked patches (preserves reverb / room
        # information that |W|-weighting would suppress).
        # IV : magnitude-weighted cosine on masked patches (|IV| weighting is
        # appropriate because IV direction is degenerate at low |IV|).
        l_yzx = masked_yzx_mse(yzx_pred, yzx_tgt, masked)
        l_iv  = cosine_iv_loss(iv_pred, iv_unit_tgt, iv_weight_m)

        total = self.yzx_loss_weight * l_yzx + self.iv_loss_weight * l_iv

        return {
            "loss": total,
            "l_yzx": l_yzx,
            "l_iv":  l_iv,
            "yzx_pred": yzx_pred,
            "iv_pred":  iv_pred,
            "yzx_target": yzx_tgt,
            "iv_target":  iv_unit_tgt,
            "iv_weight_target": iv_weight,
            "masked": masked,
            "z_vis":  z_vis,
        }

    # ----- Diagnostic / logging helpers ---------------------------
    def _fbank_to_patches(self, fbank: torch.Tensor) -> torch.Tensor:
        return self._patchify(fbank.float())                          # (B, P, C, fs, ts)

    def _patches_to_img(self, patches: torch.Tensor) -> np.ndarray:
        fs, ts = self.fshape, self.tshape
        pf, pt = self.p_f_dim, self.p_t_dim
        B, P, C, _, _ = patches.shape
        x = patches.reshape(B, pf, pt, C, fs, ts)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()                  # (B, C, pf, fs, pt, ts)
        x = x.reshape(B, C, pf * fs, pt * ts)                         # (B, C, F, T)
        x = x.transpose(2, 3)                                          # (B, C, T, F)
        return x.cpu().float().numpy()

    def _log_spectrogram(self, fbank: torch.Tensor, title: str, loss=None):
        with torch.no_grad():
            patches = self._fbank_to_patches(fbank[:1].float())
        img = self._patches_to_img(patches)[0]
        caption = f"Loss: {loss:.4f}" if loss is not None else title
        fig = plot_fbank(img, vmin=img.min(), vmax=img.max(), title=caption)
        self.logger.experiment.add_figure(title, fig, global_step=self.global_step)
        plt.close(fig)

    def _log_spectrogram_with_mask(self, fbank: torch.Tensor,
                                   visible_mask: torch.Tensor, title: str,
                                   show_visible: bool = True):
        with torch.no_grad():
            patches = self._fbank_to_patches(fbank[:1].float()).cpu()
        vis = visible_mask[0].cpu()
        zero = vis if not show_visible else ~vis
        patches[:, zero] = 0.0
        img = self._patches_to_img(patches)[0]
        fig = plot_fbank(img, vmin=img.min(), vmax=img.max(), title=title)
        self.logger.experiment.add_figure(title, fig, global_step=self.global_step)
        plt.close(fig)

    def _log_iv_angular_error(self, iv_pred: torch.Tensor,
                              iv_target: torch.Tensor,
                              visible_mask: torch.Tensor, title: str):
        with torch.no_grad():
            cos = (iv_pred[:1] * iv_target[:1]).sum(dim=2).clamp(-1.0, 1.0)
            ang = torch.acos(cos) * (180.0 / np.pi)
            ang_per_patch = ang.mean(dim=(-2, -1)).squeeze(0).cpu().float()
        pf, pt = self.p_f_dim, self.p_t_dim
        grid = ang_per_patch.view(pf, pt).numpy()

        fig, ax = plt.subplots(figsize=(max(4, pt // 2), max(3, pf // 2)))
        im = ax.imshow(grid, aspect="auto", origin="lower",
                       vmin=0, vmax=90, cmap="plasma")
        fig.colorbar(im, ax=ax, label="Angular error (°)")
        vis = visible_mask[0].cpu().view(pf, pt).numpy()
        for r in range(pf):
            for c in range(pt):
                if not vis[r, c]:
                    rect = plt.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        linewidth=0.6, edgecolor="cyan", facecolor="none"
                    )
                    ax.add_patch(rect)
        ax.set_title(f"{title}  (cyan = masked patches)")
        ax.set_xlabel("Time patch"); ax.set_ylabel("Freq patch")
        self.logger.experiment.add_figure(title, fig, global_step=self.global_step)
        plt.close(fig)

    def _log_yzx_reconstruction(self, yzx_fb: torch.Tensor,
                                yzx_pred: torch.Tensor,
                                visible_mask: torch.Tensor, title: str):
        """Stitch: visible patches from input YZX, masked patches from predicted.

        Visualizes channel 0 (Y) only, to keep the figure simple. Looking at
        Y alone is enough to confirm whether the reconstruction is sane.
        """
        with torch.no_grad():
            tgt_patches  = self._patchify(yzx_fb[:1].float())[:, :, :1]   # (1, P, 1, fs, ts)
            pred_patches = yzx_pred[:1, :, :1].float()                    # (1, P, 1, fs, ts)
            vis = visible_mask[:1].view(1, -1, 1, 1, 1)
            stitched = torch.where(vis, tgt_patches, pred_patches)
        img = self._patches_to_img(stitched)[0]                            # (1, T, F)
        fig = plot_fbank(img, vmin=img.min(), vmax=img.max(), title=title)
        self.logger.experiment.add_figure(title, fig, global_step=self.global_step)
        plt.close(fig)

    # ----- Training step ----------------------------------------
    def training_step(self, batch, batch_idx):
        yzx_fb, iv_fb, visible_mask = self._prepare_batch(batch)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = self.forward(yzx_fb, iv_fb, visible_mask)

        loss = out["loss"]

        with torch.no_grad():
            # Magnitude-weighted angular error on masked patches only.
            iv_w_masked = out["iv_weight_target"] * out["masked"].view(
                *out["masked"].shape, 1, 1, 1).float()
            cos = (out["iv_pred"] * out["iv_target"]).sum(dim=2, keepdim=True).clamp(-1.0, 1.0)
            ang = torch.acos(cos) * (180.0 / np.pi)
            ang_err_masked = (ang * iv_w_masked).sum() / iv_w_masked.sum().clamp_min(1e-6)


        self.log_dict({
            "loss": loss,
            "l_yzx": out["l_yzx"],
            "l_iv":  out["l_iv"],
            "ang_err_masked_deg": ang_err_masked,
        }, prog_bar=True)

        if self.global_step % self.log_every_n_steps == 0:
            # YZX channel 0 (Y) and IV channel 0 (IVx) as representative views.
            self._log_spectrogram(yzx_fb[:, :1], title="spec/y_input", loss=loss.item())
            self._log_spectrogram(iv_fb[:, :1],  title="spec/iv_input")
            self._log_spectrogram_with_mask(
                yzx_fb[:, :1], visible_mask,
                title="spec/visible_patches_y", show_visible=True,
            )
            self._log_spectrogram_with_mask(
                yzx_fb[:, :1], visible_mask,
                title="spec/masked_patches_y", show_visible=False,
            )
            self._log_yzx_reconstruction(
                yzx_fb, out["yzx_pred"], visible_mask,
                title="recon/yzx_stitched_y",
            )
            self._log_iv_angular_error(
                out["iv_pred"], out["iv_target"],
                visible_mask, title="iv/angular_error_heatmap",
            )
            # In training_step, occasionally (every 1000 steps):
            with torch.no_grad():
                # Full visibility forward — no masking, no decoder.
                z = self._encode_visible(
                    torch.cat([yzx_fb, iv_fb], dim=1),
                    torch.ones(yzx_fb.shape[0], self.num_patches, dtype=torch.bool, device=self.device),
                )
                z_patches = z[:, 1:, :]  # (B, P, D), all patches visible
                # The encoder's tokens at each patch — do they correlate with the IV at that patch?
                # Quick probe: linear regression encoder output → patchified IV unit vectors.
                iv_unit_tgt, _ = self._iv_target(iv_fb)  # (B, P, 3, fs, ts)
                iv_unit_per_patch = iv_unit_tgt.mean(dim=(-1, -2))  # (B, P, 3) — mean direction per patch
                iv_unit_per_patch = F.normalize(iv_unit_per_patch.float(), dim=-1)
                
                # Solve closed-form linear regression: W = (Z^T Z)^-1 Z^T Y
                Z = z_patches.float().reshape(-1, z_patches.shape[-1])  # (B*P, D)
                Y = iv_unit_per_patch.reshape(-1, 3)  # (B*P, 3)
                W = torch.linalg.lstsq(Z, Y).solution  # (D, 3)
                pred = F.normalize(Z @ W, dim=-1)
                probe_cos = (pred * Y).sum(dim=-1).mean()
                print(f"linear probe cosine (encoder->IV-direction): {probe_cos:.4f}")

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

    # ----- Inference / fine-tuning interface --------------------
    def pass_through_encoder(self, x_yzx: torch.Tensor,
                             x_iv: torch.Tensor) -> torch.Tensor:
        """Full-visibility forward (no masking) for downstream use.

        x_yzx: (B, 3, T, F)
        x_iv:  (B, 3, T, F)
        Returns (B, 1+P, D) with CLS prepended.

        No W input. Downstream fusion with any semantic backbone happens
        externally, via late fusion at the task head.
        """
        B = x_yzx.shape[0]
        x_6ch = torch.cat([x_yzx, x_iv], dim=1)                   # (B, 6, T, F)

        x = x_6ch.transpose(2, 3)                                 # (B, 6, F, T)
        embedded = self.patch_strategy.embed(x, self.patch_embed)
        embedded = embedded + self.pos_embed[:, 1:, :]
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        x = torch.cat([cls, embedded], dim=1)
        x = self.encoder_dropout(x)
        for block in self.encoder_blocks:
            x = block(x)
        return self.encoder_norm(x)

    def get_audio_representation(self, x_yzx, x_iv, strategy: str = "mean"):
        z = self.pass_through_encoder(x_yzx, x_iv)
        if strategy == "mean":
            return z[:, 1:, :].mean(axis=1)
        if strategy == "cls":
            return z[:, 0, :]
        if strategy == "raw":
            f = self.grid_size[0]
            return rearrange(z[:, 1:, :], "b (f t) d -> b t (f d)",
                             f=f, d=self.encoder_embedding_dim)
        raise ValueError(f"Unknown strategy '{strategy}'")