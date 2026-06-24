from typing import List, Optional
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
from transformers import AutoModel, AutoFeatureExtractor

from ..patching import PatchStrategy
from .pos_embed import get_2d_sincos_pos_embed
from .utils import PatchEmbed, create_pretrained_model, plot_fbank
from ..data_modules.dataset_functions import pad_or_truncate_batch
from .ambisonic_feature_extractor import FeatureExtractor

set_fused_attn(True)
use_fused_attn(True)

pad_or_truncate_batch = torch.compile(pad_or_truncate_batch)

def matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
    """Zhou et al. 2019 continuous 6D representation: first two columns of R."""
    return R[..., :, :2].reshape(*R.shape[:-2], 6)


def rotate_foa_waveform(wav: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """ACN channels 1:4 ordered (Y, Z, X). Permute -> rotate -> permute back."""
    W, YZX = wav[:, :1], wav[:, 1:4]
    XYZ = YZX[:, [2, 0, 1]]
    XYZ_rot = torch.einsum("nij,njt->nit", R, XYZ)
    YZX_rot = XYZ_rot[:, [1, 2, 0]]
    return torch.cat([W, YZX_rot], dim=1)

def geometric_rotate_iv(iv_pred: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    iv_pred: (B, P, 3, fs, ts) where channels are Y, Z, X.
    R: (B, 3, 3) standard 3D rotation matrix acting on XYZ.
    """
    iv_xyz = iv_pred[:, :, [2, 0, 1], :, :]
    
    iv_xyz_rot = torch.einsum("bij, bpjft -> bpift", R, iv_xyz)
    iv_yzx_rot = iv_xyz_rot[:, :, [1, 2, 0], :, :]
    
    # Re-normalize to ensure they remain unit vectors
    return F.normalize(iv_yzx_rot, p=2, dim=2, eps=1e-6)


def cosine_iv_loss(pred: torch.Tensor, target_unit: torch.Tensor,
                   weight: torch.Tensor,
                   sharpness: float = 1.0,
                   eps: float = 1e-4) -> torch.Tensor:
    """Weighted directional loss on unit IV directions."""
    cos = (pred * target_unit).sum(dim=-3, keepdim=True).clamp(-1.0, 1.0)
    one_minus_cos = 1.0 - cos
    per_bin = one_minus_cos + sharpness * torch.sqrt(one_minus_cos + eps)
    per_bin = per_bin * weight
    return per_bin.sum() / weight.sum().clamp_min(1e-6)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """MSE over masked patches.

    pred, target: (N, P, C, fs, ts)
    mask:         (N, P) bool — True where loss should apply (masked patches).
    """
    m = mask.view(*mask.shape, 1, 1, 1).float()
    sq = (pred - target) ** 2 * m
    denom = m.sum() * pred.shape[2] * pred.shape[3] * pred.shape[4]
    return sq.sum() / denom.clamp_min(1e-6)



# ======================================================================
# SphereV2 - IV only (semantic branch + WXYZ reconstruction removed)
# ======================================================================

class SphereV2(pl.LightningModule):
    """v2 architecture (IV-only variant):
      - Spatial encoder processes all 7 channels (W, X, Y, Z log-mels + 3 IV), masked.
        Pure self-attention (no semantic/GRAMT cross-attention branch).
      - Single decoder: IVDecoder for unit IV direction field (FiLM-R conditioned).
      - Group-theoretic consistency: predict clean IV (Identity R), geometrically
        rotate it, and enforce consistency with directly predicted rotated IV.
    """

    def __init__(
            self,
            model_size: str = "base",
            encoder_depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,

            iv_decoder_depth: int = 4,
            decoder_num_heads: int = 6,
            decoder_embedding_dim: int = 384,

            lr: float = 2e-4,
            b1: float = 0.9, b2: float = 0.95,
            weight_decay: float = 0.01,
            loss_sharpness: float = 0.5,

            patch_strategy: PatchStrategy = None,
            in_channels: int = 7,
            sr: int = 32000,
            num_mel_bins: int = 128,
            input_length: int = 500,
            target_length: int = 200,
            rotations_per_clip: int = 8,

            diag_every_n_steps: int = 100,
            log_every_n_steps: int = 500,

            **kwargs,
        ):
        super().__init__()
        self.save_hyperparameters(ignore=['patch_strategy'])

        self.sr = sr
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.input_length = input_length
        self.rotations_per_clip = rotations_per_clip
        self.in_channels = in_channels
        self.loss_sharpness = loss_sharpness
        self.lr = lr
        self.b1 = b1; self.b2 = b2; self.weight_decay = weight_decay
        self.diag_every_n_steps = diag_every_n_steps
        self.log_every_n_steps = log_every_n_steps

        # Patch geometry.
        self.patch_strategy = patch_strategy
        self.p_f_dim, self.p_t_dim = self.patch_strategy.get_patch_size()
        self.num_patches = self.p_f_dim * self.p_t_dim
        self.grid_size = (self.p_f_dim, self.p_t_dim)
        fs, ts = self.patch_strategy.fshape, self.patch_strategy.tshape
        self.patch_shape = (fs, ts)

        # --- Spatial encoder (pure self-attention, no GRAMT/semantic branch) ---
        _, self.encoder_embedding_dim = create_pretrained_model(model_size)

        self.patch_embed = PatchEmbed()
        self._update_patch_embed_layers(self.patch_embed)

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

        self.encoder_dropout = nn.Dropout(0.0)
        self.encoder_blocks = nn.ModuleList([
            Block(dim=self.encoder_embedding_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=True,
                  norm_layer=nn.LayerNorm)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(self.encoder_embedding_dim)

        # --- IV Decoder (FiLM-R conditioned, group-theoretic rotation handling) ---
        self.iv_decoder = IVDecoder(
            encoder_embed_dim=self.encoder_embedding_dim,
            decoder_embed_dim=decoder_embedding_dim,
            depth=iv_decoder_depth,
            num_heads=decoder_num_heads,
            patch_shape=self.patch_shape,
            num_patches=self.num_patches,
            grid_size=self.grid_size,
            mlp_ratio=2.0,
        )

        self.melspec = FeatureExtractor(
            sample_rate=self.sr, n_fft=1024, win_length=1024,
            hop_length=self.sr // 100, f_min=50, f_max=self.sr // 2,
            n_mels=self.num_mel_bins, power=2.0,
        ).float()

        self._init_our_weights()

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def _init_our_weights(self):
        def init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for mod in [self.patch_embed, self.encoder_blocks, self.encoder_norm, self.iv_decoder]:
            if isinstance(mod, nn.Module):
                mod.apply(init_fn)

        # FiLM zero-init lives only on the IV decoder.
        nn.init.zeros_(self.iv_decoder.film_gen[-1].weight)
        nn.init.zeros_(self.iv_decoder.film_gen[-1].bias)

    def _update_patch_embed_layers(self, patch_embed):
        patch_embed.proj = nn.Conv2d(
            self.in_channels, self.encoder_embedding_dim,
            kernel_size=(self.patch_strategy.fshape, self.patch_strategy.tshape),
            stride=(self.patch_strategy.fstride, self.patch_strategy.tstride),
        )
        patch_embed.num_patch = self.num_patches

    # ---- target utils --------------------------------------------------
    def _patchify(self, x: torch.Tensor, channels: slice) -> torch.Tensor:
        fs, ts = self.patch_strategy.fshape, self.patch_strategy.tshape
        assert self.patch_strategy.fstride == fs and self.patch_strategy.tstride == ts
        sel = x[:, channels].transpose(2, 3)
        B, C, Fm, Tm = sel.shape
        p_f, p_t = Fm // fs, Tm // ts
        assert p_f == self.p_f_dim and p_t == self.p_t_dim
        sel = sel.view(B, C, p_f, fs, p_t, ts) \
                 .permute(0, 2, 4, 1, 3, 5).contiguous() \
                 .view(B, p_f * p_t, C, fs, ts)
        return sel

    def _iv_target(self, x: torch.Tensor):
        fs, ts = self.patch_strategy.fshape, self.patch_strategy.tshape
        iv = x[:, 4:7]
        mag = torch.linalg.norm(iv, dim=1, keepdim=True)
        iv_unit = iv / mag.clamp_min(1e-6)
        weight = mag.clamp(0.0, 1.0)
        iv_unit = iv_unit.transpose(2, 3)
        weight = weight.transpose(2, 3)
        B, _, Fm, Tm = iv_unit.shape
        p_f, p_t = Fm // fs, Tm // ts
        iv_unit = iv_unit.view(B, 3, p_f, fs, p_t, ts) \
                         .permute(0, 2, 4, 1, 3, 5).contiguous() \
                         .view(B, p_f * p_t, 3, fs, ts)
        weight = weight.view(B, 1, p_f, fs, p_t, ts) \
                       .permute(0, 2, 4, 1, 3, 5).contiguous() \
                       .view(B, p_f * p_t, 1, fs, ts)
        return iv_unit, weight

    def _wav2fbank(self, waveform):
        with torch.amp.autocast('cuda', enabled=False):
            waveform = waveform.float()
            rms = torch.sqrt(waveform[:, :1].pow(2).mean(dim=-1, keepdim=True) + 1e-8)
            waveform = waveform / rms
            mel = self.melspec(waveform)
            return mel.transpose(3, 2)

    # ---- Encoder (pure self-attention, no GRAMT context) ---------------
    def _encode_visible(self, x: torch.Tensor, visible_mask: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.transpose(2, 3)
        embedded = self.patch_strategy.embed(x, self.patch_embed)
        embedded = embedded + self.pos_embed[:, 1:, :]

        L_vis = visible_mask[0].sum().item()
        assert (visible_mask.sum(dim=1) == L_vis).all(), \
            "All samples in the batch must have the same number of visible patches."
        z_vis = embedded[visible_mask].view(B, L_vis, -1)

        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        x = torch.cat([cls, z_vis], dim=1)
        x = self.encoder_dropout(x)

        for block in self.encoder_blocks:
            x = block(x)

        return self.encoder_norm(x)

    @torch.no_grad()
    def generate_clip_level_steering_plot(self, x_clean, num_steps=36,
                                        save_path="steering_sweep.png"):
        """Sweep a 360° Yaw, force the spatial decoder to rotate the scene, and
        plot the global energy-weighted clip-level IV. No masking.

        x_clean: (1, 7, T, F) — single clean 7-channel feature tensor.
        """
        self.eval()
        device = x_clean.device

        # Diffuseness weights from the clean IV (for global weighted averaging).
        _, w_clean = self._iv_target(x_clean)                   # (1, P, 1, fs, ts)

        # Fully-visible encoder pass (pure self-attn encoder).
        z = self.pass_through_encoder(x_clean)                  # (1, 1+P, D)
        z_vis = z[:, 1:, :]                                     # drop CLS

        full_mask = torch.ones((1, self.num_patches),
                            dtype=torch.bool, device=device)

        predicted_global_ivs = []
        angles_deg = np.linspace(0, 360, num_steps, endpoint=False)

        for ang in angles_deg:
            theta = np.deg2rad(ang)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            R = torch.tensor([
                [ cos_t, -sin_t, 0.0],
                [ sin_t,  cos_t, 0.0],
                [   0.0,    0.0, 1.0],
            ], dtype=x_clean.dtype, device=device).unsqueeze(0)
            R_6d = matrix_to_6d(R)

            pred_iv = self.iv_decoder(z_vis, full_mask, R_6d)
            weighted_iv = pred_iv * w_clean
            clip_vector = weighted_iv.sum(dim=(1, 3, 4))                 # (1, 3)
            clip_vector = F.normalize(clip_vector, p=2, dim=1)
            predicted_global_ivs.append(clip_vector.squeeze(0).cpu().float().numpy())

        predicted_global_ivs = np.array(predicted_global_ivs)

        # Axes: idx 2 = Front-Back (X), idx 0 = Left-Right (Y), idx 1 = Up-Down (Z).
        X_pred = predicted_global_ivs[:, 2]
        Y_pred = predicted_global_ivs[:, 0]
        Z_mean = predicted_global_ivs[:, 1].mean()
        print(f"Average Elevation (Z): {Z_mean:.4f}")

        fig, ax = plt.subplots(figsize=(7, 7))
        circle = plt.Circle((0, 0), 1, color='gray', fill=False,
                            linestyle='--', alpha=0.3)
        ax.add_patch(circle)
        ax.axhline(0, color='black', lw=0.5, alpha=0.3)
        ax.axvline(0, color='black', lw=0.5, alpha=0.3)
        scatter = ax.scatter(X_pred, Y_pred, c=angles_deg, cmap='hsv',
                            s=100, edgecolor='k', zorder=3)
        ax.quiver(0, 0, X_pred[0], Y_pred[0], color='red',
                angles='xy', scale_units='xy', scale=1,
                label="Predicted Forward (0°)")
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Injected Rotation Angle (Degrees)')
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_xlabel("Predicted Front-Back (X)")
        ax.set_ylabel("Predicted Left-Right (Y)")
        ax.set_title("Fully Visible Decoder Steering\n"
                    "(Global Weighted Intensity Vector)")
        ax.legend(loc='lower right')
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        return predicted_global_ivs
    

    @torch.no_grad()
    def _prepare_batch(self, batch):
        audio, context_idx, R_mats = batch
        audio = audio.to(self.device, non_blocking=True)
        context_idx = context_idx.to(self.device, non_blocking=True)
        R_mats = R_mats.to(self.device, non_blocking=True)

        B = audio.shape[0]
        Rn = R_mats.shape[1]

        wav_exp = audio.unsqueeze(1).expand(-1, Rn, -1, -1).reshape(B * Rn, 4, -1)
        R_flat = R_mats.reshape(B * Rn, 3, 3).to(audio.dtype)
        rotated_wav = rotate_foa_waveform(wav_exp, R_flat)

        all_wav = torch.cat([audio, rotated_wav], dim=0)
        all_fb = self._wav2fbank(all_wav)  # (N, C, T_total, n_freq)

        # Same temporal crop for clean and its Rn rotated versions
        N, C, T_total, n_freq = all_fb.shape
        T_crop = self.target_length
        max_start = T_total - T_crop

        offsets = torch.randint(0, max_start + 1, (B,), device=self.device)
        all_offsets = torch.cat([offsets, offsets.repeat_interleave(Rn)], dim=0)  # (N,)

        # Vectorized crop via gather along the time dim.
        t_range = torch.arange(T_crop, device=self.device)
        time_idx = (all_offsets.view(N, 1, 1, 1) + t_range.view(1, 1, T_crop, 1)) \
            .expand(N, C, T_crop, n_freq)
        all_fb = torch.gather(all_fb, dim=2, index=time_idx)  # (N, C, T_crop, n_freq)

        clean = all_fb[:B]
        rotated = all_fb[B:].reshape(B * Rn, self.in_channels, T_crop, n_freq)

        R_mats_flat = R_mats.reshape(B * Rn, 3, 3)

        visible_mask = torch.zeros(B, self.num_patches, dtype=torch.bool, device=self.device)
        visible_mask.scatter_(1, context_idx.long(), True)

        return (clean.to(torch.bfloat16),
                rotated.to(torch.bfloat16),
                R_mats_flat,
                visible_mask)

    def forward(self, x_clean, x_rot_flat, R_mats_flat, visible_mask):
        N = x_clean.shape[0]
        Rn = x_rot_flat.shape[0] // N

        # Encoder on clean visible patches (pure self-attention).
        z = self._encode_visible(x_clean, visible_mask)
        z_vis = z[:, 1:, :]

        # Targets (IV only)
        iv_unit_rot, w_iv_rot = self._iv_target(x_rot_flat)
        
        # Rep shapes for IV Decoder
        z_vis_rep = z_vis.unsqueeze(1).expand(-1, Rn, -1, -1).reshape(N * Rn, *z_vis.shape[1:])
        visible_mask_rep = visible_mask.unsqueeze(1).expand(-1, Rn, -1).reshape(N * Rn, -1)
        R_6d = matrix_to_6d(R_mats_flat)

        # Predictions (IV only)
        iv_pred_rot = self.iv_decoder(z_vis_rep, visible_mask_rep, R_6d)

        # --- Losses (IV only, with group-theoretic consistency) ---
        l_iv_rot = cosine_iv_loss(iv_pred_rot, iv_unit_rot, w_iv_rot, self.loss_sharpness)

        # Predict the clean (unrotated) IV using the Identity matrix
        R_eye = torch.eye(3, dtype=R_mats_flat.dtype, device=R_mats_flat.device)
        R_eye_flat = R_eye.unsqueeze(0).expand(N, -1, -1)
        iv_pred_clean = self.iv_decoder(z_vis, visible_mask, matrix_to_6d(R_eye_flat))

        # Repeat predictions to match the (N * Rn) rotated batch size
        iv_pred_clean_rep = iv_pred_clean.unsqueeze(1).expand(-1, Rn, -1, -1, -1, -1).reshape(N * Rn, *iv_pred_clean.shape[1:])

        # Mathematically rotate the clean predictions (group-theoretic geometric consistency)
        iv_pred_geom_rot = geometric_rotate_iv(iv_pred_clean_rep, R_mats_flat).detach()

        # CONSISTENCY LOSS: enforces that direct prediction under R matches
        # the geometrically rotated clean prediction.
        l_consistency = cosine_iv_loss(iv_pred_rot, iv_pred_geom_rot, w_iv_rot, self.loss_sharpness)

        # Total Loss (IV reconstruction + group-theoretic consistency)
        total = l_iv_rot + l_consistency

        return {
            "loss": total,
            "l_iv_rot": l_iv_rot,
            "l_consistency": l_consistency,
            "iv_pred": iv_pred_rot, "iv_target": iv_unit_rot,
            "iv_weight_target": w_iv_rot,
            "masked": ~visible_mask,
            "masked_rep": ~visible_mask_rep,
            "z_vis": z_vis, "R_6d": R_6d,
        }


    # ======================================================================
    # TensorBoard logging helpers
    # ======================================================================
    def _fbank_to_patches(self, fbank: torch.Tensor) -> torch.Tensor:
        """Fold (B, C, T, F) fbank into (B, P, C, fs, ts) non-overlapping patches.
        Mirrors the patching in _iv_target / _patchify.
        """
        fs, ts = self.patch_strategy.fshape, self.patch_strategy.tshape
        pf, pt = self.p_f_dim, self.p_t_dim
        B, C, T, Fm = fbank.shape
        x = fbank.transpose(2, 3)                           # (B, C, F, T)
        x = x.reshape(B, C, pf, fs, pt, ts)
        return x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(B, pf * pt, C, fs, ts)

    def _patches_to_img(self, patches: torch.Tensor) -> np.ndarray:
        """Invert _fbank_to_patches: (B, P, C, fs, ts) -> numpy (B, C, T, F)."""
        fs, ts = self.patch_strategy.fshape, self.patch_strategy.tshape
        pf, pt = self.p_f_dim, self.p_t_dim
        B, P, C, _, _ = patches.shape
        x = patches.reshape(B, pf, pt, C, fs, ts)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()        # (B, C, pf, fs, pt, ts)
        x = x.reshape(B, C, pf * fs, pt * ts)               # (B, C, F, T)
        x = x.transpose(2, 3)                               # (B, C, T, F)
        return x.cpu().float().numpy()

    def _log_spectrogram(self, fbank: torch.Tensor, title: str, loss=None):
        """Log the first sample's full multi-channel spectrogram to TensorBoard."""
        with torch.no_grad():
            patches = self._fbank_to_patches(fbank[:1].float())
        img = self._patches_to_img(patches)[0]              # (C, T, F)
        caption = f"Loss: {loss:.4f}" if loss is not None else title
        fig = plot_fbank(img, vmin=img.min(), vmax=img.max(), title=caption)
        self.logger.experiment.add_figure(title, fig, global_step=self.global_step)
        plt.close(fig)

    def _log_spectrogram_with_mask(self, fbank: torch.Tensor,
                                   visible_mask: torch.Tensor, title: str,
                                   show_visible: bool = True):
        """Log spectrogram with visible OR masked patches zeroed out."""
        with torch.no_grad():
            patches = self._fbank_to_patches(fbank[:1].float()).cpu()
        vis = visible_mask[0].cpu()
        zero = vis if not show_visible else ~vis
        patches[:, zero] = 0.0
        img = self._patches_to_img(patches)[0]
        fig = plot_fbank(img, vmin=img.min(), vmax=img.max(), title=title)
        self.logger.experiment.add_figure(title, fig, global_step=self.global_step)
        plt.close(fig)

    def _log_reconstruction(self, pred_patches: torch.Tensor,
                            target_patches: torch.Tensor,
                            title_prefix: str):
        """Log predicted vs target reconstruction spectrograms side-by-side.
        (Kept for compatibility but unused in IV-only mode.)
        """
        with torch.no_grad():
            pred_img = self._patches_to_img(pred_patches[:1].float())[0]     # (C, T, F)
            target_img = self._patches_to_img(target_patches[:1].float())[0] # (C, T, F)

        vmin = float(min(pred_img.min(), target_img.min()))
        vmax = float(max(pred_img.max(), target_img.max()))

        fig_pred = plot_fbank(pred_img, vmin=vmin, vmax=vmax, title=f"{title_prefix} (pred)")
        self.logger.experiment.add_figure(f"{title_prefix}/pred", fig_pred,
                                          global_step=self.global_step)
        plt.close(fig_pred)

        fig_tgt = plot_fbank(target_img, vmin=vmin, vmax=vmax, title=f"{title_prefix} (target)")
        self.logger.experiment.add_figure(f"{title_prefix}/target", fig_tgt,
                                          global_step=self.global_step)
        plt.close(fig_tgt)

    def _log_iv_angular_error(self, iv_pred: torch.Tensor,
                               iv_target: torch.Tensor,
                               visible_mask: torch.Tensor, title: str):
        """Per-patch angular-error heatmap with masked patches outlined in cyan."""
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
        ax.set_xlabel("Time patch")
        ax.set_ylabel("Freq patch")
        self.logger.experiment.add_figure(title, fig, global_step=self.global_step)
        plt.close(fig)

    # ---- Training step -------------------------------------------------
    def training_step(self, batch, batch_idx):
        x_clean, x_rot, R_mats_flat, visible_mask = self._prepare_batch(batch)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = self.forward(x_clean, x_rot, R_mats_flat, visible_mask)

        loss = out["loss"]

        with torch.no_grad():
            # Error on strictly masked patches
            m = out["masked_rep"].view(*out["masked_rep"].shape, 1, 1, 1).float()
            w_masked_iv = out["iv_weight_target"] * m
            cos = (out["iv_pred"] * out["iv_target"]).sum(dim=2, keepdim=True).clamp(-1.0, 1.0)
            ang = torch.acos(cos) * (180.0 / np.pi)
            ang_err_masked = (ang * w_masked_iv).sum() / w_masked_iv.sum().clamp_min(1e-6)

            # Error on non-masked (visible) patches
            v = (~out["masked_rep"]).view(*out["masked_rep"].shape, 1, 1, 1).float()
            w_vis_iv = out["iv_weight_target"] * v
            ang_err_vis = (ang * w_vis_iv).sum() / w_vis_iv.sum().clamp_min(1e-6)

        self.log_dict({
                    "loss": loss,
                    "l_iv_rot": out["l_iv_rot"],
                    "l_consistency": out["l_consistency"],
                    "ang_err_masked_deg": ang_err_masked,
                    "ang_err_vis_deg": ang_err_vis,
                }, prog_bar=True)

        # ---- Periodic image logs ---------------------------------------
        if self.global_step % self.log_every_n_steps == 0:
            # Inputs: clean and rotated (first sample, all 7 channels).
            self._log_spectrogram(
                x_clean, title="spectrogram/clean_input", loss=loss.item()
            )
            self._log_spectrogram(
                x_rot[:x_clean.shape[0]], title="spectrogram/rotated_input"
            )

            # Mask views on the clean input.
            self._log_spectrogram_with_mask(
                x_clean, visible_mask,
                title="spectrogram/visible_patches", show_visible=True
            )
            self._log_spectrogram_with_mask(
                x_clean, visible_mask,
                title="spectrogram/masked_patches", show_visible=False
            )

            # IV angular error heatmap (uses the rotated-branch masks).
            Rn = x_rot.shape[0] // x_clean.shape[0]
            visible_mask_rep = (
                visible_mask.unsqueeze(1).expand(-1, Rn, -1)
                            .reshape(x_rot.shape[0], -1)
            )
            self._log_iv_angular_error(
                out["iv_pred"], out["iv_target"],
                visible_mask_rep, title="iv/angular_error_heatmap"
            )

        # ---- Rotation gap diagnostic (group-theoretic validation) ------
        if self.global_step % self.diag_every_n_steps == 0 and self.global_step > 0:
            with torch.no_grad():
                N, Rn = x_clean.shape[0], x_rot.shape[0] // x_clean.shape[0]
                z_vis_rep = out["z_vis"].unsqueeze(1).expand(-1, Rn, -1, -1) \
                                        .reshape(N * Rn, *out["z_vis"].shape[1:])
                vmask_rep = visible_mask.unsqueeze(1).expand(-1, Rn, -1) \
                                        .reshape(N * Rn, -1)

                m_rep = (~vmask_rep).view(*vmask_rep.shape, 1, 1, 1).float()
                w_masked_iv_diag = out["iv_weight_target"] * m_rep

                # Counterfactual 1: zero-R (degenerate input, not a valid rotation).
                iv_noR = self.iv_decoder(
                    z_vis_rep, vmask_rep, torch.zeros_like(out["R_6d"])
                )
                l_noR = cosine_iv_loss(
                    iv_noR, out["iv_target"], w_masked_iv_diag, self.loss_sharpness
                )

                # Counterfactual 2: swapped-R (each sample gets another sample's rotation).
                # Both the "correct" and "swapped" inputs are valid rotations — the
                # only difference is whether R matches the target scene. This isolates
                # "is FiLM actually *using* R to rotate predictions?".
                R_6d_swapped = torch.roll(out["R_6d"], shifts=1, dims=0)
                iv_swapped = self.iv_decoder(z_vis_rep, vmask_rep, R_6d_swapped)
                l_swapped = cosine_iv_loss(
                    iv_swapped, out["iv_target"], w_masked_iv_diag, self.loss_sharpness
                )

                l_withR = out["l_iv_rot"]
                gap_noR = l_noR - l_withR
                gap_swap = l_swapped - l_withR

                # TB scalars for trajectory monitoring.
                self.log_dict({
                    "diag/l_iv_rot_masked":     l_withR,
                    "diag/l_iv_noR":     l_noR,
                    "diag/l_iv_swapR":   l_swapped,
                    "diag/gap_noR":      gap_noR,
                    "diag/gap_swapR":    gap_swap,
                }, prog_bar=False)

                print(f"--- step {self.global_step} | "
                    f"IV rot cos-loss: withR={l_withR:.4f}  "
                    f"noR={l_noR:.4f} (gap={gap_noR:+.4f})  "
                    f"swapR={l_swapped:.4f} (gap={gap_swap:+.4f})  "
                    f"ang_err_mask={ang_err_masked:.2f}° (vis={ang_err_vis:.2f}°)")

        return loss

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, self.lr, weight_decay=self.weight_decay, betas=(self.b1, self.b2),
        )
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=10000,
            num_training_steps=self.trainer.max_steps,
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    # ---- Downstream eval interface -------------------------------------
    def pass_through_encoder(self, x_input_7ch: torch.Tensor) -> torch.Tensor:
        B = x_input_7ch.shape[0]
        x = x_input_7ch.transpose(2, 3)
        embedded = self.patch_strategy.embed(x, self.patch_embed)
        embedded = embedded + self.pos_embed[:, 1:, :]
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        x = torch.cat([cls, embedded], dim=1)
        x = self.encoder_dropout(x)
        for block in self.encoder_blocks:
            x = block(x)
        return self.encoder_norm(x)

    def get_audio_representation(self, x, strategy: str = "mean"):
        z = self.pass_through_encoder(x)
        if strategy == "mean":
            return z[:, 1:, :].mean(axis=1)
        if strategy == "cls":
            return z[:, 0, :]
        if strategy == "raw":
            f = self.grid_size[0]
            return rearrange(z[:, 1:, :], "b (f t) d -> b t (f d)",
                             f=f, d=self.encoder_embedding_dim)
        raise ValueError(f"Unknown strategy '{strategy}'")