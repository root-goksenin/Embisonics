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
from transformers import AutoModel

from ..patching import PatchStrategy
from .pos_embed import get_2d_sincos_pos_embed
from .utils import create_pretrained_model, plot_fbank
from ..data_modules.dataset_functions import pad_or_truncate_batch
from .ambisonic_feature_extractor import FeatureExtractor

set_fused_attn(True)
use_fused_attn(True)

pad_or_truncate_batch = torch.compile(pad_or_truncate_batch)


class FOAFrontEnd(nn.Module):
    """Per-channel patch embed + channel-type embedding + cross-channel attention.

    Input:  (B, in_channels, F, T)
    Output: (B, num_patches, embed_dim)
    """
    def __init__(self, in_channels, embed_dim,
                 fshape, tshape, fstride, tstride,
                 xca_layers=1, xca_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels, in_channels * embed_dim,
            kernel_size=(fshape, tshape),
            stride=(fstride, tstride),
            groups=in_channels,
        )

        self.channel_emb = nn.Parameter(torch.zeros(1, in_channels, 1, 1, embed_dim))
        nn.init.trunc_normal_(self.channel_emb, std=0.02)

        xca_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=xca_heads,
            dim_feedforward=4 * embed_dim,
            batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.xca = nn.TransformerEncoder(xca_layer, num_layers=xca_layers)

        self.fuse = nn.Linear(in_channels * embed_dim, embed_dim)

        self.num_patch = None

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x)
        x = x.view(B, self.in_channels, self.embed_dim, x.size(-2), x.size(-1))
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x + self.channel_emb

        B_, C, Nf, Nt, D = x.shape
        x = x.permute(0, 2, 3, 1, 4).reshape(B_ * Nf * Nt, C, D)
        x = self.xca(x)
        x = x.view(B_, Nf, Nt, C, D).reshape(B_, Nf, Nt, C * D)
        x = self.fuse(x)
        return x.flatten(1, 2)


def matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
    return R[..., :, :2].reshape(*R.shape[:-2], 6)


def rotate_foa_waveform(wav: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    W, YZX = wav[:, :1], wav[:, 1:4]
    XYZ = YZX[:, [2, 0, 1]]
    XYZ_rot = torch.einsum("nij,njt->nit", R, XYZ)
    YZX_rot = XYZ_rot[:, [1, 2, 0]]
    return torch.cat([W, YZX_rot], dim=1)


def geometric_rotate_iv(iv_pred: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    iv_xyz = iv_pred[:, :, [2, 0, 1], :, :]
    iv_xyz_rot = torch.einsum("bij, bpjft -> bpift", R, iv_xyz)
    iv_yzx_rot = iv_xyz_rot[:, :, [1, 2, 0], :, :]
    return F.normalize(iv_yzx_rot, p=2, dim=2, eps=1e-6)


def cosine_iv_loss(pred: torch.Tensor, target_unit: torch.Tensor,
                   weight: torch.Tensor) -> torch.Tensor:
    """
    Standard weighted cosine distance loss.
    pred: (B, ..., 3, F, T) unit vectors
    target_unit: (B, ..., 3, F, T) unit vectors
    weight: (B, ..., 1, F, T) magnitude/energy weights
    """
    cos = (pred * target_unit).sum(dim=-3, keepdim=True).clamp(-1.0, 1.0)
    
    per_bin = 1.0 - cos
    per_bin = per_bin * weight
    
    return per_bin.sum() / weight.sum().clamp_min(1e-6)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        sa_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + sa_out

        q = self.norm_q(x)
        kv = self.norm_kv(context)
        ca_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = x + ca_out

        x = x + self.mlp(self.norm2(x))
        return x


class IVDecoder(nn.Module):
    """Reconstructs unit IV direction field. FiLM-R conditioned."""
    def __init__(self, encoder_embed_dim: int, decoder_embed_dim: int,
                 depth: int, num_heads: int, patch_shape: tuple,
                 num_patches: int, grid_size: tuple, mlp_ratio: float = 4.0):
        super().__init__()
        self.depth = depth
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

        self.film_gen = nn.Sequential(
            nn.Linear(6, decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_embed_dim, depth * decoder_embed_dim * 2),
        )
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)

        self.blocks = nn.ModuleList([
            Block(decoder_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.head = nn.Linear(decoder_embed_dim, 3 * patch_shape[0] * patch_shape[1])

    def forward(self, z_vis, visible_mask, R_6d):
        N, P, D = z_vis.shape[0], self.num_patches, self.decoder_embed_dim
        z_vis_proj = self.decoder_embed(z_vis)
        x = self.mask_token.type_as(z_vis_proj).expand(N, P, D).clone()
        x[visible_mask] = z_vis_proj.reshape(-1, D)
        x = x + self.pos_embed

        film = self.film_gen(R_6d).view(N, self.depth, 2, D)
        for i, blk in enumerate(self.blocks):
            gamma = film[:, i, 0, :].unsqueeze(1)
            beta  = film[:, i, 1, :].unsqueeze(1)
            x = x * (1.0 + gamma) + beta
            x = blk(x)
        x = self.norm(x)
        iv = self.head(x).view(N, P, 3, self.patch_shape[0], self.patch_shape[1])
        return F.normalize(iv, p=2, dim=2, eps=1e-6)


class SphereV2(pl.LightningModule):
    """v2 architecture (IV-only):
      - Spatial encoder processes 3 IV channels, masked.
      - GRAMT-mono (frozen) processes UNMASKED W log-mel separately, cross-attended
        at each spatial encoder layer.
      - One decoder consuming encoder output:
          IVDecoder: masked & non-masked IVs of ROTATED scene + Identity scene (FiLM-R)
    """

    def __init__(
            self,
            encoder_embedding_dim = 384,
            encoder_depth: int = 6,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,

            iv_decoder_depth: int = 2,
            decoder_num_heads: int = 6,
            decoder_embedding_dim: int = 384,

            gramt_model_id: str = "labhamlet/gramt-mono",
            gramt_freeze: bool = True,

            lr: float = 2e-4,
            b1: float = 0.9, b2: float = 0.95,
            weight_decay: float = 0.01,

            patch_strategy: PatchStrategy = None,
            in_channels: int = 3,
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
        self.lr = lr
        self.b1 = b1; self.b2 = b2; self.weight_decay = weight_decay
        self.diag_every_n_steps = diag_every_n_steps
        self.log_every_n_steps = log_every_n_steps
        self.encoder_embedding_dim = encoder_embedding_dim

        self.patch_strategy = patch_strategy
        self.p_f_dim, self.p_t_dim = self.patch_strategy.get_patch_size()
        self.num_patches = self.p_f_dim * self.p_t_dim
        self.grid_size = (self.p_f_dim, self.p_t_dim)
        fs, ts = self.patch_strategy.fshape, self.patch_strategy.tshape
        self.patch_shape = (fs, ts)

        # --- GRAMT-mono -------------------------------------------------
        self.gram = AutoModel.from_pretrained(gramt_model_id, trust_remote_code=True)
        self.gramt_freeze = gramt_freeze
        if gramt_freeze:
            for p in self.gram.parameters():
                p.requires_grad = False
            self.gram.eval()

        with torch.no_grad():
            dummy_w = torch.zeros(1, 1, target_length, num_mel_bins)
            dummy_out = self.gram(dummy_w, strategy="raw")
        T_gram, flat_dim = dummy_out.shape[1], dummy_out.shape[2]
        self.gramt_n_freq = self.p_f_dim
        assert flat_dim % self.gramt_n_freq == 0
        self.gramt_native_dim = flat_dim // self.gramt_n_freq
        assert T_gram * self.gramt_n_freq == self.num_patches
        self.gramt_t_dim = T_gram
        self.gramt_dim = self.gramt_native_dim


        self.patch_embed = FOAFrontEnd(
            in_channels=self.in_channels,
            embed_dim=self.encoder_embedding_dim,
            fshape=self.patch_strategy.fshape,
            tshape=self.patch_strategy.tshape,
            fstride=self.patch_strategy.fstride,
            tstride=self.patch_strategy.tstride,
            xca_layers=1,
            xca_heads=4,
        )
        self.patch_embed.num_patch = self.num_patches

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

        if self.gramt_native_dim != self.encoder_embedding_dim:
            self.gramt_proj = nn.Linear(self.gramt_native_dim, self.encoder_embedding_dim)
        else:
            self.gramt_proj = nn.Identity()

        self.encoder_dropout = nn.Dropout(0.0)
        self.encoder_blocks = nn.ModuleList([
            CrossAttentionBlock(
                dim=self.encoder_embedding_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(self.encoder_embedding_dim)

        self.iv_decoder = IVDecoder(
            encoder_embed_dim=self.encoder_embedding_dim,
            decoder_embed_dim=decoder_embedding_dim,
            depth=iv_decoder_depth,
            num_heads=decoder_num_heads,
            patch_shape=self.patch_shape,
            num_patches=self.num_patches,
            grid_size=self.grid_size,
            mlp_ratio=4.0,
        )

        self.melspec = FeatureExtractor(
            sample_rate=self.sr, n_fft=1024, win_length=1024,
            hop_length=self.sr // 100, f_min=50, f_max=self.sr // 2,
            n_mels=self.num_mel_bins, power=2.0,
        ).float()

        self._init_our_weights()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.gramt_freeze:
            self.gram.eval()
        return self

    def _init_our_weights(self):
        def init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for mod in [self.patch_embed, self.gramt_proj,
                    self.encoder_blocks, self.encoder_norm,
                    self.iv_decoder]:
            if isinstance(mod, nn.Module):
                mod.apply(init_fn)

        for blk in self.encoder_blocks:
            nn.init.zeros_(blk.cross_attn.out_proj.weight)
            nn.init.zeros_(blk.cross_attn.out_proj.bias)

        nn.init.zeros_(self.iv_decoder.film_gen[-1].weight)
        nn.init.zeros_(self.iv_decoder.film_gen[-1].bias)

    # ---- target utils --------------------------------------------------
    def _iv_target(self, x: torch.Tensor):
        """x: (B, 3, T, F) IVs. Returns unit IV patches and per-bin magnitude weight."""
        fs, ts = self.patch_strategy.fshape, self.patch_strategy.tshape
        iv = x
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

    # ---- GRAMT forward -------------------------------------------------
    def _gramt_tokens(self, w_log_mel_full: torch.Tensor) -> torch.Tensor:
        if self.gramt_freeze:
            with torch.no_grad():
                tokens = self.gram(w_log_mel_full, strategy="raw")
        else:
            tokens = self.gram(w_log_mel_full, strategy="raw")

        B, T, FD = tokens.shape
        F_gram = self.gramt_n_freq
        D = self.gramt_native_dim
        assert FD == F_gram * D
        assert T == self.gramt_t_dim

        tokens = tokens.view(B, T, F_gram, D)
        tokens = tokens.permute(0, 2, 1, 3).contiguous()
        tokens = tokens.reshape(B, F_gram * T, D)
        tokens = self.gramt_proj(tokens)
        tokens = tokens + self.pos_embed[:, 1:, :]
        return tokens

    # ---- Encoder -------------------------------------------------------
    def _encode_visible(self, x: torch.Tensor, visible_mask: torch.Tensor,
                        gramt_tokens: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.transpose(2, 3)
        embedded = self.patch_strategy.embed(x, self.patch_embed)
        embedded = embedded + self.pos_embed[:, 1:, :]

        L_vis = visible_mask[0].sum().item()
        assert (visible_mask.sum(dim=1) == L_vis).all()
        z_vis = embedded[visible_mask].view(B, L_vis, -1)

        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        x = torch.cat([cls, z_vis], dim=1)
        x = self.encoder_dropout(x)

        for block in self.encoder_blocks:
            x = block(x, context=gramt_tokens)

        return self.encoder_norm(x)

    @torch.no_grad()
    def generate_clip_level_steering_plot(self, x_clean_iv, x_clean_w,
                                          num_steps=36,
                                          save_path="steering_sweep.png"):
        """Sweep a 360° Yaw, plot the global energy-weighted clip-level IV.

        x_clean_iv: (1, 3, T, F)  IV channels.
        x_clean_w:  (1, 1, T, F)  W log-mel for GRAMT.
        """
        self.eval()
        device = x_clean_iv.device

        _, w_clean = self._iv_target(x_clean_iv)

        z = self.pass_through_encoder(x_clean_iv, x_clean_w)
        z_vis = z[:, 1:, :]

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
            ], dtype=x_clean_iv.dtype, device=device).unsqueeze(0)
            R_6d = matrix_to_6d(R)

            pred_iv = self.iv_decoder(z_vis, full_mask, R_6d)
            weighted_iv = pred_iv * w_clean
            clip_vector = weighted_iv.sum(dim=(1, 3, 4))
            clip_vector = F.normalize(clip_vector, p=2, dim=1)
            predicted_global_ivs.append(clip_vector.squeeze(0).cpu().float().numpy())

        predicted_global_ivs = np.array(predicted_global_ivs)
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
        ax.set_title("Fully Visible Decoder Steering")
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
        all_fb = self._wav2fbank(all_wav)  # (N_total, 7, T_total, n_freq)

        # --- Temporal crop FIRST, then split into W and IV ---
        N_total, C, T_total, n_freq = all_fb.shape
        T_crop = self.target_length
        max_start = T_total - T_crop

        offsets = torch.randint(0, max_start + 1, (B,), device=self.device)
        all_offsets = torch.cat([offsets, offsets.repeat_interleave(Rn)], dim=0)

        t_range = torch.arange(T_crop, device=self.device)
        time_idx = (all_offsets.view(N_total, 1, 1, 1) + t_range.view(1, 1, T_crop, 1)) \
            .expand(N_total, C, T_crop, n_freq)
        all_fb = torch.gather(all_fb, dim=2, index=time_idx)  # (N_total, 7, T_crop, n_freq)

        # Split AFTER crop
        w_fb  = all_fb[:, 0:1]   # (N_total, 1, T_crop, n_freq)
        iv_fb = all_fb[:, 4:7]   # (N_total, 3, T_crop, n_freq)

        clean_iv   = iv_fb[:B]
        clean_w    = w_fb[:B]
        rotated_iv = iv_fb[B:].reshape(B * Rn, 3, T_crop, n_freq)

        R_mats_flat = R_mats.reshape(B * Rn, 3, 3)

        visible_mask = torch.zeros(B, self.num_patches, dtype=torch.bool, device=self.device)
        visible_mask.scatter_(1, context_idx.long(), True)

        return (clean_iv.to(torch.bfloat16),
                clean_w.to(torch.bfloat16),
                rotated_iv.to(torch.bfloat16),
                R_mats_flat,
                visible_mask)

    def forward(self, x_clean_iv, x_clean_w, x_rot_iv, R_mats_flat, visible_mask):
        N = x_clean_iv.shape[0]
        Rn = x_rot_iv.shape[0] // N

        gramt_tokens = self._gramt_tokens(x_clean_w)

        x_clean_spatial = torch.cat([x_clean_w, x_clean_iv], dim=1)   # (B, 4, T, F)
        z = self._encode_visible(x_clean_spatial, visible_mask, gramt_tokens)
        z_vis = z[:, 1:, :]
        iv_unit_rot, w_iv_rot = self._iv_target(x_rot_iv)

        z_vis_rep = z_vis.unsqueeze(1).expand(-1, Rn, -1, -1).reshape(N * Rn, *z_vis.shape[1:])
        visible_mask_rep = visible_mask.unsqueeze(1).expand(-1, Rn, -1).reshape(N * Rn, -1)
        R_6d = matrix_to_6d(R_mats_flat)

        iv_pred_rot = self.iv_decoder(z_vis_rep, visible_mask_rep, R_6d)

        # Reconstruction loss on masked patches only (consistency keeps full weights).
        masked_indicator = (~visible_mask_rep).float().view(N * Rn, self.num_patches, 1, 1, 1)
        w_iv_rot_masked = w_iv_rot * masked_indicator
        l_iv_rot = cosine_iv_loss(iv_pred_rot, iv_unit_rot, w_iv_rot_masked)

        # Consistency: predict canonical (R=I), rotate analytically, compare
        R_eye = torch.eye(3, dtype=R_mats_flat.dtype, device=R_mats_flat.device)
        R_eye_flat = R_eye.unsqueeze(0).expand(N, -1, -1)
        iv_pred_clean = self.iv_decoder(z_vis, visible_mask, matrix_to_6d(R_eye_flat))
        iv_pred_clean_rep = iv_pred_clean.unsqueeze(1).expand(-1, Rn, -1, -1, -1, -1) \
                                         .reshape(N * Rn, *iv_pred_clean.shape[1:])
        iv_pred_geom_rot = geometric_rotate_iv(iv_pred_clean_rep, R_mats_flat).detach()

        l_consistency = cosine_iv_loss(iv_pred_rot, iv_pred_geom_rot, w_iv_rot)

        total = l_iv_rot + l_consistency

        return {
            "loss": total,
            "l_iv_rot": l_iv_rot,
            "l_consistency": l_consistency,
            "iv_pred": iv_pred_rot,
            "iv_target": iv_unit_rot,
            "iv_weight_target": w_iv_rot,
            "masked": ~visible_mask,
            "masked_rep": ~visible_mask_rep,
            "z_vis": z_vis,
            "R_6d": R_6d,
        }

    # ---- TB logging helpers (unchanged shape, IV-only inputs) ----------
    def _fbank_to_patches(self, fbank: torch.Tensor) -> torch.Tensor:
        fs, ts = self.patch_strategy.fshape, self.patch_strategy.tshape
        pf, pt = self.p_f_dim, self.p_t_dim
        B, C, T, Fm = fbank.shape
        x = fbank.transpose(2, 3)
        x = x.reshape(B, C, pf, fs, pt, ts)
        return x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(B, pf * pt, C, fs, ts)

    def _patches_to_img(self, patches: torch.Tensor) -> np.ndarray:
        fs, ts = self.patch_strategy.fshape, self.patch_strategy.tshape
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
        ax.set_xlabel("Time patch")
        ax.set_ylabel("Freq patch")
        self.logger.experiment.add_figure(title, fig, global_step=self.global_step)
        plt.close(fig)

    # ---- Training step -------------------------------------------------
    def training_step(self, batch, batch_idx):
        x_clean_iv, x_clean_w, x_rot_iv, R_mats_flat, visible_mask = self._prepare_batch(batch)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = self.forward(x_clean_iv, x_clean_w, x_rot_iv, R_mats_flat, visible_mask)

        loss = out["loss"]

        with torch.no_grad():
            m = out["masked_rep"].view(*out["masked_rep"].shape, 1, 1, 1).float()
            w_masked_iv = out["iv_weight_target"] * m
            cos = (out["iv_pred"] * out["iv_target"]).sum(dim=2, keepdim=True).clamp(-1.0, 1.0)
            ang = torch.acos(cos) * (180.0 / np.pi)
            ang_err_masked = (ang * w_masked_iv).sum() / w_masked_iv.sum().clamp_min(1e-6)

            v = (~out["masked_rep"]).view(*out["masked_rep"].shape, 1, 1, 1).float()
            w_vis_iv = out["iv_weight_target"] * v
            ang_err_vis = (ang * w_vis_iv).sum() / w_vis_iv.sum().clamp_min(1e-6)

            ca_norm = sum(
                blk.cross_attn.out_proj.weight.norm().item()
                for blk in self.encoder_blocks
            ) / len(self.encoder_blocks)

        self.log_dict({
            "loss": loss,
            "l_iv_rot": out["l_iv_rot"],
            "l_consistency": out["l_consistency"],
            "ang_err_masked_deg": ang_err_masked,
            "ang_err_vis_deg": ang_err_vis,
            "cross_attn_norm": ca_norm,
        }, prog_bar=True)

        if self.global_step % self.log_every_n_steps == 0:
            self._log_spectrogram(
                x_clean_iv, title="spectrogram/clean_iv_input", loss=loss.item()
            )
            self._log_spectrogram(
                x_rot_iv[:x_clean_iv.shape[0]], title="spectrogram/rotated_iv_input"
            )
            self._log_spectrogram_with_mask(
                x_clean_iv, visible_mask,
                title="spectrogram/visible_patches", show_visible=True
            )
            self._log_spectrogram_with_mask(
                x_clean_iv, visible_mask,
                title="spectrogram/masked_patches", show_visible=False
            )
            Rn = x_rot_iv.shape[0] // x_clean_iv.shape[0]
            visible_mask_rep = (
                visible_mask.unsqueeze(1).expand(-1, Rn, -1)
                            .reshape(x_rot_iv.shape[0], -1)
            )
            self._log_iv_angular_error(
                out["iv_pred"], out["iv_target"],
                visible_mask_rep, title="iv/angular_error_heatmap"
            )

        if self.global_step % self.diag_every_n_steps == 0 and self.global_step > 0:
            with torch.no_grad():
                N, Rn = x_clean_iv.shape[0], x_rot_iv.shape[0] // x_clean_iv.shape[0]
                z_vis_rep = out["z_vis"].unsqueeze(1).expand(-1, Rn, -1, -1) \
                                        .reshape(N * Rn, *out["z_vis"].shape[1:])
                vmask_rep = visible_mask.unsqueeze(1).expand(-1, Rn, -1) \
                                        .reshape(N * Rn, -1)

                m_rep = (~vmask_rep).view(*vmask_rep.shape, 1, 1, 1).float()
                w_masked_iv_diag = out["iv_weight_target"] * m_rep

                iv_noR = self.iv_decoder(
                    z_vis_rep, vmask_rep, torch.zeros_like(out["R_6d"])
                )
                l_noR = cosine_iv_loss(
                    iv_noR, out["iv_target"], w_masked_iv_diag
                )

                R_6d_swapped = torch.roll(out["R_6d"], shifts=1, dims=0)
                iv_swapped = self.iv_decoder(z_vis_rep, vmask_rep, R_6d_swapped)
                l_swapped = cosine_iv_loss(
                    iv_swapped, out["iv_target"], w_masked_iv_diag
                )

                l_withR = out["l_iv_rot"]
                gap_noR = l_noR - l_withR
                gap_swap = l_swapped - l_withR

                self.log_dict({
                    "diag/l_iv_rot_masked": l_withR,
                    "diag/l_iv_noR":        l_noR,
                    "diag/l_iv_swapR":      l_swapped,
                    "diag/gap_noR":         gap_noR,
                    "diag/gap_swapR":       gap_swap,
                }, prog_bar=False)

                print(f"--- step {self.global_step} | "
                      f"IV rot cos-loss: withR={l_withR:.4f}  "
                      f"noR={l_noR:.4f} (gap={gap_noR:+.4f})  "
                      f"swapR={l_swapped:.4f} (gap={gap_swap:+.4f})  "
                      f"ang_err_mask={ang_err_masked:.2f}° (vis={ang_err_vis:.2f}°)  "
                      f"ca_norm={ca_norm:.4f}")

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

    def pass_through_encoder(self, x_iv: torch.Tensor, x_w: torch.Tensor) -> torch.Tensor:
        """x_iv: (B, 3, T, F) IVs.  x_w: (B, 1, T, F) W log-mel."""
        B = x_iv.shape[0]
        gramt_tokens = self._gramt_tokens(x_w)
        x_spatial = torch.cat([x_w, x_iv], dim=1)        # (B, 4, T, F)
        x = x_spatial.transpose(2, 3)
        embedded = self.patch_strategy.embed(x, self.patch_embed)
        embedded = embedded + self.pos_embed[:, 1:, :]
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        x = torch.cat([cls, embedded], dim=1)
        x = self.encoder_dropout(x)
        for block in self.encoder_blocks:
            x = block(x, context=gramt_tokens)
        return self.encoder_norm(x)

    def get_audio_representation(self, x_iv, x_w, strategy: str = "mean"):
        z = self.pass_through_encoder(x_iv, x_w)
        if strategy == "mean":
            return z[:, 1:, :].mean(axis=1)
        if strategy == "cls":
            return z[:, 0, :]
        if strategy == "raw":
            f = self.grid_size[0]
            return rearrange(z[:, 1:, :], "b (f t) d -> b t (f d)",
                             f=f, d=self.encoder_embedding_dim)
        raise ValueError(f"Unknown strategy '{strategy}'")