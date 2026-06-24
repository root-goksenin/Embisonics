"""
SphericalVICReg — minimal cross-channel VICReg with rotation-conditioned predictor.

Token grid: (channel, freq_patch, time_patch), flattened with channel as the
outer axis: token i = (c, f, t) where i = c*P_f*P_t + f*P_t + t.
Total tokens = C * P_f * P_t (+1 for CLS). C = 4 for FOA.

Architecture:
  - Patch embed: shared 1-channel Conv2d applied per ambisonic channel.
  - Channel embedding: learned (1, C, D), broadcast over spatial axis.
  - Spatial pos embed: frozen 2D sincos over (P_f, P_t), broadcast over channel.
  - Single student encoder (ViT) over the full (C, F, T) token grid.
  - Rotation predictor: FiLM-conditioned transformer that takes student-on-clean
    and rotation R, predicts student-on-rotated.

Losses (VICReg-style):
  L_inv      — cosine: rotation_predictor(z_clean, R) vs student(x_rot).
  L_var      — VICReg per-dim variance hinge on [z_clean, z_rotated] (concat).
  L_cov      — VICReg covariance penalty on [z_clean, z_rotated] (concat).
  L_rot_var  — Per-clip variance hinge along the rotation axis: for each clip,
               the Rn rotated embeddings must spread out per-(patch, dim).
               This is the self-supervised replacement for an l_rot_rec head;
               it directly prevents the rotation-invariance failure mode that
               L_inv + marginal VICReg cannot block on its own.

Watch closely:
  diag/student_rot_sensitivity — must stay clearly above zero.
  diag/gap_noR                 — must stay clearly above zero (predictor uses R).
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from einops import rearrange
from timm.models.vision_transformer import Block

from ...patching import PatchStrategy
from ..ambisonic_feature_extractor import FeatureExtractor
from ..pos_embed import get_2d_sincos_pos_embed
from ..utils import PatchEmbed


# ━━ Utilities ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
    """Zhou et al. 2019: first two columns of rotation matrix → 6D."""
    return R[..., :, :2].reshape(*R.shape[:-2], 6)


def rotate_foa_waveform(wav: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Rotate FOA waveform (ACN order: W, Y, Z, X) by 3×3 rotation matrix."""
    W, YZX = wav[:, :1], wav[:, 1:4]
    XYZ = YZX[:, [2, 0, 1]]
    XYZ_rot = torch.einsum("nij,njt->nit", R, XYZ)
    YZX_rot = XYZ_rot[:, [1, 2, 0]]
    return torch.cat([W, YZX_rot], dim=1)


def vicreg_terms(z: torch.Tensor, eps: float = 1e-4, gamma: float = 1.0):
    """VICReg variance + covariance regularisation on (B, P, D) embeddings."""
    z_flat = z.reshape(-1, z.shape[-1])
    z_c = z_flat - z_flat.mean(0, keepdim=True)

    std = (z_c.var(0) + eps).sqrt()
    var_loss = F.relu(gamma - std).mean()

    N, D = z_flat.shape
    cov = (z_c.T @ z_c) / max(N - 1, 1)
    off = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    cov_loss = off / D
    return var_loss, cov_loss


def rotation_axis_variance(
    z_per_clip: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4,
) -> torch.Tensor:
    """Per-clip variance hinge along the rotation axis.

    z_per_clip: (B, Rn, P, D)  embeddings stacked along the rotation axis
    Returns:    scalar         hinge loss F.relu(gamma - std).mean()

    No covariance counterpart — with Rn ~ 4 the covariance estimator is too
    noisy to be useful; only the variance term carries signal.
    """
    std = (z_per_clip.var(dim=1, unbiased=True) + eps).sqrt()  # (B, P, D)
    return F.relu(gamma - std).mean()


# ━━ Sub-modules ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class RotationPredictor(nn.Module):
    """FiLM-conditioned transformer.

    Takes the student's full (channel, freq, time) patch embeddings and a
    rotation R (6D), predicts the student's embeddings of the rotated input.
    Uses the same channel + spatial position scheme as the encoder.
    FiLM weights are zero-initialised so the predictor starts unconditioned.
    """

    def __init__(
        self,
        encoder_dim: int,
        predictor_dim: int,
        depth: int,
        num_heads: int,
        num_channels: int,
        p_f: int,
        p_t: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.depth = depth
        self.predictor_dim = predictor_dim
        self.num_channels = num_channels
        self.p_f = p_f
        self.p_t = p_t
        self.spatial_patches = p_f * p_t

        self.proj_in = nn.Linear(encoder_dim, predictor_dim)

        # Spatial pos embed: frozen 2D sincos, shared across channels
        self.pos_embed_spatial = nn.Parameter(
            torch.zeros(1, self.spatial_patches, predictor_dim),
            requires_grad=False,
        )
        spatial_pe = get_2d_sincos_pos_embed(
            predictor_dim, (p_f, p_t), cls_token_num=0,
        )
        self.pos_embed_spatial.data.copy_(
            torch.from_numpy(spatial_pe).float().unsqueeze(0)
        )

        # Channel embedding: learned, broadcasts over spatial axis
        self.channel_embed = nn.Parameter(
            torch.zeros(1, num_channels, 1, predictor_dim)
        )
        nn.init.trunc_normal_(self.channel_embed, std=0.02)

        # FiLM generator: 6D rotation → (gamma, beta) per layer
        self.film_gen = nn.Sequential(
            nn.Linear(6, predictor_dim),
            nn.GELU(),
            nn.Linear(predictor_dim, depth * predictor_dim * 2),
        )
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)

        self.blocks = nn.ModuleList([
            Block(
                predictor_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, norm_layer=nn.LayerNorm,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)
        self.proj_out = nn.Linear(predictor_dim, encoder_dim)

    def _add_position(self, x: torch.Tensor) -> torch.Tensor:
        """Add channel + spatial position embeddings to (B, C*P_f*P_t, D)."""
        B = x.shape[0]
        C, PS = self.num_channels, self.spatial_patches
        x = x.reshape(B, C, PS, -1)
        x = x + self.channel_embed                        # (1, C, 1, D)
        x = x + self.pos_embed_spatial.unsqueeze(1)       # (1, 1, PS, D)
        return x.reshape(B, C * PS, -1)

    def forward(self, z_s: torch.Tensor, R_6d: torch.Tensor) -> torch.Tensor:
        """
        z_s:  (B, C*P_f*P_t, D_enc)  student patch embeddings (CLS already stripped)
        R_6d: (B, 6)
        Returns: (B, C*P_f*P_t, D_enc)
        """
        B = z_s.shape[0]
        D = self.predictor_dim

        x = self._add_position(self.proj_in(z_s))
        film = self.film_gen(R_6d).view(B, self.depth, 2, D)

        for i, blk in enumerate(self.blocks):
            gamma = film[:, i, 0].unsqueeze(1)            # (B, 1, D)
            beta = film[:, i, 1].unsqueeze(1)
            x = x * (1.0 + gamma) + beta
            x = blk(x)

        return self.proj_out(self.norm(x))


# ━━ Main Module ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SphereREG(pl.LightningModule):

    def __init__(
        self,
        # ── encoder ──
        encoder_dim: int = 768,
        encoder_depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        # ── rotation predictor ──
        predictor_dim: int = 384,
        predictor_depth: int = 4,
        predictor_num_heads: int = 6,
        # ── loss weights ──
        lambda_var: float = 0.1,
        lambda_cov: float = 0.1,
        lambda_rot_var: float = 1.0,
        gamma_rot: float = 1.0,
        # ── optimiser ──
        lr: float = 2e-4,
        b1: float = 0.9,
        b2: float = 0.95,
        weight_decay: float = 0.01,
        warmup_steps: int = 10_000,
        # ── audio / patches ──
        patch_strategy: PatchStrategy = None,
        in_channels: int = 4,
        sr: int = 16_000,
        num_mel_bins: int = 128,
        target_length: int = 200,
        rotations_per_clip: int = 4,
        # ── logging ──
        log_every_n_steps: int = 100,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["patch_strategy"])

        # store scalars
        self.encoder_dim = encoder_dim
        self.sr = sr
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.rotations_per_clip = rotations_per_clip
        self.in_channels = in_channels
        self.lr = lr
        self.b1, self.b2 = b1, b2
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.lambda_rot_var = lambda_rot_var
        self.gamma_rot = gamma_rot
        self.log_every_n_steps = log_every_n_steps

        # patch geometry
        self.patch_strategy = patch_strategy
        self.p_f, self.p_t = patch_strategy.get_patch_size()
        self.spatial_patches = self.p_f * self.p_t
        self.num_patches = in_channels * self.spatial_patches
        self.grid_size = (self.p_f, self.p_t)

        # ── mel feature extractor ──────────────────────────────────
        self.melspec = FeatureExtractor(
            sample_rate=sr, n_fft=1024, win_length=1024,
            hop_length=sr // 100, f_min=50, f_max=sr // 2,
            n_mels=num_mel_bins, power=2.0,
        ).float()

        # ── patch embedding: shared 1-channel conv ─────────────────
        self.patch_embed = PatchEmbed()
        self.patch_embed.proj = nn.Conv2d(
            1, encoder_dim,
            kernel_size=(patch_strategy.fshape, patch_strategy.tshape),
            stride=(patch_strategy.fstride, patch_strategy.tstride),
        )
        self.patch_embed.num_patch = self.num_patches

        # ── CLS token + its position ───────────────────────────────
        self.cls_token = nn.Parameter(
            nn.init.normal_(torch.empty(1, 1, encoder_dim), std=0.02)
        )
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, encoder_dim))

        # ── spatial pos embed (frozen sincos, shared across channels) ──
        self.pos_embed_spatial = nn.Parameter(
            torch.zeros(1, self.spatial_patches, encoder_dim),
            requires_grad=False,
        )
        spatial_pe = get_2d_sincos_pos_embed(
            encoder_dim, self.grid_size, cls_token_num=0,
        )
        self.pos_embed_spatial.data.copy_(
            torch.from_numpy(spatial_pe).float().unsqueeze(0)
        )

        # ── channel embedding (learned) ────────────────────────────
        self.channel_embed = nn.Parameter(
            torch.zeros(1, in_channels, 1, encoder_dim)
        )
        nn.init.trunc_normal_(self.channel_embed, std=0.02)

        # ── student encoder ────────────────────────────────────────
        self.encoder_blocks = nn.ModuleList([
            Block(
                encoder_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, norm_layer=nn.LayerNorm,
            )
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(encoder_dim)

        # ── rotation predictor ─────────────────────────────────────
        self.rotation_predictor = RotationPredictor(
            encoder_dim=encoder_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            num_channels=in_channels,
            p_f=self.p_f,
            p_t=self.p_t,
        )

    # ━━ Feature extraction / patching ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _wav2fbank(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            waveform = waveform.float()
            rms = (waveform[:, :1].pow(2).mean(-1, keepdim=True) + 1e-8).sqrt()
            waveform = waveform / rms
            mel = self.melspec(waveform)
            return mel.transpose(3, 2)  # (B, C, T, F)

    def _embed_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-channel patching with cross-channel tokens.

        x:        (B, C, T, F)   mel spectrogram
        Returns:  (B, C*P_f*P_t, D)   token grid in (c, f, t) flat order

        Bypasses self.patch_strategy.embed because that path expects multi-
        channel input. If your PatchStrategy does anything beyond a strided
        conv, you'll need to channel-ize that path here.
        """
        B, C, T, F = x.shape
        x = x.transpose(2, 3)                        # (B, C, F, T)
        x = x.reshape(B * C, 1, F, T)                # (B*C, 1, F, T)

        x = self.patch_embed.proj(x)                 # (B*C, D, P_f, P_t)
        _, D, Pf, Pt = x.shape

        x = x.reshape(B, C, D, Pf * Pt)              # (B, C, D, P_f*P_t)
        x = x.permute(0, 1, 3, 2).contiguous()       # (B, C, P_f*P_t, D)

        x = x + self.channel_embed                   # (1, C, 1, D) broadcast
        x = x + self.pos_embed_spatial.unsqueeze(1)  # (1, 1, P_f*P_t, D) broadcast

        return x.reshape(B, C * Pf * Pt, D)          # (c, f, t) flat, c outer

    # ━━ Encoder forward ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _student_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Student forward on full (unmasked) input.
        Returns (B, 1+C*P_f*P_t, D) with CLS at position 0."""
        B = x.shape[0]
        embedded = self._embed_patches(x)
        cls = self.cls_token.expand(B, -1, -1) + self.cls_pos
        h = torch.cat([cls, embedded], dim=1)
        for block in self.encoder_blocks:
            h = block(h)
        return self.encoder_norm(h)

    # ━━ Batch preparation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @torch.no_grad()
    def _prepare_batch(self, batch):
        """
        batch: (audio, R_mats)
            audio:  (B, 4, samples)
            R_mats: (B, Rn, 3, 3)
        Returns:
            x_clean:    (B, C, T_crop, F)        bf16
            x_rot_flat: (B*Rn, C, T_crop, F)     bf16
            R_flat:     (B*Rn, 3, 3)
        """
        audio, R_mats = batch
        B, Rn = audio.shape[0], R_mats.shape[1]

        # rotate waveforms
        wav_exp = (
            audio.unsqueeze(1)
            .expand(-1, Rn, -1, -1)
            .reshape(B * Rn, self.in_channels, -1)
        )
        R_flat = R_mats.reshape(B * Rn, 3, 3).to(audio.dtype)
        rot_wav = rotate_foa_waveform(wav_exp, R_flat)

        # compute mel spectrograms jointly
        all_wav = torch.cat([audio, rot_wav], dim=0)
        all_fb = self._wav2fbank(all_wav)

        # random temporal crop (same offset for clean & its rotations)
        N, C, T_total, Freq = all_fb.shape
        T_crop = self.target_length
        offsets = torch.randint(0, max(T_total - T_crop, 1), (B,), device=self.device)
        all_offsets = torch.cat([offsets, offsets.repeat_interleave(Rn)])

        t_idx = torch.arange(T_crop, device=self.device)
        idx = (
            all_offsets[:, None, None, None] + t_idx[None, None, :, None]
        ).expand(N, C, T_crop, Freq)
        all_fb = torch.gather(all_fb, 2, idx)

        x_clean = all_fb[:B]
        x_rot = all_fb[B:]
        R_flat = R_mats.reshape(B * Rn, 3, 3)

        return x_clean.to(torch.bfloat16), x_rot.to(torch.bfloat16), R_flat

    # ━━ Forward ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def forward(self, x_clean, x_rot_flat, R_mats_flat):
        B = x_clean.shape[0]
        Rn = x_rot_flat.shape[0] // B

        # ── student on clean and on all rotated views (shared weights) ──
        z_s = self._student_encode(x_clean)[:, 1:]            # (B, P, D)
        z_s_rot_all = self._student_encode(x_rot_flat)[:, 1:]  # (B*Rn, P, D)

        # ── L_inv: predict student-on-rotated from student-on-clean ──
        # No stop-grad on the target — VICReg-style; collapse handled by var/cov.
        z_s_rep = z_s.repeat_interleave(Rn, dim=0)
        R_6d_all = matrix_to_6d(R_mats_flat)
        z_hat = self.rotation_predictor(z_s_rep, R_6d_all)

        z_hat_n = F.normalize(z_hat, dim=-1)
        z_tgt_n = F.normalize(z_s_rot_all, dim=-1)
        l_inv = (2.0 - 2.0 * (z_hat_n * z_tgt_n).sum(-1)).mean()

        # ── VICReg variance + covariance on both branches ──────────
        # Concat clean and rotated so neither branch can collapse independently.
        l_var, l_cov = vicreg_terms(torch.cat([z_s, z_s_rot_all], dim=0))

        # ── Per-clip rotation-axis variance hinge ──────────────────
        # For each clip, the Rn rotated embeddings must have spread along the
        # rotation axis at every (patch, dim). This is the VICReg-style
        # replacement for the removed l_rot_rec head — it directly forbids
        # rotation-invariance, which is otherwise a global optimum of the
        # remaining loss.
        P, D = z_s_rot_all.shape[1], z_s_rot_all.shape[2]
        z_per_clip = z_s_rot_all.reshape(B, Rn, P, D)
        l_rot_var = rotation_axis_variance(z_per_clip, gamma=self.gamma_rot)

        # ── total ──────────────────────────────────────────────────
        loss = (
            l_inv
            + self.lambda_var * l_var
            + self.lambda_cov * l_cov
            + self.lambda_rot_var * l_rot_var
        )

        return dict(
            loss=loss,
            l_inv=l_inv,
            l_var=l_var,
            l_cov=l_cov,
            l_rot_var=l_rot_var,
            z_s=z_s,
            z_hat=z_hat,
            z_s_rot_all=z_s_rot_all,
            R_6d_all=R_6d_all,
        )

    # ━━ Training ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def training_step(self, batch, batch_idx):
        x_clean, x_rot, R_flat = self._prepare_batch(batch)
        out = self.forward(x_clean, x_rot, R_flat)

        self.log_dict(
            {
                "loss": out["loss"],
                "l_inv": out["l_inv"],
                "l_var": out["l_var"],
                "l_cov": out["l_cov"],
                "l_rot_var": out["l_rot_var"],
            },
            prog_bar=True,
        )

        # ── diagnostics ───────────────────────────────────────────
        # With no l_rot_rec head, these two metrics ARE the safety net.
        # Watch them every step in early training; if either trends to zero,
        # the encoder has fallen into rotation-invariance and the architecture
        # is not working.
        if self.global_step % self.log_every_n_steps == 0:
            Rn = self.rotations_per_clip
            with torch.no_grad():
                # does the predictor actually use R?
                z_hat_noR = self.rotation_predictor(
                    out["z_s"].repeat_interleave(Rn, dim=0),
                    torch.zeros_like(out["R_6d_all"]),
                )
                z_tgt_n = F.normalize(out["z_s_rot_all"].detach(), dim=-1)
                z_noR_n = F.normalize(z_hat_noR, dim=-1)
                l_noR = (2.0 - 2.0 * (z_noR_n * z_tgt_n).sum(-1)).mean()

                # is the student rotation-sensitive?
                rot_sens = (
                    out["z_s"] - out["z_s_rot_all"][: x_clean.shape[0]]
                ).pow(2).mean()

                self.log_dict(
                    {
                        "diag/l_inv_noR": l_noR,
                        "diag/gap_noR": l_noR - out["l_inv"],
                        "diag/student_rot_sensitivity": rot_sens,
                    },
                    prog_bar=False,
                )

        return out["loss"]

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            params, self.lr, betas=(self.b1, self.b2),
            weight_decay=self.weight_decay,
        )
        sched = transformers.get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.max_steps,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }

    # ━━ Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Full student encoder forward. Returns (B, 1+C*P_f*P_t, D) with CLS."""
        return self._student_encode(x)

    def get_representation(self, x: torch.Tensor, strategy: str = "concat"):
        """
        Downstream representation extraction.

        Strategies:
            "concat" → (B, T_patches, C * F_patches * D)
                       per-timestep embedding with channel + freq concatenated
                       (carries directional info into linear probes)
            "mean"   → (B, D)  global mean pool over all (c, f, t) tokens
            "cls"    → (B, D)  CLS token
            "raw"    → (B, C*P_f*P_t, D)  all patch embeddings
        """
        z = self._student_encode(x)
        z_patches = z[:, 1:]

        if strategy == "concat":
            return rearrange(
                z_patches, "b (c f t) d -> b t (c f d)",
                c=self.in_channels, f=self.p_f, d=self.encoder_dim,
            )
        if strategy == "mean":
            return z_patches.mean(1)
        if strategy == "cls":
            return z[:, 0]
        if strategy == "raw":
            return z_patches
        raise ValueError(f"Unknown strategy: {strategy}")