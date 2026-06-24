from typing import Optional

import copy
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import transformers
from einops import rearrange
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModel
from timm.layers.config import set_fused_attn, use_fused_attn
from timm.models.vision_transformer import Block

from ...patching import PatchStrategy
from ...data_modules.dataset_functions import pad_or_truncate_batch
from ..ambisonic_feature_extractor import FeatureExtractor
from ..pos_embed import get_2d_sincos_pos_embed
from ..utils import PatchEmbed, create_pretrained_model

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


def cosine_loss(pred: torch.Tensor, 
                target: torch.Tensor, 
                weight: torch.Tensor) -> torch.Tensor:
    cos_sim = (pred * target).sum(dim=2, keepdim=True)
    loss = (1.0 - cos_sim) * weight
    return loss.sum() / weight.sum().clamp_min(1e-6)


def masked_mse(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """MSE over a (B, P) bool mask. pred, target: (B, P, D)."""
    m = mask.unsqueeze(-1).float()
    sq = (pred - target).pow(2) * m
    denom = m.sum() * pred.shape[-1]
    return sq.sum() / denom.clamp_min(1e-6)



class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )

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


class Predictor(nn.Module):
    """Mask-infill + rotation-conditioned predictor.

    For R = I, behaves as a vanilla mask-infill predictor.
    For R != I, conditioned via FiLM at each layer; expected to predict the
    contextualised rep of the rotated scene given visible tokens of the clean
    scene.

    NB (per v2 §10): FiLM injection of R is *conditioning*, not equivariance —
    SpatialCNN/patch_embed has already destroyed the SH structure, so there is
    no canonical SO(3) action to preserve.  The predictor learns to use R as a
    side-input.
    """

    def __init__(self, encoder_dim: int, predictor_dim: int, depth: int,
                 num_heads: int, num_patches: int, patch_grid_size,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.depth = depth
        self.num_patches = num_patches
        self.predictor_dim = predictor_dim

        self.encoder_to_predictor = nn.Linear(encoder_dim, predictor_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_dim), requires_grad=False
        )
        pe = get_2d_sincos_pos_embed(predictor_dim, patch_grid_size, cls_token_num=0)
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

        # FiLM-R: produces (gamma, beta) per layer.  Zero-init so the predictor
        # starts as an unconditioned transformer and the rotation signal has to
        # be earned.  Identity R (passed as 6D = first two cols of I) gives
        # gamma=beta=0 by construction once weights are zero.
        self.film_gen = nn.Sequential(
            nn.Linear(6, predictor_dim),
            nn.GELU(),
            nn.Linear(predictor_dim, depth * predictor_dim * 2),
        )
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)

        self.blocks = nn.ModuleList([
            Block(predictor_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)
        self.predictor_to_encoder = nn.Linear(predictor_dim, encoder_dim, bias=True)

    def forward(self, z_vis: torch.Tensor, visible_mask: torch.Tensor,
                R_6d: torch.Tensor) -> torch.Tensor:
        """
        z_vis:        (B, N_vis, D_enc) student encoder output, CLS dropped.
        visible_mask: (B, P) bool — True where visible.
        R_6d:         (B, 6) — identity flattened to 6D for the mask target.
        Returns:      (B, P, D_enc)
        """
        B, P, D = z_vis.shape[0], self.num_patches, self.predictor_dim
        z_proj = self.encoder_to_predictor(z_vis)

        x = self.mask_token.type_as(z_proj).expand(B, P, D).clone()
        x[visible_mask] = z_proj.reshape(-1, D)
        x = x + self.pos_embed

        film = self.film_gen(R_6d).view(B, self.depth, 2, D)
        for i, blk in enumerate(self.blocks):
            gamma = film[:, i, 0, :].unsqueeze(1)
            beta = film[:, i, 1, :].unsqueeze(1)
            x = x * (1.0 + gamma) + beta
            x = blk(x)
        x = self.norm(x)
        return self.predictor_to_encoder(x)


class DirectionHead(nn.Module):
    def __init__(self, encoder_dim: int, hidden_dim: int, patch_shape):
        super().__init__()
        self.patch_shape = patch_shape
        fs, ts = patch_shape
        self.head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3 * fs * ts),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, P, _ = z.shape
        fs, ts = self.patch_shape
        out = self.head(z).view(B, P, 3, fs, ts)
        return F.normalize(out, p=2, dim=2, eps=1e-6)


class SphereJEPA(pl.LightningModule):
    teacher_encoder_blocks: nn.ModuleList
    teacher_encoder_norm: nn.LayerNorm

    def __init__(
        self,
        # Encoder / GRAMT
        model_size: str = "base",
        encoder_depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        gramt_model_id: str = "labhamlet/gramt-mono",
        gramt_freeze: bool = True,

        # Predictor
        predictor_depth: int = 4,
        predictor_num_heads: int = 6,
        predictor_dim: int = 384,
        predictor_mlp_ratio: float = 4.0,

        # Direction head
        direction_head_hidden: int = 256,

        # JEPA / EMA
        ema_decay: float = 0.996,
        ema_end_decay: float = 0.99999,
        ema_anneal_end_step: int = 25_000,
        average_top_k_layers: int = 8,

        # Loss weights
        lambda_iv: float = 1.0,
        lambda_var_std: float = 0.1,
        lambda_var_cov: float = 0.1,

        # Optim
        lr: float = 2e-4,
        b1: float = 0.9,
        b2: float = 0.95,
        weight_decay: float = 0.01,

        # Audio / patch geometry
        patch_strategy: PatchStrategy = None,
        in_channels: int = 7,
        sr: int = 16000,
        num_mel_bins: int = 128,
        input_length: int = 500,
        target_length: int = 200,
        rotations_per_clip: int = 4,

        # Logging
        log_every_n_steps: int = 100,

        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["patch_strategy"])

        self.sr = sr
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.input_length = input_length
        self.rotations_per_clip = rotations_per_clip
        self.in_channels = in_channels
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.weight_decay = weight_decay
        self.lambda_iv = lambda_iv
        self.lambda_var_std = lambda_var_std
        self.lambda_var_cov = lambda_var_cov
        self.average_top_k_layers = average_top_k_layers
        self.ema_decay = ema_decay
        self.ema_end_decay = ema_end_decay
        self.ema_end_step = ema_anneal_end_step
        self.encoder_depth = encoder_depth
        self.log_every_n_steps = log_every_n_steps

        self.patch_strategy = patch_strategy
        self.p_f_dim, self.p_t_dim = self.patch_strategy.get_patch_size()
        self.num_patches = self.p_f_dim * self.p_t_dim
        self.grid_size = (self.p_f_dim, self.p_t_dim)
        fs, ts = self.patch_strategy.fshape, self.patch_strategy.tshape
        self.patch_shape = (fs, ts)

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
        assert flat_dim % self.gramt_n_freq == 0, (
            f"GRAMT 'raw' flat dim {flat_dim} not divisible by p_f_dim={self.gramt_n_freq}."
        )
        self.gramt_native_dim = flat_dim // self.gramt_n_freq
        assert T_gram * self.gramt_n_freq == self.num_patches, (
            f"GRAMT token grid {T_gram} x {self.gramt_n_freq} != spatial num_patches {self.num_patches}."
        )
        self.gramt_t_dim = T_gram

        # ---- spatial / fused encoder -----------------------------------
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

        # ---- predictor + direction head --------------------------------
        self.predictor = Predictor(
            encoder_dim=self.encoder_embedding_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            num_patches=self.num_patches,
            patch_grid_size=self.grid_size,
            mlp_ratio=predictor_mlp_ratio,
        )
        self.direction_head = DirectionHead(
            encoder_dim=self.encoder_embedding_dim,
            hidden_dim=direction_head_hidden,
            patch_shape=self.patch_shape,
        )

        # ---- mel feature extractor (kept identical to SphereV2) --------
        self.melspec = FeatureExtractor(
            sample_rate=self.sr, n_fft=1024, win_length=1024,
            hop_length=self.sr // 100, f_min=50, f_max=self.sr // 2,
            n_mels=self.num_mel_bins, power=2.0,
        ).float()

        self._init_our_weights()
        self._init_teacher()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        if self.gramt_freeze:
            self.gram.eval()
        self.teacher_encoder_blocks.eval()
        self.teacher_encoder_norm.eval()
        return self

    def _init_our_weights(self):
        def init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for mod in [self.patch_embed, self.gramt_proj,
                    self.encoder_blocks, self.encoder_norm,
                    self.predictor, self.direction_head]:
            if isinstance(mod, nn.Module):
                mod.apply(init_fn)

        for blk in self.encoder_blocks:
            nn.init.zeros_(blk.cross_attn.out_proj.weight)
            nn.init.zeros_(blk.cross_attn.out_proj.bias)

        # FiLM zero-init re-applied (init_fn would have xavier'd it)
        nn.init.zeros_(self.predictor.film_gen[-1].weight)
        nn.init.zeros_(self.predictor.film_gen[-1].bias)

    def _init_teacher(self):
        """Deepcopy the contextualising student modules. Per v2 §3 only these
        get an EMA partner; patch_embed / gramt_proj / cls_token / pos_embed
        are shared between sides."""
        self.teacher_encoder_blocks = copy.deepcopy(self.encoder_blocks)
        self.teacher_encoder_norm = copy.deepcopy(self.encoder_norm)
        for p in self.teacher_encoder_blocks.parameters():
            p.requires_grad = False
        for p in self.teacher_encoder_norm.parameters():
            p.requires_grad = False
        self.teacher_encoder_blocks.eval()
        self.teacher_encoder_norm.eval()

    def _update_patch_embed_layers(self, patch_embed):
        patch_embed.proj = nn.Conv2d(
            self.in_channels, self.encoder_embedding_dim,
            kernel_size=(self.patch_strategy.fshape, self.patch_strategy.tshape),
            stride=(self.patch_strategy.fstride, self.patch_strategy.tstride),
        )
        patch_embed.num_patch = self.num_patches

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------
    def _get_ema_decay(self) -> float:
        if self.global_step >= self.ema_end_step:
            return self.ema_end_decay
        # Linear anneal ema_decay -> ema_end_decay over ema_end_step steps.
        r = self.ema_end_decay - self.ema_decay
        pct_remaining = 1.0 - (self.global_step / self.ema_end_step)
        return self.ema_end_decay - r * pct_remaining

    @torch.no_grad()
    def _step_teacher(self):
        r = self._get_ema_decay()
        for s, t in zip(self.encoder_blocks.parameters(),
                        self.teacher_encoder_blocks.parameters()):
            t.data.mul_(r).add_((1.0 - r) * s.detach().data)
        for s, t in zip(self.encoder_norm.parameters(),
                        self.teacher_encoder_norm.parameters()):
            t.data.mul_(r).add_((1.0 - r) * s.detach().data)

    # ------------------------------------------------------------------
    # Patchify / IV target utilities — copied from SphereV2 verbatim so
    # this module is self-contained.
    # ------------------------------------------------------------------
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
        with torch.amp.autocast("cuda", enabled=False):
            waveform = waveform.float()
            rms = torch.sqrt(waveform[:, :1].pow(2).mean(dim=-1, keepdim=True) + 1e-8)
            waveform = waveform / rms
            mel = self.melspec(waveform)
            return mel.transpose(3, 2)


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


    def _embed_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify + add patch pos embed (no CLS, no masking)."""
        x = x.transpose(2, 3)
        embedded = self.patch_strategy.embed(x, self.patch_embed)
        embedded = embedded + self.pos_embed[:, 1:, :]
        return embedded  # (B, P, D)

    def _student_encode_visible(self, x: torch.Tensor, visible_mask: torch.Tensor,
                                gramt_tokens: torch.Tensor) -> torch.Tensor:
        """Student forward on visible patches only.
        Returns (B, 1+N_vis, D) including CLS at position 0.
        """
        B = x.shape[0]
        embedded = self._embed_patches(x)

        L_vis = visible_mask[0].sum().item()
        assert (visible_mask.sum(dim=1) == L_vis).all(), (
            "All samples in the batch must have the same number of visible patches."
        )
        z_vis = embedded[visible_mask].view(B, L_vis, -1)

        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        x = torch.cat([cls, z_vis], dim=1)
        x = self.encoder_dropout(x)

        for block in self.encoder_blocks:
            x = block(x, context=gramt_tokens)
        return self.encoder_norm(x)

    @torch.no_grad()
    def _teacher_encode_full(self, x: torch.Tensor,
                             gramt_tokens: torch.Tensor) -> torch.Tensor:
        """Teacher forward on the FULL (unmasked) sequence.

        Returns the data2vec-2 style target: instance-norm the last-K teacher
        layer outputs, then average. Skips the encoder_norm — instance-norm
        across features inside _make_targets handles normalisation.

        Output: (B, P, D_enc), CLS-stripped.
        """
        B = x.shape[0]
        embedded = self._embed_patches(x)
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        h = torch.cat([cls, embedded], dim=1)

        layer_outputs = []
        K = self.average_top_k_layers
        n_layers = len(self.teacher_encoder_blocks)
        for i, block in enumerate(self.teacher_encoder_blocks):
            h = block(h, context=gramt_tokens)
            if (n_layers - i) <= K:
                layer_outputs.append(h)

        if K > 1:
            target = self._make_targets(layer_outputs)
        else:
            target = layer_outputs[-1]
        # Strip CLS: target is (B, 1+P, D_enc) -> (B, P, D_enc)
        return target[:, 1:, :]

    @staticmethod
    def _make_targets(layer_outputs):
        """data2vec-2: per-layer instance norm over (features), then mean."""
        stacked = torch.stack(layer_outputs)              # (K, B, T, D)
        transposed = stacked.transpose(2, 3)              # (K, B, D, T)
        normalized = F.instance_norm(transposed)
        normalized = normalized.transpose(2, 3)
        return normalized.mean(dim=0)                     # (B, T, D)

    # ------------------------------------------------------------------
    # Batch prep — same shape contract as SphereV2._prepare_batch
    # ------------------------------------------------------------------
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
        all_fb = self._wav2fbank(all_wav)

        N, C, T_total, n_freq = all_fb.shape
        T_crop = self.target_length
        max_start = T_total - T_crop
        offsets = torch.randint(0, max_start + 1, (B,), device=self.device)
        all_offsets = torch.cat([offsets, offsets.repeat_interleave(Rn)], dim=0)

        t_range = torch.arange(T_crop, device=self.device)
        time_idx = (all_offsets.view(N, 1, 1, 1) + t_range.view(1, 1, T_crop, 1)) \
            .expand(N, C, T_crop, n_freq)
        all_fb = torch.gather(all_fb, dim=2, index=time_idx)

        clean = all_fb[:B]
        rotated = all_fb[B:].reshape(B * Rn, self.in_channels, T_crop, n_freq)
        R_mats_flat = R_mats.reshape(B * Rn, 3, 3)

        visible_mask = torch.zeros(B, self.num_patches, dtype=torch.bool,
                                   device=self.device)
        visible_mask.scatter_(1, context_idx.long(), True)

        return (clean.to(torch.bfloat16),
                rotated.to(torch.bfloat16),
                R_mats_flat,
                visible_mask)

    # ------------------------------------------------------------------
    # Forward — produces all losses
    # ------------------------------------------------------------------
    def forward(self, x_clean, x_rot_flat, R_mats_flat, visible_mask):
        N = x_clean.shape[0]
        Rn = x_rot_flat.shape[0] // N

        w_full = x_clean[:, 0:1]
        gramt_tokens_clean = self._gramt_tokens(w_full)
        gramt_tokens_rot = gramt_tokens_clean.repeat_interleave(Rn, dim=0)

        # ----- Student: visible patches of CLEAN -----------------------
        z_student = self._student_encode_visible(
            x_clean, visible_mask, gramt_tokens_clean
        )
        z_vis = z_student[:, 1:, :]   # drop CLS for predictor

        # ----- Teacher targets (no grad) -------------------------------
        with torch.no_grad():
            z_tgt_mask = self._teacher_encode_full(x_clean, gramt_tokens_clean)
            z_tgt_rot = self._teacher_encode_full(x_rot_flat, gramt_tokens_rot)

        # ----- Predictor: two passes -----------------------------------
        # Mask target: R = I (6D form is the first two cols of I).
        I = torch.eye(3, dtype=R_mats_flat.dtype, device=R_mats_flat.device)
        I_6d = matrix_to_6d(I.unsqueeze(0).expand(N, -1, -1))
        z_hat_mask = self.predictor(z_vis, visible_mask, I_6d)            # (N, P, D)

        # Rotation target: replicate visible/encoder outputs Rn times.
        z_vis_rep = z_vis.unsqueeze(1).expand(-1, Rn, -1, -1) \
                          .reshape(N * Rn, *z_vis.shape[1:])
        vmask_rep = visible_mask.unsqueeze(1).expand(-1, Rn, -1) \
                                 .reshape(N * Rn, -1)
        R_6d = matrix_to_6d(R_mats_flat)
        z_hat_rot = self.predictor(z_vis_rep, vmask_rep, R_6d)             # (N*Rn, P, D)

        masked = ~visible_mask                          # (N, P)
        masked_rep = ~vmask_rep                         # (N*Rn, P)

        l_jepa_mask = masked_mse(z_hat_mask, z_tgt_mask.detach(), masked)
        l_jepa_rot = masked_mse(z_hat_rot, z_tgt_rot.detach(), masked_rep)

        # ----- L_iv: cosine on direction head, weighted by IV magnitude
        # IV target derived from CLEAN scene (same scene the mask predictor
        # is reconstructing). Always available everywhere — never masked.
        iv_unit_clean, w_iv_clean = self._iv_target(x_clean)
        d_hat = self.direction_head(z_hat_mask)
        m_mask = masked.view(N, -1, 1, 1, 1).float()
        combined_weight = w_iv_clean * m_mask
        l_iv = cosine_loss(d_hat, iv_unit_clean, combined_weight)


        loss = (l_jepa_mask + l_jepa_rot
                + self.lambda_iv * l_iv)
        return {
            "loss": loss,
            "l_jepa_mask": l_jepa_mask,
            "l_jepa_rot": l_jepa_rot,
            "l_iv": l_iv,
            "z_hat_mask": z_hat_mask,
            "z_hat_rot": z_hat_rot,
            "z_tgt_mask": z_tgt_mask,
            "z_tgt_rot": z_tgt_rot,
            "iv_pred": d_hat,
            "iv_target": iv_unit_clean,
            "iv_weight": w_iv_clean,
            "masked": masked,
            "masked_rep": masked_rep,
            "z_vis": z_vis,
            "R_6d": R_6d,
        }

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        x_clean, x_rot, R_mats_flat, visible_mask = self._prepare_batch(batch)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = self.forward(x_clean, x_rot, R_mats_flat, visible_mask)

        loss = out["loss"]

        with torch.no_grad():
            # IV angular error on masked + visible patches (rotation branch).
            cos = (out["iv_pred"] * out["iv_target"]).sum(dim=2, keepdim=True) \
                       .clamp(-1.0, 1.0)
            ang = torch.acos(cos) * (180.0 / np.pi)
            m = out["masked"].view(*out["masked"].shape, 1, 1, 1).float()
            w = out["iv_weight"]
            ang_err_masked = (ang * m * w).sum() / (m * w).sum().clamp_min(1e-6)

        self.log_dict({
            "loss": loss,
            "l_jepa_mask": out["l_jepa_mask"],
            "l_jepa_rot": out["l_jepa_rot"],
            "l_iv": out["l_iv"],
            "ang_err_masked_deg": ang_err_masked,
            "ema_decay": self._get_ema_decay(),
        }, prog_bar=True)

        # FiLM-R diagnostic: does conditioning actually help?
        if self.global_step % self.log_every_n_steps == 0:
            with torch.no_grad():
                # Counterfactual 1: zero R_6d (degenerate input).
                z_hat_noR = self.predictor(
                    out["z_vis"].unsqueeze(1).expand(-1, x_rot.shape[0] // x_clean.shape[0], -1, -1)
                                .reshape(x_rot.shape[0], *out["z_vis"].shape[1:]),
                    visible_mask.unsqueeze(1)
                                .expand(-1, x_rot.shape[0] // x_clean.shape[0], -1)
                                .reshape(x_rot.shape[0], -1),
                    torch.zeros_like(out["R_6d"]),
                )
                R_swapped = torch.roll(out["R_6d"], shifts=1, dims=0)
                z_vis_rep = out["z_vis"].unsqueeze(1) \
                                        .expand(-1, x_rot.shape[0] // x_clean.shape[0], -1, -1) \
                                        .reshape(x_rot.shape[0], *out["z_vis"].shape[1:])
                vmask_rep = visible_mask.unsqueeze(1) \
                                        .expand(-1, x_rot.shape[0] // x_clean.shape[0], -1) \
                                        .reshape(x_rot.shape[0], -1)
                z_hat_swapR = self.predictor(z_vis_rep, vmask_rep, R_swapped)

                l_noR = masked_mse(z_hat_noR, out["z_tgt_rot"], out["masked_rep"])
                l_swap = masked_mse(z_hat_swapR, out["z_tgt_rot"], out["masked_rep"])
                self.log_dict({
                    "diag/jepa_rot_withR": out["l_jepa_rot"],
                    "diag/jepa_rot_noR": l_noR,
                    "diag/jepa_rot_swapR": l_swap,
                    "diag/gap_noR": l_noR - out["l_jepa_rot"],
                    "diag/gap_swapR": l_swap - out["l_jepa_rot"],
                }, prog_bar=False)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.amp.autocast("cuda", enabled=False):
            self._step_teacher()

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, self.lr, weight_decay=self.weight_decay,
            betas=(self.b1, self.b2),
        )
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=10000,
            num_training_steps=self.trainer.max_steps,
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def pass_through_encoder(self, x_input_7ch: torch.Tensor) -> torch.Tensor:
        B = x_input_7ch.shape[0]
        w_full = x_input_7ch[:, 0:1]
        gramt_tokens = self._gramt_tokens(w_full)
        embedded = self._embed_patches(x_input_7ch)
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        x = torch.cat([cls, embedded], dim=1)
        x = self.encoder_dropout(x)
        for block in self.encoder_blocks:
            x = block(x, context=gramt_tokens)
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