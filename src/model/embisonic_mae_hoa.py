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

from ..patching import PatchStrategy
from .pos_embed import get_2d_sincos_pos_embed
from .utils import PatchEmbed, create_pretrained_model, plot_fbank
from ..data_modules.dataset_functions import pad_or_truncate_batch
from .ambisonic_feature_extractor import FeatureExtractor

set_fused_attn(True); use_fused_attn(True)
pad_or_truncate_batch = torch.compile(pad_or_truncate_batch)


def wigner_d1_acn(R_xyz: torch.Tensor) -> torch.Tensor:
    """XYZ-convention rotation matrix -> 3x3 matrix acting on ACN-ordered
    (Y, Z, X) first-order ambisonic channels.

    ACN index 0 (Y) corresponds to XYZ index 1
    ACN index 1 (Z) corresponds to XYZ index 2
    ACN index 2 (X) corresponds to XYZ index 0
    """
    perm = [1, 2, 0]
    return R_xyz[..., perm, :][..., :, perm]


def rotation_to_sh_features(R: torch.Tensor) -> torch.Tensor:
    """Flattened first-order Wigner D^(1) in ACN ordering.
    R: (..., 3, 3) in XYZ convention. Returns: (..., 9).
    """
    return wigner_d1_acn(R).reshape(*R.shape[:-2], 9)

def rotate_foa_waveform(wav: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """ACN channels 1:4 ordered (Y, Z, X). Permute -> rotate -> permute back."""
    W, YZX = wav[:, :1], wav[:, 1:4]
    XYZ = YZX[:, [2, 0, 1]]
    XYZ_rot = torch.einsum("nij,njt->nit", R, XYZ)
    YZX_rot = XYZ_rot[:, [1, 2, 0]]
    return torch.cat([W, YZX_rot], dim=1)


def cosine_iv_loss(pred: torch.Tensor, target_unit: torch.Tensor,
                   weight: torch.Tensor,
                   sharpness: float = 1.0,
                   eps: float = 1e-4) -> torch.Tensor:
    """Weighted directional loss on unit IV directions.

    L = w * [ (1 - cos θ)  +  sharpness * sqrt(1 - cos θ + eps) ]

    The sqrt term is ~linear in θ near the target (vs. quadratic for pure
    cosine), preventing gradient saturation on well-predicted bins while
    keeping the overall loss bounded-gradient (no arccos singularity).

    Set sharpness=0.0 to recover the original cosine loss.

    pred, target_unit: (..., 3, ...) unit vectors along the '3' axis.
    weight: same shape with 1 in the '3' slot.
    """
    cos = (pred * target_unit).sum(dim=-3, keepdim=True).clamp(-1.0, 1.0)
    one_minus_cos = 1.0 - cos
    per_bin = one_minus_cos + sharpness * torch.sqrt(one_minus_cos + eps)
    per_bin = per_bin * weight
    return per_bin.sum() / weight.sum().clamp_min(1e-6)



class MaskedIVDecoder(nn.Module):
    """Takes visible encoder tokens + a mask layout + rotation R, and
    reconstructs the unit IV direction field at EVERY patch (including masked).

    Masked positions are filled with a learned mask token. Rotation conditioning
    goes through per-layer FiLM modulation.
    """
    def __init__(self, encoder_embed_dim: int, decoder_embed_dim: int,
                 depth: int, num_heads: int, patch_shape: tuple,
                 num_patches: int, grid_size: tuple, mlp_ratio: float = 2.0):
        super().__init__()
        self.depth = depth
        self.num_patches = num_patches
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_shape = patch_shape  # (fs, ts)

        # Project encoder features into decoder space.
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        # Learned mask token.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Fixed 2D sincos positional embeddings (patch grid).
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False
        )
        pe = get_2d_sincos_pos_embed(decoder_embed_dim, grid_size, cls_token_num=0)
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

        # FiLM generator (R_6d -> per-layer gamma, beta).
        self.film_gen = nn.Sequential(
            nn.Linear(9, decoder_embed_dim),
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

        # Output head: per-patch, per-TF-bin 3-vector.
        self.head = nn.Linear(decoder_embed_dim, 3 * patch_shape[0] * patch_shape[1])

    def forward(self, z_vis: torch.Tensor, visible_mask: torch.Tensor,
                R_sh: torch.Tensor) -> torch.Tensor:
        """
        z_vis: (N, L_vis, D_enc) — encoder outputs for visible patches only.
        visible_mask: (N, num_patches) bool — True where visible.
        R_sh: (N, 9) — flattened D^(1)(R) in ACN ordering.

        Returns: (N, num_patches, 3, fs, ts) unit vectors.
        """
        N = z_vis.shape[0]
        P = self.num_patches
        D = self.decoder_embed_dim

        z_vis_proj = self.decoder_embed(z_vis)                   # (N, L_vis, D)

        # Cast mask token to match AMP precision (e.g., bfloat16) before expanding/cloning
        x = self.mask_token.type_as(z_vis_proj).expand(N, P, D).clone()

        # Now both are the exact same dtype, so this works perfectly!
        x[visible_mask] = z_vis_proj.reshape(-1, D)
        # Add positional embeddings (after scatter, so mask tokens get PE too).
        x = x + self.pos_embed

        # Generate per-layer FiLM params from R.
        film = self.film_gen(R_sh).view(N, self.depth, 2, D)

        for i, blk in enumerate(self.blocks):
            gamma = film[:, i, 0, :].unsqueeze(1)
            beta = film[:, i, 1, :].unsqueeze(1)
            x = x * (1.0 + gamma) + beta
            x = blk(x)

        x = self.norm(x)
        iv = self.head(x)                                         # (N, P, 3*fs*ts)
        iv = iv.view(N, P, 3, self.patch_shape[0], self.patch_shape[1])
        iv = F.normalize(iv, p=2, dim=2, eps=1e-6)
        return iv


# ---------------------------------------------------------------------------
#  Model
# ---------------------------------------------------------------------------
class EmbisonicsMAE(pl.LightningModule):
    """Masked + rotation-equivariant self-supervised pretraining for FOA audio.

    Signal: from visible patches of the CLEAN input only, reconstruct the
    unit IV direction field of the ROTATED scene at every patch position.
    Diffuseness weighting is ||I_norm|| of the target (from FeatureExtractor).
    """
    def __init__(
            self,
            model_size: str = "base",
            lr: float = 2e-5,
            trainer: str = "adamW",
            b1: float = 0.9, b2: float = 0.95,
            weight_decay: float = 0.01,
            mlp_ratio: float = 4.0,
            patch_strategy: PatchStrategy = None,
            in_channels: int = 7,
            encoder_depth: Optional[int] = 8,
            decoder_depth: int = 2,
            decoder_num_heads: int = 4,
            decoder_embedding_dim: int = 384,
            sr: int = 32000,
            num_mel_bins: int = 128,
            input_length: int = 500,
            rotations_per_clip: int = 8,
            masked_weight: float = 1.0,
            visible_weight: float = 0.1,
            clean_weight: float = 1.0,
            diag_every_n_steps: int = 500,
            log_every_n_steps: int = 500,
            loss_sharpness: float = 0.5,
            **kwargs,
        ):
        super().__init__()
        self.save_hyperparameters(ignore=['patch_strategy'])

        # Knobs
        self.sr = sr
        self.num_mel_bins = num_mel_bins
        self.loss_sharpness = loss_sharpness
        self.target_length = input_length
        self.input_length = input_length
        self.rotations_per_clip = rotations_per_clip
        self.in_channels = in_channels
        self.masked_weight = masked_weight
        self.visible_weight = visible_weight
        self.clean_weight = clean_weight
        self.diag_every_n_steps = diag_every_n_steps
        self.log_every_n_steps = log_every_n_steps
        self.lr = lr
        self.trainer_name = trainer
        self.b1 = b1; self.b2 = b2; self.weight_decay = weight_decay

        # Patch geometry
        self.patch_strategy = patch_strategy
        self.p_f_dim, self.p_t_dim = self.patch_strategy.get_patch_size()
        self.num_patches = self.p_f_dim * self.p_t_dim
        self.grid_size = (self.p_f_dim, self.p_t_dim)
        fs, ts = self.patch_strategy.fshape, self.patch_strategy.tshape
        self.iv_patch_shape = (fs, ts)

        # Encoder
        self.encoder, self.encoder_embedding_dim = create_pretrained_model(model_size)
        self.encoder_cls_token_num = 1
        if encoder_depth is not None:
            self.encoder.layers = nn.ModuleList(list(self.encoder.layers)[:encoder_depth])

        self.patch_embed = PatchEmbed()
        self._update_patch_embed_layers(self.patch_embed)
        self.cls_token = nn.Parameter(
            nn.init.normal_(torch.empty([1, 1, self.encoder_embedding_dim]), std=0.02)
        )
        self.encoder.pos_embedding = self._get_pos_embed_params()

        # Decoder
        self.decoder = MaskedIVDecoder(
            encoder_embed_dim=self.encoder_embedding_dim,
            decoder_embed_dim=decoder_embedding_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            patch_shape=self.iv_patch_shape,
            num_patches=self.num_patches,
            grid_size=self.grid_size,
            mlp_ratio=2.0,
        )

        self.melspec = FeatureExtractor(
            sample_rate=self.sr, n_fft=1024, win_length=1024,
            hop_length=self.sr // 100, f_min=50, f_max=self.sr // 2,
            n_mels=self.num_mel_bins, power=2.0,
        ).float()

        self.apply(self._init_weights)

    # ---- setup helpers --------------------------------------------------
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _get_pos_embed_params(self):
        pe = nn.Parameter(
            torch.zeros(1, self.num_patches + self.encoder_cls_token_num,
                        self.encoder_embedding_dim),
            requires_grad=False)
        data = get_2d_sincos_pos_embed(
            self.encoder_embedding_dim, self.grid_size,
            cls_token_num=self.encoder_cls_token_num)
        pe.data.copy_(torch.from_numpy(data).float().unsqueeze(0))
        return pe

    def _update_patch_embed_layers(self, patch_embed):
        patch_embed.proj = nn.Conv2d(
            self.in_channels, self.encoder_embedding_dim,
            kernel_size=(self.patch_strategy.fshape, self.patch_strategy.tshape),
            stride=(self.patch_strategy.fstride, self.patch_strategy.tstride))
        patch_embed.num_patch = self.num_patches

    # ---- IV target utils -------------------------------------------------
    def _iv_direction_and_weight(self, x: torch.Tensor):
        """From channels 4:7 (energy-normalized IV), split into unit direction
        and diffuseness magnitude in [0, 1]. Returned in patch layout.

        x: (B, 7, T, F)
        Returns: iv_unit (B, P, 3, fs, ts), weight (B, P, 1, fs, ts)
        """
        fs, ts = self.patch_strategy.fshape, self.patch_strategy.tshape
        assert self.patch_strategy.fstride == fs and self.patch_strategy.tstride == ts, \
            "Assumes non-overlapping patches."

        iv = x[:, 4:7]  # remains bfloat16
        mag = torch.linalg.norm(iv, dim=1, keepdim=True)
        iv_unit = iv / mag.clamp_min(1e-6)
        weight = mag.clamp(0.0, 1.0)

        iv_unit = iv_unit.transpose(2, 3)                 # (B, 3, F, T)
        weight = weight.transpose(2, 3)                   # (B, 1, F, T)
        B, C, Fm, Tm = iv_unit.shape
        p_f, p_t = Fm // fs, Tm // ts
        assert p_f == self.p_f_dim and p_t == self.p_t_dim

        iv_unit = iv_unit.view(B, C, p_f, fs, p_t, ts) \
                         .permute(0, 2, 4, 1, 3, 5).contiguous() \
                         .view(B, p_f * p_t, C, fs, ts)
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
            if self.in_channels <= 2:
                return torch.log(mel + 1e-5).transpose(3, 2)
            return mel.transpose(3, 2)

    def _encode_visible(self, x: torch.Tensor, visible_mask: torch.Tensor):
        """Encode only visible patches.

        x: (B, C, T, F)
        visible_mask: (B, num_patches) bool — True where visible.

        Returns: z_vis (B, L_vis + 1, D) — includes CLS token at position 0.
        """
        B = x.shape[0]
        x = x.transpose(2, 3)                                 # (B, C, F, T)
        embedded = self.patch_strategy.embed(x, self.patch_embed)  # (B, P, D)

        # Add patch positional embeddings BEFORE masking (so each visible
        # patch still carries its original grid position).
        embedded = embedded + self.encoder.pos_embedding[:, self.encoder_cls_token_num:, :]

        # Select visible patches. All samples in the batch share L_vis because
        # the masker produces a fixed-count visible set per sample.
        L_vis = visible_mask[0].sum().item()
        assert (visible_mask.sum(dim=1) == L_vis).all(), \
            "All samples in a batch must have the same number of visible patches."
        z_vis = embedded[visible_mask].view(B, L_vis, -1)

        cls = self.cls_token.expand(B, -1, -1) + self.encoder.pos_embedding[:, :1, :]
        x = torch.cat((cls, z_vis), dim=1)
        x = self.encoder.dropout(x)
        for block in self.encoder.layers:
            x = block(x)
        return self.encoder.ln(x)                              # (B, 1 + L_vis, D)

    @torch.no_grad()
    def _prepare_batch(self, batch):
        """batch yields:
            audio:       (B, 4, T_wav)   — raw FOA waveform
            context_idx: (B, L_vis)      — visible patch indices
            R_mats:      (B, Rn, 3, 3)   — rotation matrices
        """
        audio, context_idx, R_mats = batch

        audio       = audio.to(self.device, non_blocking=True)
        context_idx = context_idx.to(self.device, non_blocking=True)
        R_mats      = R_mats.to(self.device, non_blocking=True)

        B  = audio.shape[0]
        Rn = R_mats.shape[1]

        # 1) Rotate waveforms in bulk: (B*Rn, 4, T_wav).
        wav_exp     = audio.unsqueeze(1).expand(-1, Rn, -1, -1).reshape(B * Rn, 4, -1)
        R_flat      = R_mats.reshape(B * Rn, 3, 3).to(audio.dtype)
        rotated_wav = rotate_foa_waveform(wav_exp, R_flat)

        # 2) Feature extraction for clean + all rotations in one batched call.
        #    pad_or_truncate_batch trims/pads to target_length directly — no crop needed.
        all_wav = torch.cat([audio, rotated_wav], dim=0)      # (B + B*Rn, 4, T_wav)
        all_fb  = self._wav2fbank(all_wav)
        all_fb  = pad_or_truncate_batch(all_fb, self.target_length)

        clean   = all_fb[:B]                                  # (B, 7, T, F)
        rotated = all_fb[B:].reshape(B * Rn, self.in_channels, self.target_length, all_fb.shape[-1])
                                                              # (B*Rn, 7, T, F)

        # 3) Expand R to match the flat (B*Rn,) batch layout.
        R_mats_flat = R_mats.reshape(B * Rn, 3, 3)

        # 4) Build the boolean visible mask from patch indices.
        #    context_idx: (B, L_vis) — no sample dimension any more.
        visible_mask = torch.zeros(B, self.num_patches, dtype=torch.bool, device=self.device)
        visible_mask.scatter_(1, context_idx.long(), True)

        return clean.to(torch.bfloat16), rotated.to(torch.bfloat16), R_mats_flat, visible_mask

    # ---- forward / loss --------------------------------------------------
    def forward(self, x_clean, x_rot_flat, R_mats_flat, visible_mask):
        """
        x_clean:      (N, 7, T, F)              — N = B
        x_rot_flat:   (N*Rn, 7, T, F)
        R_mats_flat:  (N*Rn, 3, 3)
        visible_mask: (N, num_patches) bool
        """
        N = x_clean.shape[0]
        Rn = x_rot_flat.shape[0] // N

        # Targets (unit IV + diffuseness weight).
        iv_unit_clean, w_clean = self._iv_direction_and_weight(x_clean)       # (N, P, ...)
        iv_unit_rot, w_rot = self._iv_direction_and_weight(x_rot_flat)        # (N*Rn, P, ...)

        # Encode visible patches of the CLEAN input only.
        z = self._encode_visible(x_clean, visible_mask)                       # (N, 1+L_vis, D)
        z_vis = z[:, 1:, :]                                                   # drop CLS; (N, L_vis, D)


        # --- Term 1: reconstruct the CLEAN direction field (R = I) ---
        R_id = torch.eye(3, device=self.device).view(1, 3, 3).expand(N, -1, -1)
        R_id_sh = rotation_to_sh_features(R_id)
        pred_clean = self.decoder(z_vis, visible_mask, R_id_sh)               # (N, P, 3, fs, ts)

        # --- Term 2: reconstruct the ROTATED direction fields ---
        z_vis_rep = z_vis.unsqueeze(1).expand(-1, Rn, -1, -1).reshape(N * Rn, *z_vis.shape[1:])
        visible_mask_rep = visible_mask.unsqueeze(1).expand(-1, Rn, -1).reshape(N * Rn, -1)
        R_sh_flat = rotation_to_sh_features(R_mats_flat)
        pred_rot = self.decoder(z_vis_rep, visible_mask_rep, R_sh_flat)       # (N*Rn, P, 3, fs, ts)

        # --- Split losses: masked vs visible, for both clean and rotated ---
        mask_mask = ~visible_mask                                             # (N, P) True = MASKED
        mask_mask_rep = mask_mask.unsqueeze(1).expand(-1, Rn, -1).reshape(N * Rn, -1)
        def split_loss(pred, target, w, vis_mask):
            vmask = vis_mask.view(*vis_mask.shape, 1, 1, 1).float()
            mmask = 1.0 - vmask
            w_vis = w * vmask
            w_msk = w * mmask
            return (cosine_iv_loss(pred, target, w_msk, self.loss_sharpness),
                    cosine_iv_loss(pred, target, w_vis, self.loss_sharpness))

        l_clean_masked, l_clean_vis = split_loss(pred_clean, iv_unit_clean, w_clean, visible_mask)
        l_rot_masked, l_rot_vis = split_loss(pred_rot, iv_unit_rot, w_rot, visible_mask_rep)

        # Weighted combination.
        loss_rot = self.masked_weight * l_rot_masked + self.visible_weight * l_rot_vis
        loss_clean = self.masked_weight * l_clean_masked + self.visible_weight * l_clean_vis
        total = loss_rot + self.clean_weight * loss_clean

        return {
            "loss": total,
            "rot_masked": l_rot_masked,
            "rot_visible": l_rot_vis,
            "clean_masked": l_clean_masked,
            "clean_visible": l_clean_vis,
            "pred_rot": pred_rot,
            "iv_unit_rot": iv_unit_rot,
            "w_rot": w_rot,
            "mask_mask_rep": mask_mask_rep,
            "z_vis": z_vis,
            "visible_mask": visible_mask,
            "R_sh_flat": R_sh_flat,
        }

    # ---- TensorBoard logging helpers -------------------------------------
    def _fbank_to_patches(self, fbank: torch.Tensor) -> torch.Tensor:
        """Fold (B, C, T, F) fbank into (B, P, C, fs, ts) non-overlapping patches.

        Mirrors _iv_direction_and_weight exactly:
          (B, C, T, F) --transpose(2,3)--> (B, C, F, T)
          then split F into (pf, fs) and T into (pt, ts).
        fshape/pf index the F axis; tshape/pt index the T axis.
        """
        fs, ts = self.patch_strategy.fshape, self.patch_strategy.tshape
        pf, pt = self.p_f_dim, self.p_t_dim
        B, C, T, F = fbank.shape
        # match the transpose done before patching everywhere else in the model
        x = fbank.transpose(2, 3)                          # (B, C, F, T)
        x = x.reshape(B, C, pf, fs, pt, ts)               # split F and T into patches
        return x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(B, pf * pt, C, fs, ts)

    def _patches_to_img(self, patches: torch.Tensor) -> np.ndarray:
        """Invert _fbank_to_patches: (B, P, C, fs, ts) -> numpy (B, C, T, F).

        plot_fbank expects (C, T, F) — time on the row axis, freq on the col axis.
        We reconstruct (B, C, F, T) from the patch grid then swap F and T back.
        """
        fs, ts = self.patch_strategy.fshape, self.patch_strategy.tshape
        pf, pt = self.p_f_dim, self.p_t_dim
        B, P, C, _, _ = patches.shape
        x = patches.reshape(B, pf, pt, C, fs, ts)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()     # (B, C, pf, fs, pt, ts)
        x = x.reshape(B, C, pf * fs, pt * ts)             # (B, C, F, T)
        x = x.transpose(2, 3)                             # (B, C, T, F) for plot_fbank
        return x.cpu().float().numpy()

    def _log_spectrogram(self, fbank: torch.Tensor, title: str, loss=None):
        """Log the first sample's full spectrogram to TensorBoard.

        fbank: (B, C, T, F) — plot_fbank receives (C, F, T).
        """
        with torch.no_grad():
            patches = self._fbank_to_patches(fbank[:1].float())
        img = self._patches_to_img(patches)[0]             # (C, F, T)

        caption = f"Loss: {loss:.4f}" if loss is not None else title
        fig = plot_fbank(img, vmin=img.min(), vmax=img.max(), title=caption)
        self.logger.experiment.add_figure(title, fig, global_step=self.global_step)
        plt.close(fig)

    def _log_spectrogram_with_mask(self, fbank: torch.Tensor,
                                   visible_mask: torch.Tensor, title: str,
                                   show_visible: bool = True):
        """Log the first sample's spectrogram with certain patches zeroed out.

        show_visible=True  → keep VISIBLE patches, zero the masked ones.
        show_visible=False → keep MASKED  patches, zero the visible ones.

        fbank:        (B, C, T, F)
        visible_mask: (B, P) bool — True = visible to encoder.
        """
        with torch.no_grad():
            patches = self._fbank_to_patches(fbank[:1].float()).cpu()  # (1, P, C, fs, ts)

        # visible_mask: True = visible patch.
        # show_visible=True  → we want to show visible → zero patches where NOT visible
        # show_visible=False → we want to show masked  → zero patches where visible
        vis = visible_mask[0].cpu()       # (P,) True = visible
        zero = vis if not show_visible else ~vis
        patches[:, zero] = 0.0

        img = self._patches_to_img(patches)[0]   # (C, F, T)
        fig = plot_fbank(img, vmin=img.min(), vmax=img.max(), title=title)
        self.logger.experiment.add_figure(title, fig, global_step=self.global_step)
        plt.close(fig)

    def _log_iv_angular_error(self, iv_pred: torch.Tensor,
                               iv_target: torch.Tensor,
                               visible_mask: torch.Tensor, title: str):
        """Log a 2-D per-patch angular-error heatmap to TensorBoard.

        iv_pred / iv_target : (N, P, 3, fs, ts) unit vectors.
        visible_mask        : (N, P) bool — True = visible (encoder saw it).

        Masked patches are outlined in cyan so reconstruction difficulty is
        immediately apparent.
        """
        with torch.no_grad():
            cos = (iv_pred[:1] * iv_target[:1]).sum(dim=2).clamp(-1.0, 1.0)  # (1, P, fs, ts)
            ang = torch.acos(cos) * (180.0 / np.pi)                           # degrees
            ang_per_patch = ang.mean(dim=(-2, -1)).squeeze(0).cpu().float()   # (P,)

        pf, pt = self.p_f_dim, self.p_t_dim
        grid = ang_per_patch.view(pf, pt).numpy()

        fig, ax = plt.subplots(figsize=(max(4, pt // 2), max(3, pf // 2)))
        im = ax.imshow(grid, aspect="auto", origin="lower",
                       vmin=0, vmax=90, cmap="plasma")
        fig.colorbar(im, ax=ax, label="Angular error (°)")

        # Outline masked patches in cyan.
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



    @torch.no_grad()
    def generate_clip_level_steering_plot(self, x_clean, num_steps=36, save_path="steering_sweep.png"):
        """
        Sweeps a 360-degree Yaw rotation, forces the decoder to predict the rotated scenes,
        and calculates the global, energy-weighted clip-level Intensity Vector.
        
        This version uses NO MASK (100% patch visibility).
        
        x_clean: (1, 7, T, F) - A single cleanly encoded batch sample.
        """

        self.eval()
        device = x_clean.device
        
        _, w_clean = self._iv_direction_and_weight(x_clean)
        
        x_transposed = x_clean.transpose(2, 3) 
        embedded = self.patch_strategy.embed(x_transposed, self.patch_embed)
        
        z = self.pass_through_encoder(embedded, 1)
        z_vis = z[:, self.encoder_cls_token_num:, :] # (1, num_patches, D)
        
        full_mask = torch.ones((1, self.num_patches), dtype=torch.bool, device=device)
        
        predicted_global_ivs = []
        angles_deg = np.linspace(0, 360, num_steps, endpoint=False)
        angles_rad = np.deg2rad(angles_deg)
        
        for theta in angles_rad:
            # 3. Create a Yaw rotation matrix (Rotation around Z-axis)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            
            # Ambisonic ACN/SN3D standard: Rotation around Z (up) affects X (front) and Y (left)
            R = torch.tensor([
                [ cos_t, -sin_t,  0.0],
                [ sin_t,  cos_t,  0.0],
                [   0.0,    0.0,  1.0]
            ], dtype=x_clean.dtype, device=device).unsqueeze(0) 
            

            R_sh = rotation_to_sh_features(R)
            
            pred_iv = self.decoder(z_vis, full_mask, R_sh) # (1, P, 3, fs, ts)            
            
            weighted_iv = pred_iv * w_clean 
            
            clip_vector = weighted_iv.sum(dim=(1, 3, 4)) # (1, 3)
            
            # Normalize to unit length for directional visualization
            clip_vector = torch.nn.functional.normalize(clip_vector, p=2, dim=1)
            
            predicted_global_ivs.append(clip_vector.squeeze(0).cpu().float().numpy())
            
        predicted_global_ivs = np.array(predicted_global_ivs) # (num_steps, 3)
        
        fig, ax = plt.subplots(figsize=(7, 7))
        
        # Plot predicted X (Front) and Y (Left)
        X_pred = predicted_global_ivs[:, 2] # Use Index 2 for Front-Back (X)
        Y_pred = predicted_global_ivs[:, 0] # Use Index 0 for Left-Right (Y)

        # Optional: You can also calculate Z to see how much elevation exists
        Z_mean = predicted_global_ivs[:, 1].mean()
        print(f"Average Elevation (Z): {Z_mean:.4f}")
        
        # Reference Circle
        circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', alpha=0.3)
        ax.add_patch(circle)
        ax.axhline(0, color='black', lw=0.5, alpha=0.3)
        ax.axvline(0, color='black', lw=0.5, alpha=0.3)
        
        # Plot dots colored by angle
        scatter = ax.scatter(X_pred, Y_pred, c=angles_deg, cmap='hsv', s=100, edgecolor='k', zorder=3)
        
        # Add an arrow showing the very first prediction (0 degrees)
        ax.quiver(0, 0, X_pred[0], Y_pred[0], color='red', angles='xy', scale_units='xy', scale=1, 
                label="Predicted Forward (0°)")
        
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Injected Rotation Angle (Degrees)')
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_xlabel("Predicted Front-Back (X)")
        ax.set_ylabel("Predicted Left-Right (Y)")
        ax.set_title("Fully Visible Decoder Steering\n(Global Weighted Intensity Vector)")
        ax.legend(loc='lower right')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        return predicted_global_ivs
    

    @torch.no_grad()
    def generate_masked_steering_plot(self, x_clean, mask_ratio=0.75, save_path="steering_masked.png"):
        """
        Randomly masks patches of the input, then sweeps rotation.
        Proves the model can 'inpaint' the spatial field and rotate it from partial context.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        self.eval()
        device = x_clean.device
        B, C, T, F = x_clean.shape
        
        # 1. Get Ground Truth energy weights (for the global weighted average)
        _, w_clean = self._iv_direction_and_weight(x_clean)
        
        # 2. Generate a Random Mask (standard MAE style)
        num_masked = int(self.num_patches * mask_ratio)
        # Create a random permutation of patch indices
        perm = torch.randperm(self.num_patches, device=device)
        # Indices to keep (visible)
        context_idx = perm[num_masked:].unsqueeze(0) # (1, L_vis)
        
        # Create the boolean mask for the decoder
        visible_mask = torch.zeros((1, self.num_patches), dtype=torch.bool, device=device)
        visible_mask.scatter_(1, context_idx, True)
        
        # 3. Encode ONLY the visible patches
        # _encode_visible handles embedding + pos_embed + transformer blocks
        z = self._encode_visible(x_clean, visible_mask)
        z_vis = z[:, 1:, :] # (1, L_vis, D) - drop CLS
        
        predicted_global_ivs = []
        angles_deg = np.linspace(0, 360, 36, endpoint=False)
        
        for ang in angles_deg:
            theta = np.deg2rad(ang)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            
            # Rotation Matrix (Yaw)
            R = torch.tensor([
                [ cos_t, -sin_t,  0.0],
                [ sin_t,  cos_t,  0.0],
                [   0.0,    0.0,  1.0]
            ], dtype=x_clean.dtype, device=device).unsqueeze(0)
            

            R_sh = rotation_to_sh_features(R)
            
            # 4. Decode the FULL scene from PARTIAL context + Rotation
            # The decoder uses its learned mask_token for the missing (num_masked) patches
            pred_iv = self.decoder(z_vis, visible_mask, R_sh) # (1, P, 3, fs, ts)            
            
            # 5. Global Weighted Average
            weighted_iv = pred_iv * w_clean
            clip_vector = weighted_iv.sum(dim=(1, 3, 4))
            clip_vector = torch.nn.functional.normalize(clip_vector, p=2, dim=1)
            
            predicted_global_ivs.append(clip_vector.squeeze(0).cpu().float().numpy())
            
        predicted_global_ivs = np.array(predicted_global_ivs)
        
        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(7, 7))
        X_pred, Y_pred = predicted_global_ivs[:, 2], predicted_global_ivs[:, 0]
        
        circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', alpha=0.3)
        ax.add_patch(circle)
        
        scatter = ax.scatter(X_pred, Y_pred, c=angles_deg, cmap='hsv', s=100, edgecolor='k', zorder=3)
        
        plt.colorbar(scatter, label='Injected Rotation Angle')
        
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_title(f"Masked Decoder Steering (Ratio: {mask_ratio*100:.0f}%)\n"
                    f"Reconstructing {num_masked} hidden patches")
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    # ---- training / optim ------------------------------------------------
    def training_step(self, batch, batch_idx):
        x_clean, x_rot, R_mats_flat, visible_mask = self._prepare_batch(batch)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = self.forward(x_clean, x_rot, R_mats_flat, visible_mask)

        loss = out["loss"]

        # Angular error on masked, rotated patches (the hardest diagnostic).
        with torch.no_grad():
            mmask = out["mask_mask_rep"].view(*out["mask_mask_rep"].shape, 1, 1, 1).float()
            w_masked = out["w_rot"] * mmask
            cos = (out["pred_rot"] * out["iv_unit_rot"]).sum(dim=2, keepdim=True).clamp(-1.0, 1.0)
            ang = torch.acos(cos) * (180.0 / np.pi)
            ang_err_masked = (ang * w_masked).sum() / w_masked.sum().clamp_min(1e-6)

        self.log_dict({
            "loss": loss,
            "rot_masked": out["rot_masked"],
            "rot_visible": out["rot_visible"],
            "clean_masked": out["clean_masked"],
            "clean_visible": out["clean_visible"],
            "ang_err_masked_deg": ang_err_masked,
        }, prog_bar=True)

        # ---- Periodic TensorBoard image logs ----------------------------
        if self.global_step % self.log_every_n_steps == 0:
            Rn = x_rot.shape[0] // x_clean.shape[0]
            visible_mask_rep = (
                visible_mask.unsqueeze(1).expand(-1, Rn, -1)
                            .reshape(x_rot.shape[0], -1)
            )

            self._log_spectrogram(
                x_clean, title="spectrogram/clean_input", loss=loss.item()
            )
            self._log_spectrogram_with_mask(
                x_clean, visible_mask,
                title="spectrogram/visible_patches", show_visible=True
            )
            self._log_spectrogram_with_mask(
                x_clean, visible_mask,
                title="spectrogram/masked_patches", show_visible=False
            )
            self._log_spectrogram(
                x_rot[:x_clean.shape[0]],
                title="spectrogram/rotated_input"
            )
            self._log_iv_angular_error(
                out["pred_rot"], out["iv_unit_rot"],
                visible_mask_rep, title="iv/angular_error_heatmap"
            )

        if self.global_step % self.diag_every_n_steps == 0 and self.global_step > 0:
            with torch.no_grad():
                z_vis = out["z_vis"]
                N, Rn = x_clean.shape[0], x_rot.shape[0] // x_clean.shape[0]
                z_vis_rep = z_vis.unsqueeze(1).expand(-1, Rn, -1, -1) \
                                  .reshape(N * Rn, *z_vis.shape[1:])
                vmask_rep = visible_mask.unsqueeze(1).expand(-1, Rn, -1).reshape(N * Rn, -1)

                R_id_baseline = rotation_to_sh_features(
                    torch.eye(3, device=self.device).expand(out["R_sh_flat"].shape[0], -1, -1)
                )
                pred_noR = self.decoder(z_vis_rep, vmask_rep, R_id_baseline)
                w_masked = out["w_rot"] * (~vmask_rep).view(*vmask_rep.shape, 1, 1, 1).float()
                l_noR = cosine_iv_loss(pred_noR, out["iv_unit_rot"], w_masked)
                l_withR = out["rot_masked"]
                print(f"--- step {self.global_step} | "
                      f"masked cos-loss: with R={l_withR:.4f}  "
                      f"no R={l_noR:.4f}  gap={l_noR - l_withR:.4f}  "
                      f"ang_err(masked)={ang_err_masked:.2f}°")

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

    # ---- downstream eval -------------------------------------------------
    def pass_through_encoder(self, x, B):
        """Unmasked encoder pass for downstream representation extraction."""
        x = x + self.encoder.pos_embedding[:, self.encoder_cls_token_num:, :]
        cls = self.cls_token.expand(B, -1, -1) + self.encoder.pos_embedding[:, :1, :]
        x = torch.cat((cls, x), dim=1)
        x = self.encoder.dropout(x)
        for block in self.encoder.layers:
            x = block(x)
        return self.encoder.ln(x)

    def get_audio_representation(self, x, strategy="mean"):
        B = x.shape[0]
        x = x.transpose(2, 3)
        embedded = self.patch_strategy.embed(x, self.patch_embed)
        x = self.pass_through_encoder(embedded, B)
        if strategy == "mean":
            return x[:, self.encoder_cls_token_num:, :].mean(axis=1)
        if strategy == "sum":
            return x[:, self.encoder_cls_token_num:, :].sum(axis=1)
        if strategy == "cls":
            return x[:, 0, :]
        if strategy == "raw":
            x = x[:, self.encoder_cls_token_num:, :]
            f = self.grid_size[0]
            return rearrange(x, "b (f t) d -> b t (f d)",
                             f=f, d=self.encoder_embedding_dim)
        raise ValueError(f"Unknown strategy '{strategy}'")