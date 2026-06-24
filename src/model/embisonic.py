from typing import Optional
from einops import rearrange

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import transformers

import matplotlib.pyplot as plt
import numpy as np

from timm.models.vision_transformer import Block
from timm.layers.config import set_fused_attn, use_fused_attn

from ..patching import PatchStrategy
from .pos_embed import get_2d_sincos_pos_embed
from .utils import PatchEmbed, create_pretrained_model
from ..data_modules.dataset_functions import pad_or_truncate_batch
from .ambisonic_feature_extractor import FeatureExtractor

set_fused_attn(True); use_fused_attn(True)
pad_or_truncate_batch = torch.compile(pad_or_truncate_batch)




def matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
    """Zhou et al. 2019 continuous 6D representation: first two columns of R."""
    return R[..., :, :2].reshape(*R.shape[:-2], 6)


def rotate_foa_waveform(wav: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """ACN channels 1:4 are ordered (Y, Z, X). Permute to (X, Y, Z), apply the
    standard Cartesian rotation R, then permute back. Equivalent to applying
    the l=1 Wigner D in e3nn's (y,z,x) basis, but without the CPU generators."""
    W, YZX = wav[:, :1], wav[:, 1:4]
    XYZ = YZX[:, [2, 0, 1]]                              # (Y,Z,X) -> (X,Y,Z)
    XYZ_rot = torch.einsum("nij,njt->nit", R, XYZ)
    YZX_rot = XYZ_rot[:, [1, 2, 0]]                      # (X,Y,Z) -> (Y,Z,X)
    return torch.cat([W, YZX_rot], dim=1)

def vicreg_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4):
    """VICReg-style variance-hinge + off-diagonal covariance. z: (N, D)."""
    z = z - z.mean(dim=0, keepdim=True)
    N, D = z.shape
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = F.relu(gamma - std).mean()
    cov = (z.T @ z) / max(N - 1, 1)
    cov_loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / D
    return var_loss, cov_loss


class EquivarianceDecoder(nn.Module):
    """
    Equivariance Decoder using Deep FiLM conditioning and fixed 2D sincos PE.
    Replaces the R-token approach with feature-wise linear modulation.
    """
    def __init__(self, embed_dim: int, depth: int, num_heads: int, mlp_ratio: float = 4.0,
                 num_patches: int = None, grid_size: tuple = None):
        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim

        if num_patches is not None and grid_size is not None:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim), requires_grad=False
            )
            # cls_token_num=0 because FiLM doesn't use a conditioning token
            pe = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token_num=0)
            self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))
        else:
            self.pos_embed = None

        # FiLM
        # Generates (gamma, beta) pairs for every layer
        self.film_gen = nn.Sequential(
            nn.Linear(6, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, depth * embed_dim * 2),
        )
        
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, z: torch.Tensor, R_6d: torch.Tensor) -> torch.Tensor:
        """
        z: (N, L, D) - Encoder patch features
        R_6d: (N, 6) - Rotation representation
        """
        N, L, D = z.shape

        # A. Add fixed positional embeddings
        if self.pos_embed is not None:
            z = z + self.pos_embed[:, :L, :]

        # B. Generate FiLM parameters for all layers
        # Shape: (N, depth, 2, D)
        film_params = self.film_gen(R_6d).view(N, self.depth, 2, D)

        # C. Process Blocks with per-layer FiLM
        for i, blk in enumerate(self.blocks):
            # Extract gamma (scale) and beta (shift) for the current block
            # Broadcast to (N, 1, D) to apply across the sequence L
            gamma = film_params[:, i, 0, :].unsqueeze(1)
            beta = film_params[:, i, 1, :].unsqueeze(1)

            # Apply modulation: x = x * (1 + gamma) + beta
            z = z * (1.0 + gamma) + beta
            
            # Standard Transformer Block
            z = blk(z)

        z = self.norm(z)
        return z


class IVDecoder(nn.Module):
    """
    FiLM-Conditioned IV Decoder.
    Given latents (z) and a rotation (R), it reconstructs the 
    rotated physical Intensity Vector field.
    """
    def __init__(self, embed_dim: int, depth: int, num_heads: int, 
                 patch_shape: tuple, mlp_ratio: float = 4.0,
                 num_patches: int = None, grid_size: tuple = None):
        super().__init__()
        self.patch_shape = patch_shape # (fs, ts)
        self.embed_dim = embed_dim
        self.depth = depth

        # 1. Fixed Sincos PE
        if num_patches is not None and grid_size is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
            pe = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token_num=0)
            self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))
        else:
            self.pos_embed = None

        # 2. Deep FiLM for IV grounding
        # This maps the rotation R into the scaling/shifting factors for IV reconstruction
        self.film_gen = nn.Sequential(
            nn.Linear(6, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, depth * embed_dim * 2),
        )
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        
        # 3. Physical output head
        self.head = nn.Linear(embed_dim, 3 * patch_shape[0] * patch_shape[1])

    def forward(self, z: torch.Tensor, R_6d: torch.Tensor) -> torch.Tensor:
        N, L, D = z.shape
        
        # Add spatial grounding
        if self.pos_embed is not None:
            z = z + self.pos_embed[:, :L, :]

        # Generate FiLM parameters from R to ground the reconstruction in the rotated frame
        film_params = self.film_gen(R_6d).view(N, self.depth, 2, D)

        for i, blk in enumerate(self.blocks):
            gamma = film_params[:, i, 0, :].unsqueeze(1)
            beta = film_params[:, i, 1, :].unsqueeze(1)
            
            # Ground the latents using R
            z = z * (1.0 + gamma) + beta
            z = blk(z)
        
        # Project to physical IV: (N, L, 3, fs, ts)
        iv = self.head(z)
        return iv.view(N, L, 3, self.patch_shape[0], self.patch_shape[1])
    

class Embisonics(pl.LightningModule):
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
            log_every_n_steps: int = 1000,
            encoder_depth: Optional[int] = 4,
            decoder_depth: int = 2,
            decoder_num_heads: int = 8,
            sr: int = 32000,
            num_mel_bins: int = 128,
            nr_samples_per_audio: int = 8,
            target_length: int = 200,
            input_length: int = 1024,
            rotations_per_clip: int = 4,
            vicreg_gamma: float = 1.0,
            vicreg_var_weight: float = 1.0,
            vicreg_cov_weight: float = 0.01,
            rot_var_weight: float = 1.0,
            iv_weight: float = 1.0,
            iv_equiv_weight: float = 0.5,
            diag_every_n_steps: int = 10,
            **kwargs,
        ):
            super().__init__()
            self.save_hyperparameters(ignore=['patch_strategy'])
            
            # Audio & Training Knobs
            self.sr = sr
            self.num_mel_bins = num_mel_bins
            self.target_length = target_length
            self.input_length = input_length
            self.nr_samples_per_audio = nr_samples_per_audio
            self.rotations_per_clip = rotations_per_clip
            self.in_channels = in_channels
            self.vicreg_gamma = vicreg_gamma
            self.vicreg_var_weight = vicreg_var_weight
            self.vicreg_cov_weight = vicreg_cov_weight
            self.rot_var_weight = rot_var_weight
            self.iv_weight = iv_weight
            self.iv_equiv_weight = iv_equiv_weight
            self.diag_every_n_steps = diag_every_n_steps
            self.lr = lr
            self.trainer_name = trainer
            self.b1 = b1; self.b2 = b2; self.weight_decay = weight_decay

            # 2. Patch Geometry Setup
            self.patch_strategy = patch_strategy
            self.p_f_dim, self.p_t_dim = self.patch_strategy.get_patch_size()
            self.num_patches = self.p_f_dim * self.p_t_dim
            self.grid_size = (self.p_f_dim, self.p_t_dim)
            fs, ts = self.patch_strategy.fshape, self.patch_strategy.tshape
            self.iv_patch_shape = (fs, ts)

            # 3. Encoder (Shallow Pretrained ViT)
            # create_pretrained_model is assumed to return (model, embed_dim)
            self.encoder, self.encoder_embedding_dim = create_pretrained_model(model_size)
            self.encoder_cls_token_num = 1
            
            # Prune encoder depth (e.g., only keep first 4 layers)
            if encoder_depth is not None:
                self.encoder.layers = nn.ModuleList(list(self.encoder.layers)[:encoder_depth])
            
            # Custom Patch Embedding layer for Ambisonic channels
            self.patch_embed = PatchEmbed()
            self._update_patch_embed_layers(self.patch_embed)
            
            # Global CLS token
            self.cls_token = nn.Parameter(
                nn.init.normal_(torch.empty([1, 1, self.encoder_embedding_dim]), std=0.02)
            )
            
            # Fixed 2D Sincos Positional Embeddings for Encoder
            self.encoder.pos_embedding = self._get_pos_embed_params()

            # 4. Dual Grounded Decoders
            # Decoder A: Latent Equivariance (Grounds R in the embedding space)
            self.decoder = EquivarianceDecoder(
                embed_dim=self.encoder_embedding_dim,
                depth=decoder_depth,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                num_patches=self.num_patches,
                grid_size=self.grid_size,
            )

            # Decoder B: IV Reconstruction (Grounds R in the physical space)
            self.iv_decoder = IVDecoder(
                embed_dim=self.encoder_embedding_dim,
                depth=max(1, decoder_depth // 2), # IV needs less depth than Equivariance
                num_heads=decoder_num_heads,
                patch_shape=self.iv_patch_shape,
                num_patches=self.num_patches,
                grid_size=self.grid_size,
            )

            self.melspec = FeatureExtractor(
                sample_rate=self.sr, n_fft=1024, win_length=1024,
                hop_length=self.sr // 100, f_min=50, f_max=self.sr // 2,
                n_mels=self.num_mel_bins, power=2.0,
            ).float()

            # Initialize remaining linear layers
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _iv_frame_target(self, x: torch.Tensor) -> torch.Tensor:
        """Per-frame IV target, reshaped to match patch-token layout.

        x: (B, 7, T_target, F_mel) — output of _wav2fbank after pad/truncate/crop.
        Returns: (B, num_patches, 3, fshape, tshape).

        Assumes non-overlapping patches (fstride == fshape, tstride == tshape).
        """
        fs, ts = self.patch_strategy.fshape, self.patch_strategy.tshape
        assert self.patch_strategy.fstride == fs and self.patch_strategy.tstride == ts, \
            "Frame-level IV target assumes non-overlapping patches."

        iv = x[:, 4:7].float()                  # (B, 3, T, F)
        iv = iv.transpose(2, 3)                 # (B, 3, F, T) — matches _encode
        B, C, Fm, Tm = iv.shape                 # C = 3

        p_f, p_t = Fm // fs, Tm // ts
        assert p_f == self.p_f_dim and p_t == self.p_t_dim, \
            f"Mel grid {(Fm, Tm)} doesn't match expected patch grid {(self.p_f_dim, self.p_t_dim)}"

        # (B, 3, p_f, fs, p_t, ts) -> (B, p_f, p_t, 3, fs, ts) -> (B, P, 3, fs, ts)
        iv = iv.view(B, C, p_f, fs, p_t, ts)
        iv = iv.permute(0, 2, 4, 1, 3, 5).contiguous()
        iv = iv.view(B, p_f * p_t, C, fs, ts)
        return iv
        
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

    def log_ambisonic_spectrograms(self, x, title="Ambisonic_Channels"):
        """
        x: Input tensor of shape (B, C, T, F) - usually 'clean' from _prepare_batch
        """
        # Take the first sample in the batch
        # x shape is (C, T, F). We transpose to (C, F, T) for standard viz
        specs = x[0].detach().cpu().float().numpy().transpose(0, 2, 1)
        
        names = ['0: W (Omni)', '1: Y (Left/Right)', '2: Z (Up/Down)', '3: X (Front/Back)']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"{title} - Step {self.global_step}")
        
        # Use a diverging colormap (RdBu) if you move to signed intensity.
        # If using power=2.0, 'magma' or 'viridis' is fine.
        cmap = 'magma' if specs.min() >= 0 else 'RdBu_r'
        
        for i, ax in enumerate(axes.flat):
            if i < specs.shape[0]:
                # Use a symmetric vmin/vmax if the data is signed
                vmax = np.max(np.abs(specs[i]))
                vmin = 0 if cmap == 'magma' else -vmax
                
                im = ax.imshow(specs[i], aspect='auto', origin='lower', 
                            cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(names[i])
                fig.colorbar(im, ax=ax)
            ax.axis('off')

        plt.tight_layout()
        self.logger.experiment.add_figure(title, fig, global_step=self.global_step)
        plt.close(fig)
        

    def _wav2fbank(self, waveform):
        with torch.amp.autocast('cuda', enabled=False):
            waveform = waveform.float()
            # Per-clip RMS normalization on the W channel (omni = loudness reference).
            # Rotation-invariant: W is unchanged under rotation.
            rms = torch.sqrt(waveform[:, :1].pow(2).mean(dim=-1, keepdim=True) + 1e-8)
            waveform = waveform / rms
            mel = self.melspec(waveform)
            if self.in_channels <= 2:
                return torch.log(mel + 1e-5).transpose(3, 2)
            return mel.transpose(3, 2)

    def pass_through_encoder(self, x, B):
        x = x + self.encoder.pos_embedding[:, self.encoder_cls_token_num:, :]
        cls = self.cls_token.expand(B, -1, -1) + self.encoder.pos_embedding[:, :1, :]
        x = torch.cat((cls, x), dim=1)
        x = self.encoder.dropout(x)
        for block in self.encoder.layers:
            x = block(x)
        return self.encoder.ln(x)

    @torch.no_grad()
    def _foa_directionality(self, wav: torch.Tensor) -> torch.Tensor:
        """Per-clip scalar 'directionality' in ~[0, 1], rotation-invariant.
        0 = diffuse/isotropic field, 1 = ideal plane wave (SN3D/ACN).

        d = 2 * ||<W * YZX>_t|| / (<W^2>_t + <|YZX|^2>_t)

        Factor 2: for a SN3D plane wave the ratio maxes at 1/2.
        wav: (B, 4, T) in ACN ordering [W, Y, Z, X]."""
        W = wav[:, 0].float()
        YZX = wav[:, 1:4].float()
        I = (W.unsqueeze(1) * YZX).mean(dim=-1)                        # (B, 3)
        I_mag = torch.linalg.norm(I, dim=-1)                           # (B,)
        E = (W ** 2).mean(dim=-1) + (YZX ** 2).sum(dim=1).mean(dim=-1) # (B,)
        return (2.0 * I_mag / (E + 1e-8)).clamp(0.0, 1.0)
    
    def _encode(self, x):
        B, C, T, F = x.shape
        x = x.transpose(2, 3)                              # (B, C, F, T)
        embedded = self.patch_strategy.embed(x, self.patch_embed)
        return self.pass_through_encoder(embedded, B)

    @torch.no_grad()
    def _prepare_batch(self, batch):
        audio, R_mats = batch # audio: (B, 4, T), R_mats: (B, Rn, 3, 3)
        B = audio.shape[0]
        Rn = R_mats.shape[1]

        dir_scalar = self._foa_directionality(audio) 

        wav_exp = audio.unsqueeze(1).expand(-1, Rn, -1, -1).reshape(B * Rn, 4, -1)
        R_flat = R_mats.reshape(B * Rn, 3, 3).to(audio.dtype)
        rotated_wav = rotate_foa_waveform(wav_exp, R_flat)

        all_wav = torch.cat([audio, rotated_wav], dim=0)
        all_fb = self._wav2fbank(all_wav)
        all_fb = pad_or_truncate_batch(all_fb, self.input_length)
        T_mel, F_mel = all_fb.shape[-2], all_fb.shape[-1]

        fb_clean = all_fb[:B]
        fb_rot = all_fb[B:].reshape(B, Rn, self.in_channels, T_mel, F_mel)

        n = self.nr_samples_per_audio
        starts = torch.randint(0, T_mel - self.target_length + 1, (B, n), device=self.device)
        idx = starts.unsqueeze(-1) + torch.arange(self.target_length, device=self.device)

        idx_c = idx.view(B, n, 1, self.target_length, 1).expand(-1, -1, self.in_channels, -1, F_mel)
        clean = torch.gather(fb_clean.unsqueeze(1).expand(-1, n, -1, -1, -1), 3, idx_c)

        idx_r = idx.view(B, 1, n, 1, self.target_length, 1).expand(-1, Rn, -1, self.in_channels, -1, F_mel)
        rotated = torch.gather(fb_rot.unsqueeze(2).expand(-1, -1, n, -1, -1, -1), 4, idx_r)

        clean = clean.reshape(B * n, self.in_channels, self.target_length, F_mel)
        rotated = (rotated.permute(0, 2, 1, 3, 4, 5)
                          .reshape(B * n, Rn, self.in_channels, self.target_length, F_mel))

        # FIX: Return the 3x3 matrices expanded to the (B*n, Rn, 3, 3) shape
        R_mats_exp = R_mats.unsqueeze(1).expand(-1, n, -1, -1, -1).reshape(B * n, Rn, 3, 3)

        dir_scalar = dir_scalar.unsqueeze(1).expand(-1, n).reshape(B * n)

        return (clean, rotated, R_mats_exp, dir_scalar)
    
    @torch.no_grad()
    def _get_tf_directionality(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the time-frequency directionality (diffuseness) weights.
        0 = completely diffuse, 1 = perfectly directional.
        x: (B, 7, T, F) where channels are assumed to be [W, Y, Z, X, I_y, I_z, I_x]
        Returns: (B, T, F) with values in [0, 1]
        """
        # I_mag: ||(I_y, I_z, I_x)||
        I_mag = torch.linalg.norm(x[:, 4:7].float(), dim=1) # (B, T, F)
        # E: W + Y + Z + X (using Mel power)
        E = x[:, 0].float() + x[:, 1:4].float().sum(dim=1)  # (B, T, F)
        
        # 2 * ||I|| / E
        dir_tf = (2.0 * I_mag / (E + 1e-8)).clamp(0.0, 1.0)
        return dir_tf

    @torch.no_grad()
    def _get_iv_and_patch_weights(self, x: torch.Tensor):
        """
        Returns tf-bin weights for IV loss and patch-level weights for latent loss.
        x: (B, 7, T, F)
        Returns:
            iv_weight: (B, P, 1, fs, ts)
            patch_weight: (B, P, 1)
        """
        dir_tf = self._get_tf_directionality(x) # (B, T, F)
        dir_tf = dir_tf.transpose(1, 2)         # (B, F, T) -> to match patch layout
        
        fs, ts = self.patch_strategy.fshape, self.patch_strategy.tshape
        B, Fm, Tm = dir_tf.shape
        p_f, p_t = Fm // fs, Tm // ts
        
        # Reshape into patches: (B, p_f, fs, p_t, ts) -> (B, p_f, p_t, fs, ts) -> (B, P, 1, fs, ts)
        w = dir_tf.view(B, p_f, fs, p_t, ts)
        w = w.permute(0, 1, 3, 2, 4).contiguous()
        iv_weight = w.view(B, p_f * p_t, 1, fs, ts)
        
        # Average over the tf-bins within each patch for the latent weight
        patch_weight = iv_weight.mean(dim=(3, 4)) # (B, P, 1)
        
        return iv_weight, patch_weight

    def forward(self, x_clean, x_rot, R_mats, dir_weights):
        N_total, Rn = x_rot.shape[:2]
        D = self.encoder_embedding_dim
        
        R_6d = matrix_to_6d(R_mats)
        R_6d_flat = R_6d.reshape(N_total * Rn, 6)
        
        iv_target_clean = self._iv_frame_target(x_clean)
        x_rot_flat = x_rot.reshape(N_total * Rn, *x_rot.shape[2:])
        iv_target_rot = self._iv_frame_target(x_rot_flat)

        iv_weight_clean, patch_weight_clean = self._get_iv_and_patch_weights(x_clean)
        iv_weight_rot, patch_weight_rot = self._get_iv_and_patch_weights(x_rot_flat)

        z_clean = self._encode(x_clean)[:, self.encoder_cls_token_num:, :]
        z_rot_actual = self._encode(x_rot_flat)[:, self.encoder_cls_token_num:, :]

        z_clean_rep = z_clean.unsqueeze(1).expand(-1, Rn, -1, -1).reshape(N_total * Rn, -1, D)
        z_pred_rot = self.decoder(z_clean_rep, R_6d_flat)
        equi_loss = (F.mse_loss(z_pred_rot, z_rot_actual.detach(), reduction='none') * patch_weight_rot).mean()

        R_id = torch.eye(3, device=self.device).view(1, 3, 3).expand(N_total, -1, -1)
        R_id_6d = matrix_to_6d(R_id)
        iv_pred_clean = self.iv_decoder(z_clean, R_id_6d)
        iv_loss_clean = (F.mse_loss(iv_pred_clean, iv_target_clean, reduction='none') * iv_weight_clean).mean()

        iv_pred_rot_grounded = self.iv_decoder(z_clean_rep, R_6d_flat)
        iv_loss_rot = (F.mse_loss(iv_pred_rot_grounded, iv_target_rot, reduction='none') * iv_weight_rot).mean()

        R_id_rep_6d = R_id_6d.repeat_interleave(Rn, dim=0)
        iv_pred_from_equi = self.iv_decoder(z_pred_rot, R_id_rep_6d)
        iv_consist_loss = (F.mse_loss(iv_pred_from_equi, iv_pred_rot_grounded.detach(), reduction='none') * iv_weight_rot).mean()

        z_flat = z_clean.reshape(-1, D).float()
        var_loss, cov_loss = vicreg_loss(z_flat, gamma=self.vicreg_gamma)

        z_rot_stacked = z_rot_actual.view(N_total, Rn, -1, D)
        z_all = torch.cat([z_clean.unsqueeze(1), z_rot_stacked], dim=1)
        z_pooled = z_all.mean(dim=2).float() # Mean over patches
        std_across_rot = torch.sqrt(z_pooled.var(dim=1) + 1e-4)
        hinge = F.relu(self.vicreg_gamma - std_across_rot).mean(dim=-1)
        rot_var_loss = (hinge * dir_weights).mean()


        loss = (equi_loss + 
                self.iv_weight * (iv_loss_clean + iv_loss_rot) + 
                self.iv_equiv_weight * iv_consist_loss +
                self.vicreg_var_weight * var_loss + 
                self.vicreg_cov_weight * cov_loss +
                self.rot_var_weight * rot_var_loss)

        return (loss, equi_loss, var_loss, cov_loss, rot_var_loss, 
                iv_loss_rot, iv_consist_loss, z_clean_rep, z_rot_actual, patch_weight_rot)

    def training_step(self, batch, batch_idx):
        x_clean, x_rot, R_mats, dir_weights = self._prepare_batch(batch)
        
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            (loss, equi, var, cov, rot_var, 
             iv_rot, iv_consist, z_clean_rep, z_rot_actual, patch_weight_rot) = self.forward(
                x_clean, x_rot, R_mats, dir_weights
            )

        self.log_dict({
            "loss": loss,
            "equiv_latent_mse": equi,
            "vicreg_var": var,
            "vicreg_cov": cov,
            "rot_var": rot_var,
            "iv_recon_rot": iv_rot,
            "iv_consistency": iv_consist,
            "batch_directionality": dir_weights.mean(),
        }, prog_bar=True)

        if self.global_step % 500 == 0:
            self.log_ambisonic_spectrograms(x_clean, title="Train_Spectrograms")

        # Diagnostics
        if self.global_step % self.diag_every_n_steps == 0:
            with torch.no_grad():
                N, Rn = x_rot.shape[:2]
                R_6d_flat = matrix_to_6d(R_mats).reshape(N * Rn, 6)
                
                mse_identity = (F.mse_loss(z_clean_rep, z_rot_actual, reduction='none') * patch_weight_rot).mean()
                mse_with_rot = (F.mse_loss(self.decoder(z_clean_rep, R_6d_flat), z_rot_actual, reduction='none') * patch_weight_rot).mean()
                mse_without_rot = (F.mse_loss(
                    self.decoder(z_clean_rep, torch.zeros_like(R_6d_flat)), z_rot_actual, reduction='none'
                ) * patch_weight_rot).mean()
                
                print(f"""--- Step {self.global_step} Diagnostics ---
                latent/identity_baseline: {mse_identity:.6f}
                latent/decoder_with_R:    {mse_with_rot:.6f}
                latent/decoder_no_R:      {mse_without_rot:.6f}
                latent/conditioning_gap:  {mse_without_rot - mse_with_rot:.6f}
                """)

        return loss

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        audio_trainables = [p for p in self.parameters() if p.requires_grad]
        optimizer = None
        if self.trainer_name == "adamW":
            optimizer = torch.optim.AdamW(
                audio_trainables,
                self.lr,
                weight_decay=self.weight_decay,
                betas=(self.b1, self.b2),
            )

        cosine_annealing = transformers.get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps=10000, num_training_steps=self.trainer.max_steps)

        return {"optimizer": optimizer,
                'lr_scheduler' : {"scheduler": cosine_annealing, "interval": "step"}}
    # ----------------------------- downstream eval -----------------------------
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