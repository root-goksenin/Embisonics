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

set_fused_attn(True)
use_fused_attn(True)

pad_or_truncate_batch = torch.compile(pad_or_truncate_batch)


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


class Decoder(nn.Module):
    """Reconstructs the full 7-channel acoustic scene (WXYZ log-mels + 3D IV).
    """
    def __init__(self, encoder_embed_dim: int, decoder_embed_dim: int,
                 depth: int, num_heads: int, patch_shape: tuple,
                 num_patches: int, grid_size: tuple, mlp_ratio: float = 2.0):
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

        self.blocks = nn.ModuleList([
            Block(decoder_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_embed_dim)
        # 7 channels: 4 (W+X/Y/Z log-mels) + 3 (IV)
        self.head = nn.Linear(decoder_embed_dim, 7 * patch_shape[0] * patch_shape[1])

    def forward(self, z_vis, visible_mask):
        N, P, D = z_vis.shape[0], self.num_patches, self.decoder_embed_dim
        z_vis_proj = self.decoder_embed(z_vis)
        x = self.mask_token.type_as(z_vis_proj).expand(N, P, D).clone()
        x[visible_mask] = z_vis_proj.reshape(-1, D)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x).view(N, P, 7, self.patch_shape[0], self.patch_shape[1])


# ======================================================================
# SphereV2 — Pure Masked Autoencoder (No Rotation)
# ======================================================================

class SphereV2(pl.LightningModule):
    """Simple spatial MAE architecture:
      - Pure self-attention spatial encoder on 7-channel input.
      - Single Decoder reconstructs all 7 channels.
      - Objective: MSE for WXYZ channels, Cosine Loss for IV directional channels.
    """

    def __init__(
            self,
            model_size: str = "base",
            encoder_depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,

            decoder_depth: int = 4,
            decoder_num_heads: int = 6,
            decoder_embedding_dim: int = 384,

            lr: float = 2e-4,
            b1: float = 0.9, b2: float = 0.95,
            weight_decay: float = 0.01,
            loss_sharpness: float = 0.5,
            wxyz_loss_weight: float = 0.25,
            use_mse_loss_on_iv : bool = False,

            patch_strategy: PatchStrategy = None,
            in_channels: int = 7,
            sr: int = 32000,
            num_mel_bins: int = 128,
            input_length: int = 500,
            target_length: int = 200,

            log_every_n_steps: int = 500,
            **kwargs,
        ):
        super().__init__()
        self.save_hyperparameters(ignore=['patch_strategy'])

        self.sr = sr
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.input_length = input_length
        self.in_channels = in_channels
        self.loss_sharpness = loss_sharpness
        self.wxyz_loss_weight = wxyz_loss_weight
        self.lr = lr
        self.b1 = b1; self.b2 = b2; self.weight_decay = weight_decay
        self.log_every_n_steps = log_every_n_steps
        self.use_mse_loss_on_iv = use_mse_loss_on_iv

        # Patch geometry.
        self.patch_strategy = patch_strategy
        self.p_f_dim, self.p_t_dim = self.patch_strategy.get_patch_size()
        self.num_patches = self.p_f_dim * self.p_t_dim
        self.grid_size = (self.p_f_dim, self.p_t_dim)
        fs, ts = self.patch_strategy.fshape, self.patch_strategy.tshape
        self.patch_shape = (fs, ts)

        # --- Spatial encoder (pure self-attention) ---
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
            Block(self.encoder_embedding_dim, num_heads=num_heads,
                  mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(self.encoder_embedding_dim)

        # --- Decoder ---
        self.decoder = Decoder(
            encoder_embed_dim=self.encoder_embedding_dim,
            decoder_embed_dim=decoder_embedding_dim,
            depth=decoder_depth,
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
        for mod in [self.patch_embed, self.encoder_blocks, self.encoder_norm,
                    self.decoder]:
            if isinstance(mod, nn.Module):
                mod.apply(init_fn)

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

    # ---- Encoder -------------------------------------------------------
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
            x = block(x)   # pure self-attention

        return self.encoder_norm(x)

    @torch.no_grad()
    def _prepare_batch(self, batch):
        audio, context_idx, _ = batch  # Third element (rotation matrices) is ignored
        audio = audio.to(self.device, non_blocking=True)
        context_idx = context_idx.to(self.device, non_blocking=True)

        B = audio.shape[0]
        all_fb = self._wav2fbank(audio)  # (N, C, T_total, n_freq)

        # Temporal crop for clean version
        N, C, T_total, n_freq = all_fb.shape
        T_crop = self.target_length
        max_start = T_total - T_crop

        offsets = torch.randint(0, max_start + 1, (B,), device=self.device)

        # Vectorized crop via gather along the time dim
        t_range = torch.arange(T_crop, device=self.device)
        time_idx = (offsets.view(N, 1, 1, 1) + t_range.view(1, 1, T_crop, 1)) \
            .expand(N, C, T_crop, n_freq)
        clean = torch.gather(all_fb, dim=2, index=time_idx)  # (N, C, T_crop, n_freq)

        visible_mask = torch.zeros(B, self.num_patches, dtype=torch.bool, device=self.device)
        visible_mask.scatter_(1, context_idx.long(), True)

        return clean.to(torch.bfloat16), visible_mask

    def forward(self, x_clean, visible_mask):
        # Encoder on clean visible patches
        z = self._encode_visible(x_clean, visible_mask)
        z_vis = z[:, 1:, :]

        # Targets (iv_target here is the unit-normalized target)
        wxyz_target = self._patchify(x_clean, slice(0, 4))
        iv_target, w_target = self._iv_target(x_clean)

        # Predictions (Single Decoder predicts 7 channels)
        pred_7ch = self.decoder(z_vis, visible_mask)
        
        if getattr(self, "use_mse_loss_on_iv", False):
            wxyz_pred = pred_7ch[:, :, :4]
            iv_pred_raw = pred_7ch[:, :, 4:7]
            
            masked = ~visible_mask
            
            # For standard MSE, we want to reconstruct the raw unnormalized IV patches
            iv_target_raw = self._patchify(x_clean, slice(4, 7))
            
            l_wxyz = masked_mse_loss(wxyz_pred, wxyz_target, masked)
            l_iv = masked_mse_loss(iv_pred_raw, iv_target_raw, masked)
            
            total = self.wxyz_loss_weight * l_wxyz + l_iv
            
            # Normalize the predicted IV to unit vectors so the cosine angular
            # error logging in training_step() continues to work mathematically.
            iv_pred = F.normalize(iv_pred_raw, p=2, dim=2, eps=1e-6)

        else:
            wxyz_pred = pred_7ch[:, :, :4]
            iv_pred = pred_7ch[:, :, 4:7]
            
            # Normalize predicted IV to unit vectors for cosine similarity
            iv_pred = F.normalize(iv_pred, p=2, dim=2, eps=1e-6)

            # --- Losses ---
            masked = ~visible_mask
            l_wxyz = masked_mse_loss(wxyz_pred, wxyz_target, masked)
            
            # We only want to compute IV loss on masked patches
            m = masked.view(*masked.shape, 1, 1, 1).float()
            w_target_masked = w_target * m
            l_iv = cosine_iv_loss(iv_pred, iv_target, w_target_masked, self.loss_sharpness)

            # Total Loss
            total = self.wxyz_loss_weight * l_wxyz + l_iv

        return {
            "loss": total,
            "l_wxyz": l_wxyz,
            "l_iv": l_iv,
            "wxyz_pred": wxyz_pred, 
            "wxyz_target": wxyz_target,
            "iv_pred": iv_pred, 
            "iv_target": iv_target,
            "iv_weight_target": w_target,
            "masked": masked,
        }

    # ======================================================================
    # TensorBoard logging helpers
    # ======================================================================
    def _fbank_to_patches(self, fbank: torch.Tensor) -> torch.Tensor:
        fs, ts = self.patch_strategy.fshape, self.patch_strategy.tshape
        pf, pt = self.p_f_dim, self.p_t_dim
        B, C, T, Fm = fbank.shape
        x = fbank.transpose(2, 3)                           # (B, C, F, T)
        x = x.reshape(B, C, pf, fs, pt, ts)
        return x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(B, pf * pt, C, fs, ts)

    def _patches_to_img(self, patches: torch.Tensor) -> np.ndarray:
        fs, ts = self.patch_strategy.fshape, self.patch_strategy.tshape
        pf, pt = self.p_f_dim, self.p_t_dim
        B, P, C, _, _ = patches.shape
        x = patches.reshape(B, pf, pt, C, fs, ts)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()        # (B, C, pf, fs, pt, ts)
        x = x.reshape(B, C, pf * fs, pt * ts)               # (B, C, F, T)
        x = x.transpose(2, 3)                               # (B, C, T, F)
        return x.cpu().float().numpy()

    def _log_spectrogram(self, fbank: torch.Tensor, title: str, loss=None):
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
        x_clean, visible_mask = self._prepare_batch(batch)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = self.forward(x_clean, visible_mask)

        loss = out["loss"]

        with torch.no_grad():
            # Error on strictly masked patches
            m = out["masked"].view(*out["masked"].shape, 1, 1, 1).float()
            w_masked_iv = out["iv_weight_target"] * m
            cos = (out["iv_pred"] * out["iv_target"]).sum(dim=2, keepdim=True).clamp(-1.0, 1.0)
            ang = torch.acos(cos) * (180.0 / np.pi)
            ang_err_masked = (ang * w_masked_iv).sum() / w_masked_iv.sum().clamp_min(1e-6)

            # Error on non-masked (visible) patches
            v = (~out["masked"]).view(*out["masked"].shape, 1, 1, 1).float()
            w_vis_iv = out["iv_weight_target"] * v
            ang_err_vis = (ang * w_vis_iv).sum() / w_vis_iv.sum().clamp_min(1e-6)

        self.log_dict({
            "loss": loss,
            "l_wxyz": out["l_wxyz"],
            "l_iv": out["l_iv"],
            "ang_err_masked_deg": ang_err_masked,
            "ang_err_vis_deg": ang_err_vis,
        }, prog_bar=True)

        # ---- Periodic image logs ---------------------------------------
        if self.global_step % self.log_every_n_steps == 0:
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

            self._log_reconstruction(
                out["wxyz_pred"], out["wxyz_target"],
                title_prefix="recon/WXYZ_mel"
            )

            self._log_iv_angular_error(
                out["iv_pred"], out["iv_target"],
                visible_mask, title="iv/angular_error_heatmap"
            )

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