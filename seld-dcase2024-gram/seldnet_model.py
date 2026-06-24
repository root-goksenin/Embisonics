# The SELDnet architecture

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from IPython import embed


class MSELoss_ADPIT(object):
    def __init__(self):
        super().__init__()
        self._each_loss = nn.MSELoss(reduction='none')

    def _each_calc(self, output, target):
        return self._each_loss(output, target).mean(dim=(2))  # class-wise frame-level

    def __call__(self, output, target):
        """
        Auxiliary Duplicating Permutation Invariant Training (ADPIT) for 13 (=1+6+6) possible combinations
        Args:
            output: [batch_size, frames, num_track*num_axis*num_class=3*4*13]
            target: [batch_size, frames, num_track_dummy=6, num_axis=5, num_class=13]
        Return:
            loss: scalar
        """
        target_A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 1:, :]  # A0, no ov from the same class, [batch_size, frames, num_axis(act)=1, num_class=12] * [batch_size, frames, num_axis(XYZD)=4, num_class=12]
        target_B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 1:, :]  # B0, ov with 2 sources from the same class
        target_B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 1:, :]  # B1
        target_C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 1:, :]  # C0, ov with 3 sources from the same class
        target_C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 1:, :]  # C1
        target_C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 1:, :]  # C2

        target_A0A0A0 = torch.cat((target_A0, target_A0, target_A0), 2)  # 1 permutation of A (no ov from the same class), [batch_size, frames, num_track*num_axis=3*4, num_class=12]
        target_B0B0B1 = torch.cat((target_B0, target_B0, target_B1), 2)  # 6 permutations of B (ov with 2 sources from the same class)
        target_B0B1B0 = torch.cat((target_B0, target_B1, target_B0), 2)
        target_B0B1B1 = torch.cat((target_B0, target_B1, target_B1), 2)
        target_B1B0B0 = torch.cat((target_B1, target_B0, target_B0), 2)
        target_B1B0B1 = torch.cat((target_B1, target_B0, target_B1), 2)
        target_B1B1B0 = torch.cat((target_B1, target_B1, target_B0), 2)
        target_C0C1C2 = torch.cat((target_C0, target_C1, target_C2), 2)  # 6 permutations of C (ov with 3 sources from the same class)
        target_C0C2C1 = torch.cat((target_C0, target_C2, target_C1), 2)
        target_C1C0C2 = torch.cat((target_C1, target_C0, target_C2), 2)
        target_C1C2C0 = torch.cat((target_C1, target_C2, target_C0), 2)
        target_C2C0C1 = torch.cat((target_C2, target_C0, target_C1), 2)
        target_C2C1C0 = torch.cat((target_C2, target_C1, target_C0), 2)

        output = output.reshape(output.shape[0], output.shape[1], target_A0A0A0.shape[2], target_A0A0A0.shape[3])  # output is set the same shape of target, [batch_size, frames, num_track*num_axis=3*4, num_class=12]
        pad4A = target_B0B0B1 + target_C0C1C2
        pad4B = target_A0A0A0 + target_C0C1C2
        pad4C = target_A0A0A0 + target_B0B0B1
        loss_0 = self._each_calc(output, target_A0A0A0 + pad4A)  # padded with target_B0B0B1 and target_C0C1C2 in order to avoid to set zero as target
        loss_1 = self._each_calc(output, target_B0B0B1 + pad4B)  # padded with target_A0A0A0 and target_C0C1C2
        loss_2 = self._each_calc(output, target_B0B1B0 + pad4B)
        loss_3 = self._each_calc(output, target_B0B1B1 + pad4B)
        loss_4 = self._each_calc(output, target_B1B0B0 + pad4B)
        loss_5 = self._each_calc(output, target_B1B0B1 + pad4B)
        loss_6 = self._each_calc(output, target_B1B1B0 + pad4B)
        loss_7 = self._each_calc(output, target_C0C1C2 + pad4C)  # padded with target_A0A0A0 and target_B0B0B1
        loss_8 = self._each_calc(output, target_C0C2C1 + pad4C)
        loss_9 = self._each_calc(output, target_C1C0C2 + pad4C)
        loss_10 = self._each_calc(output, target_C1C2C0 + pad4C)
        loss_11 = self._each_calc(output, target_C2C0C1 + pad4C)
        loss_12 = self._each_calc(output, target_C2C1C0 + pad4C)

        loss_min = torch.min(
            torch.stack((loss_0,
                         loss_1,
                         loss_2,
                         loss_3,
                         loss_4,
                         loss_5,
                         loss_6,
                         loss_7,
                         loss_8,
                         loss_9,
                         loss_10,
                         loss_11,
                         loss_12), dim=0),
            dim=0).indices

        loss = (loss_0 * (loss_min == 0) +
                loss_1 * (loss_min == 1) +
                loss_2 * (loss_min == 2) +
                loss_3 * (loss_min == 3) +
                loss_4 * (loss_min == 4) +
                loss_5 * (loss_min == 5) +
                loss_6 * (loss_min == 6) +
                loss_7 * (loss_min == 7) +
                loss_8 * (loss_min == 8) +
                loss_9 * (loss_min == 9) +
                loss_10 * (loss_min == 10) +
                loss_11 * (loss_min == 11) +
                loss_12 * (loss_min == 12)).mean()

        return loss


# The SELDnet architecture

from dataclasses import dataclass, field
import sys
from typing import Callable, Optional 
sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================================================
# Shared SELD head for the embedding-probe paths (ii, iii, ours)
#   GRU -> GLU -> MHSA -> FNN -> ACCDOA.  Internals identical to SeldModel.
# ============================================================================
class SeldHead(nn.Module):
    def __init__(self, in_dim, out_shape, params):
        super().__init__()
        self.gru = nn.GRU(in_dim, params['rnn_size'], params['nb_rnn_layers'],
                          batch_first=True, dropout=params['dropout_rate'],
                          bidirectional=True)
        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for _ in range(params['nb_self_attn_layers']):
            self.mhsa_block_list.append(nn.MultiheadAttention(
                params['rnn_size'], params['nb_heads'],
                dropout=params['dropout_rate'], batch_first=True))
            self.layer_norm_list.append(nn.LayerNorm(params['rnn_size']))
        self.fnn_list = nn.ModuleList()
        if params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(nn.Linear(
                    params['fnn_size'] if fc_cnt else params['rnn_size'],
                    params['fnn_size'], bias=True))
        self.fnn_list.append(nn.Linear(
            params['fnn_size'] if params['nb_fnn_layers'] else params['rnn_size'],
            out_shape[-1], bias=True))

    def forward(self, x):                       # (B, T_seld, in_dim)
        x, _ = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1] // 2:] * x[:, :, :x.shape[-1] // 2]
        for mhsa, ln in zip(self.mhsa_block_list, self.layer_norm_list):
            xin = x
            x, _ = mhsa(xin, xin, xin)
            x = ln(x + xin)
        for fnn_cnt in range(len(self.fnn_list) - 1):
            x = self.fnn_list[fnn_cnt](x)
        return torch.tanh(self.fnn_list[-1](x))


# ============================================================================
# Injectable mono-encoder spec  (swap GRAM-T / Dasheng / ATST via config)
# ============================================================================
@dataclass
class MonoEncoderSpec:
    name: str
    model: nn.Module                                  # frozen W-only backbone
    embed_dim: int                                    # D_mono
    n_freq: int                                       # F_mono
    # (model, w_logmel)->(B, T_mono, F_mono*D_mono), F-major / D-minor flatten
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor] = \
        field(default=lambda m, w: m(w, strategy="raw"))

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x

def gramt_mono_spec(gram_module, n_freq, embed_dim, name="gram-t"):
    """GRAM-T pulled from a SphereV4 (sphere.gram)."""
    return MonoEncoderSpec(name, gram_module, embed_dim, n_freq,
                           lambda m, w: m(w, strategy="raw"))

# ============================================================================
# Baseline (iii): learnt spatial tokens via a 2-layer Conv2D front-end
# ============================================================================
class LearntSpatialFrontEnd(nn.Module):
    """Two Conv2D layers on 7-ch FOA -> tokens of dim (B, T, F_sp*D_sp),
    matching the pre-trained spatial stream's layout exactly."""
    def __init__(self, in_ch, D_sp, F_sp, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or D_sp
        self.conv1 = ConvBlock(in_ch, mid_ch)
        self.conv2 = ConvBlock(mid_ch, D_sp)
        self.F_sp, self.D_sp = F_sp, D_sp

    def forward(self, x):                           # (B, 7, T, F)
        x = self.conv2(self.conv1(x))               # (B, D_sp, T, F)
        B, D, T, _ = x.shape
        x = F.adaptive_avg_pool2d(x, (T, self.F_sp))  # (B, D_sp, T, F_sp)
        x = x.permute(0, 2, 3, 1).contiguous()        # (B, T, F_sp, D_sp)
        return x.reshape(B, T, self.F_sp * self.D_sp)



class SphereV4SELD(nn.Module):
    """Frozen-encoder SELD probe.

    inject_spatial_tokens:
      "True"  -> frozen pre-trained SphereV4 spatial encoder   (ours)
      "False" -> mono stream only                              (baseline ii)
      "Learn" -> trainable 2-Conv2D spatial front-end          (baseline iii)

    Mono encoder is always frozen and injected via `mono_spec`.
    """
    def __init__(self, out_shape, params, mono_spec: MonoEncoderSpec,
                 spatial_encoder: Optional[nn.Module] = None,
                 inject_spatial_tokens: str = "True"):
        super().__init__()
        assert inject_spatial_tokens in ("True", "False", "Learn")
        self.params = params
        self.inject = inject_spatial_tokens
        self.has_spatial = inject_spatial_tokens != "False"
        self.T_seld = out_shape[-2]
        proj_dim = 256
        dropout = params.get('dropout_rate', 0.1)

        # ---- Mono stream (always frozen) ----
        self.mono = mono_spec
        self.add_module("mono_model", mono_spec.model)      # registered for .to()/.eval()
        for p in mono_spec.model.parameters():
            p.requires_grad = False
        mono_spec.model.eval()
        self.D_gr, self.F_gr = mono_spec.embed_dim, mono_spec.n_freq

        # ---- Spatial stream ----
        self.sphere = None
        self.learnt_front = None
        if self.has_spatial:
            self.D_sp = spatial_encoder.encoder_embedding_dim
            self.F_sp = spatial_encoder.p_f_dim
            if inject_spatial_tokens == "True":
                assert spatial_encoder is not None, "True needs a SphereV4 encoder"
                self.sphere = spatial_encoder
                for p in self.sphere.parameters():
                    p.requires_grad = False
                self.sphere.eval()
            else:
                self.learnt_front = LearntSpatialFrontEnd(
                    in_ch=7, D_sp=self.D_sp, F_sp=self.F_sp,
                    mid_ch=params.get('nb_cnn2d_filt', None))

            self.input_norm_sp = nn.LayerNorm(self.F_sp * self.D_sp)
            self.q_sp = nn.Parameter(torch.randn(self.D_sp) * 0.02)
            self.k_proj_sp = nn.Linear(self.D_sp, self.D_sp, bias=False)
            self.sphere_proj = nn.Sequential(
                nn.LayerNorm(self.D_sp), nn.Linear(self.D_sp, proj_dim),
                nn.GELU(), nn.Dropout(dropout))

        # mono projection plumbing
        self.input_norm_gr = nn.LayerNorm(self.F_gr * self.D_gr)
        self.q_gr = nn.Parameter(torch.randn(self.D_gr) * 0.02)
        self.k_proj_gr = nn.Linear(self.D_gr, self.D_gr, bias=False)
        self.gram_proj = nn.Sequential(
            nn.LayerNorm(self.D_gr), nn.Linear(self.D_gr, proj_dim),
            nn.GELU(), nn.Dropout(dropout))

        # fuse: concat only when a spatial stream is present (ii feeds mono directly)
        self.feat_proj = nn.Linear(2 * proj_dim, proj_dim) if self.has_spatial else None

        self.head = SeldHead(proj_dim, out_shape, params)

    def train(self, mode: bool = True):
        super().train(mode)
        self.mono.model.eval()                       # always frozen
        if self.sphere is not None:
            self.sphere.eval()                       # frozen pre-trained spatial
        return self

    @staticmethod
    def _freq_pool(z_flat, query, k_proj, F_dim, D_dim):
        B, T, _ = z_flat.shape
        z = z_flat.view(B, T, F_dim, D_dim)
        attn = (k_proj(z) @ query) / (D_dim ** 0.5)
        attn = attn.softmax(dim=-1).unsqueeze(-1)
        return (z * attn).sum(dim=2)                 # (B, T, D)

    def _avg_pool_time(self, z):                     # (B,T,D)->(B,T_seld,D)
        return F.adaptive_avg_pool1d(z.transpose(1, 2), self.T_seld).transpose(1, 2)

    def _spatial_tokens(self, x):
        x_w, x_yzx, x_iv = x[:, 0:1], x[:, 1:4], x[:, 4:7]
        if self.inject == "True":
            with torch.no_grad():
                x_logmel4 = torch.cat([x_w, x_yzx], dim=1)        # W,Y,Z,X
                z = self.sphere.pass_through_encoder(x_logmel4, x_iv)
                z = z[:, 1:, :]                                   # drop CLS
                z = rearrange(z, "b (f t) d -> b t (f d)",
                              f=self.sphere.grid_size[0])
            return z
        return self.learnt_front(x)                  # "Learn": trainable

    def forward(self, x):
        """x: (B, 7, T, F): ch0=W, ch1:4=YZX, ch4:7=IV."""
        # ---- Mono stream (frozen) ----
        with torch.no_grad():
            z_gr = self.mono.forward_fn(self.mono.model, x[:, 0:1])
        z_gr = self.input_norm_gr(z_gr)
        z_gr = self._freq_pool(z_gr, self.q_gr, self.k_proj_gr, self.F_gr, self.D_gr)
        z_gr = self.gram_proj(self._avg_pool_time(z_gr))         # (B,T_seld,256)

        if not self.has_spatial:                                 # baseline (ii)
            return self.head(z_gr)

        z_sp = self._spatial_tokens(x)
        z_sp = self.input_norm_sp(z_sp)
        z_sp = self._freq_pool(z_sp, self.q_sp, self.k_proj_sp, self.F_sp, self.D_sp)
        z_sp = self.sphere_proj(self._avg_pool_time(z_sp))       # (B,T_seld,256)
        z = self.feat_proj(torch.cat([z_sp, z_gr], dim=-1))      # (B,T_seld,256)
        
        return self.head(z)