# The SELDnet architecture — Dasheng (mono) + Sphere (spatial)
#
# Refactored to mirror the GRAM-T / SphereV4 probe (document 1):
#   * Dasheng is injected as a frozen MonoEncoderSpec, exactly like gramt_mono_spec
#   * every stream is  norm -> (freq-pool) -> time-align -> project-to-256
#   * the two 256-d streams are concatenated, fused, and decoded by the shared SeldHead
#
# Dasheng differs from GRAM-T in two ways the structure accommodates:
#   * it has no frequency axis (F_mono = 1), so the mono stream skips freq-pooling
#   * it consumes its own mel front-end, passed to forward() as `dash`

from dataclasses import dataclass, field
import sys
from typing import Callable 

import sys
sys.path.append("..")
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModel


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



# ============================================================================
# Injectable mono-encoder: Dasheng  (frozen, F_mono = 1)
# ============================================================================
class DashengMono(nn.Module):
    """Frozen Dasheng backbone returning per-frame embeddings (B, T_mono, D)."""
    def __init__(self, hf_id="mispeech/dasheng-base"):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            hf_id, outputdim=None, trust_remote_code=True)
        self.embed_dim = 768        # 768 (base)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()                                       # always frozen
        return self

    @torch.no_grad()
    def forward(self, mel):                                     # mel -> (B, T_mono, D)
        return self.model(input_values=mel).hidden_states


def dasheng_mono_spec(dasheng_module: DashengMono, name="dasheng"):
    """Dasheng wrapped as a MonoEncoderSpec (no frequency axis -> n_freq = 1)."""
    return MonoEncoderSpec(
        name=name,
        model=dasheng_module,
        embed_dim=dasheng_module.embed_dim,                     # D_mono
        n_freq=1,                                               # F_mono
        forward_fn=lambda m, mel: m(mel),                       # (B, T_mono, D_mono)
    )


# ============================================================================
# Sphere (spatial) + Dasheng (mono) SELD probe
#   Mirrors SphereV4SELD: per-stream projection plumbing -> fuse -> SeldHead.
# ============================================================================
class DashengSphereSELD(nn.Module):
    """Frozen-/trainable-encoder SELD probe.

    Spatial stream : pre-trained Sphere encoder (frozen when freeze_backbone).
    Mono stream    : frozen Dasheng, injected via `mono_spec`.
    The Sphere conditioner (`sphere.gram`) is always frozen.
    """
    def __init__(self, out_shape, params, mono_spec: MonoEncoderSpec,
                 spatial_encoder: nn.Module, freeze_backbone: bool = True):
        super().__init__()
        self.params = params
        self.freeze_backbone = freeze_backbone
        self.T_seld = out_shape[-2]
        proj_dim = 256
        dropout = params.get('dropout_rate', 0.1)

        # ---- Mono stream (Dasheng, always frozen) ----
        self.mono = mono_spec
        self.add_module("mono_model", mono_spec.model)          # registered for .to()/.eval()
        for p in mono_spec.model.parameters():
            p.requires_grad = False
        mono_spec.model.eval()
        self.D_mo, self.F_mo = mono_spec.embed_dim, mono_spec.n_freq

        # ---- Spatial stream (Sphere) ----
        self.sphere = spatial_encoder
        if freeze_backbone:
            for p in self.sphere.parameters():
                p.requires_grad = False
            self.sphere.eval()
        for p in self.sphere.gram.parameters():                 # conditioner stays frozen
            p.requires_grad = False
        self.sphere.gram.eval()
        self.D_sp, self.F_sp = self.sphere.encoder_embedding_dim, self.sphere.p_f_dim

        # ---- spatial projection plumbing (freq-pool over F_sp) ----
        self.input_norm_sp = nn.LayerNorm(self.F_sp * self.D_sp)
        self.q_sp = nn.Parameter(torch.randn(self.D_sp) * 0.02)
        self.k_proj_sp = nn.Linear(self.D_sp, self.D_sp, bias=False)
        self.sphere_proj = nn.Sequential(
            nn.LayerNorm(self.D_sp), nn.Linear(self.D_sp, proj_dim),
            nn.GELU(), nn.Dropout(dropout))

        # ---- mono projection plumbing (no freq-pool, F_mono = 1) ----
        self.input_norm_mo = nn.LayerNorm(self.D_mo)
        self.mono_proj = nn.Sequential(
            nn.LayerNorm(self.D_mo), nn.Linear(self.D_mo, proj_dim),
            nn.GELU(), nn.Dropout(dropout))

        # ---- fuse + decode ----
        self.feat_proj = nn.Linear(2 * proj_dim, proj_dim)
        self.head = SeldHead(proj_dim, out_shape, params)

    def train(self, mode: bool = True):
        super().train(mode)
        self.mono.model.eval()                                  # Dasheng always frozen
        if self.freeze_backbone:
            self.sphere.eval()                                  # frozen pre-trained spatial
        self.sphere.gram.eval()                                 # conditioner always frozen
        return self

    @staticmethod
    def _freq_pool(z_flat, query, k_proj, F_dim, D_dim):
        B, T, _ = z_flat.shape
        z = z_flat.view(B, T, F_dim, D_dim)
        attn = (k_proj(z) @ query) / (D_dim ** 0.5)
        attn = attn.softmax(dim=-1).unsqueeze(-1)
        return (z * attn).sum(dim=2)                            # (B, T, D)

    def _avg_pool_time(self, z):                                # (B,T,D)->(B,T_seld,D)
        return F.adaptive_avg_pool1d(z.transpose(1, 2), self.T_seld).transpose(1, 2)

    def _spatial_tokens(self, x):
        x_w, x_yzx, x_iv = x[:, 0:1], x[:, 1:4], x[:, 4:7]
        ctx = torch.no_grad() if self.freeze_backbone else contextlib.nullcontext()
        with ctx:
            with torch.no_grad():
                x_logmel4 = torch.cat([x_w, x_yzx], dim=1)        # W,Y,Z,X
                z = self.sphere.pass_through_encoder(x_logmel4, x_iv)
                z = z[:, 1:, :]                                   # drop CLS
                z = rearrange(z, "b (f t) d -> b t (f d)",
                              f=self.sphere.grid_size[0])
        return z

    def forward(self, x, dash):
        """x: (B,7,T,F) ch0=W, ch1:4=YZX, ch4:7=IV;  dash: Dasheng mel front-end."""
        # ---- Mono stream (Dasheng, frozen; no freq-pool, F_mono = 1) ----
        with torch.no_grad():
            z_mo = self.mono.forward_fn(self.mono.model, dash)  # (B, T_mono, D_mo)
        z_mo = self.input_norm_mo(z_mo)
        pooled = self._avg_pool_time(z_mo)
        z_mo = self.mono_proj(pooled)        # (B, T_seld, 256)

        # ---- Spatial stream (Sphere; freq-pool over F_sp) ----
        z_sp = self._spatial_tokens(x)
        z_sp = self.input_norm_sp(z_sp)
        z_sp = self._freq_pool(z_sp, self.q_sp, self.k_proj_sp, self.F_sp, self.D_sp)
        pooled = self._avg_pool_time(z_sp)
        z_sp = self.sphere_proj(pooled)      # (B, T_seld, 256)

        # ---- Fuse + decode ----
        z = self.feat_proj(torch.cat([z_sp, z_mo], dim=-1))     # (B, T_seld, 256)
        return self.head(z)

