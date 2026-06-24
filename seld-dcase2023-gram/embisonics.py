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
from typing import Callable,Optional

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
# Injectable mono-encoder: SPEAR  (frozen, F_mono = 1)
#   Zipformer backbone (93M params), 512-d embeddings at ~50 Hz.
#   Unlike Dasheng it consumes the *raw 16 kHz waveform* (B, n_samples) instead
#   of a mel front-end, and exposes its features under outputs["encoder_out"].
# ============================================================================
class SpearMono(nn.Module):
    """Frozen SPEAR backbone returning per-frame embeddings (B, T_mono, D)."""
    def __init__(self, hf_id="marcoyang/spear-base-speech-audio-v2"):
        super().__init__()
        self.model = AutoModel.from_pretrained(hf_id, trust_remote_code=True)
        self.embed_dim = 512        # 512 (base)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()                                       # always frozen
        return self

    @torch.no_grad()
    def forward(self, wav, wav_len=None):                       # wav -> (B, T_mono, D)
        if wav_len is None:                                     # fixed-length SELD clips
            wav_len = wav.new_full((wav.shape[0],), wav.shape[-1], dtype=torch.long)
        return self.model(wav, wav_len)["encoder_out"]          # (B, T_mono, D)


def spear_mono_spec(spear_module: SpearMono, name="spear"):
    """SPEAR wrapped as a MonoEncoderSpec (no frequency axis -> n_freq = 1)."""
    return MonoEncoderSpec(
        name=name,
        model=spear_module,
        embed_dim=spear_module.embed_dim,                       # D_mono
        n_freq=1,                                               # F_mono
        forward_fn=lambda m, wav: m(wav),                       # (B, T_mono, D_mono)
    )


# ============================================================================
# Sphere (spatial) + SPEAR (mono) SELD probe
#   Mirrors DashengSphereSELD: per-stream projection plumbing -> fuse -> SeldHead.
#   Only the mono front-end differs — SPEAR consumes the raw 16 kHz waveform
#   (B, n_samples) rather than a mel spectrogram.
# ============================================================================
class SpearSphereSELD(nn.Module):
    """Frozen-/trainable-encoder SELD probe.

    Spatial stream : pre-trained Sphere encoder (frozen when freeze_backbone).
    Mono stream    : frozen SPEAR, injected via `mono_spec`.
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

        # ---- Mono stream (SPEAR, always frozen) ----
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
        self.mono.model.eval()                                  # SPEAR always frozen
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

    def forward(self, x, spear):
        """x: (B,7,T,F) ch0=W, ch1:4=YZX, ch4:7=IV;  spear: raw 16 kHz waveform (B, n_samples)."""
        # ---- Mono stream (SPEAR, frozen; no freq-pool, F_mono = 1) ----
        with torch.no_grad():
            z_mo = self.mono.forward_fn(self.mono.model, spear)  # (B, T_mono, D_mo)
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


def gramt_mono_spec(gram_module, n_freq, embed_dim, name="gram-t"):
    """GRAM-T pulled from a SphereV4 (sphere.gram)."""
    return MonoEncoderSpec(name, gram_module, embed_dim, n_freq,
                           lambda m, w: m(w, strategy="raw"))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


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
                print("Learning front-end")
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