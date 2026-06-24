# The SELDnet architecture
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field
import sys
from typing import Callable, Optional 
sys.path.append("..")

from einops import rearrange
from transformers import AutoModel


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

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class SeldModel(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.nb_classes = params['unique_classes']
        self.params=params
        self.conv_block_list = nn.ModuleList()
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(ConvBlock(in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1], out_channels=params['nb_cnn2d_filt']))
                self.conv_block_list.append(nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt])))
                self.conv_block_list.append(nn.Dropout2d(p=params['dropout_rate']))

        self.gru_input_dim = params['nb_cnn2d_filt'] * int(np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
        self.gru = torch.nn.GRU(input_size=self.gru_input_dim, hidden_size=params['rnn_size'],
                                num_layers=params['nb_rnn_layers'], batch_first=True,
                                dropout=params['dropout_rate'], bidirectional=True)

        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for mhsa_cnt in range(params['nb_self_attn_layers']):
            self.mhsa_block_list.append(nn.MultiheadAttention(embed_dim=self.params['rnn_size'], num_heads=params['nb_heads'], dropout=params['dropout_rate'],  batch_first=True))
            self.layer_norm_list.append(nn.LayerNorm(self.params['rnn_size']))

        self.fnn_list = torch.nn.ModuleList()
        if params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(nn.Linear(params['fnn_size'] if fc_cnt else self.params['rnn_size'], params['fnn_size'], bias=True))
        self.fnn_list.append(nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else self.params['rnn_size'], out_shape[-1], bias=True))

    def forward(self, x):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]

        for mhsa_cnt in range(len(self.mhsa_block_list)):
            x_attn_in = x 
            x, _ = self.mhsa_block_list[mhsa_cnt](x_attn_in, x_attn_in, x_attn_in)
            x = x + x_attn_in
            x = self.layer_norm_list[mhsa_cnt](x)

        for fnn_cnt in range(len(self.fnn_list) - 1):
            x = self.fnn_list[fnn_cnt](x)
        doa = torch.tanh(self.fnn_list[-1](x))
        return doa


class GramtAmbiSELD(nn.Module):
    """SELD model built on a frozen, pre-trained GRAMT-Ambisonics encoder
    loaded from the HuggingFace Hub (`labhamlet/gramt-ambisonics`).

    The Hub wrapper exposes a single raw representation via
    `gramt(log_mel, strategy="raw")` -> (B, T, F*D); it does not return
    per-layer hidden states, so there is no SUPERB layer weighting here.

    Pipeline
    --------
        log_mel (B, 7, T_in, F_mel)   ambisonic log-mel + IVs (from the extractor)
          |
          v  frozen GRAMT raw output      gramt(log_mel, strategy="raw")
        (B, T, F*D)
          |
          v  attention frequency pooling  learned query over F patches
        (B, T, D)
          |
          v  time adaptation to SELD len  adaptive_avg_pool -> T_seld
        (B, T_seld, D)
          |
          v  BiGRU + tanh multiplicative gating
        (B, T_seld, rnn_size)
          |
          v  MHSA blocks (residual + LayerNorm)
          |
          v  FNN head -> tanh
        doa (B, T_seld, out_shape[-1])

    Parameters
    ----------
    out_shape : tuple
        Target shape; out_shape[-2] is the SELD time resolution, out_shape[-1]
        the (multi-)ACCDOA output dimension (e.g. 3*4*13 for ADPIT).
    params : dict
        SELDnet hyper-params: 'rnn_size', 'nb_rnn_layers', 'nb_self_attn_layers',
        'nb_heads', 'nb_fnn_layers', 'fnn_size', 'dropout_rate', 'unique_classes'.
    gramt : nn.Module, optional
        Pre-built GRAMT-Ambisonics model whose `forward(log_mel, strategy="raw")`
        returns (B, T, F*D). If None (default), it is loaded from the Hub via
        `AutoModel.from_pretrained(hf_model_id, trust_remote_code=True)`.
    hf_model_id : str
        Hub repo to load when `gramt` is None (default "labhamlet/gramt-ambisonics").
    embed_dim : int
        Token dim D of the GRAMT encoder (768 for the "base" ambisonics model).
    f_patches : int
        Number of frequency patches F in the raw output (8 for the base model,
        so F*D = 6144).
    freeze_encoder : bool
        If True (default) the GRAMT encoder is frozen and kept in eval mode.
    """

    def __init__(self, out_shape, params, gramt: Optional[nn.Module] = None,
                 hf_model_id: str = "labhamlet/gramt-ambisonics",
                 trust_remote_code: bool = True,
                 embed_dim: int = 768, f_patches: int = 8,
                 freeze_encoder: bool = True):
        super().__init__()
        self.params = params
        self.nb_classes = params['unique_classes']
        self.T_seld = out_shape[-2]
        self.freeze_encoder = freeze_encoder

        # ---- Pre-trained GRAMT-Ambisonics encoder (HuggingFace Hub) ----
        # Loaded here by default; pass `gramt=` to inject an already-built model.
        if gramt is None:
            gramt = AutoModel.from_pretrained(
                hf_model_id, trust_remote_code=trust_remote_code)
        self.gramt = gramt
        if freeze_encoder:
            for p in self.gramt.parameters():
                p.requires_grad = False
            self.gramt.eval()

        self.D = embed_dim              # token dim
        self.F = f_patches              # nr. frequency patches  (F*D == raw last dim)

        # ---- Attention frequency pooling (learned query over F patches) ----
        self.input_norm = nn.LayerNorm(self.F * self.D)
        self.q_freq = nn.Parameter(torch.randn(self.D) * 0.02)
        self.k_proj_freq = nn.Linear(self.D, self.D, bias=False)

        # ---- BiGRU ----
        self.gru = nn.GRU(
            input_size=self.D, hidden_size=params['rnn_size'],
            num_layers=params['nb_rnn_layers'], batch_first=True,
            dropout=params['dropout_rate'], bidirectional=True,
        )

        # ---- Multi-head self-attention ----
        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for _ in range(params['nb_self_attn_layers']):
            self.mhsa_block_list.append(
                nn.MultiheadAttention(
                    embed_dim=params['rnn_size'], num_heads=params['nb_heads'],
                    dropout=params['dropout_rate'], batch_first=True,
                )
            )
            self.layer_norm_list.append(nn.LayerNorm(params['rnn_size']))

        # ---- FNN head (mirrors SeldModel) ----
        self.fnn_list = nn.ModuleList()
        if params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(
                    nn.Linear(params['fnn_size'] if fc_cnt else params['rnn_size'],
                              params['fnn_size'], bias=True)
                )
        self.fnn_list.append(
            nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else params['rnn_size'],
                      out_shape[-1], bias=True)
        )

    # keep the frozen encoder in eval mode regardless of .train()/.eval()
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.gramt.eval()
        return self

    @staticmethod
    def _freq_pool(z_flat, query, k_proj, F_dim, D_dim):
        """Attention pooling over the frequency-patch axis. (B,T,F*D)->(B,T,D)."""
        B, T, _ = z_flat.shape
        z = z_flat.view(B, T, F_dim, D_dim)
        attn = (k_proj(z) @ query) / (D_dim ** 0.5)   # (B, T, F)
        attn = attn.softmax(dim=-1).unsqueeze(-1)      # (B, T, F, 1)
        return (z * attn).sum(dim=2)                   # (B, T, D)

    def _adapt_time(self, z):                          # (B,T,D)->(B,T_seld,D)
        if z.shape[1] == self.T_seld:
            return z
        return F.adaptive_avg_pool1d(z.transpose(1, 2), self.T_seld).transpose(1, 2)

    def forward(self, log_mel):
        """log_mel: extractor `input_values`, (B, 7, T_in, F_mel) ambisonic features."""
        # ---- Frozen GRAMT: raw (last-layer) representation ----
        if self.freeze_encoder:
            with torch.no_grad():
                z = self.gramt(log_mel, strategy="raw")   # (B, T, F*D)
        else:
            z = self.gramt(log_mel, strategy="raw")
        assert z.shape[-1] == self.F * self.D, (
            f"raw dim {z.shape[-1]} != F*D ({self.F}*{self.D}); "
            f"set embed_dim/f_patches to match the checkpoint")

        # ---- Attention frequency pooling ----
        z = self.input_norm(z)
        z = self._freq_pool(z, self.q_freq, self.k_proj_freq, self.F, self.D)  # (B, T, D)

        # ---- Time adaptation to SELD resolution ----
        z = self._adapt_time(z)                               # (B, T_seld, D)

        # ---- BiGRU + tanh multiplicative gating (as in SeldModel) ----
        z, _ = self.gru(z)
        z = torch.tanh(z)
        z = z[:, :, z.shape[-1] // 2:] * z[:, :, :z.shape[-1] // 2]   # (B, T_seld, rnn_size)

        # ---- MHSA (residual + LayerNorm) ----
        for mhsa, ln in zip(self.mhsa_block_list, self.layer_norm_list):
            z_in = z
            z, _ = mhsa(z_in, z_in, z_in)
            z = z + z_in
            z = ln(z)

        # ---- FNN head ----
        for fnn in self.fnn_list[:-1]:
            z = fnn(z)
        doa = torch.tanh(self.fnn_list[-1](z))                # (B, T_seld, out_shape[-1])
        return doa