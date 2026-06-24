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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class SeldModel(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params, in_vid_feat_shape=None):
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
            self.mhsa_block_list.append(nn.MultiheadAttention(embed_dim=self.params['rnn_size'], num_heads=self.params['nb_heads'], dropout=self.params['dropout_rate'], batch_first=True))
            self.layer_norm_list.append(nn.LayerNorm(self.params['rnn_size']))

        # fusion layers
        if in_vid_feat_shape is not None:
            self.visual_embed_to_d_model = nn.Linear(in_features = int(in_vid_feat_shape[2]*in_vid_feat_shape[3]), out_features = self.params['rnn_size'] )
            self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.params['rnn_size'], nhead=self.params['nb_heads'], batch_first=True)
            self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=self.params['nb_transformer_layers'])

        self.fnn_list = torch.nn.ModuleList()
        if params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(nn.Linear(params['fnn_size'] if fc_cnt else self.params['rnn_size'], params['fnn_size'], bias=True))
        print(out_shape[-1])
        self.fnn_list.append(nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else self.params['rnn_size'], out_shape[-1], bias=True))

        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()

    def forward(self, x, vid_feat=None):
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

        if vid_feat is not None:
            vid_feat = vid_feat.view(vid_feat.shape[0], vid_feat.shape[1], -1)  # b x 50 x 49
            vid_feat = self.visual_embed_to_d_model(vid_feat)
            x = self.transformer_decoder(x, vid_feat)

        for fnn_cnt in range(len(self.fnn_list) - 1):
            x = self.fnn_list[fnn_cnt](x)
        doa = self.fnn_list[-1](x)

        return doa



# ============================================================================
# Embisonics SELD model
# ----------------------------------------------------------------------------
# A First-Order-Ambisonics ("embisonics") network for joint localisation +
# distance estimation. It keeps the SELDnet backbone (Conv front-end ->
# GRU -> GLU -> MHSA -> FNN) but restructures the front-end into two parallel
# streams that respect the physical structure of an Ambisonics signal:
#
#   * mono / energy stream  : the omni W channel only -> "what & when"
#   * spatial stream        : the full FOA tensor (directional channels +
#                             active-intensity vectors) -> "where & how far"
#
# The two streams are pooled across frequency with a learned-query attention
# pool, projected to a common width, fused, and decoded by the shared
# recurrent-attentive head. The output is the multi-ACCDOA + distance vector
# [x, y, z, dist] per track/class, with Tanh on the Cartesian DOA part and
# ReLU on the (non-negative) distance part. It is shape-compatible with
# MSELoss_ADPIT above (num_track=3, num_axis=4 -> XYZD).
# ============================================================================
class FreqAttentionPool(nn.Module):
    """Learned-query attention pooling across the frequency axis.

    Collapses a conv feature map (B, C, T, F) into a per-frame embedding
    (B, T, C) by weighting frequency bins with a content-dependent score.
    This is the Ambisonics-friendly replacement for the baseline's
    "flatten all frequency bins into the GRU input" step: spatial cues live
    in a handful of frequency regions, so a soft frequency selector keeps the
    head dimension fixed (= nb_cnn2d_filt) regardless of mel resolution.
    """
    def __init__(self, channels):
        super().__init__()
        self.q = nn.Parameter(torch.randn(channels) * 0.02)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.scale = channels ** 0.5

    def forward(self, x):                       # (B, C, T, F)
        x = x.permute(0, 2, 3, 1)               # (B, T, F, C)
        attn = (self.k_proj(x) @ self.q) / self.scale   # (B, T, F)
        attn = attn.softmax(dim=-1).unsqueeze(-1)       # (B, T, F, 1)
        return (x * attn).sum(dim=2)            # (B, T, C)


class EmbisonicsSeldModel(nn.Module):
    """Embisonics (FOA) SELD network for localisation + distance estimation.

    Args:
        in_feat_shape: (batch, channels, time, mel). `channels` is the number
            of Ambisonics feature channels, e.g. 7 = 4 FOA log-mel (W,Y,Z,X)
            + 3 active-intensity-vector channels.
        out_shape:     (batch, frames_out, num_track*num_axis*num_class).
                       num_axis must be 4 -> [x, y, z, dist].
        params:        the usual SELDnet param dict (nb_cnn2d_filt,
                       t_pool_size, f_pool_size, rnn_size, nb_rnn_layers,
                       nb_self_attn_layers, nb_heads, nb_fnn_layers,
                       fnn_size, dropout_rate, unique_classes).

    Input  : (B, C, T, F)  -- channel 0 is assumed to be the omni W component.
    Output : (B, T_out, num_track*4*num_class) multi-ACCDOA + distance.
    """
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.params = params
        self.nb_classes = params['unique_classes']
        self.nb_tracks = 3        # ADPIT expects exactly 3 tracks (A / B / C)
        self.nb_axis = 4          # x, y, z, distance

        in_ch = in_feat_shape[1]
        nb_filt = params['nb_cnn2d_filt']
        drop = params['dropout_rate']
        proj_dim = params['rnn_size']

        # ---- Conv front-ends: one per stream, identical t/f pooling so the
        #      two streams stay frame-aligned for fusion. ----
        def make_stream(stream_in_ch):
            layers = nn.ModuleList()
            for conv_cnt in range(len(params['f_pool_size'])):
                layers.append(ConvBlock(nb_filt if conv_cnt else stream_in_ch, nb_filt))
                layers.append(nn.MaxPool2d((params['t_pool_size'][conv_cnt],
                                            params['f_pool_size'][conv_cnt])))
                layers.append(nn.Dropout2d(p=drop))
            return layers

        self.spatial_convs = make_stream(in_ch)   # full FOA + intensity vectors
        self.mono_convs = make_stream(1)          # omni W channel only

        self.spatial_pool = FreqAttentionPool(nb_filt)
        self.mono_pool = FreqAttentionPool(nb_filt)

        self.spatial_proj = nn.Sequential(
            nn.LayerNorm(nb_filt), nn.Linear(nb_filt, proj_dim),
            nn.GELU(), nn.Dropout(drop))
        self.mono_proj = nn.Sequential(
            nn.LayerNorm(nb_filt), nn.Linear(nb_filt, proj_dim),
            nn.GELU(), nn.Dropout(drop))
        self.fuse = nn.Linear(2 * proj_dim, proj_dim)

        # ---- Shared recurrent-attentive head (same as SeldModel backbone) ----
        self.gru = nn.GRU(input_size=proj_dim, hidden_size=params['rnn_size'],
                          num_layers=params['nb_rnn_layers'], batch_first=True,
                          dropout=params['dropout_rate'], bidirectional=True)

        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for _ in range(params['nb_self_attn_layers']):
            self.mhsa_block_list.append(nn.MultiheadAttention(
                embed_dim=params['rnn_size'], num_heads=params['nb_heads'],
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

        # localisation head uses Tanh (bounded Cartesian DOA),
        # distance head uses ReLU (distance is non-negative).
        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()

    @staticmethod
    def _run_stream(convs, x):
        for layer in convs:
            x = layer(x)
        return x                                  # (B, nb_filt, T', F')

    def forward(self, x):
        """x: (B, C, T, F). Channel 0 is the omni W component."""
        # ---- two parallel Ambisonics streams ----
        spat = self._run_stream(self.spatial_convs, x)        # (B, nb_filt, T', F')
        mono = self._run_stream(self.mono_convs, x[:, 0:1])   # (B, nb_filt, T', F')

        spat = self.spatial_proj(self.spatial_pool(spat))     # (B, T', proj_dim)
        mono = self.mono_proj(self.mono_pool(mono))           # (B, T', proj_dim)
        z = self.fuse(torch.cat([spat, mono], dim=-1))        # (B, T', proj_dim)

        # ---- shared head ----
        z, _ = self.gru(z)
        z = torch.tanh(z)
        z = z[:, :, z.shape[-1] // 2:] * z[:, :, :z.shape[-1] // 2]   # GLU

        for mhsa, ln in zip(self.mhsa_block_list, self.layer_norm_list):
            z_in = z
            z, _ = mhsa(z_in, z_in, z_in)
            z = ln(z + z_in)

        for fnn_cnt in range(len(self.fnn_list) - 1):
            z = self.fnn_list[fnn_cnt](z)
        out = self.fnn_list[-1](z)                            # (B, T', 3*4*C)

        # ---- split localisation (x,y,z) and distance (d) heads ----
        B, T, _ = out.shape
        out = out.view(B, T, self.nb_tracks, self.nb_axis, self.nb_classes)
        doa = self.doa_act(out[:, :, :, :3, :])               # x, y, z  -> Tanh
        dist = self.dist_act(out[:, :, :, 3:, :])             # distance -> ReLU
        out = torch.cat([doa, dist], dim=3)                   # (B, T', 3, 4, C)
        return out.reshape(B, T, -1)                          # (B, T', 3*4*C)