import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Optional, Callable
from torchaudio.transforms import Spectrogram, MelScale


class FeatureExtractor(nn.Module):
    """8-channel FOA features for ACN/SN3D ambiX [W, Y, Z, X].

    encoder input (forward) : (B, 8, M, T) = [logmel(W,Y,Z,X) | AIV(3) | diffuseness(1)]
      - AIV channels are in extractor order (IV_y, IV_z, IV_x), aligned with the
        (Y,Z,X) logmel channels. RouteATarget reorders to (x,y,z) for grounding.
    extract_all also returns intensity_mel / energy_mel for the Route A target.

    Conventions verified against STARSS23 steering vectors:
      I = 2 Re(conj(W)*[Y,Z,X]) points TOWARD the source (DOA = +I/||I||, no minus).
      ||I_bin|| <= E_bin pointwise (AM-GM), so AIV magnitude and diffuseness are in [0,1].
    """
    def __init__(self, sample_rate=32000, n_fft=1024, win_length=1024,
                 hop_length=None, f_min=50.0, f_max=None, n_mels=128,
                 power=2.0, mel_scale="htk",
                 diffuseness_tau_t=11, diffuseness_tau_f=3, diffuseness_c=1.0,
                 window_fn: Callable[..., Tensor] = torch.hann_window):
        super().__init__()
        self.eps = 1e-6
        self.power = power
        self.hop_length = hop_length or (win_length // 2)
        self.tau_t = diffuseness_tau_t
        self.tau_f = diffuseness_tau_f
        self.c = diffuseness_c
        assert self.tau_t % 2 == 1 and self.tau_f % 2 == 1
        self.spectrogram = Spectrogram(n_fft=n_fft, win_length=win_length,
                                       hop_length=self.hop_length, power=None,
                                       window_fn=window_fn, center=True,
                                       pad_mode="reflect", onesided=True)
        self.mel_scale = MelScale(n_mels, sample_rate, f_min,
                                  f_max if f_max is not None else sample_rate // 2,
                                  n_fft // 2 + 1, None, mel_scale)

    def _intensity_energy(self, spec):
        W = spec[:, [0]]
        YZX = spec[:, 1:]                                       # [Y, Z, X]
        I = 2.0 * torch.real(torch.conj(W) * YZX)              # (B,3,F,T) order (y,z,x)
        E = torch.abs(W) ** 2 + (torch.abs(YZX) ** 2).sum(1, keepdim=True)
        return I, E

    def _tf_smooth(self, x):
        """Mel+time average for the diffuseness coherence window (stride 1, 'same')."""
        return F.avg_pool2d(x, (self.tau_f, self.tau_t), stride=1,
                            padding=(self.tau_f // 2, self.tau_t // 2),
                            count_include_pad=False)

    def _diffuseness(self, intensity_mel, energy_mel):
        I = self._tf_smooth(intensity_mel.float())
        E = self._tf_smooth(energy_mel.float())
        num = torch.linalg.norm(I, dim=1, keepdim=True)         # ||<I>||
        psi = 1.0 - num / (self.c * E + self.eps)
        return psi.clamp(0.0, 1.0)                              # (B,1,M,T)

    def extract_all(self, audio: Tensor) -> dict:
        spec = self.spectrogram(audio)                          # (B,4,F,T) complex
        logmel = torch.log(self.mel_scale(torch.abs(spec) ** self.power) + self.eps)
        I, E = self._intensity_energy(spec)
        intensity_mel = self.mel_scale(I)                       # (B,3,M,T) order (y,z,x)
        energy_mel = self.mel_scale(E)                          # (B,1,M,T)
        aiv = intensity_mel / (energy_mel + self.eps)           # (B,3,M,T) ||.||<=1
        diffuseness = self._diffuseness(intensity_mel, energy_mel)
        return dict(logmel=logmel, aiv=aiv, intensity_mel=intensity_mel,
                    energy_mel=energy_mel, diffuseness=diffuseness)

    def forward(self, audio: Tensor) -> Tensor:
        """(B, 8, M, T) encoder input."""
        d = self.extract_all(audio)
        return torch.cat([d["logmel"], d["aiv"], d["diffuseness"]], dim=1)