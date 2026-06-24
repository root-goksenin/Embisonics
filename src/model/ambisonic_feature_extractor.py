import torch
from torch import Tensor
from typing import Optional, Callable
from torchaudio.transforms import Spectrogram, MelScale


class FeatureExtractor(torch.nn.Module):
    """FOA feature extractor.

    Input audio is ACN/SN3D-ordered first-order Ambisonics: channels
    (W, Y, Z, X). The "XYZ" name used internally for channels 1:4 is just a
    label for "the three directional channels"; the intensity computation is
    order-agnostic and the resulting 3 AIV components stay aligned with the
    (Y, Z, X) log-mel channels.
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        n_mels: int = 128,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        power: float = None,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: Optional[bool] = None,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.power = power
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.normalized = normalized
        self.n_mels = n_mels
        self.f_max = f_max
        self.f_min = f_min
        self.eps = 1e-6
        self.spectrogram = Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            pad=self.pad,
            window_fn=window_fn,
            power=None,                       # complex STFT
            normalized=self.normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
            onesided=True,
        )
        self.mel_scale = MelScale(
            self.n_mels, self.sample_rate, self.f_min, self.f_max,
            self.n_fft // 2 + 1, norm, mel_scale,
        )

    # ------------------------------------------------------------------
    def _intensity_energy(self, spec: Tensor):
        """Raw active intensity and energy at each linear TF bin.

        spec: complex (B, 4, F, T), channels (W, Y, Z, X).
        Returns:
          I : (B, 3, F, T)  I = 2 Re[conj(W) (Y,Z,X)]
          E : (B, 1, F, T)  E = |W|^2 + sum |YZX|^2
        """
        W = spec[:, [0]]
        YZX = spec[:, 1:]
        I = 2.0 * torch.real(torch.conj(W) * YZX)             # (B, 3, F, T)
        E = (torch.abs(W) ** 2) + (torch.abs(YZX) ** 2).sum(dim=1, keepdim=True)
        return I, E

    def _get_foa_intensity_vectors(self, linear_spectra: Tensor) -> Tensor:
        """Per-band energy-normalized active intensity (the DIRECTION feature).

        We mel-aggregate the raw intensity I and the raw energy E *separately*
        and divide afterwards:

            Ihat(b) = mel(I)(b) / (mel(E)(b) + eps)

        Because ||I(f)|| <= E(f) holds pointwise at every linear bin and the
        mel weights m_b(f) are nonnegative, this guarantees ||Ihat(b)|| <= 1
        for every band, with magnitude -> 1 for a plane wave and -> 0 for a
        diffuse field, *independent* of the mel filterbank normalization (the
        per-band row-sum cancels between numerator and denominator). The
        magnitude is thus a genuine per-band direct-energy fraction (an
        instantaneous 1 - diffuseness).

        Note this is energy-weighted pooling: low-energy / noisy bins are
        down-weighted automatically, which is the desired behaviour for a
        direction cue. It deliberately discards absolute level; that is fine --
        it only shapes the direction stream and does not touch the
        distance-relevant cues carried elsewhere.
        """
        I, E = self._intensity_energy(linear_spectra)
        I_mel = self.mel_scale(I)                              # (B, 3, M, T)
        E_mel = self.mel_scale(E)                              # (B, 1, M, T)
        return I_mel / (E_mel + self.eps)                      # (B, 3, M, T)

    # ------------------------------------------------------------------
    def forward(self, audio: Tensor) -> Tensor:
        """Original interface: (B, 7, M, T) = [log-mel(4), AIV(3)]."""
        spec = self.spectrogram(audio)
        power_spec = torch.abs(spec) ** self.power
        mel_spec = torch.log(self.mel_scale(power_spec) + self.eps)
        foa_aiv = self._get_foa_intensity_vectors(spec)
        return torch.cat([mel_spec, foa_aiv], dim=1)

    def extract_all(self, audio: Tensor) -> dict:
        """Returns a dict of (B, C, T, F) tensors (note: T, F order).

          logmel        (B, 4, T, F)  log-mel power of [W, Y, Z, X]
          aiv           (B, 3, T, F)  normalized active intensity (input feat)
          intensity_mel (B, 3, T, F)  RAW mel intensity      (diffuseness)
          energy_mel    (B, 1, T, F)  mel energy             (diffuseness)

        intensity_mel / energy_mel are NOT meant as encoder inputs; they are
        the ingredients for the diffuseness target. mel_scale is linear over
        frequency, so it applies validly to the signed intensity I. The input
        AIV feature is derived from exactly these same mel-pooled quantities
        (aiv = intensity_mel / energy_mel), so the input direction feature and
        the diffuseness target now share one physical pipeline.
        """
        spec = self.spectrogram(audio)                         # (B, 4, F, T) complex

        power_spec = torch.abs(spec) ** self.power
        logmel = torch.log(self.mel_scale(power_spec) + self.eps)   # (B, 4, M, T)

        I, E = self._intensity_energy(spec)                    # (B,3,F,T), (B,1,F,T)
        intensity_mel = self.mel_scale(I)                      # (B, 3, M, T)
        energy_mel = self.mel_scale(E)                         # (B, 1, M, T)
        aiv = intensity_mel / (energy_mel + self.eps)          # (B, 3, M, T)

        def tf(x: Tensor) -> Tensor:                           # (B,C,M,T) -> (B,C,T,F)
            return x.transpose(-1, -2)

        return {
            "logmel": tf(logmel),
            "aiv": tf(aiv),
            "intensity_mel": tf(intensity_mel),
            "energy_mel": tf(energy_mel),
        }


if __name__ == "__main__":
    audio = torch.zeros([2, 4, 160000])
    extractor = FeatureExtractor(
        sample_rate=16000, n_fft=1024, win_length=1024,
        hop_length=16000 // 100, f_min=50, f_max=16000 // 2, n_mels=128,
        power=2.0,
    )
    out = extractor(audio)
    print("forward:", out.shape)                               # (2, 7, M, T)
    d = extractor.extract_all(audio)
    for k, v in d.items():
        print(k, tuple(v.shape))