import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


def pad_or_truncate_1d(feature: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Adjust the length of a feature tensor by padding or truncating.

    Parameters
    ----------
    feature : torch.Tensor
        A tensor containing the feature to be adjusted. Expected shape is `(n_frames, ...)`.
    target_length : int
        The desired length of the feature along the first dimension.

    Returns
    -------
    torch.Tensor
        A tensor of shape `(target_length, ...)`, padded or truncated as needed.

    Notes
    -----
    Padding is applied using zero-padding. Truncation is performed along the first dimension
    by slicing the tensor.
    """
    n_frames = feature.shape[1]
    padding = target_length - n_frames
    if padding > 0:
        pad = torch.nn.ZeroPad1d((0, padding))
        return pad(feature)
    elif padding < 0:
        return feature[:, :target_length]
    return feature

def pad_or_truncate(feature: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Adjust the length of a feature tensor by padding or truncating.

    Parameters
    ----------
    feature : torch.Tensor
        A tensor containing the feature to be adjusted. Expected shape is `(n_frames, ...)`.
    target_length : int
        The desired length of the feature along the first dimension.

    Returns
    -------
    torch.Tensor
        A tensor of shape `(target_length, ...)`, padded or truncated as needed.

    Notes
    -----
    Padding is applied using zero-padding. Truncation is performed along the first dimension
    by slicing the tensor.
    """
    n_frames = feature.shape[1]
    padding = target_length - n_frames
    if padding > 0:
        pad = torch.nn.ZeroPad2d((0, 0, 0, padding))
        return pad(feature)
    elif padding < 0:
        return feature[:, :target_length, :]
    return feature



def pad_or_truncate_batch(x: torch.Tensor, target_length: int) -> torch.Tensor:
    B, C, T, F_bins = x.shape
    if T == target_length:
        return x
    elif T > target_length:
        return x[:, :, :target_length, :]
    else:
        # F.pad format: (last_dim_left, last_dim_right, second_last_dim_left, second_last_dim_right)
        # We want to pad the T dimension (second to last) on the right by (target_length - T)
        return F.pad(x, (0, 0, 0, target_length - T))



def instance_normalize(feature: torch.Tensor) -> torch.Tensor:
    """
    Normalize a feature tensor using the specified mean and standard deviation.

    Parameters
    ----------
    feature : torch.Tensor
        A tensor containing the feature to normalize.
    mean : float
        The mean value for normalization.
    std : float
        The standard deviation value for normalization.

    Returns
    -------
    torch.Tensor
        A tensor where each element is normalized as:
        `(feature - mean) / (std)`.

    Notes
    -----
    This normalization scales the data to have a mean of 0 and reduces the amplitude
    by the factor of `2 * std`.
    """
    return (feature - feature.mean()) / (feature.std() + 1e-8)


def pre_process_audio_visage(waveform, audio_sr, resample_sr):
    # waveform: (4, T) — ambisonics channels [W, Y, Z, X] with SN3D
    waveform = (
        torchaudio.functional.resample(waveform, audio_sr, resample_sr)
        if audio_sr != resample_sr
        else waveform
    )
    # Ensure 4 channels
    assert waveform.shape[0] == 4, f"Expected 4 channels, got {waveform.shape[0]}"
    # Make sure audio is 5 seconds
    target_len = resample_sr * 5
    padding = target_len - waveform.shape[1]
    if padding > 0:
        waveform = F.pad(waveform, (0, padding), "constant", 0)
    elif padding < 0:
        waveform = waveform[:, :target_len]
    return waveform  # (4, resample_sr * 5)

def pre_process_audio(audio, audio_sr, resample_sr):
    waveform = audio[0, :] if audio.ndim > 1 else audio
    # Resample the audio
    waveform = (
        torchaudio.functional.resample(waveform, audio_sr, resample_sr)
        if audio_sr != resample_sr
        else waveform
    )
    # Normalize the audio using RMSE
    waveform = normalize_audio(waveform, -14.0)
    waveform = waveform.reshape(1, -1)
    # Make sure audio is 10 seconds
    padding = resample_sr * 10 - waveform.shape[1]
    if padding > 0:
        waveform = F.pad(waveform, (0, padding), "constant", 0)
    elif padding < 0:
        waveform = waveform[:, : resample_sr * 10]
    return waveform[0]

def pre_process_noise(audio, audio_sr, resample_sr):
    waveform = audio[0, :] if audio.ndim > 1 else audio
    # Resample the audio
    waveform = (
        torchaudio.functional.resample(waveform, audio_sr, resample_sr)
        if audio_sr != resample_sr
        else waveform
    )
    # Normalize the audio using RMSE
    waveform = normalize_audio(waveform, -14.0)
    return waveform

def normalize_audio(audio_data, target_dBFS=-14.0):
    rms = torch.sqrt(torch.mean(audio_data**2))  # Calculate the RMS of the audio
    if rms == 0:  # Avoid division by zero in case of a completely silent audio
        return audio_data
    current_dBFS = 20 * torch.log10(rms)  # Convert RMS to dBFS
    gain_dB = target_dBFS - current_dBFS  # Calculate the required gain in dB
    gain_linear = 10 ** (gain_dB / 20)  # Convert gain from dB to linear scale
    normalized_audio = audio_data * gain_linear  # Apply the gain to the audio data
    return normalized_audio


def _wav2fbank(waveform, sr, num_mel_bins, input_length):
    """
    Compute FBANK features from a waveform file with optional RIR transformation.

    Parameters
    ----------
    filename : str
        Path to the audio file to process.

    Returns
    -------
    Tuple[Tuple[torch.Tensor, torch.Tensor], Optional[dict]]
        A tuple containing:
        - FBANK features as a tuple of tensors (for stereo or mono channels).
        - Metadata for the RIR point used, or `None` if no RIR was applied.

    Notes
    -----
    - FBANK features are computed using `torchaudio.compliance.kaldi.fbank`.
    - For stereo audio, features are computed separately for each channel.
    - For mono audio, features are duplicated to mimic a stereo structure.
    - If an RIR is provided, it is applied to the waveform via convolution
        before computing FBANK features.
    """

    # Python garbage collector caches this, so it is okay to keep it for now.
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        win_length=1024,
        hop_length=320,
        f_min=50,
        f_max=sr // 2,
        n_mels=num_mel_bins,
        power=2.0,
    ).cuda()

    mel = melspec(waveform).transpose(3, 2)
    log_mel = (mel + torch.finfo().eps).log()
    # Handle stereo/mono channels consistently
    if waveform.shape[0] == 1:
        # For mono audio, duplicate the channel to create stereo
        log_mel = torch.cat((log_mel, log_mel), dim=0)
    return log_mel


