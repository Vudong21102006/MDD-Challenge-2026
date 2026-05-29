"""
Audio processing utilities for the MDD Challenge.

CRITICAL: All audio file reading MUST use ``soundfile`` (``sf.read()``).
Do NOT use ``torchaudio.load()`` or ``torchcodec`` — they cause
WinError 127 on Windows due to backend DLL resolution issues.
"""

from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF


def load_and_resample(
    audio_path: str, target_sr: int = 16000
) -> Tuple[torch.Tensor, int]:
    """
    Read a ``.wav`` file via ``soundfile``, convert to a PyTorch tensor,
    and resample to ``target_sr`` if the original sample rate differs.

    Args:
        audio_path:  Path to the ``.wav`` file.
        target_sr:   Desired sample rate in Hz (default: 16000).

    Returns:
        A tuple ``(waveform, sample_rate)`` where ``waveform`` is a
        PyTorch tensor of shape ``(num_channels, num_samples)`` and
        ``sample_rate`` is the (possibly resampled) sample rate.
    """
    # soundfile returns (samples, sample_rate) — channels_last shape
    samples, orig_sr = sf.read(str(audio_path))

    # Convert NumPy array → torch tensor (float32).
    # sf.read returns shape (num_samples,) for mono or (num_samples, num_channels) for stereo.
    waveform = torch.from_numpy(samples.copy()).float()

    # Reshape to (num_channels, num_samples) so downstream code has a consistent layout.
    if waveform.ndim == 1:
        # Mono: add channel dimension.
        waveform = waveform.unsqueeze(0)  # (1, num_samples)
    else:
        # Stereo: transpose from (num_samples, num_channels) → (num_channels, num_samples).
        waveform = waveform.transpose(0, 1)

    # Resample if the original rate differs from the target.
    if orig_sr != target_sr:
        waveform = AF.resample(waveform, orig_sr, target_sr)

    return waveform, target_sr


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    Convert a multi-channel waveform to mono by averaging across channels.

    Args:
        waveform:  A PyTorch tensor of shape ``(num_channels, num_samples)``.

    Returns:
        A tensor of shape ``(1, num_samples)`` containing the
        channel-averaged mono signal.
    """
    if waveform.ndim < 2 or waveform.shape[0] == 1:
        # Already mono or degenerate shape — return as-is (with channel dim).
        return waveform if waveform.ndim == 2 else waveform.unsqueeze(0)

    # Mean across the channel dimension, keep the dimension.
    return waveform.mean(dim=0, keepdim=True)
