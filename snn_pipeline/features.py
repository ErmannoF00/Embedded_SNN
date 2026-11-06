"""Lightweight EEG window features implemented identically in C++.

Input: (windows, 250 samples, 8 channels) or (250, 8), sampled at 250 Hz.
Output: 32 floats ordered [channel0: four features, channel1: four, ...].
No filtering or presumed medical interpretation is performed here.
"""
import numpy as np

SAMPLES = 250
CHANNELS = 8
FEATURES_PER_CHANNEL = 4
FEATURE_COUNT = CHANNELS * FEATURES_PER_CHANNEL


def extract_features(windows: np.ndarray) -> np.ndarray:
    data = np.asarray(windows, dtype=np.float32)
    if data.shape == (SAMPLES, CHANNELS):
        data = data[None, :, :]
    if data.ndim != 3 or data.shape[1:] != (SAMPLES, CHANNELS):
        raise ValueError(f"Expected (n, {SAMPLES}, {CHANNELS}), got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("EEG windows contain NaN or infinity")

    # Calculations are performed in float32, matching the Cortex-M runtime.
    rms = np.sqrt(np.mean(data * data, axis=1, dtype=np.float32))
    mean_abs = np.mean(np.abs(data), axis=1, dtype=np.float32)
    diff = np.diff(data, axis=1)
    diff_rms = np.sqrt(np.mean(diff * diff, axis=1, dtype=np.float32))
    # Deterministic sign crossing rule: >=0 is nonnegative.
    crossings = (data[:, 1:, :] >= 0) != (data[:, :-1, :] >= 0)
    zcr = crossings.sum(axis=1).astype(np.float32) / np.float32(SAMPLES - 1)
    return np.stack((rms, mean_abs, diff_rms, zcr), axis=2).reshape(-1, FEATURE_COUNT)
