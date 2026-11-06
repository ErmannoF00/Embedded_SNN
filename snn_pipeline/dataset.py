"""Explicitly labeled EEG windows and a separately identified synthetic demo."""
from pathlib import Path

import numpy as np

from .features import CHANNELS, SAMPLES


def load_labeled_windows(path: Path):
    with np.load(path, allow_pickle=False) as z:
        if "windows" not in z or "labels" not in z:
            raise ValueError("Dataset requires windows and labels arrays")
        x = np.asarray(z["windows"], dtype=np.float32)
        original_labels = z["labels"]
        if original_labels.dtype.kind not in "iu":
            raise ValueError("Ground-truth labels must be integer 0/1, not probabilities or strings")
        y = np.asarray(original_labels, dtype=np.int64)
        groups = np.asarray(z["groups"]) if "groups" in z else None
        if "sample_rate_hz" not in z:
            raise ValueError("Real EEG dataset must declare sample_rate_hz=250")
        rate = np.asarray(z["sample_rate_hz"])
        if rate.ndim != 0 or not np.isclose(float(rate), 250.0, atol=1e-5):
            raise ValueError("Only exactly 250 Hz window recordings are supported")
    if x.ndim != 3 or x.shape[1:] != (SAMPLES, CHANNELS):
        raise ValueError(f"Expected windows (N,{SAMPLES},{CHANNELS}); got {x.shape}")
    if y.shape != (len(x),) or not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be N binary 0/1 ground-truth values")
    if not np.isfinite(x).all():
        raise ValueError("Input windows contain nonfinite data")
    if groups is not None and groups.shape != (len(x),):
        raise ValueError("groups must have one group id per window")
    if len(np.unique(y)) != 2:
        raise ValueError("Both classes must be present")
    return x, y, groups


def synthetic_demo(n_per_class: int = 90, seed: int = 41):
    """Two different oscillatory signal classes. NOT human EEG or clinical data."""
    rng = np.random.default_rng(seed)
    t = np.arange(SAMPLES, dtype=np.float32) / np.float32(250)
    x = np.empty((2*n_per_class, SAMPLES, CHANNELS), dtype=np.float32)
    y = np.repeat(np.array([0, 1], dtype=np.int64), n_per_class)
    for i, label in enumerate(y):
        for c in range(CHANNELS):
            freq = rng.uniform(8, 12) if label == 0 else rng.uniform(19, 25)
            amp = rng.uniform(0.7, 1.3)
            phase = rng.uniform(0, 2*np.pi)
            slow_phase = rng.uniform(0, 2*np.pi)
            wave = (amp * np.sin(2*np.pi*freq*t + phase)
                    + 0.20*np.sin(2*np.pi*3*t + slow_phase)
                    + rng.normal(0, 0.20, SAMPLES))
            x[i, :, c] = wave.astype(np.float32)
    indices = rng.permutation(len(y))
    return x[indices], y[indices], None
