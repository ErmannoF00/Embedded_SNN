from pathlib import Path

import numpy as np
import pytest

from snn_pipeline.dataset import load_labeled_windows
from snn_pipeline.splitting import split_indices


def test_group_disjoint_and_repeatable():
    labels = np.tile([0, 1] * 10, 8)
    groups = np.repeat(np.arange(8), 20)
    first = split_indices(labels, groups, seed=41)
    again = split_indices(labels, groups, seed=41)
    for a, b in zip(first, again):
        np.testing.assert_array_equal(a, b)
        assert len(set(labels[a])) == 2
    for i in range(3):
        for j in range(i+1, 3):
            assert not set(groups[first[i]]) & set(groups[first[j]])
    assert np.unique(np.concatenate(first)).size == labels.size


def test_stratified_partition():
    labels = np.array([0] * 100 + [1] * 80)
    train, val, test = split_indices(labels, None, seed=3)
    assert abs(float(labels[train].mean()) - labels.mean()) < 0.025
    assert set(train).isdisjoint(val) and set(val).isdisjoint(test)


def test_invalid_labels_are_rejected(tmp_path: Path):
    f = tmp_path / "bad.npz"
    np.savez(f, windows=np.zeros((20, 250, 8), dtype=np.float32),
             labels=np.arange(20) % 2 + 0.1, sample_rate_hz=250)
    with pytest.raises(ValueError, match="integer"):
        load_labeled_windows(f)


def test_missing_group_protection():
    labels = np.tile([0, 1], 10)
    with pytest.raises(ValueError, match="6 distinct"):
        split_indices(labels, np.repeat([0, 1], 10))


def test_nonfinite_windows_are_rejected(tmp_path: Path):
    f = tmp_path / "nan.npz"
    x = np.zeros((30, 250, 8), dtype=np.float32)
    x[3, 5, 0] = np.nan
    np.savez(f, windows=x, labels=np.arange(30) % 2, sample_rate_hz=250)
    with pytest.raises(ValueError, match="nonfinite"):
        load_labeled_windows(f)


def test_wrong_sample_rate_rejected(tmp_path: Path):
    f = tmp_path / "wrong_rate.npz"
    np.savez(f, windows=np.zeros((30, 250, 8), np.float32),
             labels=np.arange(30) % 2, sample_rate_hz=200)
    with pytest.raises(ValueError, match="250 Hz"):
        load_labeled_windows(f)


def test_missing_rate_rejected(tmp_path: Path):
    f = tmp_path / "missing_rate.npz"
    np.savez(f, windows=np.zeros((30, 250, 8), np.float32), labels=np.arange(30) % 2)
    with pytest.raises(ValueError, match="sample_rate_hz"):
        load_labeled_windows(f)
