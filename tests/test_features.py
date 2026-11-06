import numpy as np
import pytest

from snn_pipeline.features import extract_features
from snn_pipeline.dataset import synthetic_demo


def test_constant_windows():
    sample = np.ones((250, 8), dtype=np.float32)
    got = extract_features(sample).reshape(8, 4)
    np.testing.assert_allclose(got, np.tile([1, 1, 0, 0], (8, 1)), atol=1e-6)


def test_sign_crossing():
    sample = np.ones((250, 8), dtype=np.float32)
    sample[1::2] = -1
    np.testing.assert_allclose(extract_features(sample)[0, 3::4], 1)


def test_bad_shapes_rejected():
    with pytest.raises(ValueError):
        extract_features(np.zeros((250, 7)))


def test_demo_has_ground_truth_and_correct_shapes():
    x, y, groups = synthetic_demo(10)
    assert x.shape == (20, 250, 8)
    assert np.bincount(y).tolist() == [10, 10]
    assert groups is None
