import numpy as np
import pytest
from research_upgrade.feature_protocol import spectral_features,compute_features,contract


def test_spectral_signature_peak_and_channel_order():
    fs=250
    t=np.arange(fs)/fs
    x=np.zeros((2,250,8),dtype=np.float32)
    x[0,:,0]=np.sin(2*np.pi*10*t)
    x[1,:,0]=np.sin(2*np.pi*18*t)
    z=spectral_features(x)
    assert z.shape==(2,32)
    assert z[0,0]>z[0,2]
    assert z[1,2]>z[1,0]
    assert np.isfinite(z).all()
    assert contract('spectral32')['cxx_raw_window_extractor_supported'] is False
    assert compute_features(x,'amplitude32').shape==(2,32)


def test_spectral_invalid():
    with pytest.raises(ValueError):
        spectral_features(np.zeros((1,200,8)))
    with pytest.raises(ValueError):
        compute_features(np.zeros((1,250,8)), 'magic')
