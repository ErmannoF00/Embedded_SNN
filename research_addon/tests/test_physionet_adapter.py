import numpy as np
import pytest
from research_upgrade.physionet_mi import extract_from_raw,CHANNELS


def test_annotations_to_250hz_real_labels():
    mne=pytest.importorskip('mne')
    sfreq=250
    t=np.arange(8*sfreq)/sfreq
    channels=list(CHANNELS)
    signal=np.vstack([np.sin(2*np.pi*(8+i)*t)*1e-6 for i in range(8)])
    raw=mne.io.RawArray(signal,mne.create_info(channels,sfreq,ch_types='eeg'),verbose='ERROR')
    raw.set_annotations(mne.Annotations(onset=[0.,4.],duration=[4.,4.],description=['T1','T2']))
    x,y,g,trial=extract_from_raw(raw,17,run=4)
    assert x.shape==(6,250,8)
    assert y.tolist()==[0,0,0,1,1,1]
    assert set(g)=={'17'}
    assert np.max(np.abs(x)) <=1.0001
    assert len(set(trial))==2


def test_trial_identifiers_do_not_collide_across_runs():
    mne=pytest.importorskip('mne')
    fs=250
    channels=list(CHANNELS)
    raw=mne.io.RawArray(np.zeros((8,8*fs)),mne.create_info(channels,fs,ch_types='eeg'),verbose='ERROR')
    raw.set_annotations(mne.Annotations(onset=[0.,4.],duration=[4.,4.],description=['T1','T2']))
    *_, t1=extract_from_raw(raw,17,run=4)
    *_, t2=extract_from_raw(raw,17,run=8)
    assert set(t1).isdisjoint(set(t2))
    assert all(':R04:' in s for s in t1)
    assert all(':R08:' in s for s in t2)
    with pytest.raises(ValueError):
        extract_from_raw(raw,17,run=3)
