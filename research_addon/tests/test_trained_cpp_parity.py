import numpy as np
import shutil
import pytest
from research_upgrade.verify_trained_cpp import verify


def test_cpp_parity_with_saved_model_file(tmp_path):
    if not shutil.which('g++'):
        pytest.skip('Needs C++ compiler')
    rng=np.random.default_rng(73)
    folder=tmp_path/'fold_00';folder.mkdir()
    np.savez_compressed(folder/'weights.npz',
        w1=rng.normal(0,.08,(48,32)).astype(np.float32),
        b1=rng.normal(0,.08,48).astype(np.float32),
        w2=rng.normal(0,.08,(2,48)).astype(np.float32),
        b2=rng.normal(0,.08,2).astype(np.float32))
    np.savez_compressed(folder/'test.npz',
        z=rng.normal(0,1,(18,32)).astype(np.float32),
        labels=rng.integers(0,2,18))
    outcome=verify(folder,tmp_path/'report.json',limit=18)
    assert outcome['status']=='PASS'
    assert outcome['class_mismatches']==0
