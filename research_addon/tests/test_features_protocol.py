import numpy as np
import pytest
from research_upgrade.features import extract_features
from research_upgrade.data_protocol import validate_dataset,split_groups,assert_split_integrity,Split


def test_features_semantics():
    x=np.zeros((250,8),dtype='f4')
    x[:,0]=np.where(np.arange(250)%2==0,-2.,2.)
    f=extract_features(x)
    assert f.shape==(32,)
    np.testing.assert_allclose(f[:4],[2,2,4,1],rtol=1e-7,atol=1e-7)
    assert (f[4:]==0).all()


def test_groups_disjoint():
    rng=np.random.default_rng(8)
    x=rng.normal(0,1,(120,250,8)).astype('f4')
    y=np.tile([0,1],60);g=np.repeat(np.arange(12),10)
    assert validate_dataset(x,y,g,250.)
    s=split_groups(y,g)
    assert_split_integrity(x,g,s)
    a,b,c=(set(g[getattr(s,k)]) for k in ('train','val','test'))
    assert not a&b and not a&c and not b&c


def test_exact_duplicate_across_subjects_rejected():
    rng=np.random.default_rng(8)
    x=rng.normal(0,1,(12,250,8)).astype('f4')
    x[2]=x[0]
    with pytest.raises(ValueError,match='duplicated'):
        assert_split_integrity(x,np.arange(12),Split(np.array([0]),np.array([2]),np.array([4])))


def test_invalid_groups_and_rate():
    x=np.zeros((100,250,8),'f4');y=np.tile([0,1],50)
    with pytest.raises(ValueError): validate_dataset(x,y,np.zeros(100),250.)
    with pytest.raises(ValueError): validate_dataset(x,y,np.arange(100),160.)
