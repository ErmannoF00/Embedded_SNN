import numpy as np
from research_upgrade.evaluate_baselines import run


def test_baseline_smoke(tmp_path):
    rng=np.random.default_rng(19)
    n=120
    y=np.tile([0,1],60).astype('i8')
    x=rng.normal(0,1,(n,250,8)).astype('f4')
    x[y==1,:,:]+=0.4
    g=np.repeat(np.arange(12),10).astype('U4')
    f=tmp_path/'synthetic.npz'
    np.savez_compressed(f,windows=x,labels=y,groups=g,sample_rate_hz=np.array(250.))
    r=run(f,tmp_path/'metrics.json')
    assert set(r['models'])=={'logistic','mlp_48'}
    assert (tmp_path/'metrics.json').exists()
    # Synthetic data gives no information about EEG classification quality.
