import itertools
import numpy as np
import pytest
from research_upgrade.certified_exit import Config,Parameters,infer,final_margin_interval,evaluate_exit


def params(seed=123, bias=None):
    g=np.random.default_rng(seed)
    return Parameters(g.normal(0,.25,(48,32)).astype('f4'),
                      g.normal(0,.1,48).astype('f4'),
                      g.normal(0,.3,(2,48)).astype('f4'),
                      np.array([0.,0.] if bias is None else bias,dtype='f4'))


def test_full_vs_certified_many_random_inputs():
    g=np.random.default_rng(48)
    for seed in range(8):
        p=params(seed)
        z=g.normal(0,1,(35,32)).astype('f4')
        r=evaluate_exit(z,p,mode='certified',numerical_guard=1e-4)
        assert r['disagreements']==0
        assert (r['steps']>=2).all() and (r['steps']<=20).all()


def test_certified_exits_in_easy_case():
    p=params(24,bias=[-6.,6.])
    z=np.zeros(32,'f4')
    a=infer(z,p,mode='full')
    b=infer(z,p,mode='certified',min_steps=1)
    assert a.pred==b.pred==1 and b.certified
    assert 1<=b.steps_used<=20


def test_interval_encloses_all_future_binary_patterns():
    g=np.random.default_rng(12)
    dw=g.normal(0,0.3,4)
    m=.7;bias=-.12;h=3;r=.9
    lo,hi=final_margin_interval(m,h,dw,bias,r)
    for raw in itertools.product((0.,1.),repeat=4*h):
        v=m
        for t in range(h):
            v=r*v+bias+np.dot(dw,raw[t*4:(t+1)*4])
        assert lo-1e-12<=v<=hi+1e-12


def test_fixed_and_tie_break():
    p=Parameters(np.zeros((48,32),'f4'),np.zeros(48,'f4'),
                 np.zeros((2,48),'f4'),np.zeros(2,'f4'))
    a=infer(np.zeros(32),p,mode='fixed',fixed_steps=5)
    b=infer(np.zeros(32),p,mode='full')
    assert a.steps_used==5 and a.pred==b.pred==0
    with pytest.raises(ValueError):
        infer(np.zeros(32),p,mode='fixed',fixed_steps=21)


def test_reject_nan_and_parameter_layout():
    p=params()
    x=np.zeros(32,'f4');x[3]=np.nan
    with pytest.raises(ValueError): infer(x,p)
    with pytest.raises(ValueError): Parameters(np.ones((12,31)),p.b1,p.w2,p.b2)


def test_certification_can_differ_from_current_margin_sign():
    # A positive future guaranteed readout can reverse early readout.
    # The certificate must emit FINAL sign rather than current argmax.
    dw=np.ones(4,dtype=np.float64);lo,hi=final_margin_interval(-1.,1,dw,5.,.9)
    assert lo>0 and hi>0


def test_reject_arithmetic_overflow():
    p=Parameters(np.full((48,32),1e38,dtype='f4'),np.zeros(48,'f4'),
                 np.zeros((2,48),'f4'),np.zeros(2,'f4'))
    with pytest.raises(FloatingPointError):
        infer(np.ones(32,'f4'),p)
