"""ALIF reference inference with a conservative final-class stopping bound."""
from dataclasses import dataclass
from typing import Literal
import numpy as np


@dataclass(frozen=True)
class Config:
    steps: int = 20
    leak: float = 0.85
    adapt_decay: float = 0.90
    adapt_gain: float = 0.20
    threshold: float = 1.0
    readout_leak: float = 0.90

    def __post_init__(self):
        if self.steps < 1 or not 0 <= self.readout_leak <= 1:
            raise ValueError('steps>=1 and 0<=readout_leak<=1 required')
        if not 0 <= self.leak <= 1 or not 0 <= self.adapt_decay <= 1:
            raise ValueError('state leakage factors must lie in [0,1]')


@dataclass(frozen=True)
class Parameters:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    def __post_init__(self):
        w1 = np.asarray(self.w1, dtype=np.float32)
        b1 = np.asarray(self.b1, dtype=np.float32)
        w2 = np.asarray(self.w2, dtype=np.float32)
        b2 = np.asarray(self.b2, dtype=np.float32)
        if w1.ndim != 2 or w1.shape[1] != 32 or w1.shape[0] < 1:
            raise ValueError('w1 must have shape (hidden,32)')
        h = w1.shape[0]
        if b1.shape != (h,) or w2.shape != (2,h) or b2.shape != (2,):
            raise ValueError('expected b1=(hidden,), w2=(2,hidden), b2=(2,)')
        if not all(np.isfinite(x).all() for x in (w1,b1,w2,b2)):
            raise ValueError('parameters must be finite')
        for key,val in (('w1',w1),('b1',b1),('w2',w2),('b2',b2)):
            object.__setattr__(self,key,val)

    @classmethod
    def from_npz(cls, path: str):
        with np.load(path, allow_pickle=False) as npz:
            return cls(*(npz[k] for k in ('w1','b1','w2','b2')))


@dataclass(frozen=True)
class Result:
    scores: np.ndarray
    pred: int
    steps_used: int
    spike_count: int
    certified: bool
    certificate_interval: tuple[float,float] | None


def final_margin_interval(margin: float, remaining: int,
                          delta_w: np.ndarray, delta_bias: float,
                          readout_leak: float) -> tuple[float,float]:
    """Exact-real-arithmetic enclosure over ALL future binary spike patterns.

    Uses float64 for analysis; this is NOT a verified float32 rounding bound.
    """
    if remaining < 0 or not 0 <= readout_leak <= 1:
        raise ValueError('bad horizon or leak')
    dw = np.asarray(delta_w, dtype=np.float64)
    if dw.ndim != 1 or not np.isfinite(dw).all() or not np.isfinite(margin):
        raise ValueError('invalid margin/weight vector')
    if not np.isfinite(delta_bias):
        raise ValueError('invalid bias')
    r = float(readout_leak)
    decay = r ** remaining
    geom = float(remaining) if r == 1.0 else (1.0-decay)/(1.0-r)
    lower_step = float(delta_bias)+float(np.minimum(dw,0).sum(dtype=np.float64))
    upper_step = float(delta_bias)+float(np.maximum(dw,0).sum(dtype=np.float64))
    center = decay * float(margin)
    return center+geom*lower_step, center+geom*upper_step


def infer(z: np.ndarray, p: Parameters, config: Config = Config(),
          mode: Literal['full','certified','margin','fixed'] = 'full',
          margin_threshold: float = 1.0, min_steps: int = 2,
          fixed_steps: int | None = None,
          numerical_guard: float = 1e-4) -> Result:
    """Run standalone reference. Input z must already be normalized 32 features.

    `certified`: can return a class consistent with full-T in exact arithmetic.
    `margin`: heuristic, NOT class-preserving, calibrate only on validation.
    `fixed`: ablation; requires fixed_steps.

    The returned `scores` are EARLY-T scores for exits and cannot be compared
    to the full-T score vector as if numerically equivalent.
    """
    if mode not in ('full','certified','margin','fixed'):
        raise ValueError('bad mode')
    z = np.asarray(z, dtype=np.float32)
    if z.shape != (32,) or not np.isfinite(z).all():
        raise ValueError('z must be 32 finite features')
    if min_steps < 1 or numerical_guard < 0 or margin_threshold < 0:
        raise ValueError('invalid exit parameters')
    if mode == 'fixed' and (fixed_steps is None or not 1 <= fixed_steps <= config.steps):
        raise ValueError('fixed_steps must lie in [1,T]')
    h = p.w1.shape[0]
    membrane = np.zeros(h,np.float32)
    adaptation = np.zeros(h,np.float32)
    scores = np.zeros(2,np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        current = (p.w1 @ z + p.b1).astype(np.float32)
    delta_w = (p.w2[1].astype(np.float64) - p.w2[0].astype(np.float64))
    delta_b = float(np.float64(p.b2[1])-np.float64(p.b2[0]))
    n_spikes = 0
    steps = config.steps if mode != 'fixed' else fixed_steps
    interval = None
    if not np.isfinite(current).all():
        raise FloatingPointError('Input projection overflowed')
    for t in range(1,steps+1):
        candidate = (np.float32(config.leak)*membrane + current -
                     np.float32(config.adapt_gain)*adaptation).astype(np.float32)
        if not np.isfinite(candidate).all():
            raise FloatingPointError('Membrane state overflowed')
        spikes = (candidate >= np.float32(config.threshold))
        membrane = (candidate-spikes.astype(np.float32)).astype(np.float32)
        adaptation = (np.float32(config.adapt_decay)*adaptation+
                      spikes.astype(np.float32)).astype(np.float32)
        scores = (np.float32(config.readout_leak)*scores + p.b2 +
                  p.w2 @ spikes.astype(np.float32)).astype(np.float32)
        if not np.isfinite(scores).all() or not np.isfinite(adaptation).all():
            raise FloatingPointError('Readout/adaptation overflowed')
        n_spikes += int(spikes.sum())
        if t < config.steps and t >= min_steps:
            margin = float(np.float64(scores[1])-np.float64(scores[0]))
            if mode == 'certified':
                interval = final_margin_interval(margin,config.steps-t,
                                                  delta_w,delta_b,config.readout_leak)
                if interval[0] > numerical_guard or interval[1] < -numerical_guard:
                    return Result(scores.copy(),1 if interval[0]>numerical_guard else 0,t,n_spikes,True,interval)
            elif mode == 'margin' and abs(margin) >= margin_threshold:
                return Result(scores.copy(),int(scores[1]>scores[0]),t,n_spikes,False,None)
    return Result(scores.copy(),int(scores[1]>scores[0]),steps,n_spikes,
                  mode=='certified' and steps==config.steps,None)


def evaluate_exit(z_matrix, p, config=Config(), mode='certified', **kwargs):
    z_matrix = np.asarray(z_matrix,dtype=np.float32)
    if z_matrix.ndim != 2 or z_matrix.shape[1] != 32:
        raise ValueError('expected (N,32) normalized features')
    base = [infer(z,p,config,'full') for z in z_matrix]
    policy = [infer(z,p,config,mode=mode,**kwargs) for z in z_matrix]
    return {
        'n': len(base),
        'full_predictions': np.asarray([x.pred for x in base],dtype=np.int64),
        'policy_predictions': np.asarray([x.pred for x in policy],dtype=np.int64),
        'full_scores': np.asarray([x.scores for x in base],dtype=np.float32),
        'policy_scores': np.asarray([x.scores for x in policy],dtype=np.float32),
        'steps': np.asarray([x.steps_used for x in policy],dtype=np.int64),
        'spikes': np.asarray([x.spike_count for x in policy],dtype=np.int64),
        'disagreements': sum(x.pred!=y.pred for x,y in zip(base,policy)),
        'certified_exits': sum(x.certified and x.steps_used<config.steps for x in policy),
    }
