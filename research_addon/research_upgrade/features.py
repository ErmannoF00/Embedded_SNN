"""Frozen 250x8 -> 32 feature contract from Embedded_SNN handoff."""
import numpy as np


def extract_features(windows: np.ndarray) -> np.ndarray:
    x=np.asarray(windows,dtype=np.float32)
    squeeze=False
    if x.shape == (250,8):
        x=x[None,:,:]; squeeze=True
    if x.ndim!=3 or x.shape[1:]!=(250,8) or not np.isfinite(x).all():
        raise ValueError('windows must be finite shape (N,250,8)')
    # Float64 accumulation for stable offline evaluation; parity with the
    # actual C++ reduction ORDER is a separate required deployment test.
    a=x.astype(np.float64)
    rms=np.sqrt(np.mean(a*a,axis=1))
    mean_abs=np.mean(np.abs(a),axis=1)
    delta=np.diff(a,axis=1)
    diff_rms=np.sqrt(np.mean(delta*delta,axis=1))
    crossing=np.mean((a[:,1:,:]>=0)!=(a[:,:-1,:]>=0),axis=1)
    out=np.stack((rms,mean_abs,diff_rms,crossing),axis=-1).reshape(-1,32).astype(np.float32)
    return out[0] if squeeze else out
