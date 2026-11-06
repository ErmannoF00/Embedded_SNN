"""32-feature extraction contracts for host-side EEG experiments."""
import numpy as np
from scipy.signal import welch
from .features import extract_features

BANDS_HZ=((8.0,12.0),(12.0,16.0),(16.0,22.0),(22.0,30.0))


def spectral_features(windows, sample_rate_hz=250.0):
    x=np.asarray(windows,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(250,8) or not np.isfinite(x).all():
        raise ValueError('Expected finite EEG windows shaped (N,250,8)')
    if float(sample_rate_hz)!=250.0:
        raise ValueError('spectral32 requires 250 Hz input')
    f, psd=welch(x,fs=250.0,nperseg=250,noverlap=0,axis=1,
                 window='hann',detrend='constant',scaling='density')
    bands=[]
    for lo,hi in BANDS_HZ:
        band=(f>=lo)&(f<hi)
        power=psd[:,band,:].sum(axis=1)*(f[1]-f[0])
        bands.append(np.log10(np.maximum(power,1e-12)))
    result=np.stack(bands,axis=-1).reshape(len(x),32).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite spectral features')
    return result


def compute_features(windows, mode='amplitude32'):
    if mode=='amplitude32':
        return extract_features(windows)
    if mode=='spectral32':
        return spectral_features(windows)
    raise ValueError('mode must be amplitude32 or spectral32')


def contract(mode):
    if mode not in ('amplitude32','spectral32'):
        raise ValueError('Unknown feature contract')
    return {'name':mode, 'input_shape':[250,8], 'feature_shape':[32],
            'sample_rate_hz':250.0, 'channel_order':'FC3,C3,CP3,FC4,C4,CP4,Cz,Pz',
            'unit':'microvolt',
            'cxx_raw_window_extractor_supported': mode=='amplitude32',
            'explanation': ('Four channel-local time-domain moments' if mode=='amplitude32'
                            else 'Welch Hann log10 band powers 8-12,12-16,16-22,22-30 Hz; host only')}
