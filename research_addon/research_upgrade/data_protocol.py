"""Leakage and split checks intended for genuine, experiment-labeled EEG."""
from dataclasses import dataclass
import hashlib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def validate_dataset(windows, labels, groups, sample_rate_hz):
    x=np.asarray(windows);y=np.asarray(labels);g=np.asarray(groups)
    if x.ndim!=3 or x.shape[1:]!=(250,8) or not np.isfinite(x).all():
        raise ValueError('Expected (N,250,8) finite EEG windows')
    if x.dtype!=np.float32:
        raise ValueError('Expected float32 EEG windows')
    if y.shape!=(len(x),) or not np.isin(y,[0,1]).all():
        raise ValueError('Expected binary experimentally derived labels')
    if g.shape!=(len(x),) or len(np.unique(g))<6:
        raise ValueError('Need >=6 independent group identities')
    if float(sample_rate_hz)!=250.0:
        raise ValueError('Expected 250Hz, must resample with declared process')
    if len(x)<30 or len(np.unique(y))!=2:
        raise ValueError('Not enough labeled examples')
    return True


def split_groups(labels,groups,seed=41,max_tries=100) -> Split:
    y=np.asarray(labels);g=np.asarray(groups)
    if y.shape!=g.shape or len(np.unique(g))<6:
        raise ValueError('invalid labels/groups')
    X=np.zeros((len(y),1),dtype=np.float32)
    for attempt in range(max_tries):
        s=seed+attempt
        trval,te=next(GroupShuffleSplit(n_splits=1,test_size=0.15,random_state=s).split(X,y,g))
        tr_rel,va_rel=next(GroupShuffleSplit(n_splits=1,test_size=0.15/0.85,random_state=s).split(X[trval],y[trval],g[trval]))
        tr=trval[tr_rel];va=trval[va_rel]
        if all(len(np.unique(y[idx]))==2 for idx in (tr,va,te)):
            return Split(tr,va,te)
    raise ValueError('Cannot find group-disjoint, 2-class train/val/test splits')


def assert_split_integrity(windows, groups, split:Split):
    x=np.asarray(windows);g=np.asarray(groups)
    seen={}
    for phase,indices in [('train',split.train),('val',split.val),('test',split.test)]:
        for gi in np.unique(g[indices]):
            if gi in seen and seen[gi]!=phase:
                raise ValueError('Group identity leaks across phases')
            seen[gi]=phase
    hashes={}
    for phase,indices in [('train',split.train),('val',split.val),('test',split.test)]:
        for i in indices:
            digest=hashlib.sha256(np.ascontiguousarray(x[i]).tobytes()).hexdigest()
            if digest in hashes and hashes[digest]!=phase:
                raise ValueError('Exact duplicated window across splits')
            hashes[digest]=phase
    # Does not detect overlapping windows, near duplicates, shared session
    # metadata or temporal autocorrelation. Group correctly at SOURCE.
