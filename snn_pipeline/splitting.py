"""Immutable, explicitly group-disjoint train/validation/test partitioning.

Window-wise splitting is only valid for independently sampled examples.
For EEG recordings with overlapping windows, groups must identify the
subject/session/recording that cannot be present in multiple partitions.
"""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def split_indices(labels: np.ndarray, groups: np.ndarray | None, seed: int = 41):
    labels = np.asarray(labels)
    index = np.arange(labels.size)
    if labels.ndim != 1 or labels.size < 20 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Need at least 20 properly labeled binary examples")
    if groups is None:
        # Approximate 70/15/15 split; both holdout partitions remain untouched.
        tr, pending = train_test_split(index, test_size=0.30, stratify=labels, random_state=seed)
        va, te = train_test_split(pending, test_size=0.50, stratify=labels[pending], random_state=seed + 1)
    else:
        groups = np.asarray(groups)
        if groups.shape != labels.shape or np.unique(groups).size < 6:
            raise ValueError("Group validation needs aligned IDs and >=6 distinct groups")
        first = GroupShuffleSplit(test_size=0.30, n_splits=1, random_state=seed)
        tr, pending = next(first.split(index, labels, groups))
        second = GroupShuffleSplit(test_size=0.50, n_splits=1, random_state=seed + 1)
        va_rel, te_rel = next(second.split(pending, labels[pending], groups[pending]))
        va, te = pending[va_rel], pending[te_rel]
        group_sets = [set(groups[v].tolist()) for v in (tr, va, te)]
        if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise AssertionError("Group leakage in split")
    partitions = (np.asarray(tr), np.asarray(va), np.asarray(te))
    if len(set(np.concatenate(partitions))) != len(index):
        raise AssertionError("Missing/duplicated indices in split")
    if any(len(p) == 0 or len(np.unique(labels[p])) != 2 for p in partitions):
        raise ValueError("Every partition must contain both classes; collect more diverse subjects/sessions")
    return partitions
