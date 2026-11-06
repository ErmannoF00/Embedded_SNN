"""Preflight real EEG data: schema, sample rate, class balance and duplicates."""
import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .dataset import load_labeled_windows
from .splitting import split_indices


def validate(path: Path):
    x, labels, groups = load_labeled_windows(path)
    if groups is None:
        raise ValueError("For real EEG, provide subjects/recording IDs as 'groups' "
                         "to prevent overlapping-window leakage")
    partitions = split_indices(labels, groups)
    print(f"windows={len(x)}, shape={x.shape[1:]}, sample_rate_hz=250")
    print(f"classes={dict(Counter(labels.tolist()))}, groups={len(np.unique(groups))}")
    # Exact-duplicate windows MUST NOT cross dataset partitions.
    hashes = defaultdict(set)
    for split_id, indices in enumerate(partitions):
        print(f"split_{split_id}: windows={len(indices)}, groups={len(np.unique(groups[indices]))}, "
              f"class_counts={dict(Counter(labels[indices].tolist()))}")
        for ix in indices:
            fingerprint = hashlib.sha256(np.ascontiguousarray(x[ix]).tobytes()).digest()
            hashes[fingerprint].add(split_id)
    overlap = sum(len(where) > 1 for where in hashes.values())
    if overlap:
        raise ValueError(f"{overlap} exact duplicate windows span train/val/test partitions")
    print("PASS: schema, group-disjoint split and duplicate-window leakage check")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path)
    validate(ap.parse_args().dataset)
