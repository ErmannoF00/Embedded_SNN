"""Adversarial deterministic raw-window parity suite for Python/C++ inference.

This is a NUMERICAL and input-contract test, not real EEG validation.
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch

from .features import extract_features, SAMPLES, CHANNELS
from .model import AdaptiveLIF


def stress_windows(seed: int = 19):
    rng = np.random.default_rng(seed)
    t = np.arange(SAMPLES, dtype=np.float32) / np.float32(250.0)
    out = [np.zeros((SAMPLES, CHANNELS), np.float32),
           np.ones((SAMPLES, CHANNELS), np.float32),
           -np.ones((SAMPLES, CHANNELS), np.float32),
           np.full((SAMPLES, CHANNELS), np.float32(1e-6)),
           np.full((SAMPLES, CHANNELS), np.float32(500.0))]
    for sign in (-1, 1):
        arr = np.zeros((SAMPLES, CHANNELS), dtype=np.float32)
        arr[0, 0] = 10.0 * sign
        arr[-1, -1] = 10.0 * sign
        out.append(arr)
    for hz in (0.5, 5, 10, 25, 60, 110):
        arr = np.broadcast_to(np.sin(2 * np.pi * hz * t)[:, None],
                              (SAMPLES, CHANNELS)).copy()
        out.append(arr.astype(np.float32))
    for magnitude in (1e-3, 1.0, 100.0):
        out.append(rng.normal(0, magnitude, (SAMPLES, CHANNELS)).astype(np.float32))
    out.append(np.where((np.arange(SAMPLES) % 2)[:, None], -1.0, 1.0)
               .repeat(CHANNELS, axis=1).astype(np.float32))
    out.append(np.broadcast_to(np.linspace(-1, 1, SAMPLES, dtype=np.float32)[:, None],
                               (SAMPLES, CHANNELS)).copy())
    return np.stack(out)


def check(binary: Path, checkpoint: Path, workdir: Path):
    torch.set_num_threads(1)
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = AdaptiveLIF().eval()
    model.load_state_dict(data["state_dict"])
    raw = stress_windows()
    features = extract_features(raw)
    normalized = (features - data["mean"].numpy()) / data["std"].numpy()
    with torch.no_grad():
        expected = model(torch.from_numpy(normalized.astype(np.float32)))[0].numpy()
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "adversarial_windows.csv"
    np.savetxt(path, raw.reshape(len(raw), -1), fmt="%.9g", delimiter=",")
    def call(mode):
        r = subprocess.run([str(binary.resolve()), mode, str(path.resolve())],
                           check=True, capture_output=True, text=True)
        rows = np.array([[float(x) for x in line.split(",")] for line in r.stdout.splitlines()])
        return rows[:, 1:]
    cpp_features = call("--extract")
    outputs = call("--raw")
    max_feature_abs = float(np.max(np.abs(cpp_features - features)))
    max_score_abs = float(np.max(np.abs(outputs[:, 1:] - expected)))
    disagreements = int(np.count_nonzero(outputs[:, 0] != expected.argmax(axis=1)))
    print(f"STRESS: windows={len(raw)}, feature_max_abs={max_feature_abs:.6g}, "
          f"score_max_abs={max_score_abs:.6g}, class_mismatches={disagreements}")
    if not np.isfinite(outputs).all():
        raise AssertionError("Nonfinite scores on finite stress inputs")
    if max_feature_abs > 5e-4 or max_score_abs > 5e-3 or disagreements:
        raise AssertionError("C++ disagrees with trained Python network")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, default=Path("build/snn_host"))
    ap.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoint.pt"))
    ap.add_argument("--workdir", type=Path, default=Path("build/stress"))
    args = ap.parse_args()
    check(args.binary, args.checkpoint, args.workdir)
