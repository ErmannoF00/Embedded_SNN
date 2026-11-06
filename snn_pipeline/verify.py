"""Verify C++ vs Python predictions on the held-out, never-trained windows."""
import argparse
import subprocess
from pathlib import Path

from .features import extract_features

import numpy as np



def parity(binary: Path, artifact_dir: Path, work_dir: Path):
    with np.load(artifact_dir / "holdout.npz") as data:
        raw = data["raw_windows"]
        normalized = data["normalized_features"]
        scores_python = data["python_scores"]
        labels = data["labels"]

    work_dir.mkdir(parents=True, exist_ok=True)
    def invoke(mode, inputs, name):
        input_file = work_dir / name
        np.savetxt(input_file, inputs.reshape(len(inputs), -1), delimiter=",", fmt="%.9g")
        result = subprocess.run([str(binary.resolve()), mode, str(input_file)],
                                capture_output=True, text=True, check=True)
        output = np.array([[float(item) for item in row.split(",")]
                           for row in result.stdout.strip().splitlines()])
        return output[:, 1].astype(int), output[:, 2:]

    pred_features, scores_features = invoke("--features", normalized, "features.csv")
    pred_raw, scores_raw = invoke("--raw", raw, "raw_windows.csv")
    # Compare C++ on-device extraction directly, not only downstream decisions.
    raw_file = work_dir / "raw_windows.csv"
    result = subprocess.run([str(binary.resolve()), "--extract", str(raw_file)],
                            capture_output=True, text=True, check=True)
    cpp_features = np.array([[float(item) for item in row.split(",")[1:]]
                             for row in result.stdout.strip().splitlines()])
    feature_err = np.max(np.abs(extract_features(raw) - cpp_features))
    expected = scores_python.argmax(axis=1)
    err_features = np.max(np.abs(scores_python - scores_features))
    err_raw = np.max(np.abs(scores_python - scores_raw))
    print(f"Holdout samples: {len(labels)}")
    print(f"Python/C++ raw feature extraction max error: {feature_err:.8g}")
    print(f"Python/C++ normalized-feature max score error: {err_features:.8g}")
    print(f"Python/C++ raw-window max score error:         {err_raw:.8g}")
    print(f"C++ vs Python classification mismatches: features={(pred_features != expected).sum()}, "
          f"raw={(pred_raw != expected).sum()}")
    print(f"Python held-out accuracy: {(expected == labels).mean():.3f}")
    assert feature_err < 3e-5, "Python/C++ feature extraction differs"
    assert err_features < 3e-4, "Python/C++ feature-input scores differ"
    assert err_raw < 3e-4, "Python/C++ raw-window scores differ"
    assert np.array_equal(pred_features, expected)
    assert np.array_equal(pred_raw, expected)
    print("PASS: C++ runtime matches Python model on held-out windows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/snn_host"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--work-dir", type=Path, default=Path("build/parity"))
    args = parser.parse_args()
    parity(args.binary, args.artifacts, args.work_dir)
