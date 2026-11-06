# EEG motor-imagery SNN research pipeline

Research extension for [Embedded_SNN](https://github.com/ErmannoF00/Embedded_SNN). Implements subject-held-out evaluation of a supervised adaptive leaky integrate-and-fire (ALIF) network, non-spiking baselines, and host-side C++ numerical inference.

**Status:** experimental software. No MCU firmware, physical latency/energy measurement, or clinically reliable EEG decoding is demonstrated.

## Architecture

```
PhysioNet EDF (64 channels, 160 Hz)
    -> 8 EEG channels -> resample to 250 Hz -> 250 x 8 windows
    -> 32 amplitude or spectral features -> train-only scaler
    -> 32 -> 48 ALIF (20 steps) -> 2-class membrane readout
    -> Python metrics / host C++ parity / early-exit experiments
```

For motor imagery, runs 4, 8 and 12 map `T1` to left fist and `T2` to right fist. Windows from the same trial are correlated. Evaluation separates subjects, not individual windows.

The C++ raw-window feature extractor supports **amplitude32 only**. `spectral32` uses Welch log-bandpower features on the host; the C++ model accepts already-normalized spectral features but cannot extract them from raw EEG.

## Setup

From the repository's `research_addon/` folder, using the existing project `.venv`:

```bash
source ../.venv/bin/activate
python -m pip install -r requirements-train.txt
make check
```

`make check` executes software tests and builds the host C++ protocol parser. These checks do not flash a microcontroller.

## Reproduce experiments

```bash
python -m research_upgrade.physionet_mi \
  --subjects $(seq 1 20) --output data/physionet_mi_v03.npz

python -m research_upgrade.spectral_diagnostic \
  --data data/physionet_mi_v03.npz \
  --output results/spectral_diagnostic.json

python -m research_upgrade.evaluate_cv \
  --data data/physionet_mi_v03.npz --features spectral32 \
  --folds 5 --output results/spectral_baselines_cv.json

python -m research_upgrade.train_snn_cv \
  --data data/physionet_mi_v03.npz --features spectral32 \
  --folds 5 --epochs 30 --artifacts artifacts/snn_spectral \
  --output results/snn_spectral_cv.json

python -m research_upgrade.run_fold_exits \
  --artifacts artifacts/snn_spectral \
  --output results/spectral_exit_cv.json

python -m research_upgrade.verify_trained_cpp \
  --fold artifacts/snn_spectral/fold_00 \
  --output results/spectral_cpp_parity.json
```

Each fold exports its own weights, scaler, validation data, test data and feature contract. The host C++ test consumes **normalized 32-dimensional features**; it is not a raw EEG sensor-to-prediction test.

## Reported 20-subject results

The results below were run on the user's WSL installation (6 November 2026), not reproduced from source EDFs inside this archive. They use 2,700 windows from 900 trials and 20 subjects, five held-out-subject folds.

| Method | Window accuracy | Balanced accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Original-feature ALIF | 51.56% | 51.45% | 51.03% |
| Spectral-feature ALIF | 54.04% | 54.08% | 53.98% |
| Spectral logistic regression | 54.26% | 54.27% | 54.26% |
| Spectral 48-unit MLP | 52.30% | 52.32% | 52.29% |

Spectral-feature ALIF performs close to chance and does not outperform the logistic baseline. Its validation-calibrated margin policy reduced the mean recurrent steps from 20.00 to 16.02 (19.89%) with zero observed class disagreements against the full-T model in these 2,700 windows. This is **not** an accuracy, latency or energy gain. The conservative certificate reduced mean steps by only 0.15%.

See [`docs/RESULTS.md`](docs/RESULTS.md) for data provenance and interpretation.

## Code map

| Location | Purpose |
| --- | --- |
| `research_upgrade/physionet_mi.py` | Label-aware EDF conversion |
| `research_upgrade/feature_protocol.py` | Frozen amplitude/spectral feature semantics |
| `research_upgrade/train_snn_cv.py` | Supervised ALIF model and fold-wise training |
| `research_upgrade/evaluate_cv.py` | Logistic/MLP cross-subject baselines |
| `research_upgrade/certified_exit.py` | ALIF recurrence and stopping certificate |
| `research_upgrade/run_fold_exits.py` | Fold-wise early-exit comparison |
| `research_upgrade/verify_trained_cpp.py` | Trained-weight C++ numerical regression |
| `cpp/` | Static-buffer host inference and wire parser |
| `tests/` | Unit, protocol and numerical regression tests |

## Research boundary

The model uses 32 window-derived static features, not a continuously encoded EEG spike stream. It is not proven to save energy on hardware. A certified early exit preserves the reference's final class under an exact-arithmetic envelope, not true EEG labels or IEEE-754 bit-equivalence. The heuristic margin policy has no formal correctness guarantee.

Primary next work: EEG artifact filtering and trial-level evaluation, CSP/covariance baselines, temporal event encoding, matching C++ spectral preprocessing, on-board cycle/RAM/flash/energy measurements.

Dataset: [PhysioNet EEG Motor Movement/Imagery Database](https://physionet.org/content/eegmmidb/1.0.0/). The project does not redistribute EDF recordings or trained model artifacts.
