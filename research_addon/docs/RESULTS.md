# Experimental results and interpretation

## Dataset and protocol

- Source: PhysioNet EEG Motor Movement/Imagery Database v1.0.0, runs 4, 8, 12 (imagined left/right fist).
- Acquisition: 64 electrodes at 160 Hz; eight channels selected (`FC3,C3,CP3,FC4,C4,CP4,Cz,Pz`) and offline-resampled to 250 Hz.
- Windows: 1 s with three 1 s windows extracted per annotated motor-imagery trial, starting 0.5, 1.5 and 2.5 s after onset. All three share one label.
- Evaluation: 20 subjects, 900 trials, 2,700 windows, five outer GroupKFold subject-held-out folds; training/validation subjects disjoint from test subjects. The training fold fits the feature scaler.
- `spectral32`: 1 s Hann-window Welch band log-power in 8–12, 12–16, 16–22 and 22–30 Hz, four bands per EEG channel.
- `amplitude32`: per-channel RMS, mean absolute amplitude, RMS adjacent-sample difference and zero-crossing fraction.

## WSL runs on 6 November 2026

The following numbers were copied from the user's terminal reports, not recomputed from EDF recordings here. They should be archived with the corresponding JSON and code commit for an independently reproducible publication.

| Pipeline | Accuracy | Balanced accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Original-feature SNN | 0.515556 | 0.514467 | 0.510289 |
| Spectral-feature SNN | 0.540370 | 0.540828 | 0.539801 |
| Spectral logistic | 0.542593 | 0.542664 | 0.542593 |
| Spectral MLP (48 hidden) | 0.522963 | 0.523176 | 0.522878 |
| Spectral diagnostic logistic (amplitude32) | 0.496296 | 0.495394 | 0.492484 |
| Spectral diagnostic logistic (spectral32) | 0.544444 | 0.544652 | 0.544372 |

The two logistic spectral results reflect different evaluation procedures; do not treat their small difference as meaningful.

### Stopping policies on spectral ALIF

| Policy | Accuracy | Mean ALIF steps | Fraction exiting before step 20 | Disagreements with full-T |
| --- | ---: | ---: | ---: | ---: |
| Full 20 steps | 0.540370 | 20.000000 | 0 | 0 |
| Conservative certificate | 0.540370 | 19.969259 | 0.030000 | 0 |
| Validation-calibrated margin | 0.540370 | 16.021111 | 0.297778 | 0 |

The 19.89% recurrent-step reduction for the margin policy excludes preprocessing, input projection, stopping-controller overhead and CPU/MCU execution time. Zero *observed* disagreements is not a formal bound on error risk.

### Numeric verification

The user's C++ comparison returned `status: PASS` on a trained model with float score tolerance `atol=5e-4`, `rtol=1e-5`. The full report's sample count and maximum discrepancy were not included in the pasted excerpt; they are not asserted here. This checks normalized-feature inference on the host, not MCU/raw-window parity.

## Limitations

Accuracy is low, only 20 of 109 potential subjects were evaluated, and three windows per trial are correlated. A publication should additionally report trial-level and per-subject metrics, cluster-aware uncertainty intervals, multiple training seeds, and matched CSP/covariance and filter-bank baselines. There are no electrical energy or embedded cycle measurements. A future experiment must preserve the dataset, split, software and training seeds to make comparison valid.
