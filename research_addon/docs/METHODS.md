# Methods and numerical contracts

## Input and preprocessing

Each window is 250 samples × 8 channels, sample-major, `float32`, in microvolts. Channel order: `FC3,C3,CP3,FC4,C4,CP4,Cz,Pz`. The original PhysioNet files are 160 Hz; resampling to 250 Hz is performed offline. The `spectral32` host feature calculation uses a 250-point Hann-window Welch periodogram and four log-power bands per channel. The `amplitude32` features are channel-local RMS, mean absolute value, RMS consecutive difference and sign-crossing fraction.

Feature scaling uses the mean and standard deviation computed **only on the training subjects** in each fold. The scaler and model parameters are exported together.

## ALIF recurrence

For each 32-feature vector `z`, precompute `I = W1 z + b1`, then reset all hidden states and readout scores to zero. For steps `t=1..20`:

```
v_candidate = 0.85*v + I - 0.20*a
spikes      = (v_candidate >= 1.0)
v           = v_candidate - spikes
a           = 0.90*a + spikes
scores      = 0.90*scores + W2*spikes + b2
```

Model dimensions are `W1: 48×32`, `b1: 48`, `W2: 2×48`, `b2: 2`. Surrogate gradients are used in training; inference uses hard binary spikes. The final predicted label is the `argmax` of the two output scores.

## Class-preserving stopping rule

Let `d_t = scores[t,1] - scores[t,0]`, `H=20-t`, `r=0.9`, and `Δw=W2[1,:]-W2[0,:]`, `Δb=b2[1]-b2[0]`. Define

```
G_H = sum(r**k for k in 0..H-1)
L   = Δb + sum(min(0, Δw_j))
U   = Δb + sum(max(0, Δw_j))
```

All possible *binary* future spike streams satisfy, in exact arithmetic,

```
r**H*d_t + G_H*L <= d_20 <= r**H*d_t + G_H*U.
```

If the lower bound is positive, the full-20-step class is 1; if the upper bound is negative, it is 0. The implementation adds a configurable numerical guard, but that is not an interval-arithmetic proof for IEEE-754 rounding. A calibrated margin stopping policy can disagree with the final model and has no certificate.

## Software verification

Tests cover fold integrity, feature semantics, annotation-to-window conversion, packet CRC, supervised training smoke tests, Python/C++ inference parity and stopping-bound cases. `cpp/research_runtime.hpp` is portable host C++17 code, not a board support package. The host numeric regression accepts pre-normalized 32-element features. The raw-window C++ extraction path implements amplitude32 only.
