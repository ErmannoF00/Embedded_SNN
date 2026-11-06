# Source audit and correction log

## Issues found in original upload

1. Dataset parser assigned synthetic class `label = idx % 2` to OpenBCI windows.
   There was no valid EEG target variable.
2. Brian2 script used `Synapses.w_syn = rand()` but never implemented any
   supervised learning, STDP or optimizer update. 'Training' simply ran input
   windows without altering the exported weights.
3. `G_input` had threshold `False`, so its `on_pre` synapse events were never
   triggered by the observed sample values.
4. Python experiment expected 2,000 input features and 128 hidden cells;
   C++ entry points instantiated 784 inputs and 64 hidden cells.
5. The Brian2 synapses were sparse and exported as one-dimensional arrays;
   C++ loaded them as dense matrices, with no sparse index metadata or shape
   validation. Failed loads silently left random weights in place.
6. The C++ AdEx reset was `v=-0.5`, while Python reset was `v=0`.
7. C++ inference used `std::vector`, fstreams and filesystem, despite README
   claims of static bare-metal storage and no C++ standard library.
8. The CMake configuration changed compiler variables after `project()` and
   mixed semihosting with a startup file missing runtime initialization.
9. Original included ~27 MB of data with invalid alternating labels; excluded
   from cleaned distributable, while original upload remains untouched.

## Rebuild decisions

- Use a narrow, documented adaptive-LIF model whose recurrence is identical in
  Python and C++, instead of claiming unsupported AdEx simulation equivalence.
- Train with actual class labels using cross-entropy and surrogate gradients.
- Correctly enforce shape and finite-value preconditions.
- Split into train/validation/final test **before** fitting feature normalization.
- Enforce group-disjoint splits when subject/session/recording identities are supplied; reject missing sampling-rate metadata and cross-partition exact duplicates during dataset preflight.
- Fit 32 simple time-domain features so the board-side feature extractor is
  auditable, bounded, and independent of an FFT runtime.
- Export all weights, biases, feature statistics and numerical parameters.
- Validate predictions/score differences on previously unseen windows and 18 adversarial windows in
  both Python and compiled C++ using a continuous integration-style command.

## Remaining research/firmware gates

- Use an ethically obtained, correctly labeled EEG dataset and specify the task.
- Subject-level cross validation, class imbalance, trial timing and leakage
  analysis, calibration and meaningful baselines.
- Confirm sampling units, DC removal, anti-alias filtering and artifact handling.
- Select a specific MCU; implement BSP, ADC/DMA, clock, interrupt and UART/SWO.
- Measure cycle counts, RAM/flash use, real-time deadline, energy and robustness.
- Prefer int8 or fixed-point quantization only after measuring accuracy loss;
  no quantization/precision claim is made for the current float32 implementation.
