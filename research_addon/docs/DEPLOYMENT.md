# Embedded deployment boundary

Current C++ code is a host reference and replay framing parser, not flashable MCU firmware. There is no verified ADC/EEG transport, DMA, vector table, linker script, interrupt-driven scheduler, hardware clock, flash/RAM report or electrical power measurement in this release.

Suggested demonstrator: Cortex-M4F-class development board with single-precision FPU, DWT cycle counter and a host-fed EEG replay interface. An eight-channel 250-Hz acquisition front end can be integrated later, but data reference, unit scaling and electrode configuration must match the trained model.

**Acceptance gates:**

1. Freeze checkpoint hash, scaler and feature contract; reject incompatible input channel mapping or non-finite data.
2. Implement the matching spectral32 host/MCU extractor before deploying spectral weights from raw EEG; current C++ raw-window extractor supports amplitude32 only.
3. Implement board-specific startup, buffers and packet receive with CRC and recovery, and verify score parity on the device.
4. Report flash and RAM from the linker map, cycle count and p95 timing from the MCU, and energy per inference from a calibrated measurement setup.
5. Measure the total sensor-endpoint-to-decision path and compare it against the 1-second data acquisition interval.

The wire parser and host C++ ALIF runtime are reusable components, not evidence of any of these measurements.
