# MCU integration boundary

The `snn_runtime` static library needs only C++17, `float` and `<cmath>`.
It has no dynamic allocation, dynamic objects, file I/O, exceptions or OS calls.
The model header is generated from a trained checkpoint and compiled into flash.
Use `embedded_snn::infer_window()` when your board acquires 250 x 8 float
samples at 250 Hz, **after** appropriate analog preprocessing and ADC unit
conversion. If the samples originate elsewhere, communicate explicitly how
sensor channels, calibration, time bases and normalization match training data.

**No board is selected in this repository.** To flash a Cortex-M4 board, provide
a vendor board support package, its startup/vector table, RAM/flash linker
script, clock/ADC drivers, C++ runtime initialization and transport. Do not
reuse the old generic STM32 startup script: it did not initialize `.data` or
`.bss`, set up clocks or use a correct board-specific vector table.

A cross-compiled library does not by itself establish executable firmware,
real-time deadlines, memory footprint or sensor validity. Those require
an identified part number, toolchain and measured hardware validation.
