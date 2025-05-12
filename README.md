# Embedded SNN for EEG Classification

A lightweight spiking neural network (SNN) engine tailored for **EEG classification** on **embedded devices** (e.g., Cortex-M4).
The training is done in Python using **Brian2**, and the final model is deployed using a minimal **C++ runtime**.

---

## 📁 Project Structure

```
Embedded_SNN/
├── data/
│   ├── OpenBCI_EEG/               # Raw EEG dataset
│   └── eeg_process.py             # Preprocessing script
├── scripts/
│   ├── train.py                   # Brian2 training of AdEx SNN
│   └── export_weights.py         # NumPy-to-text weight export
├── deployment/
│   ├── src/
│   │   ├── main.cpp               # C++ entrypoint (optional for PC test)
│   │   ├── embedded_main.cpp      # Embedded target main
│   │   ├── snn_core.cpp/.h        # Minimal AdEx SNN C++ engine
│   │   └── startup_stm32f4xx.s    # Assembly startup for Cortex-M4
│   ├── weights/                  # Text-formatted weight files
│   ├── linker.ld                 # Custom linker script for embedded
│   └── CMakeLists.txt            # Build config for CMake
├── run_all.py                    # Run preprocessing + training + export
└── README.md                     # You're reading it!
```

---

## Python: Preprocess, Train, and Export

1. **Install Dependencies**

   ```bash
   pip install numpy scipy pandas brian2
   ```

2. **Preprocess EEG**

   ```bash
   python data/eeg_process.py
   ```

3. **Train the SNN**

   ```bash
   python scripts/train.py
   ```

4. **Export weights to text format**

   ```bash
   python scripts/export_weights.py
   ```

---

## C++: Embedded Inference

### Build for Cortex-M4 with CMake

```bash
cd deployment
cmake -S . -B build
cmake --build build
```

> ⚠️ Requires `arm-none-eabi-g++`, `newlib`, and `cmake`.

### Run on QEMU

```bash
cd build
qemu-system-arm -M olimex-stm32-h405 -cpu cortex-m4 -nographic -semihosting -kernel snn_embedded.elf
```

---

## Output

You will see semihosted debug prints such as:

```
Booting embedded SNN...
Neuron 0: spikes = 3
Neuron 1: spikes = 6
Prediction = 1
```

---

## 📌 Notes

* All dynamic memory is removed from the embedded C++ version.
* Weights are loaded statically into arrays (`float[]`) to fit embedded constraints.
* STL, file I/O, and exceptions are stripped for bare-metal compatibility.

---

## 📜 License

This project is released under the MIT License.
