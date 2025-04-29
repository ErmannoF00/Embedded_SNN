# Embedded SNN for EEG classification

An embedded-friendly Spiking Neural Network (SNN) pipeline for EEG-based data classification using OpenBCI signals. The project uses **Brian2** for SNN training and a **lightweight C++ engine** for deployment on embedded hardware (e.g., Cortex-M4).

## 📁 Project Structure
```
Embedded_SNN/
├── data/
│   ├── OpenBCI_EEG/               # EEG dataset folder (recorded from OpenBCI)
│   └── eeg_process.py             # Python script to preprocess EEG data
├── scripts/
│   ├── train.py                   # Trains the AdEx SNN model using Brian2
│   └── export_weights.py         # Converts NumPy weights to .txt format for C++
├── deployment/
│   ├── src/
│   │   ├── main.cpp               # Inference entry point
│   │   ├── snn_core.cpp          # Core logic for AdEx neurons
│   │   └── snn_core.h            # Header file for the SNN engine
│   ├── weights/                  # Text-formatted weight files
│   └── CMakeLists.txt            # CMake build configuration
├── run_all.py                    # Runs preprocessing, training, and export
└── README.md
```

## 🧠 Train and Export the SNN Model (Python)

1. **Install Dependencies:**
   ```bash
   pip install numpy scipy pandas brian2
   ```

2. **Preprocess EEG Data:**
   ```bash
   python data/eeg_process.py
   ```

3. **Train AdEx SNN Model:**
   ```bash
   python scripts/train.py
   ```

4. **Export Weights for C++ Inference:**
   ```bash
   python scripts/export_weights.py
   ```

## ⚙️ Build and Run Inference (C++)

### 1. **Build with CMake (MinGW or MSVC):**
```bash
cd deployment
cmake -S . -B build
cmake --build build
```

### 2. **Run Executable:**
```bash
cd build
./snn_infer.exe
```

## 🖥️ Output

The C++ engine will print:
- True label
- Output spike count per neuron
- Predicted label (argmax)

## 📌 Notes
- **All Python logic** (training/preprocessing/export) is separated from the C++ deployment.
- You can retrain using real EEG or synthetic OpenBCI data.
