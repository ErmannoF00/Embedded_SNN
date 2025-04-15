# Embedded SNN

An embedded-friendly Spiking Neural Network (SNN) pipeline for gesture recognition using simulated Dynamic Vision Sensor (DVS) inputs. Trained using Brian2 in Python, deployed as a custom C++ inference engine.

## 📁 Project Structure
```
gesture-snn/
├── data/
│   └── dvs_gestures.bin        # (optional real/simulated data)
├── model/
│   └── snn_model.brian2.py     # Brian2 SNN training
├── cpp_inference/
│   ├── snn_core.cpp/.h         # C++ SNN engine
│   ├── main.cpp                # Inference on Raspberry Pi
│   └── weights_input_hidden.txt / weights_hidden_output.txt
├── deployment/
│   └── simulate_input.cpp      # Random input generation
├── scripts/
│   └── convert_weights.py      # Convert .npy weights → .txt
├── CMakeLists.txt              # Build config
└── README.md
```

## 🧠 Train the SNN (Python)

1. Install dependencies:
   ```bash
   pip install brian2 numpy
   ```
2. Run the model:
   ```bash
   python model/snn_model.brian2.py
   ```
3. Convert weights to C++-friendly format:
   ```bash
   python scripts/convert_weights.py
   ```

## 🧪 Simulate Gesture Input
```bash
cd deployment
./simulate_input
```
Outputs `sample_input.txt` for inference.

## ⚙️ Build and Run (Raspberry Pi or Linux)
```bash
mkdir build && cd build
cmake ..
make
./simulate_input      # Generates fake DVS input
./snn_main            # Runs SNN inference on input
```

## 🖥️ Output
Console prints spike output of each output neuron per timestep.

## 📌 Notes
- C++ engine uses LIF neurons (reset after spike)
- Full dense weight matrices used for now; optimization possible with sparse matrices or bitwise ops
- Model complexity can be tuned via `snn_core.h`

---

Made with ❤️ for embedded neuromorphic computing.

