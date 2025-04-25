
# Embedded SNN for EEG Classification

An embedded-friendly Spiking Neural Network (SNN) pipeline for EEG-based data classification using BCI data from OpenBCI. This project uses **NEST simulator** for model simulation and a **C++ inference engine** for deployment on embedded hardware, such as Cortex M4.

## 📁 Project Structure
```
eeg-snn/
├── data/
│   ├── preprocessing/
│   │   └── eeg_process.py           # Preprocessing of EEG data (Filtering, feature extraction)
│   ├── data.py                      # Data loading and conversion for SNN input
│   └── OpenBCI_EEG/                 # Forked OpenBCI repository or submodule for EEG data
├── model/
│   ├── model.py                    # Neural network model (optional high-level model)
│   └── snn_model.py                # SNN model definition (for NEST simulation)
├── nest-simulator/                 # Forked NEST simulator for SNN simulation
├── SRC/
│   ├── snn_core.cpp                # C++ SNN engine core
│   ├── snn_core.h                  # Header file for C++ SNN engine
│   ├── main.cpp                    # Main file to run inference on embedded hardware
│   ├── simulate_input.cpp          # Simulate input for testing and generating EEG data
├── scripts/
│   ├── convert_weights.py          # Convert model weights to C++-friendly format
│   ├── setup.sh                    # Setup script for dependencies
│   └── simulate.py                 # Simulate and test the entire pipeline
├── CMakeLists.txt                  # Build configuration for C++ files
└── README.md
```

## 🧠 Train and Simulate the SNN Model (Python)

1. **Install Dependencies:**
   Ensure Python dependencies are installed for EEG processing and model simulation.
   ```bash
   pip install numpy scipy matplotlib brian2
   ```

2. **Train the SNN Model (NEST Simulator):**
   The model is designed in Python using NEST for spiking neural network simulation.
   ```bash
   python model/snn_model.py
   ```
   This script trains the SNN model using EEG data. Make sure that the OpenBCI data is preprocessed and loaded correctly.

3. **Convert Model Weights:**
   After training, convert the trained model weights into a C++-friendly format.
   ```bash
   python scripts/convert_weights.py
   ```

4. **Simulate EEG Input Data:**
   For testing purposes, you can simulate EEG input data using `simulate.py`. This script can generate fake data or use real OpenBCI data.
   ```bash
   python scripts/simulate.py
   ```

## ⚙️ Build and Run on Embedded Hardware (Cortex M4 or similar)

### 1. **Setup and Dependencies:**
   Install any necessary dependencies and build tools:
   ```bash
   ./scripts/setup.sh
   ```

### 2. **Build the C++ Inference Engine:**
   The C++ engine uses LIF (Leaky Integrate-and-Fire) neurons and runs inference based on the weights converted from the trained model.
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

### 3. **Run the Inference Engine:**
   Once the model is built, you can run the inference engine to classify gestures using EEG data:
   ```bash
   ./simulate_input         # Generates synthetic EEG input data
   ./main                   # Runs the SNN inference on generated input
   ```

   The output will print the spike response of each output neuron per timestep, which corresponds to the classified gesture.

## 📂 Notes and Considerations

- **NEST Simulator:** The NEST simulator is forked from the [original NEST repository](https://www.nest-simulator.org/). You may want to either include it as a submodule or ignore it using `.gitignore` depending on whether you need to make modifications or not.
- **EEG Data:** The `OpenBCI_EEG` folder is a forked repository or submodule. It provides the data required for EEG processing and classification. Ensure that the data is preprocessed properly before using it in the model.
- **Optimization:** The current C++ engine uses full dense weight matrices. Consider optimizing this by using sparse matrices or bitwise operations for deployment on embedded systems.

## 🖥️ Output

The inference engine will print the spike output of each output neuron for each timestep during EEG gesture classification.

## 📌 Additional Information

- **Preprocessing:** Make sure EEG data is preprocessed correctly in the `data/preprocessing/eeg_process.py` file. This step is crucial for preparing the data for the SNN model.
- **Model Complexity:** The complexity of the model can be adjusted in the `snn_core.h` file to better suit the target embedded hardware's capabilities (e.g., the Cortex M4).
- **Hardware Deployment:** The code is intended for deployment on embedded systems like the Cortex M4. The `main.cpp` file is where the inference engine runs on the embedded device.

---

**Made with ❤️ for embedded neuromorphic computing and BCI-based gesture recognition.**
