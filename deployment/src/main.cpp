// ========================= main.cpp =========================
#include "snn_core.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// Load input vector from file and extract ground-truth label
std::vector<float> load_input(const std::string& filename, int& label) {
    std::ifstream file(filename);
    std::string line;
    std::vector<float> input(784, 0.0f);  // Modify for EEG vector size if needed

    if (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        getline(ss, value, ',');
        label = stoi(value);

        for (int i = 0; i < 784 && getline(ss, value, ','); ++i)
            input[i] = stof(value);
    }
    return input;
}

int main() {
    int input_size = 784;     // Adapt to EEG segment size
    int hidden_size = 64;
    int output_size = 2;      // Binary classifier (e.g., meditative vs not)

    SNN snn(input_size, hidden_size, output_size);
    snn.load_weights("weights/weights_input_hidden.txt", "weights/weights_hidden_output.txt");

    int true_label = -1;
    std::vector<float> input = load_input("data/simulate_input.txt", true_label);

    for (int t = 0; t < 20; ++t)  // Simulate for 20 time steps
        snn.forward(input);

    int predicted = snn.classify();
    snn.print_output();

    std::cout << "True label: " << true_label << "\n";
    std::cout << "Predicted label: " << predicted << "\n";

    return 0;
}
