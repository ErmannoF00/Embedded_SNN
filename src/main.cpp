#include "snn_core.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

vector<float> parse_input(const string& filename, int& label) {
    ifstream file(filename);
    vector<float> input(784, 0.0f);
    string line;

    if (!file.is_open()) {
        cerr << "Could not open " << filename << endl;
        return input;
    }

    if (getline(file, line)) {
        stringstream ss(line);
        string value;
        getline(ss, value, ',');
        label = stoi(value); // first value is label

        for (int i = 0; i < 784 && getline(ss, value, ','); ++i) {
            input[i] = stof(value) / 255.0f; // normalize pixel
        }
    }

    return input;
}

int main() {
    int input_size = 784;
    int hidden_size = 100;
    int output_size = 10;
    SNN snn(input_size, hidden_size, output_size);

    // Optional: load pretrained weights if you have them
    // snn.load_weights("weights_input_hidden.txt", "weights_hidden_output.txt");

    int true_label = -1;
    vector<float> input = parse_input("simulate_input.txt", true_label);

    for (int t = 0; t < 20; ++t) {
        snn.forward(input);
    }

    int predicted = snn.classify();
    snn.print_output();

    cout << "True label: " << true_label << "\n";
    cout << "Predicted label: " << predicted << "\n";

    return 0;
}
