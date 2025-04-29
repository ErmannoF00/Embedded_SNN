#include "snn_core.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;
using namespace snn;

// Load input from CSV with label as first column
std::vector<float> load_input(const std::string& filename, int& label, int input_size) {
    std::ifstream file(filename);
    std::string line;
    std::vector<float> input(input_size, 0.0f);

    if (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        getline(ss, value, ',');
        label = std::stoi(value);
        for (int i = 0; i < input_size && getline(ss, value, ','); ++i)
            input[i] = std::stof(value);
    }
    return input;
}

int main() {
    const int input_size = 784;
    const int hidden_size = 64;
    const int output_size = 2;

    SNN snn(input_size, hidden_size, output_size);
    snn.load_weights("../weights/weights_input_hidden.txt", "../weights/weights_hidden_output.txt");

    std::ofstream result_csv("classification_result.csv");
    result_csv << "true,predicted";
    for (int i = 0; i < output_size; ++i)
        result_csv << ",spike_" << i;
    result_csv << "\n";

    std::string input_dir = "../../data/";
    int sample_id = 0;

    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.path().extension() == ".txt") {
            int true_label = -1;
            int time = 20;
            std::vector<float> input = load_input(entry.path().string(), true_label, input_size);

            snn.reset();
            snn.forward(input, time);

            int predicted = snn.classify();

            result_csv << true_label << "," << predicted;
            for (int i = 0; i < output_size; ++i)
                result_csv << "," << snn.get_output_spikes(i);
            result_csv << "\n";

            snn.log_spikes("spikes_sample" + std::to_string(sample_id) + ".csv");
            ++sample_id;
        }
    }

    result_csv.close();
    std::cout << "Classification and spike logs completed.\n";
    return 0;
}
