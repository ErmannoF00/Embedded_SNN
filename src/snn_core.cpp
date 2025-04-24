#include "snn_core.h"
#include <iostream>
#include <fstream>
#include <limits>
#include <cstdlib>  // for rand()

using namespace std;

// ---------------- Neuron ----------------

void Neuron::update(float input_current) {
    membrane_potential = decay * membrane_potential + input_current;
    spiked = membrane_potential >= threshold;
    if (spiked) {
        membrane_potential = reset_value;
        spike_count++;
    }
}

void Neuron::reset() {
    membrane_potential = 0.0f;
    spiked = false;
    spike_count = 0;
}

// ---------------- SNN ----------------

SNN::SNN(int input_size, int hidden_size, int output_size)
    : input_layer(input_size),
      hidden_layer(hidden_size),
      output_layer(output_size) {
    weights_input_hidden.resize(input_size, std::vector<float>(hidden_size));
    weights_hidden_output.resize(hidden_size, std::vector<float>(output_size));
    initialize_weights();
}

void SNN::initialize_weights(float scale) {
    for (auto& row : weights_input_hidden)
        for (auto& w : row)
            w = static_cast<float>(rand()) / RAND_MAX * scale;

    for (auto& row : weights_hidden_output)
        for (auto& w : row)
            w = static_cast<float>(rand()) / RAND_MAX * scale;
}

void SNN::forward(const std::vector<float>& input) {
    // Input layer update
    for (size_t i = 0; i < input_layer.size(); ++i) {
        input_layer[i].update(input[i]);
    }

    // Hidden layer input accumulation
    std::vector<float> hidden_input(hidden_layer.size(), 0.0f);
    for (size_t i = 0; i < input_layer.size(); ++i) {
        if (input_layer[i].spiked) {
            for (size_t j = 0; j < hidden_layer.size(); ++j) {
                hidden_input[j] += weights_input_hidden[i][j];
            }
        }
    }

    // Hidden layer update
    for (size_t j = 0; j < hidden_layer.size(); ++j) {
        hidden_layer[j].update(hidden_input[j]);
    }

    // Output layer input accumulation
    std::vector<float> output_input(output_layer.size(), 0.0f);
    for (size_t j = 0; j < hidden_layer.size(); ++j) {
        if (hidden_layer[j].spiked) {
            for (size_t k = 0; k < output_layer.size(); ++k) {
                output_input[k] += weights_hidden_output[j][k];
            }
        }
    }

    // Output layer update
    for (size_t k = 0; k < output_layer.size(); ++k) {
        output_layer[k].update(output_input[k]);
    }
}

void SNN::reset_network() {
    for (auto& n : input_layer) n.reset();
    for (auto& n : hidden_layer) n.reset();
    for (auto& n : output_layer) n.reset();
}

int SNN::classify() const {
    int max_index = -1;
    int max_spikes = std::numeric_limits<int>::min();

    for (size_t i = 0; i < output_layer.size(); ++i) {
        if (output_layer[i].spike_count > max_spikes) {
            max_spikes = output_layer[i].spike_count;
            max_index = static_cast<int>(i);
        }
    }

    return max_index;
}

void SNN::print_output() const {
    std::cout << "Output spikes: ";
    for (const auto& neuron : output_layer) {
        std::cout << neuron.spike_count << " ";
    }
    std::cout << std::endl;
}

void SNN::load_weights(const std::string& path_ih, const std::string& path_ho) {
    std::ifstream file_ih(path_ih);
    std::ifstream file_ho(path_ho);

    if (!file_ih.is_open() || !file_ho.is_open()) {
        std::cerr << "Error: Could not open weight files.\n";
        return;
    }

    for (auto& row : weights_input_hidden)
        for (auto& w : row)
            file_ih >> w;

    for (auto& row : weights_hidden_output)
        for (auto& w : row)
            file_ho >> w;
}
