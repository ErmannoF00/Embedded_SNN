// ========================= snn_core.cpp =========================
#include "snn_core.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <cstdlib>

void AdExNeuron::update() {
    float dv = (-(v - EL) + DeltaT * std::exp((v - VT) / DeltaT) - w + I) / tau_m;
    float dw = (a * (v - EL) - w) / tau_w;
    v += dv;
    w += dw;
    spiked = v > 1.0f;
    if (spiked) {
        v = 0.0f;
        w += 0.1f;
        spike_count++;
    }
}

void AdExNeuron::reset() {
    v = 0.0f;
    w = 0.0f;
    I = 0.0f;
    spiked = false;
    spike_count = 0;
}

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
    for (auto& neuron : hidden_layer) neuron.I = 0.0f;
    for (auto& neuron : output_layer) neuron.I = 0.0f;

    for (size_t i = 0; i < input_layer.size(); ++i)
        for (size_t j = 0; j < hidden_layer.size(); ++j)
            hidden_layer[j].I += input[i] * weights_input_hidden[i][j];

    for (auto& neuron : hidden_layer) neuron.update();

    for (size_t j = 0; j < hidden_layer.size(); ++j)
        if (hidden_layer[j].spiked)
            for (size_t k = 0; k < output_layer.size(); ++k)
                output_layer[k].I += weights_hidden_output[j][k];

    for (auto& neuron : output_layer) neuron.update();
}

void SNN::reset_network() {
    for (auto& n : hidden_layer) n.reset();
    for (auto& n : output_layer) n.reset();
}

int SNN::classify() const {
    int max_idx = -1;
    int max_spikes = -1;
    for (size_t i = 0; i < output_layer.size(); ++i) {
        if (output_layer[i].spike_count > max_spikes) {
            max_spikes = output_layer[i].spike_count;
            max_idx = static_cast<int>(i);
        }
    }
    return max_idx;
}

void SNN::print_output() const {
    std::cout << "Output spikes: ";
    for (const auto& neuron : output_layer)
        std::cout << neuron.spike_count << " ";
    std::cout << "\n";
}

void SNN::load_weights(const std::string& path_ih, const std::string& path_ho) {
    std::ifstream file_ih(path_ih);
    std::ifstream file_ho(path_ho);
    if (!file_ih.is_open() || !file_ho.is_open()) {
        std::cerr << "Error loading weights.\n";
        return;
    }
    for (auto& row : weights_input_hidden)
        for (auto& w : row)
            file_ih >> w;
    for (auto& row : weights_hidden_output)
        for (auto& w : row)
            file_ho >> w;
}
