// ========================= snn_core.cpp =========================
#include "snn_core.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <cstdlib>

namespace snn {

void AdExNeuron::update() {
    float dv = (-(v - EL) + DeltaT * std::exp((v - VT) / DeltaT) - w + I) / tau_m;
    float dw = (a * (v - EL) - w) / tau_w;
    v += dv;
    w += dw;
    spiked = v > 1.0f;
    if (spiked) {
        v = -0.5f; // hyperpolarize to prevent immediate next spike
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

void SpikeRecorder::record(const std::vector<AdExNeuron>& layer) {
    std::vector<int> spikes;
    for (const auto& n : layer) spikes.push_back(n.spiked ? 1 : 0);
    spike_log.push_back(spikes);
}

void SpikeRecorder::save_to_csv(const std::string& filename) const {
    std::ofstream out(filename);
    for (const auto& row : spike_log) {
        for (size_t i = 0; i < row.size(); ++i) {
            out << row[i];
            if (i != row.size() - 1) out << ",";
        }
        out << "\n";
    }
}

SNN::SNN(int input_sz, int hidden_sz, int output_sz)
    : input_size(input_sz), hidden_size(hidden_sz), output_size(output_sz),
      input_layer(input_sz),
      hidden_layer(hidden_sz),
      output_layer(output_sz) {
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

void SNN::forward(const std::vector<float>& input, int timesteps = 20) {
    if (input.size() != static_cast<size_t>(input_size)) {
        std::cerr << "Input size mismatch! Expected: " << input_size << ", Got: " << input.size() << "\n";
        return;
    }

    for (int t = 0; t < timesteps; ++t) {
        // Reset currents each timestep
        for (auto& neuron : hidden_layer) neuron.I = 0.0f;
        for (auto& neuron : output_layer) neuron.I = 0.0f;

        // Feed input into hidden layer each timestep
        for (int i = 0; i < input_size; ++i)
            for (int j = 0; j < hidden_size; ++j)
                hidden_layer[j].I += input[i] * weights_input_hidden[i][j];

        for (auto& neuron : hidden_layer) neuron.update();

        for (int j = 0; j < hidden_size; ++j)
            if (hidden_layer[j].spiked)
                for (int k = 0; k < output_size; ++k)
                    output_layer[k].I += weights_hidden_output[j][k];

        for (auto& neuron : output_layer) neuron.update();

        recorder.record(output_layer);
    }
}


void SNN::reset() {
    for (auto& n : hidden_layer) n.reset();
    for (auto& n : output_layer) n.reset();
    recorder = SpikeRecorder();
}

int SNN::classify() const {
    int max_idx = -1;
    int max_spikes = -1;
    for (int i = 0; i < output_size; ++i) {
        if (output_layer[i].spike_count > max_spikes) {
            max_spikes = output_layer[i].spike_count;
            max_idx = i;
        }
    }
    return max_idx;
}


void SNN::load_weights(const std::string& path_ih, const std::string& path_ho) {
    std::ifstream file_ih(path_ih);
    std::ifstream file_ho(path_ho);
    if (!file_ih.is_open() || !file_ho.is_open()) {
        std::cerr << "Error loading weights.\n";
        return;
    }
    for (int i = 0; i < input_size; ++i)
        for (int j = 0; j < hidden_size; ++j)
            file_ih >> weights_input_hidden[i][j];

    for (int j = 0; j < hidden_size; ++j)
        for (int k = 0; k < output_size; ++k)
            file_ho >> weights_hidden_output[j][k];
}

void SNN::dump_weights() const {
    std::cout << "Input-Hidden Weights:\n";
    for (const auto& row : weights_input_hidden) {
        for (float w : row) std::cout << w << " ";
        std::cout << "\n";
    }
    std::cout << "\nHidden-Output Weights:\n";
    for (const auto& row : weights_hidden_output) {
        for (float w : row) std::cout << w << " ";
        std::cout << "\n";
    }
}

void SNN::log_spikes(const std::string& filename) const {
    recorder.save_to_csv(filename);
}

int SNN::get_output_spikes(int neuron_index) const {
    if (neuron_index >= 0 && neuron_index < static_cast<int>(output_layer.size())) {
        return output_layer[neuron_index].spike_count;
    } else {
        std::cerr << "Invalid neuron index: " << neuron_index << "\n";
        return -1;
    }
}

} // namespace snn
