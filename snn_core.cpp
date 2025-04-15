// File: cpp_inference/snn_core.h
#ifndef SNN_CORE_H
#define SNN_CORE_H

#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

struct Neuron {
    float membrane_potential;
    float threshold;
    float reset_value;
    float decay;
    bool spiked;

    Neuron(float threshold = 1.0f, float reset_value = 0.0f, float decay = 0.95f)
        : membrane_potential(0.0f), threshold(threshold), reset_value(reset_value), decay(decay), spiked(false) {}

    void update(float input_current) {
        membrane_potential = decay * membrane_potential + input_current;
        spiked = membrane_potential >= threshold;
        if (spiked) {
            membrane_potential = reset_value;
        }
    }
};

class SNN {
public:
    std::vector<Neuron> input_layer;
    std::vector<Neuron> hidden_layer;
    std::vector<Neuron> output_layer;
    
    std::vector<std::vector<float>> weights_input_hidden;
    std::vector<std::vector<float>> weights_hidden_output;

    SNN(int input_size, int hidden_size, int output_size)
        : input_layer(input_size), hidden_layer(hidden_size), output_layer(output_size) {

        weights_input_hidden.resize(input_size, std::vector<float>(hidden_size));
        weights_hidden_output.resize(hidden_layer.size(), std::vector<float>(output_layer.size()));

        for (auto& row : weights_input_hidden)
            for (auto& w : row)
                w = static_cast<float>(rand()) / RAND_MAX * 0.5f;

        for (auto& row : weights_hidden_output)
            for (auto& w : row)
                w = static_cast<float>(rand()) / RAND_MAX * 0.5f;
    }

    void forward(const std::vector<float>& input) {
        for (size_t i = 0; i < input_layer.size(); ++i) {
            input_layer[i].update(input[i]);
        }

        std::vector<float> hidden_input(hidden_layer.size(), 0.0f);
        for (size_t i = 0; i < input_layer.size(); ++i) {
            if (input_layer[i].spiked) {
                for (size_t j = 0; j < hidden_layer.size(); ++j) {
                    hidden_input[j] += weights_input_hidden[i][j];
                }
            }
        }

        for (size_t j = 0; j < hidden_layer.size(); ++j) {
            hidden_layer[j].update(hidden_input[j]);
        }

        std::vector<float> output_input(output_layer.size(), 0.0f);
        for (size_t j = 0; j < hidden_layer.size(); ++j) {
            if (hidden_layer[j].spiked) {
                for (size_t k = 0; k < output_layer.size(); ++k) {
                    output_input[k] += weights_hidden_output[j][k];
                }
            }
        }

        for (size_t k = 0; k < output_layer.size(); ++k) {
            output_layer[k].update(output_input[k]);
        }
    }

    void print_output() {
        std::cout << "Output neuron spikes: ";
        for (const auto& neuron : output_layer) {
            std::cout << neuron.spiked << " ";
        }
        std::cout << std::endl;
    }

    void load_weights(const std::string& path_ih, const std::string& path_ho) {
        std::ifstream file_ih(path_ih);
        std::ifstream file_ho(path_ho);
        if (!file_ih.is_open() || !file_ho.is_open()) {
            std::cerr << "Failed to open weight files." << std::endl;
            return;
        }

        for (size_t i = 0; i < weights_input_hidden.size(); ++i) {
            for (size_t j = 0; j < weights_input_hidden[i].size(); ++j) {
                file_ih >> weights_input_hidden[i][j];
            }
        }

        for (size_t i = 0; i < weights_hidden_output.size(); ++i) {
            for (size_t j = 0; j < weights_hidden_output[i].size(); ++j) {
                file_ho >> weights_hidden_output[i][j];
            }
        }

        file_ih.close();
        file_ho.close();
    }
};

#endif // SNN_CORE_H
