#ifndef SNN_CORE_H
#define SNN_CORE_H

#include <vector>
#include <string>

// ---------------- Neuron ----------------

struct Neuron {
    float membrane_potential = 0.0f;
    float threshold = 1.0f;
    float reset_value = 0.0f;
    float decay = 0.9f;
    bool spiked = false;
    int spike_count = 0;

    void update(float input_current);
    void reset();
};

// ---------------- Spiking Neural Network ----------------

class SNN {
public:
    SNN(int input_size, int hidden_size, int output_size);

    void initialize_weights(float scale = 1.0f);
    void forward(const std::vector<float>& input);
    void reset_network();
    int classify() const;
    void print_output() const;
    void load_weights(const std::string& path_ih, const std::string& path_ho);

private:
    std::vector<Neuron> input_layer;
    std::vector<Neuron> hidden_layer;
    std::vector<Neuron> output_layer;

    std::vector<std::vector<float>> weights_input_hidden;
    std::vector<std::vector<float>> weights_hidden_output;
};

#endif // SNN_CORE_H
