// ========================= snn_core.h =========================
#ifndef SNN_CORE_H
#define SNN_CORE_H

#include <vector>
#include <string>

struct AdExNeuron {
    float v = 0.0f;
    float w = 0.0f;
    float EL = 0.0f;
    float VT = 1.0f;
    float DeltaT = 0.5f;
    float tau_m = 20.0f;
    float tau_w = 100.0f;
    float a = 0.01f;
    float I = 0.0f;
    bool spiked = false;
    int spike_count = 0;

    void update();
    void reset();
};

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
    std::vector<float> input_layer;
    std::vector<AdExNeuron> hidden_layer;
    std::vector<AdExNeuron> output_layer;
    std::vector<std::vector<float>> weights_input_hidden;
    std::vector<std::vector<float>> weights_hidden_output;
};

#endif
