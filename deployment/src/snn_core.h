// ========================= snn_core.h =========================
#ifndef SNN_CORE_H
#define SNN_CORE_H

#include <vector>
#include <string>
#include <fstream>

namespace snn {

struct AdExNeuron {
    float v = 0.0f;
    float w = 0.0f;
    float I = 0.0f;
    int spike_count = 0;
    bool spiked = false;

    static constexpr float EL = 0.0f;
    static constexpr float VT = 1.0f;
    static constexpr float DeltaT = 0.5f;
    static constexpr float tau_m = 20.0f;
    static constexpr float tau_w = 100.0f;
    static constexpr float a = 0.01f;

    void update();
    void reset();
};

class SpikeRecorder {
public:
    void record(const std::vector<AdExNeuron>& layer);
    void save_to_csv(const std::string& filename) const;
private:
    std::vector<std::vector<int>> spike_log;
};

class SNN {
public:
    SNN(int input_size, int hidden_size, int output_size);

    void forward(const std::vector<float>& input, int timesteps);
    int classify() const;
    void reset();

    void load_weights(const std::string& path_ih, const std::string& path_ho);
    void dump_weights() const;
    void log_spikes(const std::string& filename) const;
    int get_output_spikes(int neuron_index) const;
    void initialize_weights(float scale = 0.5f);

private:
    int input_size;
    int hidden_size;
    int output_size;

    std::vector<float> input_layer;
    std::vector<AdExNeuron> hidden_layer;
    std::vector<AdExNeuron> output_layer;

    std::vector<std::vector<float>> weights_input_hidden;
    std::vector<std::vector<float>> weights_hidden_output;

    SpikeRecorder recorder;
};

} // namespace snn

#endif
