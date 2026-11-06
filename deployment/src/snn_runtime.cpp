#include "snn_runtime.hpp"
#include "model_weights.hpp"

#include <array>
#include <cmath>

namespace embedded_snn {
namespace w = model_weights;
static_assert(w::INPUTS == kFeatures && w::OUTPUTS == 2,
              "Exported model/input interface mismatch");

FeatureVector extract_features(const RawWindow& data) {
    FeatureVector out{};
    for (int ch = 0; ch < kChannels; ++ch) {
        float sum_sq = 0.0f, sum_abs = 0.0f, diff_sq = 0.0f;
        int crossings = 0;
        float prior = data[ch];
        for (int t = 0; t < kSamples; ++t) {
            const float value = data[t * kChannels + ch];
            sum_sq += value * value;
            sum_abs += std::fabs(value);
            if (t > 0) {
                const float difference = value - prior;
                diff_sq += difference * difference;
                crossings += ((value >= 0.0f) != (prior >= 0.0f));
            }
            prior = value;
        }
        out[4*ch]     = std::sqrt(sum_sq / static_cast<float>(kSamples));
        out[4*ch + 1] = sum_abs / static_cast<float>(kSamples);
        out[4*ch + 2] = std::sqrt(diff_sq / static_cast<float>(kSamples-1));
        out[4*ch + 3] = static_cast<float>(crossings) / static_cast<float>(kSamples-1);
    }
    return out;
}

FeatureVector normalize(const FeatureVector& features) {
    FeatureVector result{};
    for (int i = 0; i < kFeatures; ++i)
        result[i] = (features[i] - w::FEATURE_MEAN[i]) / w::FEATURE_STD[i];
    return result;
}

Scores infer_features(const FeatureVector& features) {
    std::array<float, w::HIDDEN> current{}, v{}, adapt{}, spikes{};
    Scores out{};
    for (int j = 0; j < w::HIDDEN; ++j) {
        float value = w::B1[j];
        for (int i = 0; i < kFeatures; ++i)
            value += features[i] * w::W1[j*kFeatures + i];
        current[j] = value;
    }
    for (int t = 0; t < w::TIMESTEPS; ++t) {
        for (int j = 0; j < w::HIDDEN; ++j) {
            v[j] = w::LEAK*v[j] + current[j] - w::ADAPT_GAIN*adapt[j];
            const float spike = v[j] >= w::THRESHOLD ? 1.0f : 0.0f;
            v[j] -= spike * w::THRESHOLD;
            adapt[j] = w::ADAPT_DECAY*adapt[j] + spike;
            spikes[j] = spike;
        }
        for (int k = 0; k < w::OUTPUTS; ++k) {
            float current_out = w::B2[k];
            for (int j = 0; j < w::HIDDEN; ++j)
                current_out += spikes[j] * w::W2[k*w::HIDDEN + j];
            out[k] = w::READOUT_LEAK*out[k] + current_out;
        }
    }
    return out;
}

Scores infer_window(const RawWindow& samples) {
    return infer_features(normalize(extract_features(samples)));
}

int predict(const Scores& scores) {
    return scores[1] > scores[0] ? 1 : 0;
}
} // namespace embedded_snn
