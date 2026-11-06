#pragma once

#include <array>
#include <cstddef>

namespace embedded_snn {
inline constexpr int kChannels = 8;
inline constexpr int kSamples = 250;
inline constexpr int kFeatures = 32;

using RawWindow = std::array<float, kSamples * kChannels>; // sample-major
using FeatureVector = std::array<float, kFeatures>;
using Scores = std::array<float, 2>;

// No allocation. Floats and loop order deliberately match the Python model.
FeatureVector extract_features(const RawWindow& samples);
FeatureVector normalize(const FeatureVector& raw_features);
Scores infer_features(const FeatureVector& standardized_features);
Scores infer_window(const RawWindow& samples);
int predict(const Scores& scores);
} // namespace embedded_snn
