#include "snn_runtime.hpp"

#include <cmath>
#include <iostream>

int main() {
    using namespace embedded_snn;
    RawWindow zero{};
    const auto features = extract_features(zero);
    for (const float value : features) {
        if (value != 0.0f) return 1;
    }
    const auto first = infer_window(zero);
    const auto second = infer_window(zero);
    for (int i = 0; i < 2; ++i) {
        if (!std::isfinite(first[i]) || first[i] != second[i]) return 2;
    }
    if (predict(first) != 0 && predict(first) != 1) return 3;
    std::cout << "PASS: raw feature extraction, finite scores, deterministic reset\n";
    return 0;
}
