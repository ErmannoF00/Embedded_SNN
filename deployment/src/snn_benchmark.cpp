#include "snn_runtime.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <vector>

// HOST-ONLY benchmark. These wall-clock numbers are not Cortex-M timings.
int main() {
    using namespace embedded_snn;
    RawWindow window{};
    for (std::size_t i = 0; i < window.size(); ++i)
        window[i] = 0.15f * std::sin(static_cast<float>(i) * 0.17f);
    constexpr int warmup = 100;
    constexpr int repetitions = 1000;
    volatile float sink = 0.0f;
    for (int i = 0; i < warmup; ++i) sink = sink + infer_window(window)[0];
    std::vector<double> micros;
    micros.reserve(repetitions);
    for (int i = 0; i < repetitions; ++i) {
        const auto before = std::chrono::steady_clock::now();
        const auto result = infer_window(window);
        const auto after = std::chrono::steady_clock::now();
        sink = sink + result[0];
        micros.push_back(std::chrono::duration<double, std::micro>(after - before).count());
    }
    std::sort(micros.begin(), micros.end());
    const double avg = std::accumulate(micros.begin(), micros.end(), 0.0) / repetitions;
    std::cout << "host-only inference; n=" << repetitions
              << " avg_us=" << avg
              << " p50_us=" << micros[repetitions / 2]
              << " p95_us=" << micros[static_cast<int>(0.95 * repetitions)]
              << " min_us=" << micros.front()
              << " max_us=" << micros.back()
              << " input_bytes=" << sizeof(RawWindow)
              << " sink=" << sink << '\n';
}
