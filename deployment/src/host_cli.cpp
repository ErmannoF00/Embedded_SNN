#include "snn_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace embedded_snn;

int main(int argc, char** argv) {
    if (argc != 3 || (std::string(argv[1]) != "--features" &&
                      std::string(argv[1]) != "--raw" &&
                      std::string(argv[1]) != "--extract")) {
        std::cerr << "Usage: snn_host (--features|--raw|--extract) csv_path\n";
        return 2;
    }
    const bool extract = std::string(argv[1]) == "--extract";
    const bool raw = std::string(argv[1]) == "--raw" || extract;
    const std::size_t expected = raw ? kSamples*kChannels : kFeatures;
    std::ifstream input(argv[2]);
    if (!input) {
        std::cerr << "Cannot open " << argv[2] << '\n';
        return 2;
    }
    std::cout << std::setprecision(9);
    std::string line;
    std::size_t row = 0;
    try {
        while (std::getline(input, line)) {
            if (line.empty()) continue;
            std::vector<float> values;
            std::stringstream fields(line);
            std::string token;
            while (std::getline(fields, token, ',')) {
                std::size_t consumed = 0;
                const float value = std::stof(token, &consumed);
                if (consumed != token.size() || !std::isfinite(value))
                    throw std::runtime_error("Invalid floating-point input");
                values.push_back(value);
            }
            if (values.size() != expected)
                throw std::runtime_error("Wrong number of columns in CSV");
            Scores scores;
            if (raw) {
                RawWindow samples{};
                std::copy(values.begin(), values.end(), samples.begin());
                if (extract) {
                    const FeatureVector extracted = extract_features(samples);
                    std::cout << row++;
                    for (float value : extracted) std::cout << ',' << value;
                    std::cout << '\n';
                    continue;
                }
                scores = infer_window(samples);
            } else {
                FeatureVector features{};
                std::copy(values.begin(), values.end(), features.begin());
                scores = infer_features(features);
            }
            std::cout << row++ << ',' << predict(scores)
                      << ',' << scores[0] << ',' << scores[1] << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << "Invalid row " << row << ": " << e.what() << '\n';
        return 2;
    }
    return 0;
}
