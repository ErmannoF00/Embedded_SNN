// === simulate_input.cpp ===
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

int main() {
    std::ifstream eeg_file("eeg_encoded_input.txt");
    if (!eeg_file.is_open()) {
        std::cerr << "Could not open eeg_encoded_input.txt\n";
        return 1;
    }

    std::ofstream sim_file("simulate_input.txt");
    if (!sim_file.is_open()) {
        std::cerr << "Could not write to simulate_input.txt\n";
        return 1;
    }

    std::string line;
    if (std::getline(eeg_file, line)) {
        sim_file << line << std::endl;
        std::cout << "simulate_input.txt generated from EEG sample.\n";
    } else {
        std::cerr << "eeg_encoded_input.txt is empty or malformed.\n";
        return 1;
    }

    return 0;
}
