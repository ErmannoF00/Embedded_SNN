#include <iostream>
#include <fstream>
#include <vector>
#include <string>

int main() {
    // Open the eeg_encoded_input.txt file for reading
    std::ifstream eeg_file("data/eeg_encoded_input.txt");  // Assuming it's in the data folder
    if (!eeg_file.is_open()) {
        std::cerr << "Could not open eeg_encoded_input.txt\n";
        return 1;
    }

    // Open the simulate_input.txt file for writing
    std::ofstream sim_file("data/simulate_input.txt");  // Saving output to the 'data' folder
    if (!sim_file.is_open()) {
        std::cerr << "Could not write to simulate_input.txt\n";
        return 1;
    }

    // Read the eeg_encoded_input.txt line by line and write it to simulate_input.txt
    std::string line;
    while (std::getline(eeg_file, line)) {
        sim_file << line << std::endl;  // Write each line from EEG data to simulate_input.txt
    }

    std::cout << "simulate_input.txt has been generated from eeg_encoded_input.txt.\n";

    return 0;
}
