// File: deployment/simulate_input.cpp
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

void generate_sample_input(const std::string& path, int input_size = 10) {
    std::ofstream file(path);
    std::srand(static_cast<unsigned>(std::time(0)));

    for (int i = 0; i < input_size; ++i) {
        float val = static_cast<float>(std::rand()) / RAND_MAX;  // Random between 0 and 1
        file << val << " ";
    }

    file.close();
    std::cout << "Sample input written to " << path << std::endl;
}

int main() {
    generate_sample_input("deployment/sample_input.txt");
    return 0;
}
