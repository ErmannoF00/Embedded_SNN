// src/embedded_main.cpp
#include "snn_core.h"
#include <cstdio>

extern "C" void initialise_monitor_handles(void);

extern "C" int main() {
    initialise_monitor_handles();  // Required for QEMU semihosting printf
    printf("Booting embedded SNN...\n");

    snn::SNN snn(784, 64, 2);
    snn.load_weights("weights/weights_input_hidden.txt", "weights/weights_hidden_output.txt");

    float input[784];
    for (int i = 0; i < 784; ++i) input[i] = 0.5f;

    snn.forward(input, 20);
    int prediction = snn.classify();

    printf("Prediction = %d\n", prediction);

    while (1);
}
