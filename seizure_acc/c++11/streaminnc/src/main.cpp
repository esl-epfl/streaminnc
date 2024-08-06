#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>

// #include "conv_model.h"
#include "conv_model_buffered.h"

#include "test_data_multiple_samples.h"

void printOutput(std::vector<double> output){
    std::cout << "Output: ";
    for (const auto& val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

int main() {

    const int N_samples = 20; //6
    const int N_time_samples = 48; //160
    const int N_channels = 4;

    // Model model = Model(N_time_samples, 960, 256);
    ModelBuffered modelBuffered = ModelBuffered(N_time_samples, 960, 256);
    modelBuffered.init_weights();
    // model.init_weights();
    // Dummy input
    std::vector<std::vector<std::vector<double>>> input(N_samples, std::vector<std::vector<double>>(N_time_samples, std::vector<double> (N_channels, 0.0)));
    
    for (int i = 0; i < N_samples; i++){
        for (int j = 0; j < N_time_samples; j++){
            for (int k = 0; k < N_channels; k ++) {
                input[i][j][k] = input1[i][j][k];
            }
        }
    }
    std::vector<double> output;
    
    for (int i = 0; i < N_samples - 1; i++)
    {
        output = modelBuffered.forward(input[i]);
        printOutput(output);
    }

    auto start = std::chrono::high_resolution_clock::now();
    output = modelBuffered.forward(input[N_samples - 1]);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    printOutput(output);

    std::cout <<"Done in " << duration.count() << " microseconds" << std::endl;
    
    return 0;
}
