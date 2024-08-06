#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>

#include "conv_model.h"

#include "test_data.h"

void printOutput(std::vector<double> output){
    std::cout << "Output: ";
    for (const auto& val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

int main() {

    const int N_time_samples = 960;
    const int N_channels = 4;

    Model model;

    model.init_weights();
    // Dummy input
    std::vector<std::vector<double>> input(N_time_samples, std::vector<double>(N_channels, 0.0));
    for (int i = 0; i < N_time_samples; i++){
        for (int j = 0; j < N_channels; j++){
            input[i][j] = input1[i][j];
        }
    }
    
    std::vector<double> output;
    auto start = std::chrono::high_resolution_clock::now();
    output = model.forward(input);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    printOutput(output);

    std::cout <<"Done in " << duration.count() << " microseconds" << std::endl;
    
    return 0;
}
