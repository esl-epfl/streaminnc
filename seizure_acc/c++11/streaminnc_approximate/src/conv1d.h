#ifndef CONV1D // include guard
#define CONV1D

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

class Conv1D {
public:
    Conv1D();
    Conv1D(int filters, int kernel_size, int input_length, int input_depth);
    void init(const double *weights, const double *bias);
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& input);

private:
    int filters, kernel_size, input_depth, input_length;
    int output_length;
    int pad_size;
    const double *weights;
    const double *bias;

    std::vector<std::vector<double>> output;

    std::vector<std::vector<double>> pad_input(const std::vector<std::vector<double>>& input, int pad_size);
};

#endif /* CONV1D_H */
