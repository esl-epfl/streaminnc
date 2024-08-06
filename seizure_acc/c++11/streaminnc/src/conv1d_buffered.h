#ifndef CONV1DBUFFERED // include guard
#define CONV1DBUFFERED

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

class Conv1DBuffered {
public:
    Conv1DBuffered();
    Conv1DBuffered(int filters, int kernel_size, int input_length, int input_depth);
    void init(const double *weights, const double *bias);
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& input);

private:
    int filters, kernel_size, input_length, input_depth;
    int output_length;
    int pad_size;
    const double *weights;
    const double *bias;

    std::vector<std::vector<double>> output;
    std::vector<std::vector<double>> buffer;
    std::vector<std::vector<double>> pad_input(const std::vector<std::vector<double>>& input, int pad_size);
};

#endif /* CONV1DBUFFERED_H */
