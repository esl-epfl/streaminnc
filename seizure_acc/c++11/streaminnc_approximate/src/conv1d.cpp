#include "conv1d.h"

Conv1D::Conv1D(){}

Conv1D::Conv1D(int filters, int kernel_size, int input_length, int input_depth)
    : filters(filters), kernel_size(kernel_size), input_length(input_length), input_depth(input_depth) {
    
    this->output_length = this->input_length;  // Same padding
    this->pad_size = this->kernel_size / 2;

    this->output.resize(output_length, std::vector<double>(this->filters, 0.0));
}

void Conv1D::init(const double *weights, const double *bias){
    this-> weights = weights;
    this-> bias = bias;
}

std::vector<std::vector<double>> Conv1D::pad_input(const std::vector<std::vector<double>>& input, int pad_size) {
    int length = input.size();
    int channels = input[0].size();

    std::vector<std::vector<double>> padded_input(length + 2 * pad_size, std::vector<double>(channels, 0.0));
    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < length; ++i) {
            padded_input[i + pad_size][c] = input[i][c];
        }
    }
    return padded_input;
}

std::vector<std::vector<double>> Conv1D::forward(const std::vector<std::vector<double>>& input) {
    std::vector<std::vector<double>> padded_input = pad_input(input, pad_size);
    
    for (int f = 0; f < this->filters; ++f) {
        for (int i = 0; i < output_length; ++i) {
            double sum = 0;
            for (int c = 0; c < this->input_depth; ++c) {
                for (int k = 0; k < this->kernel_size; ++k) {
                    sum += padded_input[i + k][c]
                                  * this->weights[f * this->kernel_size * this->input_depth + k * this->input_depth + c];// this->weights[f][k][d];
                }
            }
            this->output[i][f] = sum + this->bias[f];
            
        }
    }
    
    return output;
}