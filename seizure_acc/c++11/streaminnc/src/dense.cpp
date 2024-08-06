#include "dense.h"


Dense::Dense(){}

Dense::Dense(int input_dim, int output_dim)
    : input_dim(input_dim), output_dim(output_dim) {
        this->output.resize(this->output_dim, 0.0);
    }

std::vector<double> Dense::forward(const std::vector<double>& input) {
    // std::vector<double> output(this->output_dim, 0.0);
    for (int j = 0; j < this->output_dim; ++j) {
        double sum = this->bias[j];
        for (int i = 0; i < this->input_dim; ++i) {
            sum += input[i] * this->weights[i * this->output_dim + j];//this->weights[i][j];
        }
        output[j] = sum;
    }
    return output;
}
void Dense::init(const double *weights,
                 const double *bias){
    this->weights = weights;
    this->bias = bias;
}