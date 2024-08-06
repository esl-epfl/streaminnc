#include "batchnormalization.h"

BatchNormalization::BatchNormalization(size_t size, double epsilon, double momentum)
    : size(size), epsilon(epsilon), momentum(momentum){}

std::vector<std::vector<double>> BatchNormalization::forward(const std::vector<std::vector<double>>& x) {
    size_t rows = x.size();
    size_t cols = x[0].size();
    std::vector<std::vector<double>> output(rows, std::vector<double>(cols, 0.0));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double normalized = (x[i][j] - this->moving_mean[j]) / sqrt(this->moving_variance[j] + this->epsilon);
            output[i][j] = normalized * this->gamma[j] + this->beta[j];
        }
    }
    return output;
}

std::vector<double> BatchNormalization::forward(const std::vector<double>& x) {
    size_t size = x.size();
    std::vector<double> output(size, 0.0);

    for (size_t i = 0; i < size; ++i) {
        double normalized = (x[i] - this->moving_mean[i]) / sqrt(this->moving_variance[i] + this->epsilon);
        output[i] = normalized * this->gamma[i] + this->beta[i];
    }
    return output;
}

void BatchNormalization::init(const double *moving_mean,
                              const double *moving_variance,
                              const double *beta,
                              const double *gamma){

    this->moving_mean = moving_mean;
    this->moving_variance = moving_variance;
    this->beta = beta;
    this->gamma = gamma;

}