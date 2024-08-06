#include "ops.h"

double relu(double x) {
    return std::max(0.0, x);
}

// Overload for 2D inputs (from convolutions)
std::vector<std::vector<double>> batch_norm(std::vector<std::vector<double>> x, double epsilon, double momentum) {
    size_t rows = x.size();
    size_t cols = x[0].size();

    std::vector<double> mean(cols, 0.0);
    std::vector<double> variance(cols, 0.0);
    std::vector<std::vector<double>> output(rows, std::vector<double>(cols, 0.0));

    // Calculate mean
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            mean[j] += x[i][j];
        }
    }
    for (size_t j = 0; j < cols; ++j) {
        mean[j] /= rows;
    }

    // Calculate variance
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            variance[j] += (x[i][j] - mean[j]) * (x[i][j] - mean[j]);
        }
    }
    for (size_t j = 0; j < cols; ++j) {
        variance[j] = sqrt(variance[j] / rows + epsilon);
    }

    // Normalize
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            output[i][j] = (x[i][j] - mean[j]) / variance[j];
        }
    }

    return output;
}

// Overload for 1D inputs (from dense layers)
std::vector<double> batch_norm(std::vector<double> x, double epsilon, double momentum) {
    size_t size = x.size();
    double mean = 0.0;
    double variance = 0.0;
    std::vector<double> output(size, 0.0);

    // Calculate mean
    for (size_t i = 0; i < size; ++i) {
        mean += x[i];
    }
    mean /= size;

    // Calculate variance
    for (size_t i = 0; i < size; ++i) {
        variance += (x[i] - mean) * (x[i] - mean);
    }
    variance = sqrt(variance / size + epsilon);

    // Normalize
    for (size_t i = 0; i < size; ++i) {
        output[i] = (x[i] - mean) / variance;
    }

    return output;
}

std::vector<double> global_average_pooling(std::vector<std::vector<double>>& x) {
    size_t rows = x.size();
    size_t cols = x[0].size();
    std::vector<double> output(cols, 0.0);

    for (size_t j = 0; j < cols; ++j) {
        for (size_t i = 0; i < rows; ++i) {
            output[j] += x[i][j];
        }
        output[j] /= rows;
    }

    return output;
}