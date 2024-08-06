#ifndef BATCHNORMALIZATION // include guard
#define BATCHNORMALIZATION

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

class BatchNormalization {
public:
    BatchNormalization(size_t size = 1, double epsilon = 1e-5, double momentum = 0.1);

    std::vector<double> forward(const std::vector<double>& x);
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& x);

    void init(const double *moving_mean,
              const double *moving_variance,
              const double *beta,
              const double *gamma);
private:
    double epsilon;
    double momentum;
    int size;
    const double *moving_mean;
    const double *moving_variance;
    const double *beta;
    const double *gamma;
};

#endif /* BATCHNORMALIZATION_H */
