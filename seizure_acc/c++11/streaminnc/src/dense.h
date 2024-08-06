#ifndef DENSE // include guard
#define DENSE

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>


class Dense {
public:
    Dense();
    Dense(int input_dim, int output_dim);
    std::vector<double> forward(const std::vector<double>& input);
    
    void init(const double *weights,
              const double *bias);

private:
    int input_dim, output_dim;
    const double *weights;
    const double *bias;
    std::vector<double> output;

};

#endif /* DENSE_H */
