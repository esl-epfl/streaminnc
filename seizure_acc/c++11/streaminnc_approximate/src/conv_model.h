#ifndef CONV_MODEL // include guard
#define CONV_MODEL

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "ops.h"
#include "conv1d.h"
#include "dense.h"
#include "batchnormalization.h"

#include "weight_definitions.h"


class Model {
public:
    Model();
    std::vector<double> forward(const std::vector<std::vector<double>>& input);

    void init_weights();
private:
    Conv1D conv1, conv2, conv3, conv4, conv5, conv6;
    Dense dense1, dense2, dense3, dense4;
    BatchNormalization bn1, bn2, bn3, bn4, bn5, bn6, bn7, bn8;

    std::vector<std::vector<double>> apply_relu(const std::vector<std::vector<double>>& x);
    std::vector<double> apply_relu(const std::vector<double>& x);
};

#endif /* CONV_MODEL_H */
