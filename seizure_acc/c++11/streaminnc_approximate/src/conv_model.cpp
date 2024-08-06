#include "conv_model.h"

Model::Model() {
    // Initialize layers
    this->conv1 = Conv1D(8, 3, 960, 4);
    this->bn1 = BatchNormalization(8);

    this->conv2 = Conv1D(16, 3, 960, 8);
    this->bn2 = BatchNormalization(16);

    this->conv3 = Conv1D(32, 3, 960, 16);
    this->bn3 = BatchNormalization(32);

    this->conv4 = Conv1D(64, 3, 960, 32);
    this->bn4 = BatchNormalization(64);

    this->conv5 = Conv1D(128, 3, 960, 64);
    this->bn5 = BatchNormalization(128);

    this->conv6 = Conv1D(256, 3, 960, 128);
    this->bn6 = BatchNormalization(256);

    this->dense1 = Dense(256, 256);
    this->dense2 = Dense(256, 128);
    this->bn7 = BatchNormalization(128);

    this->dense3 = Dense(128, 64);
    this->bn8 = BatchNormalization(64);

    this->dense4 = Dense(64, 2);
}

std::vector<double> Model::forward(const std::vector<std::vector<double>>& input) {
    std::vector<std::vector<double>> x = this->conv1.forward(input);
    x = bn1.forward(x);
    x = apply_relu(x);

    x = this->conv2.forward(x);
    x = bn2.forward(x);
    x = apply_relu(x);
    
    x = this->conv3.forward(x);
    x = bn3.forward(x);
    x = apply_relu(x);
    
    x = this->conv4.forward(x);
    x = bn4.forward(x);
    x = apply_relu(x);
    
    x = this->conv5.forward(x);
    x = bn5.forward(x);
    x = apply_relu(x);
    
    x = this->conv6.forward(x);
    x = bn6.forward(x);
    x = apply_relu(x);

    std::vector<double> pooled = global_average_pooling(x);    
    
    std::vector<double> y = this->dense1.forward(pooled);
    y = this->dense2.forward(y);
    y = bn7.forward(y);
    y = apply_relu(y);
    
    y = this->dense3.forward(y);
    y = bn8.forward(y);
    y = apply_relu(y);
    
    y = this->dense4.forward(y);
    return y;
}

std::vector<std::vector<double>> Model::apply_relu(const std::vector<std::vector<double>>& x) {
    std::vector<std::vector<double>> output = x;
    for (auto& row : output) {
        for (auto& val : row) {
            val = relu(val);
        }
    }
    return output;
}

std::vector<double> Model::apply_relu(const std::vector<double>& x) {
    std::vector<double> output = x;
    for (auto& val : output) {
        val = relu(val);
    }
    return output;
}

void Model::init_weights(){
    this->conv1.init(&conv1_weights[0][0][0], &conv1_bias[0]);
    this->conv2.init(&conv2_weights[0][0][0], &conv2_bias[0]);
    this->conv3.init(&conv3_weights[0][0][0], &conv3_bias[0]);
    this->conv4.init(&conv4_weights[0][0][0], &conv4_bias[0]);
    this->conv5.init(&conv5_weights[0][0][0], &conv5_bias[0]);
    this->conv6.init(&conv6_weights[0][0][0], &conv6_bias[0]);

    this->bn1.init(&bn1_mean[0], &bn1_variance[0], &bn1_beta[0], &bn1_gamma[0]);
    this->bn2.init(&bn2_mean[0], &bn2_variance[0], &bn2_beta[0], &bn2_gamma[0]);
    this->bn3.init(&bn3_mean[0], &bn3_variance[0], &bn3_beta[0], &bn3_gamma[0]);
    this->bn4.init(&bn4_mean[0], &bn4_variance[0], &bn4_beta[0], &bn4_gamma[0]);
    this->bn5.init(&bn5_mean[0], &bn5_variance[0], &bn5_beta[0], &bn5_gamma[0]);
    this->bn6.init(&bn6_mean[0], &bn6_variance[0], &bn6_beta[0], &bn6_gamma[0]);
    this->bn7.init(&bn7_mean[0], &bn7_variance[0], &bn7_beta[0], &bn7_gamma[0]);
    this->bn8.init(&bn8_mean[0], &bn8_variance[0], &bn8_beta[0], &bn8_gamma[0]);

    this->dense1.init(&dense1_weights[0][0], &dense1_bias[0]);
    this->dense2.init(&dense2_weights[0][0], &dense2_bias[0]);
    this->dense3.init(&dense3_weights[0][0], &dense3_bias[0]);
    this->dense4.init(&dense4_weights[0][0], &dense4_bias[0]);
}