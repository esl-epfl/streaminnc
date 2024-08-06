#ifndef OPS // include guard
#define OPS

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

double relu(double x);

std::vector<std::vector<double>> batch_norm(std::vector<std::vector<double>> x, 
                                            double epsilon, double momentum);
// Overload for 1D inputs (from dense layers)
std::vector<double> batch_norm(std::vector<double> x, double epsilon, double momentum);
std::vector<double> global_average_pooling(std::vector<std::vector<double>>& x);
#endif /* OPS_H */
