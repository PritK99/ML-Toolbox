#ifndef METRICS_HPP
#define METRICS_HPP

#include <iostream>
#include <vector>
#include <cmath>

struct RegressionMetrics{
    float mse;
    float mae;
    float rmse;
    float r2;
};

struct ClassificationMetrics{
    float accuracy;
    float precision;
    float recall;
    float f1;
};

RegressionMetrics get_regression_metrics(const std::vector <float> &predictions, const std::vector <float> &true_labels);
ClassificationMetrics get_classification_metrics(const std::vector <float> &predictions, const std::vector <float> &true_labels);

#endif