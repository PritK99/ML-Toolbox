#include "metrics.hpp"

RegressionMetrics get_regression_metrics(const std::vector <float> &predictions, const std::vector <float> &true_labels){
    float true_average = 0;
    for (int i = 0; i < predictions.size(); i++){
        true_average += true_labels[i];
    }
    true_average /= predictions.size();

    float mae = 0;
    float mse = 0;
    float r2_denominator = 0;
    for (int i = 0; i < predictions.size(); i++){
        float diff = predictions[i] - true_labels[i];
        float diff_squared = diff*diff;

        mae += std::abs(diff);
        mse += diff_squared;

        r2_denominator +=  (true_labels[i] - true_average)*(true_labels[i] - true_average);
    }

    float r2 = 1 - (mse/r2_denominator);    // mse is still just summation
    mae /= predictions.size();
    mse /= predictions.size();

    RegressionMetrics regression_metrics;
    regression_metrics.mae = mae;
    regression_metrics.mse = mse;
    regression_metrics.rmse = std::sqrt(mse);
    regression_metrics.r2 = r2;

    return regression_metrics;
}