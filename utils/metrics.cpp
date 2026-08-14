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

ClassificationMetrics get_classification_metrics(const std::vector <float> &predictions, const std::vector <float> &true_labels){
    // We denote positive class as 1 and negative class as -1
    // Hence, precison, recall and f1 are with respect to positive class
    int tp = 0;
    int fp = 0;
    int fn = 0;
    int tn = 0;

    for (int i = 0; i < predictions.size(); i++){
        if (predictions[i] == true_labels[i]){
            if (predictions[i] == 1){
                tp ++;
            }
            else{
                tn ++;
            }
        }
        else{
            if (predictions[i] == 1){
                fp ++;
            }
            else{
                fn ++;
            }
        }
    }

    float accuracy = (tp + tn)*1.0 / (tp + tn + fp + fn);
    float precision = (tp)*1.0 / (tp + fp);
    float recall = (tp)*1.0 / (tp + fn);
    float f1 = (2*precision*recall) / (precision + recall);

    ClassificationMetrics classification_metrics;
    classification_metrics.accuracy = accuracy;
    classification_metrics.precision = precision;
    classification_metrics.recall = recall;
    classification_metrics.f1 = f1;

    return classification_metrics;
}