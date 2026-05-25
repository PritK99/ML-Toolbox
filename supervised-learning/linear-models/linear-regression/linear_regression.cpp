#include "../../../utils/csv.hpp"

std::pair<std::vector<std::vector<float>>, std::vector<int>> extract_features(std::vector<std::vector<std::string>>& raw_data, const int num_features){
    std::vector<std::vector<float>> data;
    std::vector<int> labels;

    for (int i = 0; i < raw_data.size(); i++){
        std::vector<float> row (num_features);
        
        std::string essay = raw_data[i][1];

        row[0] = essay.size();

        row[num_features - 1] = 1;    // bias term

        data.push_back(row);
        labels.push_back(std::stoi(raw_data[i][2]));
    }

    return {data, labels};
}

std::vector <float> fit(const std::vector<std::vector<float>> &data, const std::vector<int> &labels, std::vector <float> &weights, const float lr, const int max_iters){
    std::vector <float> mean_sqaured_errors;
    std::vector <float> mean_absolute_errors;
    int num_features = data[0].size();

    for (int i = 0; i < max_iters; i++){
        std::vector<float> dw(num_features, 0.0f);

        for (int j = 0; j < data.size(); j++){
            std::vector<float> data_point  = data[j];
            float pred = std::inner_product(weights.begin(), weights.end(), data_point.begin(), 0.0f);
            float error = pred - labels[j];

            for (int k = 0; k < num_features; k++){
                dw[k] += data_point[k]*error;
            }
        }

        for (int j = 0; j < num_features; j++){
            dw[j] /= data.size();
            weights[j] -= lr*dw[j];
        }

        float total_error = 0;
        for (int j = 0; j < data.size(); j++){
            std::vector<float> data_point  = data[j];
            float pred = std::inner_product(weights.begin(), weights.end(), data_point.begin(), 0.0f);
            float error = std::pow(pred - labels[j], 2);
            total_error += error;
        }
        total_error /= data.size();

        mean_sqaured_errors.push_back(total_error);
    }

    return mean_sqaured_errors;
}

int main(){
    std::string csv_path = "../../../data/essays.csv";

    auto result = read_csv(csv_path);
    std::vector<std::string> column_names = result.first;
    std::vector<std::vector<std::string>> raw_data = result.second;

    int num_features = 1 + 1;    // length + bias    
    std::vector <float> weights (num_features);

    // Extracting features from raw dataset
    auto feature_result = extract_features(raw_data, num_features);
    std::vector<std::vector<float>> data = feature_result.first;
    std::vector<int> labels = feature_result.second;

    data = normalize_data(data);

    // Printing a sample data point
    for (int i = 0; i < data[0].size(); i++){
        std::cout << data[0][i] << " ";
    }
    std::cout << std::endl;

    float val_ratio = 0.1;
    float test_ratio = 0.1;
    std::vector<std::pair<std::vector <std::vector <float>>, std::vector<int>>> splits = split_data(data, labels, val_ratio, test_ratio);

    std::vector<std::vector<float>> train_data = splits[0].first;
    std::vector<int> train_labels = splits[0].second;
    std::vector<std::vector<float>> val_data = splits[1].first;
    std::vector<int> val_labels = splits[1].second;
    std::vector<std::vector<float>> test_data = splits[2].first;
    std::vector<int> test_labels = splits[2].second;

    std::cout << "Train data: " << train_data.size() << std::endl;
    std::cout << "Val data: " << val_data.size() << std::endl;
    std::cout << "Test data: " << test_data.size() << std::endl;

    float lr = 0.0000001;
    float max_iters = 100;
    std::vector <float>  mean_sqaured_errors = fit(train_data, train_labels, weights, lr, max_iters);

    for (int i = 0; i < mean_sqaured_errors.size(); i++){
        std::cout << mean_sqaured_errors[i] << std::endl;
    }
    
    return 0;
}