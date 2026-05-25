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
    
    // Printing a sample data point
    for (int i = 0; i < data[0].size(); i++){
        std::cout << data[0][i] << " ";
    }
    std::cout << std::endl;

    return 0;
}