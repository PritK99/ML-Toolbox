#include "../../../utils/csv.hpp"

std::vector<std::vector<int>> extract_features(std::vector<std::vector<std::string>>& raw_data, int num_features){
    std::vector<std::vector<int>> data;
    std::vector<int> labels;

    for (int i = 0; i < raw_data.size(); i++){
        std::vector<int> row (num_features);
        std::string name = raw_data[i][0];

        for (char &c : name) {    
            c = std::tolower(c);
            int idx = c - 'a';    // This makes 'a' as 0 and so on
            row[idx] += 1;
        }

        data.push_back(row);
        labels.push_back(std::stoi(raw_data[i][1])); 

        break;
    }

    return data;
}

int main(){
    std::string csv_path = "../../../data/gender.csv";

    auto result = read_csv(csv_path);
    std::vector<std::string> column_names = result.first;
    std::vector<std::vector<std::string>> raw_data = result.second;

    int num_features = 26;    // 26 unigrams
    std::vector<std::vector<int>> data = extract_features(raw_data, num_features);
    
    for (int i = 0; i < data[0].size(); i++){
        std::cout << data[0][i] << " ";
    }
    std::cout << std::endl;

    float val_ratio = 0.1;
    float test_ratio = 0.1;
    std::vector<std::vector <std::vector <std::string>>> splits = split_data(raw_data, val_ratio, test_ratio);

    std::vector<std::vector<std::string>> train_data = splits[0];
    std::vector<std::vector<std::string>> val_data = splits[1];
    std::vector<std::vector<std::string>> test_data = splits[2];

    std::cout << "Train data: " << train_data.size() << std::endl;
    std::cout << "Val data: " << val_data.size() << std::endl;
    std::cout << "Test data: " << test_data.size() << std::endl;

    return 0;
}