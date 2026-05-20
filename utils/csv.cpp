#include "csv.hpp"

std::pair<std::vector <std::string>, std::vector <std::vector <std::string>>> read_csv(const std::string& csv_path){
    std::vector <std::string> column_names;
    std::vector <std::vector <std::string>> data;
    std::ifstream file(csv_path);

    if (!file.is_open()) {
        std::cerr << "Failed to open file.\n";
        return {column_names, data};
    }

    std::string line;

    // The first row is assumed to have column names
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            column_names.push_back(cell);
        }
    }

    // After first iteration, we start reading the data
    while (std::getline(file, line)) {

        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;

        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }

        data.push_back(row);
    }

    return {column_names, data};
}

std::vector<std::vector <std::vector <std::string>>> split_data(std::vector <std::vector <std::string>>& data, float val_ratio, float test_ratio){
    int num_samples = data.size();
    std::vector<std::vector <std::vector <std::string>>> splits;

    // Shuffling dataset
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);

    int num_test_samples = int(test_ratio*num_samples);
    int num_val_samples = int(val_ratio*num_samples);

    std::vector <std::vector <std::string>> val_data (data.begin(), data.begin() + num_val_samples);
    std::vector <std::vector <std::string>> test_data (data.begin() + num_val_samples, data.begin() + num_val_samples + num_test_samples);
    std::vector <std::vector <std::string>> train_data (data.begin() + num_val_samples + num_test_samples, data.end());

    splits.push_back(train_data);
    splits.push_back(val_data);
    splits.push_back(test_data);

    return splits;
}

// // This is for testing functions
// int main(){
//     std::string csv_path = "../data/gender.csv";

//     auto result = read_csv(csv_path);
//     std::vector<std::string> column_names = result.first;
//     std::vector<std::vector<std::string>> data = result.second;

//     float val_ratio = 0.1;
//     float test_ratio = 0.1;
//     std::vector<std::vector <std::vector <std::string>>> splits = split_data(data, val_ratio, test_ratio);

//     std::vector<std::vector<std::string>> train_data = splits[0];
//     std::vector<std::vector<std::string>> val_data = splits[1];
//     std::vector<std::vector<std::string>> test_data = splits[2];

//     std::cout << "Train data: " << train_data.size() << std::endl;
//     std::cout << "Val data: " << val_data.size() << std::endl;
//     std::cout << "Test data: " << test_data.size() << std::endl;

//     return 0;
// }