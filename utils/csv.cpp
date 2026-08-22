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
        
        // We could have simply split at commas
        // But this would break the string fields with commas such as in essay dataset
        // Thus, we explicitly check if the comma is inside double quotes, indicating it is part of string and not a separator
        bool is_string = false;
        for (int i = 0; i < line.size(); i++){
            if (line[i] == '"'){
                is_string = !is_string;
            }

            if (line[i] == ',' && !is_string){
                row.push_back(cell);
                cell = "";
                continue;
            }

            cell += line[i];
        }
        row.push_back(cell);    // This is for last column which doesnt have a comma

        // We need to do this additional handling since windows uses \r\n instead of \n
        // getline handles \n, but returns an extra \r in the end which made all the difference
        for (auto& c : row) {
            while (!c.empty() && (c.back() == '\r' || c.back() == '\n' || c.back() == ' ')) {
                c.pop_back();
            }
        }

        data.push_back(row);
    }

    return {column_names, data};
}

std::vector<std::pair<std::vector <std::vector <float>>, std::vector<float>>> split_data(std::vector <std::vector <float>>& data, std::vector<float> &labels, const float val_ratio, const float test_ratio){
    int num_samples = data.size();
    std::vector<std::pair<std::vector <std::vector <float>>, std::vector<float>>> splits;

    // Shuffling dataset
    std::vector<int> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);

    // std::random_device rd;
    // std::mt19937 g(rd());
    std::mt19937 g(99);
    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<std::vector<float>> shuffled_data(num_samples);
    std::vector<float> shuffled_labels(num_samples);

    for (int i = 0; i < num_samples; i++) {
        shuffled_data[i] = data[indices[i]];
        shuffled_labels[i] = labels[indices[i]];
    }

    int num_test_samples = int(test_ratio*num_samples);
    int num_val_samples = int(val_ratio*num_samples);

    std::vector <std::vector <float>> val_data (shuffled_data.begin(), shuffled_data.begin() + num_val_samples);
    std::vector<float> val_labels (shuffled_labels.begin(), shuffled_labels.begin() + num_val_samples);
    std::vector <std::vector <float>> test_data (shuffled_data.begin() + num_val_samples, shuffled_data.begin() + num_val_samples + num_test_samples);
    std::vector<float> test_labels (shuffled_labels.begin() + num_val_samples, shuffled_labels.begin() + num_val_samples + num_test_samples);
    std::vector <std::vector <float>> train_data (shuffled_data.begin() + num_val_samples + num_test_samples, shuffled_data.end());
    std::vector<float> train_labels (shuffled_labels.begin() + num_val_samples + num_test_samples, shuffled_labels.end());

    splits.push_back({train_data, train_labels});
    splits.push_back({val_data, val_labels});
    splits.push_back({test_data, test_labels});

    return splits;
}

// This function returns the mean and standard deviation for all features in the dataset
std::pair <std::vector <float>, std::vector <float>> compute_normalization_stats(const std::vector <std::vector <float>> &train_data){
    int num_features = train_data[0].size();
    std::vector <float> mean (num_features); 
    std::vector <float> std_dev (num_features);
    
    // Calculating mean
    for (int i = 0; i < train_data.size(); i++){
        for (int j = 0; j < num_features; j++){
            mean[j] += train_data[i][j];
        }
    }

    for (int j = 0; j < num_features; j++){
        mean[j] /= train_data.size();
    }

    // Calculating standard deviation
    for (int i = 0; i < train_data.size(); i++){
        for (int j = 0; j < num_features; j++){
            std_dev[j] += std::pow(train_data[i][j] - mean[j], 2);
        }
    }

    for (int j = 0; j < num_features; j++){
        std_dev[j] /= train_data.size();
        std_dev[j] = std::pow(std_dev[j], 0.5);
    }

    return {mean, std_dev};
}

std::vector <std::vector <float>> normalize_data(const std::vector <std::vector <float>> &unnormalized_data, const std::vector <float> &mean, const std::vector <float> &std_dev){
    std::vector <std::vector <float>> normalized_data;

    // Normalizing
    for (int i = 0; i < unnormalized_data.size(); i++){
        std::vector <float> data_point = unnormalized_data[i];
        for (int j = 0; j < unnormalized_data[0].size(); j++){
            if (std_dev[j] != 0){
                data_point[j] = (data_point[j]  - mean[j]) / std_dev[j];
            }
        }

        normalized_data.push_back(data_point);
    }

    return normalized_data;
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