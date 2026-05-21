#include "../../../utils/csv.hpp"

bool is_vowel(const char c){
    const std::string vowels = "aeiou";

    if (vowels.find(c) != std::string::npos){
        return true;
    }

    return false;
}

std::pair<std::vector<std::vector<float>>, std::vector<int>> extract_features(std::vector<std::vector<std::string>>& raw_data, const int num_features){
    std::vector<std::vector<float>> data;
    std::vector<int> labels;

    for (int i = 0; i < raw_data.size(); i++){
        std::vector<float> row (num_features);
        std::string name = raw_data[i][0];

        // The first feature is for checking if last char is a vowel
        // This is knowledge we add while crafting features since in India, many girls name end with vowel
        char c = std::tolower(name[name.size() - 1]);
        if (is_vowel(c)){
            row[0] = 1;
        }
        
        // Unigrams
        for (int j = 0; j < name.size(); j++){
            char c = std::tolower(name[j]);
            int idx = c - 'a';    // This makes 'a' as 0 and so on
            row[1 + idx] += 1;
        }

        // Bigrams
        for (int j = 0; j < name.size() - 1; j++){
            char c1 = std::tolower(name[j]);
            char c2 = std::tolower(name[j+1]);

            int idx1 = c1 - 'a';
            int idx2 = c2 - 'a';

            row[1 + 26 + idx1*26 + idx2] += 1;
        }

        // Trigrams
        // for (int j = 0; j < name.size() - 2; j++){
        //     char c1 = std::tolower(name[j]);
        //     char c2 = std::tolower(name[j+1]);
        //     char c3 = std::tolower(name[j+2]);

        //     int idx1 = c1 - 'a';
        //     int idx2 = c2 - 'a';
        //     int idx3 = c3 - 'a';

        //     row[1 + 26 + 26*26 + idx1*26*26 + idx2*26 + idx3] += 1;
        // }
        
        row[num_features - 1] = 1;    // bias term

        data.push_back(row);
        
        // Perceptron requires labels to be {+1, -1}
        if (raw_data[i][1] == "1"){
            labels.push_back(1); 
        }
        else{
            labels.push_back(-1); 
        }
    }

    return {data, labels};
}

std::vector <int> fit(const std::vector<std::vector<float>> &data, const std::vector<int> &labels, std::vector <float> &weights, const int max_iters){
    std::vector <int> all_misclassifications;
    bool is_converged = false;

    for (int i = 0; i < max_iters; i++){
        int misclassifications = 0;

        for (int j = 0; j < data.size(); j++){
            const std::vector<float>& data_point = data[j];
            float dot = std::inner_product(weights.begin(), weights.end(), data_point.begin(), 0.0f);

            if (labels[j]*dot <= 0){
                misclassifications += 1;
                
                // Update rule
                for (int k = 0; k < weights.size(); k++){
                    weights[k] += labels[j]*data_point[k];
                }
            }
        }

        all_misclassifications.push_back(misclassifications);
        if (i % 10 == 0){
            std::cout << "Misclassifications at iteration " << i << ": " << misclassifications << std::endl;
        }

        if (misclassifications == 0){
            std::cout << "Perceptron converged at iteration " << i << std::endl;
            is_converged = true;
            break;
        }
    }

    if (!is_converged){
        std::cout << "Perceptron did not converge." << std::endl;
    }

    return all_misclassifications;
}

void validate(const std::vector<std::vector<float>> &data, const std::vector<int> &labels, const std::vector <float> &weights){
    std::vector<int> preds;

    for (int i = 0; i < data.size(); i++){
        const std::vector<float>& data_point = data[i];
        float dot = std::inner_product(weights.begin(), weights.end(), data_point.begin(), 0.0f);

        if (dot >= 0){
            preds.push_back(1);
        }
        else{
            preds.push_back(-1);
        }
    }

    std::array<std::array<int, 2>, 2> confusion_matrix{};    // This will be [[TP, FP], [FN, TN]]
    for (int i = 0; i < labels.size(); i++){
        if (labels[i] == 1){
            if (preds[i] == 1){
                confusion_matrix[0][0] += 1;    // TP
            }
            else{
                confusion_matrix[1][0] += 1;    // FN
            }
        }
        else{
            if (preds[i] == -1){
                confusion_matrix[1][1] += 1;    // TN
            }
            else{
                confusion_matrix[0][1] += 1;    // FP
            }
        }
    }

    std::cout << confusion_matrix[0][0] << " " << confusion_matrix[0][1] << " " << confusion_matrix[1][0] << " " << confusion_matrix[1][1] << std::endl;
}

int main(){
    std::string csv_path = "../../../data/gender.csv";

    auto result = read_csv(csv_path);
    std::vector<std::string> column_names = result.first;
    std::vector<std::vector<std::string>> raw_data = result.second;

    int num_features = 1 + 26 + 26*26 + 26*26*26 + 1;    // 1 is_last_character_vowel + 26 unigrams + 26*26 bigrams + 26*26*26 trigrams + 1 bias
    std::vector <float> weights (num_features);

    // Extracting features from raw dataset
    auto feature_result = extract_features(raw_data, num_features);
    std::vector<std::vector<float>> data = feature_result.first;
    std::vector<int> labels = feature_result.second;
    
    // // Printing a sample data point
    // for (int i = 0; i < data[0].size(); i++){
    //     std::cout << data[0][i] << " ";
    // }
    // std::cout << std::endl;

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

    int max_iters = 500;
    std::vector <int> all_misclassifications = fit(train_data, train_labels, weights, max_iters);

    validate(train_data, train_labels, weights);
    validate(val_data, val_labels, weights);

    return 0;
}