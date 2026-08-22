/*
We consider the following statistical features from an essay

1. Number of words in essay
2. Number of short words (i.e. words with length <= 4)
3. Number of medium words (i.e. words with length > 4 and < 9)
4. Number of long words (i.e. words with length >= 9)
5. Number of sentences in essay
6. Number of short sentences (i.e. sentences with length < 10)
7. Number of medium sentences (i.e. sentences with length < 20)
8. Number of long sentences (i.e. sentences with length >= 20)
9. Average sentence length

There are many other statistical features we can think of, such as variance in word length, variance in sentence length, use of numbers, frequency of stop words, simple punctuation errors etc.
*/
#include "../../../utils/csv.hpp"
#include "../../../utils/distances.hpp"
#include "../../../utils/metrics.hpp"
#include "kd.hpp"
#include <cctype>
#include <utility>
#include <algorithm>

// Given a dataset of essay_id, essay and score, this function computes the feature vectors for each essay
std::pair<std::vector<std::vector<float>>, std::vector<float>> extract_features(std::vector<std::vector<std::string>>& raw_data, const int num_features){
    std::vector<std::vector<float>> data;
    std::vector<float> labels;

    for (int i = 0; i < raw_data.size(); i++){
        std::vector<float> row (num_features);
        std::string essay = raw_data[i][1];

        int num_words = 0;
        int num_short_words = 0;
        int num_medium_words = 0;
        int num_long_words = 0;
        int num_sentences = 0;
        int num_short_sentences = 0;
        int num_medium_sentences = 0;
        int num_long_sentences = 0;

        std::string word = "";    // These are local counters
        int num_words_in_curr_sentence = 0;
        for (int j = 0; j < essay.size(); j++){

            // Detecting word boundaries
            if ((essay[j] == ' ' || essay[j] == '.' || essay[j] == '?' || essay[j] == '!') && word != ""){
                num_words ++;

                if (word.size() <= 4){
                    num_short_words ++;
                }
                else if (word.size() < 9){
                    num_medium_words ++;
                }
                else{
                    num_long_words ++;
                }
                
                num_words_in_curr_sentence ++;
                word = "";
            }
            else{    // We only consider alphabets and numbers towards word
                if (std::isalnum(static_cast<unsigned char>(essay[j]))){    // Turns out char is signed, but isalnum only deals with unsigned
                    word += essay[j];
                }
            }

            // Detecting sentence boundaries
            if (essay[j] == '.' || essay[j] == '?' || essay[j] == '!'){    // We only consider the common possibilities
                if (j == essay.size() - 1 || essay[j+1] == '"' || essay[j+1] == ' '){
                    num_sentences ++; 

                    if (num_words_in_curr_sentence < 10){
                        num_short_sentences ++;
                    }
                    else if (num_words_in_curr_sentence < 20){
                        num_medium_sentences ++;
                    }
                    else{
                        num_long_sentences ++;
                    }

                    num_words_in_curr_sentence = 0;
                }
            }
        }

        // Creating feature vector using computed statistical features
        row[0] = num_words;
        row[1] = num_short_words;
        row[2] = num_medium_words;
        row[3] = num_long_words;
        row[4] = num_sentences;
        row[5] = num_short_sentences;
        row[6] = num_medium_sentences;
        row[7] = num_long_sentences;
        row[8] = num_words*1.0 / num_sentences;

        data.push_back(row);
        labels.push_back(float(std::stoi(raw_data[i][2])));
    }

    return {data, labels};
}


int main(){
    std::string csv_path = "../../../data/essays.csv";

    auto result = read_csv(csv_path);
    std::vector<std::string> column_names = result.first;
    std::vector<std::vector<std::string>> raw_data = result.second;

    int num_features = 9;   
    std::vector <float> weights (num_features);

    // Extracting features from raw dataset
    auto feature_result = extract_features(raw_data, num_features);
    std::vector<std::vector<float>> data = feature_result.first;
    std::vector<float> labels = feature_result.second;

    // Splitting data into train-val-test sets
    float val_ratio = 0.1;
    float test_ratio = 0.1;
    std::vector<std::pair<std::vector <std::vector <float>>, std::vector<float>>> splits = split_data(data, labels, val_ratio, test_ratio);

    std::vector<std::vector<float>> unnormalized_train_data = splits[0].first;
    std::vector<float> train_labels = splits[0].second;
    std::vector<std::vector<float>> unnormalized_val_data = splits[1].first;
    std::vector<float> val_labels = splits[1].second;
    std::vector<std::vector<float>> unnormalized_test_data = splits[2].first;
    std::vector<float> test_labels = splits[2].second;

    // Normalizing the data
    auto normalization_stats = compute_normalization_stats(unnormalized_train_data);
    std::vector <float> mean = normalization_stats.first;
    std::vector <float> std_dev = normalization_stats.second;

    std::vector <std::vector <float>> normalized_train_data = normalize_data(unnormalized_train_data, mean, std_dev);
    std::vector <std::vector <float>> normalized_val_data = normalize_data(unnormalized_val_data, mean, std_dev);
    std::vector <std::vector <float>> normalized_test_data = normalize_data(unnormalized_test_data, mean, std_dev);

    std::cout << "Train data: " << normalized_train_data.size() << std::endl;
    std::cout << "Val data: " << normalized_val_data.size() << std::endl;
    std::cout << "Test data: " << normalized_test_data.size() << std::endl << std::endl;

    int min_samples_per_node = 30;
    Node* root = new Node();
    root->data = normalized_train_data;
    build_kd_tree(root, min_samples_per_node);

    return 0;
}