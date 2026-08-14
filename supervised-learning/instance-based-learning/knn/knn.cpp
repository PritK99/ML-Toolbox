/*
We consider the following statistical features from an essay

1. Number of words in essay
2. Number of short words (i.e. words with length <= 4)
3. Number of medium words (i.e. words with length > 4 and < 9)
4. Number of long words (i.e. words with length >= 9)
5. Number of sentences in essay
6. Number of short sentences (i.e. sentences with length < 10)
7. Number of medium wsentences(i.e. sentences with length < 20)
8. Number of long sentences (i.e. sentences with length >= 20)
9. Average sentence length

There are many other statistical features we can think of, such as variance in word length, variance in sentence length, use of numbers, frequency of stop words, simple punctuation errors etc.
*/
#include "../../../utils/csv.hpp"
#include "../../../utils/distances.hpp"
#include "../../../utils/metrics.hpp"
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
        row[8] = num_sentences*1.0 / num_words;

        data.push_back(row);
        labels.push_back(float(std::stoi(raw_data[i][2])));
    }

    return {data, labels};
}

// For a given query feature vector, this function computes the score
float predict(const int k, const std::vector <float> &query, const std::vector <std::vector <float>> &train_data, const std::vector<float> &train_labels, const std::string metric = "euclidean"){
    std::vector <std::pair <float, int>> distances;

    for (int i = 0; i < train_data.size(); i++){
        float distance = 0;
        if (metric == "manhattan"){
            distance = manhattan_distance(query, train_data[i]);
        }
        else if (metric == "cosine"){
            distance = cosine_distance(query, train_data[i]);
        }
        else{
            distance = euclidean_distance(query, train_data[i]);
        }

        distances.push_back({distance, i});
    }

    std::sort(distances.begin(), distances.end());

    float score = 0;
    for (int i = 0; i < k; i++){
        int neighbor_label = train_labels[distances[i].second];
        score += neighbor_label;
    }
    score = score*1.0/k;

    return score;
}

float inference(const std::string &essay, const std::vector <std::vector <float>> &normalized_train_data, const std::vector<float> &train_labels, const int num_features, const int k, const std::string test_metric){
    std::vector <std::vector <std::string>> inference_data;
    
    std::string dummy_essay1_id = "0";    // We need this dummy values because of the way extract_features is defined
    std::string dummy_score = "0";

    std::vector <std::string> inference_point = {dummy_essay1_id, essay, dummy_score};
    inference_data.push_back(inference_point);

    auto feature_result = extract_features(inference_data, num_features);
    std::vector<std::vector<float>> data = feature_result.first;
    std::vector<float> labels = feature_result.second;

    float prediction = predict(k, data[0], normalized_train_data, train_labels, test_metric);

    return prediction;
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

    std::vector<std::vector<float>> train_data = splits[0].first;
    std::vector<float> train_labels = splits[0].second;
    std::vector<std::vector<float>> val_data = splits[1].first;
    std::vector<float> val_labels = splits[1].second;
    std::vector<std::vector<float>> test_data = splits[2].first;
    std::vector<float> test_labels = splits[2].second;

    // Normalizing the data
    std::vector <std::vector <std::vector <float>>> normalized_data = normalize_data(train_data, val_data, test_data);
    std::vector <std::vector <float>> normalized_train_data = normalized_data[0];
    std::vector <std::vector <float>> normalized_val_data = normalized_data[1];
    std::vector <std::vector <float>> normalized_test_data = normalized_data[2];

    std::cout << "Train data: " << normalized_train_data.size() << std::endl;
    std::cout << "Val data: " << normalized_val_data.size() << std::endl;
    std::cout << "Test data: " << normalized_test_data.size() << std::endl << std::endl;

    // Validation for tuning K and distance metric
    // int k_max = 19;
    // std::vector <std::string> metrics = {"manhattan", "euclidean", "cosine"};

    // for (int i = 0; i < metrics.size(); i++){
    //     for (int k = 3; k < k_max; k += 2){    // This is just 3, 5, 7, ... k_max
    //         std::string metric = metrics[i];
    //         std::cout << "Evaluating K = " << k << " using " << metric << "." << std::endl;

    //         std::vector <float> predictions (val_data.size());
    //         for (int j = 0; j < predictions.size(); j++){
    //             predictions[j] = predict(k, normalized_val_data[j], normalized_train_data, train_labels, metric);
    //         }

    //         RegressionMetrics val_regression_metrics = get_regression_metrics(predictions, val_labels);
    //         std::cout << "MAE: " << val_regression_metrics.mae << std::endl;
    //         std::cout << "RMSE: " << val_regression_metrics.rmse << std::endl;
    //         std::cout << "R2: " << val_regression_metrics.r2 << std::endl << std::endl;
    //     }
    // }

    // Testing
    std::string test_metric = "euclidean";
    int test_k = 15;
    std::cout << "Testing" << std::endl;

    std::vector <float> predictions (val_data.size());
    for (int j = 0; j < predictions.size(); j++){
        predictions[j] = predict(test_k, normalized_val_data[j], normalized_train_data, train_labels, test_metric);
    }

    RegressionMetrics test_regression_metrics = get_regression_metrics(predictions, val_labels);
    std::cout << "MAE: " << test_regression_metrics.mae << std::endl;
    std::cout << "RMSE: " << test_regression_metrics.rmse << std::endl;
    std::cout << "R2: " << test_regression_metrics.r2 << std::endl << std::endl;

    // Now, we run the model on our essay
    std::string my_essay1 = "This is a test essay. Lets see where it goes! This is a poem from the family guy. Oh, squiggly line in my eye fluid, I see you lurking there on the periphery of my vision. But when I try to look at you, you scurry away. Are you shy, squiggly line? Why only when I ignore you, do you return to the center of my eye? Oh, squiggly line, it's alright, you are forgiven.";

    float score = inference(my_essay1, normalized_train_data, train_labels, num_features, test_k, test_metric);
    std::cout << "Your essay score is: " << score << std::endl; 

    return 0;
}