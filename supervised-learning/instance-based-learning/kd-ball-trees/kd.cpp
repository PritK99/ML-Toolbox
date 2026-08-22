#include "kd.hpp"

/* 
We use a rough heuristic to find the spread
We choose the 0th data point and find the index that has maximum difference in value
Then we repeat the same for this newly found index
We loosely get the extremas, and now subtracting them gives us spread 
*/
int find_split_dimension(const std::vector <std::vector <float>> &data){
    int split_dimension = 0;
    float spread = 0;
    for (int i = 0; i < data[0].size(); i++){
        std::vector <float> values;
        for (int j = 0; j < data.size(); j++){
            values.push_back(data[j][i]);
        }

        float curr_value = values[0];
        float max_diff = 0;
        int max_diff_index = 0;
        for (int j = 0; j < values.size(); j++){
            if (std::abs(values[j] - curr_value) > max_diff){
                max_diff = std::abs(values[j] - curr_value);
                max_diff_index = j;
            }
        }

        curr_value = values[max_diff_index];
        max_diff = 0;
        max_diff_index = 0;
        for (int j = 0; j < values.size(); j++){
            if (std::abs(values[j] - curr_value) > max_diff){
                max_diff = std::abs(values[j] - curr_value);
                max_diff_index = j;
            }
        }

        // Now, max_diff is spread
        if (max_diff > spread){
            spread = max_diff;
            split_dimension = i;
        }
    }

    return split_dimension;
}

float find_split_value(const int feature_dimension, const std::vector <std::vector <float>> &data){
    std::vector <float> values;

    for (int i = 0; i < data.size(); i++){
        values.push_back(data[i][feature_dimension]);
    }

    std::sort(values.begin(), values.end());

    int middle = int(values.size() / 2);
    float split_value = values[middle];

    return split_value;
}

std::pair <std::vector <std::vector <float>>, std::vector <std::vector <float>>> split_data(const int split_dim, const float split_value, const std::vector <std::vector <float>> &data){
    std::vector <std::vector <float>> left;
    std::vector <std::vector <float>> right;

    for (int i = 0; i < data.size(); i++){
        if (data[i][split_dim] >= split_value){
            right.push_back(data[i]);
        }
        else{
            left.push_back(data[i]);
        }
    }

    return {left, right};
}

std::pair <Node*, Node*> split_node(Node* curr_node){
    int split_dim = find_split_dimension(curr_node->data);
    float split_val = find_split_value(split_dim, curr_node->data);

    curr_node->split_dim = split_dim;
    curr_node->split_val = split_val;

    auto data_splits = split_data(split_dim, split_val, curr_node->data);
    std::vector <std::vector <float>> left_data = data_splits.first;
    std::vector <std::vector <float>> right_data = data_splits.second;

    Node* left_node = new Node();
    left_node->data = left_data;
    left_node->parent = curr_node;
    curr_node->left = left_node;

    Node* right_node = new Node();
    right_node->data = right_data;
    right_node->parent = curr_node;
    curr_node->right = right_node;

    return {left_node, right_node};
}

void build_kd_tree(Node* curr_node, const int min_samples_per_node){
    if (curr_node->data.size() <= min_samples_per_node){
        return;    // This is the stopping criteria
    }

    auto children = split_node(curr_node);
    Node* left_child = children.first;
    Node* right_child = children.second;
    
    build_kd_tree(left_child, min_samples_per_node);    // We recursively build the KD tree
    build_kd_tree(right_child, min_samples_per_node);
}