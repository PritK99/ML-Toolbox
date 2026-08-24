#include "kd.hpp"

/* 
We use a rough heuristic to find the spread
We choose the 0th data point and find the index that has maximum difference in value
Then we repeat the same for this newly found index
We loosely get the extremas, and now subtracting them gives us spread 
*/
namespace KD{

int find_split_dimension(const std::vector <Point> &data){
    int split_dimension = 0;
    float spread = 0;
    for (int i = 0; i < data[0].feature_vector.size(); i++){
        std::vector <float> values;
        for (int j = 0; j < data.size(); j++){
            values.push_back(data[j].feature_vector[i]);
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

float find_split_value(const int feature_dimension, const std::vector <Point> &data){
    std::vector <float> values;

    for (int i = 0; i < data.size(); i++){
        values.push_back(data[i].feature_vector[feature_dimension]);
    }

    std::sort(values.begin(), values.end());

    int middle = int(values.size() / 2);
    float split_value = values[middle];

    return split_value;
}

std::pair <std::vector <Point>, std::vector <Point>> split_data(const int split_dim, const float split_value, const std::vector <Point> &data){
    std::vector <Point> left;
    std::vector <Point> right;

    for (int i = 0; i < data.size(); i++){
        if (data[i].feature_vector[split_dim] >= split_value){
            Point p;
            p.feature_vector = data[i].feature_vector;
            p.label = data[i].label;
            right.push_back(p);
        }
        else{
            Point p;
            p.feature_vector = data[i].feature_vector;
            p.label = data[i].label;
            left.push_back(p);
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
    std::vector <Point> left_data = data_splits.first;
    std::vector <Point> right_data = data_splits.second;

    Node* left_node = new Node();
    left_node->data = left_data;
    left_node->parent = curr_node;
    curr_node->left = left_node;

    Node* right_node = new Node();
    right_node->data = right_data;
    right_node->parent = curr_node;
    curr_node->right = right_node;

    if (left_node->data.empty() || right_node->data.empty()){
        delete left_node;
        delete right_node;
        curr_node->left = NULL;
        curr_node->right = NULL;
        return {NULL, NULL};
    }

    return {left_node, right_node};
}

void build_tree(Node* curr_node, const int min_samples_per_node){
    if (curr_node == NULL || curr_node->data.size() <= min_samples_per_node){
        return;    // This is the stopping criteria
    }

    auto children = split_node(curr_node);
    Node* left_child = children.first;
    Node* right_child = children.second;
    
    build_tree(left_child, min_samples_per_node);    // We recursively build the KD tree
    build_tree(right_child, min_samples_per_node);
}

std::vector <Point> traverse_tree(const std::vector <float> &query, const Node* curr_node, std::vector <Point> &neighbors, int k){
    if (curr_node == NULL){    // If root itself is NULL
        return neighbors;
    }

    if (curr_node->left == NULL || curr_node->right == NULL){    // This is a leaf node
        std::vector <Point> new_neighbors;
        std::vector <std::pair <float, Point>> new_points;
        
        // We need to consider each point in this leaf node
        for (int i = 0; i < curr_node->data.size(); i++){
            float distance = euclidean_distance(curr_node->data[i].feature_vector, query);
            new_points.push_back({distance, curr_node->data[i]});
        }
        for (int i = 0; i < neighbors.size(); i++){
            float distance = euclidean_distance(neighbors[i].feature_vector, query);
            new_points.push_back({distance, neighbors[i]});
        }

        std::sort(new_points.begin(), new_points.end(), [](const auto& a, const auto& b){return a.first < b.first;});    // Turns out we need a comparator, otherwise the code tries to do tie breaking using Point

        int limit = std::min((size_t)k, new_points.size());
        for (int i = 0; i < limit; i++){
            new_neighbors.push_back(new_points[i].second);
        }

        return new_neighbors;
    }
    else{
        if (query[curr_node->split_dim] >= curr_node->split_val){
            std::vector <Point> new_neighbors = traverse_tree(query, curr_node->right, neighbors, k);

            float kth_neighbor_distance = euclidean_distance(query, new_neighbors[new_neighbors.size()-1].feature_vector);
            float distance_to_plane = std::abs(query[curr_node->split_dim] - curr_node->split_val);

            if (new_neighbors.size() < k || kth_neighbor_distance > distance_to_plane){
                new_neighbors = traverse_tree(query, curr_node->left, new_neighbors, k);
            }

            return new_neighbors;
        }
        else{
            std::vector <Point> new_neighbors = traverse_tree(query, curr_node->left, neighbors, k);

            float kth_neighbor_distance = euclidean_distance(query, new_neighbors[new_neighbors.size()-1].feature_vector);
            float distance_to_plane = std::abs(query[curr_node->split_dim] - curr_node->split_val);

            if (new_neighbors.size() < k || kth_neighbor_distance > distance_to_plane){
                new_neighbors = traverse_tree(query, curr_node->right, new_neighbors, k);
            }

            return new_neighbors;
        }
    }
}

}