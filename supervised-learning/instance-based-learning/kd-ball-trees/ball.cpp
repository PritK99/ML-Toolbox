#include "ball.hpp"

namespace Ball{

/* 
We use a rough heuristic to find the spread
We choose the 0th data point and find the farthest data point from it
Then we repeat the same for this newly found data point
We loosely get the extremas, and now subtracting them gives us direction of highest variance
In KD Trees, we were concerned about finding the axis among features with highest spread
Here, our concern is to find a direction of highest spread, not necessarily the axis
*/
std::vector <float> find_split_direction(const std::vector <Point> &data){
    Point curr_value = data[0];
    float max_diff = 0;
    int max_diff_index = 0;
    for (int i = 0; i < data.size(); i++){
        if (euclidean_distance(data[i].feature_vector, curr_value.feature_vector) > max_diff){
            max_diff = euclidean_distance(data[i].feature_vector, curr_value.feature_vector);
            max_diff_index = i;
        }
    }
    Point extrema1 = data[max_diff_index];

    curr_value = data[max_diff_index];
    max_diff = 0;
    max_diff_index = 0;
    for (int i = 0; i < data.size(); i++){
        if (euclidean_distance(data[i].feature_vector, curr_value.feature_vector) > max_diff){
            max_diff = euclidean_distance(data[i].feature_vector, curr_value.feature_vector);
            max_diff_index = i;
        }
    }
    Point extrema2 = data[max_diff_index];

    std::vector <float> split_direction (extrema1.feature_vector.size());
    for (int i = 0; i < curr_value.feature_vector.size(); i++){
        split_direction[i] = extrema1.feature_vector[i] - extrema2.feature_vector[i];
    }

    return split_direction;
}

/*
This function computes the left and right splits, and their corresponding mean and radius
*/
std::pair <std::vector <std::vector <Point>>, std::pair <std::vector <std::vector <float>>, std::vector <float>>> find_partitions(const std::vector <float> &split_direction, const std::vector <Point> &data){
    std::vector <float> values;

    for (int i = 0; i < data.size(); i++){
        float projection = std::inner_product(split_direction.begin(), split_direction.end(), data[i].feature_vector.begin(), 0.0f);
        values.push_back(projection);
    }
    std::sort(values.begin(), values.end());

    int middle = int(values.size() / 2);
    float split_value = values[middle];

    std::vector <Point> left_partition;
    std::vector <Point> right_partition;
    for (int i = 0; i < data.size(); i++){
        float projection = std::inner_product(split_direction.begin(), split_direction.end(), data[i].feature_vector.begin(), 0.0f);

        if (projection >= split_value){
            right_partition.push_back(data[i]);
        }
        else{
            left_partition.push_back(data[i]);
        }
    }

    if (left_partition.empty() || right_partition.empty()){
        std::vector <std::vector <float>> means;
        std::vector <float> radii;
        std::vector <std::vector <Point>> partitions;

        return {partitions, {means, radii}};
    }

    std::vector <float> left_mean (data[0].feature_vector.size());
    std::vector <float> right_mean (data[0].feature_vector.size());

    for (int i = 0; i < left_partition.size(); i++){
        for (int j = 0; j < left_mean.size(); j++){
            left_mean[j] += left_partition[i].feature_vector[j];
        }
    }

    for (int i = 0; i < left_mean.size(); i++){
        left_mean[i] /= left_partition.size();
    }

    for (int i = 0; i < right_partition.size(); i++){
        for (int j = 0; j < right_mean.size(); j++){
            right_mean[j] += right_partition[i].feature_vector[j];
        }
    }

    for (int i = 0; i < right_mean.size(); i++){
        right_mean[i] /= right_partition.size();
    }

    float left_radius = 0;
    for (int i = 0; i < left_partition.size(); i++){
        float distance = euclidean_distance(left_partition[i].feature_vector, left_mean);
        if (distance > left_radius){
            left_radius = distance;
        }
    }

    float right_radius = 0;
    for (int i = 0; i < right_partition.size(); i++){
        float distance = euclidean_distance(right_partition[i].feature_vector, right_mean);
        if (distance > right_radius){
            right_radius = distance;
        }
    }

    std::vector <std::vector <float>> means (2);    // This will hold radius for left and right partition
    means[0] = left_mean;
    means[1] = right_mean;

    std::vector <float> radii (2);    // This will hold radius for left and right partition
    radii[0] = left_radius;
    radii[1] = right_radius;

    std::vector <std::vector <Point>> partitions (2);    // This will hold the left and right partitions
    partitions[0] = left_partition;
    partitions[1] = right_partition;

    return {partitions, {means, radii}};
}

std::pair <Node*, Node*> split_node(Node* curr_node){
    std::vector <float> split_direction = find_split_direction(curr_node->data);
    auto result = find_partitions(split_direction, curr_node->data);

    std::vector <std::vector <Point>> partitions = result.first;
    std::vector <std::vector <float>> means = result.second.first;
    std::vector <float> radii = result.second.second;

    if (means.size() == 0){
        return {NULL, NULL};
    }

    std::vector <Point> left_partition = partitions[0];
    std::vector <Point> right_partition = partitions[1];

    std::vector <float> left_mean = means[0];
    std::vector <float> right_mean = means[1];

    float left_radius = radii[0];
    float right_radius = radii[1];

    curr_node->split_direction = split_direction;
    curr_node->left_mean = left_mean;
    curr_node->left_radius = left_radius;
    curr_node->right_mean = right_mean;
    curr_node->right_radius = right_radius;

    Node* left_Node = new Node();
    left_Node->data = left_partition;
    left_Node->parent = curr_node;
    curr_node->left = left_Node;

    Node* right_Node = new Node();
    right_Node->data = right_partition;
    right_Node->parent = curr_node;
    curr_node->right = right_Node;

    return {left_Node, right_Node};
}

void build_tree(Node* curr_node, const int min_samples_per_Node){
    if (curr_node == NULL || curr_node->data.size() <= min_samples_per_Node){
        return;    // This is the stopping criteria
    }

    auto children = split_node(curr_node);
    Node* left_child = children.first;
    Node* right_child = children.second;
    
    build_tree(left_child, min_samples_per_Node);    // We recursively build the Ball tree
    build_tree(right_child, min_samples_per_Node);
}

std::vector <Point> traverse_tree(const std::vector <float> &query, const Node* curr_node, std::vector <Point> &neighbors, int k){
    if (curr_node == NULL){    // If root itself is NULL
        return neighbors;
    }

    if (curr_node->left == NULL || curr_node->right == NULL){    // This is a leaf Node
        std::vector <Point> new_neighbors;
        std::vector <std::pair <float, Point>> new_points;
        
        // We need to consider each point in this leaf Node
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
        float dist_to_left_ball = euclidean_distance(query, curr_node->left_mean) - curr_node->left_radius;
        float dist_to_right_ball = euclidean_distance(query, curr_node->right_mean) - curr_node->right_radius;

        if (dist_to_left_ball < dist_to_right_ball){
            std::vector <Point> new_neighbors = traverse_tree(query, curr_node->left, neighbors, k);
            float kth_neighbor_distance = euclidean_distance(query, new_neighbors[new_neighbors.size()-1].feature_vector);

            if (new_neighbors.size() < k || kth_neighbor_distance > dist_to_right_ball){
                new_neighbors = traverse_tree(query, curr_node->right, new_neighbors, k);
            }

            return new_neighbors;
        }
        else{
            std::vector <Point> new_neighbors = traverse_tree(query, curr_node->right, neighbors, k);
            float kth_neighbor_distance = euclidean_distance(query, new_neighbors[new_neighbors.size()-1].feature_vector);

            if (new_neighbors.size() < k || kth_neighbor_distance > dist_to_left_ball){
                new_neighbors = traverse_tree(query, curr_node->left, new_neighbors, k);
            }

            return new_neighbors;
        }
    }
}

}