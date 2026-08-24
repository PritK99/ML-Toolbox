#ifndef KD_HPP
#define KD_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <utility>
#include "../../../utils/distances.hpp"

namespace KD{

struct Point{
    std::vector <float> feature_vector;
    float label;
};

class Node{
    public:
    Node* parent;
    Node* left;
    Node* right;

    int split_dim;
    float split_val;

    std::vector <Point> data;

    Node(){
        parent = NULL;
        left = NULL;
        right = NULL;
    }

    ~Node(){
        delete left;
        delete right;
    }
};

int find_split_dimension(const std::vector <Point> &data);
float find_split_value(const int feature_dimension, const std::vector <Point> &data);
std::pair <std::vector <Point>, std::vector <Point>> split_data(const int split_dim, const float split_value, const std::vector <Point> &data);
std::pair <Node*, Node*> split_node(Node* curr_node);
void build_tree(Node* curr_node, const int min_samples_per_node);
std::vector <Point> traverse_tree(const std::vector <float> &query, const Node* curr_node, std::vector <Point> &neighbors, int k);

}

#endif