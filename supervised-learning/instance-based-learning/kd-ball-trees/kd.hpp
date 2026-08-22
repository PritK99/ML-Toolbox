#ifndef KD_HPP
#define KD_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <utility>

class Node{
    public:
    Node* parent;
    Node* left;
    Node* right;

    int split_dim;
    float split_val;

    std::vector <std::vector <float>> data;

    Node(){
        parent = NULL;
        left = NULL;
        right = NULL;
    }
};

float find_split_value(const int feature_dimension, const std::vector <std::vector <float>> &data);
int find_split_dimension(const std::vector <std::vector <float>> &data);
std::pair <std::vector <std::vector <float>>, std::vector <std::vector <float>>> split_data(const int split_dim, const float split_value, const std::vector <std::vector <float>> &data);
std::pair <Node*, Node*> split_node(Node* curr_node);
void build_kd_tree(Node* curr_node, const int min_samples_per_node);

#endif