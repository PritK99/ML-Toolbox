#ifndef BALL_HPP
#define BALL_HPP

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include "../../../utils/distances.hpp"

namespace Ball{


struct Point{
    std::vector <float> feature_vector;
    float label;
};

class Node{
    public:
    Node* parent;
    Node* left;
    Node* right;

    std::vector <float> split_direction;
    std::vector <float> left_mean;
    std::vector <float> right_mean;
    float left_radius;
    float right_radius;

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

std::vector <float> find_split_direction(const std::vector <Point> &data);
std::pair <std::vector <std::vector <Point>>, std::pair <std::vector <std::vector <float>>, std::vector <float>>> find_partitions(const std::vector <float> &split_direction, const std::vector <Point> &data);
std::pair <Node*, Node*> split_node(Node* curr_node);
void build_tree(Node* curr_node, const int min_samples_per_node);
std::vector <Point> traverse_tree(const std::vector <float> &query, const Node* curr_node, std::vector <Point> &neighbors, int k);

}

#endif