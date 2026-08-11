#ifndef DISTANCES_HPP
#define DISTANCES_HPP

#include <iostream>
#include <vector>
#include <cmath>

float euclidean_distance(std::vector <float> v1, std::vector <float> v2);
float manhattan_distance(std::vector <float> v1, std::vector <float> v2);
float cosine_distance(std::vector <float> v1, std::vector <float> v2);

#endif