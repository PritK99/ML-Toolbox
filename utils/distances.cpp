#include "distances.hpp"

float euclidean_distance(std::vector <float> v1, std::vector <float> v2){
    float distance = 0;

    for (int i = 0; i < v1.size(); i++){
        float diff = (v1[i] - v2[i]);
        distance += diff*diff;
    }

    distance = std::sqrt(distance);

    return distance;
}

float manhattan_distance(std::vector <float> v1, std::vector <float> v2){
    float distance = 0;

    for (int i = 0; i < v1.size(); i++){
        float diff = (v1[i] - v2[i]);
        distance += std::abs(diff);
    }

    return distance;
}

float cosine_distance(std::vector <float> v1, std::vector <float> v2){
    float numerator = 0;
    float v1_norm = 0;
    float v2_norm = 0;

    for (int i = 0; i < v1.size(); i++){
        numerator += v1[i]*v2[i];
        v1_norm += v1[i]*v1[i];
        v2_norm += v2[i]*v2[i];
    }

    float denominator = std::sqrt(v1_norm*v2_norm);

    if (denominator < 1e-5){    // Very small denominators
        return 1;
    }

    float cosine_similarity = numerator / denominator;
    return 1 - cosine_similarity;
}