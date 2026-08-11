#ifndef CSV_HPP
#define CSV_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <array>
#include <numeric> 

std::pair<std::vector <std::string>, std::vector <std::vector <std::string>>> read_csv(const std::string& csv_path);
std::vector<std::pair<std::vector <std::vector <float>>, std::vector<float>>> split_data(std::vector <std::vector <float>>& data, std::vector<float> &labels, const float val_ratio, const float test_ratio);
std::vector <std::vector <std::vector <float>>> normalize_data(const std::vector <std::vector <float>> &train_data, const std::vector <std::vector <float>> &val_data, const std::vector <std::vector <float>> &test_data);

#endif