#ifndef CSV_HPP
#define CSV_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

std::pair<std::vector <std::string>, std::vector <std::vector <std::string>>> read_csv(const std::string& csv_path);
std::vector<std::vector <std::vector <std::string>>> split_data(std::vector <std::vector <std::string>>& data, float val_ratio, float test_ratio);

#endif