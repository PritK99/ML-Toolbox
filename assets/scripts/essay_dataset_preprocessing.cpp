/*
The issue with raw essay dataset is that each essay spans over many lines. Thus, the way we read csv does not hold since we expect each line to have a new row.
Hence, we will have to condense each essay o a single row.
*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric> 

void read_file(const std::string& raw_file_path, const std::string& save_path){
    std::ifstream file(raw_file_path);
    std::ofstream out(save_path);

    if (!file.is_open()) {
        std::cerr << "Failed to open file.\n";
        return;
    }

    if (!out.is_open()) {
        std::cerr << "Failed to create output file.\n";
        return;
    }

    std::string line;
    std::vector <std::string> rows;
    std::string row;
    int count = 0;

    // The first row is assumed to have column names
    if (std::getline(file, line)) {
        row += line;
        rows.push_back(row);
        row = "";
    }
    
    while (std::getline(file, line)) {
        
        int last_char = line[line.size() - 1] - '0';
        bool is_label = (last_char >= 0 && last_char <= 6) ? true : false;
        row += line;

        // This indicates end of an essay
        if (line[line.size() - 2] == ',' && is_label){
            count += 1;
            rows.push_back(row);
            row = "";
        }
    }

    std::cout << "Total numbers of essays: " << count << std::endl;

    for (const auto& r : rows) {
        out << r << "\n";
    }

    file.close();
    out.close();
}

int main(){
    std::string file_path = "../../data/raw/raw_essays.csv";
    std::string save_path = "../../data/essays.csv";

    read_file(file_path, save_path);
}