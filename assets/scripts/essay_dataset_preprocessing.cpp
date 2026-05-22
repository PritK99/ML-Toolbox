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

void read_file(const std::string& file_path){
    std::ifstream file(file_path);

    if (!file.is_open()) {
        std::cerr << "Failed to open file.\n";
    }

    std::string line;
    std::vector <std::string> rows;
    std::string row;
    while (std::getline(file, line)) {
        row += line;
        std::cout << line[line.size() - 1] << std::endl;

        // All this does is to check if the line ends with ,1 to ,6
        // if ((line[line.size() - 2] == ',') && (line[line.size() - 1] - '0' >= 1 && line[line.size() - 1] - '0' <= 6)){
        //     rows.push_back(row);
        //     row = "";
        // }
    }

    file.close();
}

int main(){
    std::string file_path = "../../data/essays.csv";

    read_file(file_path);
}