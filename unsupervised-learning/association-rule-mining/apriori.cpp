#include <unordered_map>
#include "../../utils/csv.hpp"

int main(){
    std::string students_csv_path = "../../data/iiith_course_preferences.csv";
    std::string courses_csv_path = "../../data/iiith_course_mapping.csv";

    auto result = read_csv(students_csv_path);
    std::vector<std::string> students_column_names = result.first;
    std::vector<std::vector<std::string>> students_raw_data = result.second;

    result = read_csv(courses_csv_path);
    std::vector<std::string> courses_column_names = result.first;
    std::vector<std::vector<std::string>> courses_raw_data = result.second;

    // for (int i = 0; i < students_column_names.size(); i++)
    // {
    //     std::cout << students_column_names[i] << std::endl;
    // }

    // for (int i = 0; i < courses_column_names.size(); i++)
    // {
    //     std::cout << courses_column_names[i] << std::endl;
    // }

    std::unordered_map <std::string, int> itemset_1;
    for (int i = 0; i < students_raw_data.size(); i++)
    {
        std::string courses_taken_string = students_raw_data[i][1];
        std::cout << courses_taken_string << std::endl;
        // itemset_1[courses_raw_data[i][0]] += 1;
    }

    // for (const auto &i: itemset_1)
    // {
    //     std::cout << i.first << " " << i.second << std::endl;
    // }

    return 0;
}