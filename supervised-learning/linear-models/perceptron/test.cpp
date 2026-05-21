#include <fstream>
#include <vector>

int main() {

    std::vector<double> y = {1,4,9,16,25};

    std::ofstream file("data.txt");

    for (int i = 0; i < y.size(); i++)
        file << i << " " << y[i] << "\n";

    file.close();

    system("gnuplot -p -e \"plot 'data.txt' with lines\"");

    return 0;
}