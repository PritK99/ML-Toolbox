#include <cstdio>
#include <vector>

int main() {

    std::vector<double> y = {1,4,9,16,25};

    FILE* gnuplot = popen("gnuplot -persistent", "w");

    fprintf(gnuplot, "set title 'Quadratic Growth'\n");
    fprintf(gnuplot, "set xlabel 'x'\n");
    fprintf(gnuplot, "set ylabel 'y'\n");

    fprintf(gnuplot, "plot '-' with linespoints title 'y = x^2'\n");

    for (int i = 0; i < y.size(); i++) {
        fprintf(gnuplot, "%d %f\n", i, y[i]);
    }

    fprintf(gnuplot, "e\n");

    pclose(gnuplot);

    return 0;
}