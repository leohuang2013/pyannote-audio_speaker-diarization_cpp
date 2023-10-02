#include <iostream>
#include <vector>
#include <cstring>
#include "clustering.h"

int main() 
{
    std::vector<std::vector<double>> input = {
        {0, 0}, {0, 1}, {1, 0},
        {0, 4}, {0, 3}, {1, 4},
        {4, 0}, {3, 0}, {4, 1},
        {4, 4}, {3, 4}, {4, 3}};
    double cutoff = 1.1;

    auto res = cluster( input, cutoff );
    // Print the cluster assignments (T)
    std::cout << "Cluster Assignments:" << std::endl;
    for (size_t i = 0; i < input.size(); ++i) {
        std::cout << "Data Point " << i << " -> Cluster " << res[i] << std::endl;
    }


    return 0;
}

