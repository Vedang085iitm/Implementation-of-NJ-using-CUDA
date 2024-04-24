#include <iostream>
#include <fstream>
#include <vector>

int main() {
    int n;
    std::cout << "Enter the size of the matrix: ";
    std::cin >> n;

    std::vector<std::vector<int>> matrix(n, std::vector<int>(n, 0));

    // Fill the matrix
    for(int i = 0; i < n; i++) {
        for(int j = i+1; j < n; j++) {
            matrix[i][j] = matrix[j][i] = rand() % 100; // Fill with random values
        }
    }

    // Write the matrix to a file
    std::ofstream outfile("rand_input.txt");
    if (!outfile) {
        std::cerr << "Unable to open file";
        return 1;
    }

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            outfile << matrix[i][j] << " ";
        }
        outfile << "\n";
    }

    outfile.close();

    return 0;
}