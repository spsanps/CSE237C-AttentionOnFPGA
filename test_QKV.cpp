#include <iostream>
#include <iomanip>
#include "attention.h"
#include "QKV.h"

// Function to print a 1D array (vector)
void printVector(const data_t vec[DMODEL], const std::string &name) {
    std::cout << name << ": ";
    for (int i = 0; i < DMODEL; ++i) {
        std::cout << std::setw(5) << vec[i].to_uint() << " ";
    }
    std::cout << std::endl;
}

// Function to print a 2D array (matrix)
void printMatrix(const data_t mat[N][DMODEL], const std::string &name) {
    std::cout << name << ":" << std::endl;
    for (int i = 0; i < N; ++i) {
        printVector(mat[i], "Row " + std::to_string(i));
    }
}

int main4() {
    // Test data
    data_t Q[DMODEL];
    data_t K[N][DMODEL];
    int max_index = -1;

    // Initialize test data for Q and K
    for (int i = 0; i < DMODEL; ++i) {
        Q[i] = i; // Example values for Q
    }

    static hls::stream<data_t> Q_stream("tokens_stream");
    #pragma HLS STREAM variable=Q_stream depth=DMODEL

    for (int i=DMODEL; i >= 0; --i) Q_stream << Q[i];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < DMODEL; ++j) {
            K[i][j] = (2*(i -10) + j*5 + 3) % DMODEL; // Example values for K
        }
    }

    // Call the function under test
    singleQK(Q_stream, K, max_index);

    // Print the input and the result
    printVector(Q, "Q Vector");
    printMatrix(K, "K Matrix");
    std::cout << "Max Index: " << max_index << std::endl;

    // Optionally, add a check for the correctness of max_index

    return 0;
}
