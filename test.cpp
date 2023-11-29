#include <iostream>
#include <ap_int.h>
#include "attention.h"

// Helper function to initialize matrices with some test data
void initialize_matrix(data_t matrix[N][DMODEL]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < DMODEL; j++) {
            matrix[i][j] = (data_t)(i * DMODEL + j); // Simple sequential initialization
        }
    }
}

// Helper function to print matrices
void print_matrix(const char* name, data_t matrix[N][DMODEL]) {
    std::cout << name << ":\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < DMODEL; j++) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    data_t tokens[N][DMODEL];
    data_t weightsQ[DMODEL][DMODEL];
    data_t weightsK[DMODEL][DMODEL];
    data_t weightsV[DMODEL][DMODEL];
    data_t output[N][DMODEL];

    // Initialize matrices with test data
    initialize_matrix(tokens);
    initialize_matrix(weightsQ);
    initialize_matrix(weightsK);
    initialize_matrix(weightsV);

    // Print the initialized matrices
    print_matrix("Tokens", tokens);
    print_matrix("WeightsQ", weightsQ);
    print_matrix("WeightsK", weightsK);
    print_matrix("WeightsV", weightsV);

    // Call the attention function
    attention(tokens, weightsQ, weightsK, weightsV, output);

    // Print the output matrix
    print_matrix("Output", output);

    return 0;
}
