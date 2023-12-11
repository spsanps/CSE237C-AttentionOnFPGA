#include <iostream>
#include "QKVProj.h"
#include <iomanip>
#include "attention.h"

// Function to print a 1D array (vector)
void printVector(const data_t vec[DMODEL]) {
    for (int i = 0; i < DMODEL; ++i) {
        std::cout << std::setw(5) << vec[i].to_uint() << " ";
    }
    std::cout << std::endl;
}

// Function to print a 2D array (matrix)
void printMatrix(const data_t mat[DMODEL][DMODEL]) {
    for (int i = 0; i < DMODEL; ++i) {
        printVector(mat[i]);
    }
}

// Function to print a 3D array (used for N tokens)
void print3DMatrix(const data_t mat[N][DMODEL]) {
    for (int i = 0; i < N; ++i) {
        std::cout << "Token " << i << ":" << std::endl;
        printVector(mat[i]);
    }
}


int main1() {
    data_t tokens[N][DMODEL];
    data_t weights[DMODEL][DMODEL];
    data_t outputs[N][DMODEL];

    // Initialize tokens and weights with some test data
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < DMODEL; ++j) {
            tokens[i][j] = (i + j - 5)%16; // Example values
        }
    }

    for (int i = 0; i < DMODEL; ++i) {
        for (int j = 0; j < DMODEL; ++j) {
            weights[i][j] = (i+3+j*2)%16; // Identity matrix as example
        }
    }

    // Perform projection
    static hls::stream<data_t> output_stream("output_stream");
    #pragma HLS STREAM variable=output_stream depth=DMODEL
    project_all(tokens, weights, output_stream);
    for (int i=0; i<N; i++) {
    	for (int j=0; j<DMODEL; j++) {
    		outputs[i][j] = output_stream.read();
    	}
    }


    // Print the results
    std::cout << "Tokens:" << std::endl;
    print3DMatrix(tokens);

    std::cout << "Weights:" << std::endl;
    printMatrix(weights);

    std::cout << "Outputs:" << std::endl;
    print3DMatrix(outputs);

    return 0;
}
