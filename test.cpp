#include <iostream>
#include <ap_int.h>
#include "attention.h"
// #include "weights16.h"


// Helper function to initialize matrices with some test data
void initialize_matrix(data_t matrix[N][DMODEL]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < DMODEL; j++) {
            matrix[i][j] = (data_t)(i + 5*j); // Simple sequential initialization
        }
    }
}

// Helper function to print matrices
void print_matrix_simple(const char* name, data_t matrix[N][DMODEL]) {
    std::cout << name << ":\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < DMODEL; j++) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << "\n";
    }
}

int main5() {
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

    // intialize weights from Q_W, K_W, V_W

    /*for (int i = 0; i < DMODEL; i++) {
        for (int j = 0; j < DMODEL; j++) {
            weightsQ[i][j] = (data_t)Q_W[i][j];
            weightsK[i][j] = (data_t)K_W[i][j];
            weightsV[i][j] = (data_t)V_W[i][j];
        }
    }*/
    // Print the initialized matrices
    print_matrix_simple("Tokens", tokens);
    print_matrix_simple("WeightsQ", weightsQ);
    print_matrix_simple("WeightsK", weightsK);
    print_matrix_simple("WeightsV", weightsV);

    static hls::stream<data_t> tokens_stream("tokens_stream");
    #pragma HLS STREAM variable=tokens_stream depth=N*DMODEL

    for (int i=N; i > 0; --i) {
    	for (int j=DMODEL; j>0; --j) {
    		tokens_stream << tokens[i-1][j-1];
    	}
    }

    static hls::stream<data_t> output_stream("output_stream");
    #pragma HLS STREAM variable=output_stream depth=DMODEL
    // Call the attention function
    attention(tokens_stream, weightsQ, weightsK, weightsV, output_stream);
    for (int i=0; i<N; i++) {
    	for (int j=0; j<DMODEL; j++) {
    		output[i][j] = output_stream.read();
    	}
    }

    // Print the output matrix
    print_matrix_simple("Output", output);

    return 0;
}
