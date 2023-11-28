#include <ap_int.h>

#define MATRIX_SIZE 128
#define NUM_CHANNELS 3
#define BITWIDTH 4

typedef ap_int<BITWIDTH> data_t;

void process_data(data_t input_matrix[MATRIX_SIZE][MATRIX_SIZE],
                  data_t input_channels[MATRIX_SIZE][NUM_CHANNELS],
                  data_t output_matrices[NUM_CHANNELS][MATRIX_SIZE][MATRIX_SIZE]) {
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE m_axi depth=65536 port=input_matrix offset=slave
#pragma HLS INTERFACE m_axi depth=65536 port=input_channels offset=slave
#pragma HLS INTERFACE m_axi depth=65536 port=output_matrices offset=slave

    // Your code to process the input data and generate the output data here

    // Example: Copy input_matrix to each output matrix
    for (int c = 0; c < NUM_CHANNELS; c++) {
        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
#pragma HLS PIPELINE
                output_matrices[c][i][j] = input_matrix[i][j];
            }
        }
    }
}
