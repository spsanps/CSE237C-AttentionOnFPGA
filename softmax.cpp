#include <ap_int.h>
#include <hls_math.h>

#define NUM_VS 128 // Number of V vectors
#define BITWIDTH 4 // Bitwidth of input data

typedef ap_int<BITWIDTH> data_t;

void softmax(data_t input[NUM_VS], data_t output[NUM_VS]) {
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE m_axi depth=65536 port=input offset=slave
#pragma HLS INTERFACE m_axi depth=65536 port=output offset=slave

    // Your code for softmax operation here

    // Example: Softmax calculation (replace with your actual implementation)
    data_t sum = 0;
    for (int i = 0; i < NUM_VS; i++) {
#pragma HLS PIPELINE
        sum += hls::exp(input[i]);
    }
    for (int i = 0; i < NUM_VS; i++) {
#pragma HLS PIPELINE
        output[i] = hls::exp(input[i]) / sum;
    }
}
