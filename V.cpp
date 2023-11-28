#include <ap_int.h>

#define NUM_VS 128 // Number of V vectors
#define BITWIDTH 4 // Bitwidth of input data

typedef ap_int<BITWIDTH> data_t;

void apply_vector(data_t qk_result[NUM_VS], data_t v[NUM_VS], data_t output[NUM_VS]) {
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE m_axi depth=65536 port=qk_result offset=slave
#pragma HLS INTERFACE m_axi depth=65536 port=v offset=slave
#pragma HLS INTERFACE m_axi depth=65536 port=output offset=slave

    // Your code for applying vector V to QK result here

    // Example: Element-wise multiplication of QK result and V (replace with actual logic)
    for (int i = 0; i < NUM_VS; i++) {
#pragma HLS PIPELINE
        output[i] = qk_result[i] * v[i];
    }
}
