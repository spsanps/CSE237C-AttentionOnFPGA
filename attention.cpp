#include <ap_int.h>

#define SEQ_LENGTH 128
#define BITWIDTH 4

typedef ap_int<BITWIDTH> data_t;

void attention(data_t input_tokens[SEQ_LENGTH],
               data_t output_tokens[SEQ_LENGTH]) {
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE m_axi depth=65536 port=input_tokens offset=slave
#pragma HLS INTERFACE m_axi depth=65536 port=output_tokens offset=slave

    // Your attention mechanism logic here

    // Example: Pass input tokens to output directly (replace with actual attention logic)
    for (int i = 0; i < SEQ_LENGTH; i++) {
#pragma HLS PIPELINE
        output_tokens[i] = input_tokens[i];
    }
}
