#include <ap_int.h>

#define Q_BITWIDTH 4
#define K_BITWIDTH 4
#define NUM_K_TOKENS 128

typedef ap_int<Q_BITWIDTH> q_token_t;
typedef ap_int<K_BITWIDTH> k_token_t;

void process_query(q_token_t query, k_token_t k_tokens[NUM_K_TOKENS], int matching_indices[NUM_K_TOKENS]) {
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=query
#pragma HLS INTERFACE m_axi depth=65536 port=k_tokens offset=slave
#pragma HLS INTERFACE m_axi depth=65536 port=matching_indices offset=slave

    // Your code to process the query token and match it against K tokens here

    // Example: Find matching K tokens and store their indices
    for (int i = 0; i < NUM_K_TOKENS; i++) {
#pragma HLS PIPELINE
        if (query == k_tokens[i]) {
            matching_indices[i] = 1; // Set to 1 if there is a match
        } else {
            matching_indices[i] = 0; // Set to 0 if there is no match
        }
    }
}
