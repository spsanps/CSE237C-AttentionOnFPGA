#include <ap_int.h>
#include "attention.h"
#include "dotProd.cpp"

// take token (1 x DMODEL), weight (DMODEL x DMODEL) and return (1 x DMODEL)output
void project(data_t token[DMODEL],
             data_t weight[DMODEL][DMODEL],
             data_t output[DMODEL])
{
    data3_t result = 0;
    for (int i = 0; i < DMODEL; i++)
    {
        #pragma HLS unroll off=true
        // Keep only the BITWIDTH len most significant bits
        dotProd(token, weight[i], result);

        output[i] = result >> (BITWIDTH3 - BITWIDTH); // Corrected syntax
    }
}

void project_all(data_t tokens[SEQ_LENGTH][DMODEL],
                 data_t weights[DMODEL][DMODEL],
                 data_t outputs[SEQ_LENGTH][DMODEL])
{
    for (int i = 0; i < SEQ_LENGTH; i++)
    {
        #pragma HLS unroll off=true
        project(tokens[i], weights, outputs[i]);
    }
}