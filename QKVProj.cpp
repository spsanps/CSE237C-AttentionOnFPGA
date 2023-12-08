#include <ap_int.h>
#include "attention.h"
#include "dotProd.h"

// take token (1 x DMODEL), weight (DMODEL x DMODEL) and return (1 x DMODEL)output
void project(data_t token[DMODEL],
             data_t weight[DMODEL][DMODEL],
             data_t output[DMODEL])
{
    data3_t result = 0;
    data3_t max_result = 0;
    data3_t temp_results[DMODEL];
    data3_t min_result = (1ULL << BITWIDTH3) - 1;
    for (int i = 0; i < DMODEL; i++)
    {
        #pragma HLS unroll off=true
        // Keep only the BITWIDTH len most significant bits
        dotProd(token, weight[i], result);

        // normalize layer
        max_result = result > max_result ? result : max_result;
        min_result = result < min_result ? result : min_result;

        // this is bad actually, should be done in dotProd
        temp_results[i] = result; // Corrected syntax



    }
    // Normalize
    // find shift amount
    int shift_amount = 0;
    max_result = max_result - min_result;
    while (max_result > 16)
    {
        max_result = max_result >> 1;
        shift_amount++;
    }
    // shift and quantize
    for (int i = 0; i < DMODEL; i++)
    {
        // unroll completely
        #pragma HLS UNROLL
        output[i] = (temp_results[i] - min_result) >> shift_amount;
    }

}

void project_all(hls::stream<data_t> &tokens,
                 data_t weights[DMODEL][DMODEL],
				 hls::stream<data_t> &outputs)
{
    for (int i = 0; i < N; i++)
    {
        #pragma HLS unroll off=true

    	data_t tokens_arr[DMODEL];
    	for (int j=0; j<DMODEL; j++) tokens_arr[j] = tokens.read();

    	data_t outputs_arr[DMODEL];
        project(tokens_arr, weights, outputs_arr);

       for (int j=0; j<DMODEL; j++) outputs << outputs_arr[j];

    }
}
