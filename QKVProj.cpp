#include <ap_int.h>
#include "attention.h"
#include "dotProd.h"

void project(data_t token[DMODEL],
             data_t weight[DMODEL][DMODEL],
             data_t output[DMODEL])
			 // hls::stream<data_t> &output)
{
    data3_t result = 0;
    data3_t max_result = 0;
    data3_t temp_results[DMODEL];
    data3_t min_result = (1ULL << BITWIDTH3) - 1;
    for (int i = 0; i < DMODEL; i++)
    {
        // #pragma HLS unroll off=true
        #pragma HLS pipeline II = N
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
        #pragma HLS pipeline II = DMODEL
        // unroll completely
        // #pragma HLS UNROLL
        // output << ((temp_results[i] - min_result) >> shift_amount);
        output[i] = ((temp_results[i] - min_result) >> shift_amount);
    }

}

void project_all(data_t tokens[N][DMODEL],
                 data_t weights[DMODEL][DMODEL],
                 data_t outputs[N][DMODEL])
				 // hls::stream<data_t> &outputs)
{
    for (int i = 0; i < N; i++)
    {
        //#pragma HLS unroll off=true
        #pragma HLS pipeline II = (N*N*2)

//    	data_t outputs_arr[DMODEL];
        project(tokens[i], weights, outputs[i]);


//       for (int j=0; j<DMODEL; j++) outputs << outputs_arr[j];

    }
}
