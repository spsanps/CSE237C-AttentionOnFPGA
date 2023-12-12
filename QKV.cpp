#include "attention.h"
#include "dotProd.h"

// function to take in Q: 1 x DMODEL input, K (N x DMODEL) and and return max_index
void singleQK(hls::stream<data_t> &Q_stream, data_t K[N][DMODEL], int &max_index)
{

#pragma HLS dataflow
    data3_t result = 0;
    data3_t max_result = 0;
    max_index = 0;
    data_t Q[DMODEL];

    for(int j=DMODEL-1; j>=0; j--) Q[j] = Q_stream.read();

    for (int i = 0; i < N; i++)
    {
        #pragma HLS unroll off=true
        // Keep only the BITWIDTH len most significant bits
        // result = 0; // Initialize result for each dot product calculation
        // initialized in dotProd.cpp
        dotProd(Q, K[i], result);

        if (result > max_result)
        {
            max_result = result;
            max_index = i;
        }
    }
}

