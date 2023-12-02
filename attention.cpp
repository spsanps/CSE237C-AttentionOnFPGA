#include <ap_int.h>
#include "attention.h"
#include "QKVProj.h"
#include "QKV.h"

#if DMODEL == 16
#include "weights16.h"
#elif DMODEL == 32
#include "weights32.h"
#elif DMODEL == 64
#include "weights64.h"
#else
#error "Unsupported DMODEL value"
#endif

// taken tokens (N x DMODEL), 3 weights (DMODEL x DMODEL) and return (N x DMODEL) output

void attention(data_t tokens[N][DMODEL],
               data_t weightsQ[DMODEL][DMODEL],
               data_t weightsK[DMODEL][DMODEL],
               data_t weightsV[DMODEL][DMODEL],
               data_t output[N][DMODEL])
{
    data_t K[N][DMODEL];

    printf("Beginning the attention");

    static hls::stream<data_t> tokens_stream("tokens_stream");
    #pragma HLS STREAM variable=tokens_stream depth=N*DMODEL

    for (int i=N; i > 0; --i) {
    	for (int j=DMODEL; j>0; --j) {
    		tokens_stream << tokens[i-1][j-1];
    	}
    }

    printf("Populated tokens stream, beginning projection");

    // compute K
    project_all(tokens_stream, weightsK, K);

    for (int i = 0; i < N; i++)
    {
        #pragma HLS unroll off=true
        // compute Q
        data_t Q[DMODEL];
        project(tokens[i], weightsQ, Q);

        // compute max_index
        int max_index = 0;
        singleQK(Q, K, max_index);

        // compute V
        project(tokens[max_index], weightsV, output[i]);
    }
}
