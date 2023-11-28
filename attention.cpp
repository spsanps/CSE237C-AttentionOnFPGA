#include <ap_int.h>
#include "attention.h"
#include "QKVProj.cpp"
#include "QKV.cpp"

// taken tokens (N x DMODEL), 3 weights (DMODEL x DMODEL) and return (N x DMODEL) output

void attention(data_t tokens[N][DMODEL],
               data_t weightsQ[DMODEL][DMODEL],
               data_t weightsK[DMODEL][DMODEL],
               data_t weightsV[DMODEL][DMODEL],
               data_t output[N][DMODEL])
{
    data_t K[N][DMODEL];
    // compute K
    project_all(tokens, weightsK, K);

    for (int i = 0; i < N; i++)
    {
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