#include <ap_int.h>
#include "attention.h"
#include "QKVProj.h"
#include "QKV.h"

// taken tokens (N x DMODEL), 3 weights (DMODEL x DMODEL) and return (N x DMODEL) output

void print_vector(data_t vector[DMODEL])
{
    for (int i = 0; i < DMODEL; i++)
    {
#pragma HLS unroll off = true
        std::cout << vector[i] << " ";
    }
    std::cout << "\n";
}

void print_matrix(data_t matrix[N][DMODEL])
{
    for (int i = 0; i < N; i++)
    {
#pragma HLS unroll off = true
        print_vector(matrix[i]);
    }
}

void attention(data_t tokens[N][DMODEL],
               data_t weightsQ[DMODEL][DMODEL],
               data_t weightsK[DMODEL][DMODEL],
               data_t weightsV[DMODEL][DMODEL],
               data_t output[N][DMODEL])
{
    data_t K[N][DMODEL];
    data_t temp_token[DMODEL];

    // compute K
    project_all(tokens, weightsK, K);
    // print the K matrix
    // std::cout << "K:\n";
    // (K);

    for (int i = 0; i < N; i++)
    {
#pragma HLS unroll off = true
        // compute Q
        data_t Q[DMODEL];
        for (int j = 0; j < DMODEL; j++)
        {
            // Unroll completely
#pragma HLS unroll
            temp_token[j] = tokens[i][j];
        }
        project(temp_token, weightsQ, Q);
        // print_vector(Q);

        // compute max_index
        int max_index = 0;
        singleQK(Q, K, max_index);
        // std::cout << "max_index: " << max_index << "\n";

        // compute V
        for (int j = 0; j < DMODEL; j++)
        {
            // Unroll completely
#pragma HLS unroll
            temp_token[j] = tokens[max_index][j];
        }
        project(temp_token, weightsV, output[i]);
    }
}
