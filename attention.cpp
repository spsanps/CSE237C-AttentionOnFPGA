#include <ap_int.h>
#include "attention.h"
#include "QKVProj.h"
#include "QKV.h"
#include "weights16.h"

// taken tokens (N x DMODEL), 3 weights (DMODEL x DMODEL) and return (N x DMODEL) output
void attention_nostream(data_t tokens[N][DMODEL],
                        data_t weightsQ[DMODEL][DMODEL],
                        data_t weightsK[DMODEL][DMODEL],
                        data_t weightsV[DMODEL][DMODEL],
                        data_t output[N][DMODEL])
{

#pragma HLS ARRAY_PARTITION variable = weightsQ complete dim = 2
#pragma HLS ARRAY_PARTITION variable = weightsK complete dim = 2
#pragma HLS ARRAY_PARTITION variable = weightsV complete dim = 2
#pragma HLS ARRAY_PARTITION variable = tokens complete dim = 2
#pragma HLS ARRAY_PARTITION variable = output complete dim = 2

    data_t K[N][DMODEL];
    data_t temp_token[DMODEL];

    // compute K
    project_all(tokens, weightsK, K);

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

        // compute max_index
        int max_index = 0;
        singleQK(Q, K, max_index);

        // compute V
        for (int j = 0; j < DMODEL; j++)
        {
#pragma HLS unroll
            temp_token[j] = tokens[max_index][j];
        }
        project(temp_token, weightsV, output[i]);
    }
}

void attention(hls::stream<axis_t> &in,
               hls::stream<axis_t> &out,
               unsigned int control)
{
#pragma HLS INTERFACE s_axilite port = control bundle = control
#pragma HLS INTERFACE axis port = in depth = SIZE
#pragma HLS INTERFACE axis port = out depth = SIZE

#pragma HLS ARRAY_PARTITION variable = Q_W complete dim = 2
#pragma HLS ARRAY_PARTITION variable = K_W complete dim = 2
#pragma HLS ARRAY_PARTITION variable = V_W complete dim = 2

    // Stage 1: Read all input data into temporary arrays
    int i = 0;

    axis_t inp_temp[SIZE];
    int temp_data = 0;

    data_t tokens[N][DMODEL];
    data_t output[N][DMODEL];

    // Read tokens
    while (i < SIZE)
    {
        inp_temp[i] = in.read();
        temp_data = inp_temp[i].data;
        tokens[i / DMODEL][i % DMODEL] = (data_t)temp_data;
        i++;
    }
    // Stage 2: Compute the attention
    attention_nostream(tokens, Q_W, K_W, V_W, output);

    // Stage 3: Write the output
    i = 0;
    while (i < SIZE)
    {
        temp_data = (int)output[i / DMODEL][i % DMODEL];
        inp_temp[i].data = temp_data;
        out.write(inp_temp[i]);
        i++;
    }
}