#include <ap_int.h>
#include "attention.h"
#include "QKVProj.h"
#include "QKV.h"

// taken tokens (N x DMODEL), 3 weights (DMODEL x DMODEL) and return (N x DMODEL) output

void print_vector(data_t vector[DMODEL])
{
    //    for (int i = 0; i < DMODEL; i++)
    //    {
    // #pragma HLS unroll off = true
    //        std::cout << vector[i] << " ";
    //    }
    //    std::cout << "\n";
}

void print_matrix(data_t matrix[N][DMODEL])
{
    for (int i = 0; i < N; i++)
    {
        print_vector(matrix[i]);
    }
}

void attention(
    // hls::stream<data_t> &tokens_stream,
    data_t tokens[N][DMODEL],
    data_t weightsQ[DMODEL][DMODEL],
    data_t weightsK[DMODEL][DMODEL],
    data_t weightsV[DMODEL][DMODEL],
    data_t output[N][DMODEL])
// hls::stream<data_t> &output_stream)
{

    #pragma HLS pipeline II = (N * N*N)

#pragma HLS ARRAY_PARTITION variable = weightsQ complete dim = 2
#pragma HLS ARRAY_PARTITION variable = weightsK complete dim = 2
#pragma HLS ARRAY_PARTITION variable = weightsV complete dim = 2

    /*data_t tokens[N][DMODEL];

    for (int i = N; i > 0; --i)
    {
        for (int j = DMODEL; j > 0; --j)
        {
            tokens[i - 1][j - 1] = tokens_stream.read();
        }
    }*/

    data_t temp_token[DMODEL];

    // static hls::stream<data_t> K("K");
    // #pragma HLS STREAM variable=K depth=16

    data_t K[N][DMODEL];

    // compute K
    project_all(tokens, weightsK, K);

    /*data_t single_qk_k[N][DMODEL];

    for (int i=N; i > 0; --i) {
        for (int j=DMODEL; j>0; --j) {
            single_qk_k[i-1][j-1] = K.read();
        }
    }*/

    for (int i = 0; i < N; i++)
    {
// #pragma HLS unroll off = true
#pragma HLS pipeline II = (N*N*4)
        // compute Q
        data_t Q[DMODEL];
        for (int j = 0; j < DMODEL; j++)
        {
            // Unroll completely
#pragma HLS unroll
            temp_token[j] = tokens[i][j];
        }
        // hls::stream<data_t> Q_stream("Q_stream");
        //  #pragma HLS STREAM variable=Q_stream depth=16
        //  data_t Q[DMODEL];
        project(temp_token, weightsQ, Q);

        // save Q_stream to Q
        /* for (int j=DMODEL; j>0; --j) {
            Q[j-1] = Q_stream.read();
        } */
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

        //data_t output[DMODEL];

        project(temp_token, weightsV, output[i]);

        // save output to output_stream
        // for (int j = 0; j < DMODEL; j++)
        //{
        //    output_stream << output[j];
        //}
    }
}
