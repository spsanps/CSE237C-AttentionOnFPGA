#include "attention.h"

// function to take in Q: 1 x DMODEL input, K (N x DMODEL) and V (N x DMODEL) and return 1 x DMODEL output

void singleQKV(data_t Q[DMODEL], data_t K[N]][DMODEL], data_t V[N][DMODEL], data_t output[DMODEL])
{
    // keep track of max index
    int max_index = 0;
    // keep track of max value
    data2_t max_value = 0;
    // code to do QK and max index
    // we don't need intermediate QK matrix, so we can just do it in one loop
    for (int i = 0; i < N; i++)
    {
        // calculate QK
        data2_t QK = 0;
        for (int j = 0; j < DMODEL; j++)
        {
            QK += Q[j] * K[i][j];
        }
        // check if QK is greater than max value
        if (QK > max_value)
        {
            max_value = QK;
            max_index = i;
        }
    }

    // softmax is just approximated with max
    // place the max index of V into output
    for (int i = 0; i < DMODEL; i++)
    {
        output[i] = V[max_index][i];
    }
}