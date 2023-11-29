#include "attention.h"

data2_t mult_4bit(data_t a, data_t b)
{
	return a * b;
}

void dotProd(data_t row1[DMODEL], data_t row2[DMODEL], data3_t &result)
{   
    result = 0;
#pragma HLS DATAFLOW
    for (int i = 0; i < DMODEL; i++)
    {
#pragma HLS UNROLL
#pragma HLS ARRAY_PARTITION variable=row1 type=complete
#pragma HLS ARRAY_PARTITION variable=row2 type=complete
//        result += row1[i] * row2[i];
    	result += mult_4bit(row1[i], row2[i]);
    }
}
