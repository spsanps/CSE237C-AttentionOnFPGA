#ifndef ATTENTION_H
#define ATTENTION_H

#include <ap_int.h>

// Define the bit width
// Bit Width of each value in the token vector
#define BITWIDTH 4

// Define the value of N
// Number of tokens (each token is a vector)
#define N 16
#define LOG2N 4 // Should be log2(DMODEL) ?
// used in dot product
// dot product used for weight DMODEL x DMODEL
// dot product used for Q, K which is also DMODEL x DMODEL

// define 4 bit data type unsigned
typedef ap_uint<BITWIDTH> data_t;

// intermediate data types
#define BITWIDTH2 (BITWIDTH * 2)
#define BITWIDTH3 (BITWIDTH * 2 + LOG2N)

// multiplications of 4 bit values can be 8 bits
typedef ap_uint<BITWIDTH2> data2_t;
// additions of 8 bit values can overflow 8 bits
typedef ap_uint<BITWIDTH3> data3_t;

// Define the value of DMODEL
// length of a single token vector
// Ie. each token is a vector of length DMODEL
// only 16, 32 or 64 are supported
#define DMODEL 16

void attention(data_t tokens[N][DMODEL],
               data_t weightsQ[DMODEL][DMODEL],
               data_t weightsK[DMODEL][DMODEL],
               data_t weightsV[DMODEL][DMODEL],
               data_t output[N][DMODEL]);

// Include the appropriate weights file based on DMODEL value
//#if DMODEL == 16
//#include "weights16.h"
//#elif DMODEL == 32
//#include "weights32.h"
//#elif DMODEL == 64
//#include "weights64.h"
//#else
//#error "Unsupported DMODEL value"
//#endif

#endif // ATTENTION_H
