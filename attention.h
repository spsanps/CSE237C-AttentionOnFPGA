
#ifndef ATTENTION_H
#define ATTENTION_H

#include <ap_int.h>


// Define the bit width
// Width of each value in the token vector
#define BITWIDTH 4

// Define the value of N
// Number of tokens (each token is a vector)
#define N 16

// Define the value of DMODEL
// length of a single token vector
// Ie. each token is a vector of length DMODEL
#define DMODEL 16

// define 4 bit data type unsigned
typedef ap_uint<BITWIDTH> data_t;

// intermediate data type
typedef ap_uint<BITWIDTH*2> data2_t;

#endif // ATTENTION_H
