
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
// only 16, 32 or 64 are supported
#define DMODEL 16
// Include the appropriate weights file based on DMODEL value
#if DMODEL == 16
#include "weights16.h"
#elif DMODEL == 32
#include "weights32.h"
#elif DMODEL == 64
#include "weights64.h"
#else
#error "Unsupported DMODEL value"
#endif

// define 4 bit data type unsigned
typedef ap_uint<BITWIDTH> data_t;

// intermediate data type
typedef ap_uint<BITWIDTH * 2> data2_t;

#endif // ATTENTION_H
