
#ifndef DOTPROD_H
#define DOTPROD_H

#include "attention.h"

data2_t mult_4bit(data_t a, data_t b);

void dotProd(data_t row1[DMODEL], data_t row2[DMODEL], data3_t &result);

#endif // DOTPROD_H
