
#ifndef QKV_H
#define QKV_H

#include "attention.h"
#include "dotProd.h"

void singleQK(data_t Q[DMODEL], data_t K[N][DMODEL], int &max_index);

#endif // QKV_H
