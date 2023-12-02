#ifndef QKVPROJ_H
#define QKVPROJ_H

#include <ap_int.h>
#include "attention.h"
#include "dotProd.h"

void project(data_t token[DMODEL],
             data_t weight[DMODEL][DMODEL],
             data_t output[DMODEL]);

// void project_all(hls::stream<data_t> &tokens
// void project_all(data_t tokens[N][DMODEL]

void project_all(hls::stream<data_t> &tokens,
                 data_t weights[DMODEL][DMODEL],
                 data_t outputs[N][DMODEL]);

#endif // QKVPROJ_H
