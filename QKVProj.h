#ifndef QKVPROJ_H
#define QKVPROJ_H

#include <ap_int.h>
#include "attention.h"
#include "dotProd.h"

void project(data_t token[DMODEL],
             data_t weight[DMODEL][DMODEL],
                data_t output[DMODEL]);
             // hls::stream<data_t> &output);

void project_all(data_t tokens[N][DMODEL],
                 data_t weights[DMODEL][DMODEL],
                 data_t outputs[N][DMODEL]);
                 // hls::stream<data_t> &outputs);

#endif // QKVPROJ_H
