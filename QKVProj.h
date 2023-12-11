#ifndef QKVPROJ_H
#define QKVPROJ_H

#include <ap_int.h>
#include "attention.h"
#include "dotProd.h"

void project_stream(data_t token[DMODEL],
             data_t weight[DMODEL][DMODEL],
             hls::stream<data_t> &output);

void project(data_t token[DMODEL],
             data_t weight[DMODEL][DMODEL],
             data_t output[DMODEL]);

void project_all(hls::stream<data_t> &tokens,
                 data_t weights[DMODEL][DMODEL],
                 hls::stream<data_t> &outputs);

#endif // QKVPROJ_H
