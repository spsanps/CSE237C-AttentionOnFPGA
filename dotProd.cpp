#include <ap_int.h>
#include "attention.h"

// placeholder code
// do 4 bit and in one cycle
void dotProd(data_t row1[DMODEL], data_t row2[DMODEL], data3_t &result)
{   
    result = 0;
    for (int i = 0; i < DMODEL; i++)
    {
        result += row1[i] * row2[i];
    }
}