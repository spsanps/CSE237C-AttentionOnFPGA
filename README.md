# CSE237C-Project

Take N x DMODEL size input, and output N x DMODEL size output.

Following steps are taken:

Input is projected using weights of size DMODEL x 3*DMODEL, and then split into 3 parts of size N x DMODEL. (Q, K, V)

Each Q (1 X DMODEL) is the multiplied with K.T (DMODEL x  N) and then max is taken along the last dimension. This gives a max index value in a vector of size 1 x N. This is then used to index the V (N x DMODEL) to get the output (1 x DMODEL). 

*Softmax is done with max* : Please check softmaxVsMax.ipynb for simulation of softmax vs max. We identify that there is a ~ 8.5% error in the output of max compared to softmax. The error is not higher as this is 4 bit quantized.

This is done for all the N inputs, and the output is of size N x DMODEL.

## Code Structure

### attention.h
Header file.

### 4bit.cpp
Contains efficient implementation of 4 bit quantized multiplication with 8 bit output.

### rowColMul.cpp
Contains efficient implementation of row x col multiplication with 8 bit output to be done in 1 cycle.

### QKVProj.cpp
Takes input tokens (1 X DMODEL) and projects them to Q, K, V (1 X DMODEL) using weights of size DMODEL x 3*DMODEL. (1 Token projection per cycle)


### QKV.cpp
Takes Q, K, V (N X DMODEL) and does the following:
Q (1 X DMODEL) is the multiplied with K.T (DMODEL x  N) and then max is taken along the last dimension. This gives a max index value in a vector of size 1 x N. This is then used to index the V (N x DMODEL) to get the output (1 x DMODEL).

### attention.cpp
Puts all the above functions together to do the attention operation. Takes input of size N x DMODEL and outputs N x DMODEL.




