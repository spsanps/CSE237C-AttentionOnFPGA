# CSE237C-Project

Take N x DMODEL size input, and output N x DMODEL size output.

Following steps are taken:

K is projected using weights of size DMODEL x DMODEL to get K.T (DMODEL x N). K is done once and stored in memory. Q and V are computed on the fly and not stored in memory.

Each Q (1 X DMODEL) is the multiplied with K.T (DMODEL x  N) and then max is taken along the last dimension. This gives a max index value in a vector of size 1 x N. This is then used to generate V (1 x DMODEL) at the max index to get the output (1 x DMODEL). 

*Softmax is done with max* : Please check softmaxVsMax.ipynb for simulation of softmax vs max. We identify that there is a ~ 8.5% error in the output of max compared to softmax. The error is not higher as this is 4 bit quantized.

This is done for all the N inputs, and the output is of size N x DMODEL.

In memory, we store 
- Input of size N x DMODEL 
- 3 weights of size DMODEL x DMODEL
- K.T of size DMODEL x N
- Output of size N x DMODEL

If N == DMODEL then our memory requirement is 3 x DMODEL^2 + DMODEL^2 + DMODEL^2 + DMODEL^2 = 6 x DMODEL^2

At DMODEL = 16, we get 6 x 16^2 = 1536 values
At DMODEL = 32, we get 6 x 32^2 = 6144 values
At DMODEL = 64, we get 6 x 64^2 = 24576 values

## Code Structure

### attention.h
Header file.

### weights*.h
Contains weights of size DMODEL x DMODEL for Q, K, V. Each file is for different DMODEL value.

### 4bit.cpp
Contains efficient implementation of 4 bit quantized multiplication with 8 bit output.

### rowColMul.cpp
Contains efficient implementation of row x col multiplication with 8 bit output to be done in 1 cycle.

### QKVProj.cpp
Takes input tokens (1 X DMODEL) and projects them to Q, K, V (1 X DMODEL) using weights of size DMODEL x DMODEL. Separate functions are used for Q, K, V.

### QKV.cpp
Takes TOKEN (1 X DMODEL) and K.T (DMODEL x N)  and does the following:
 - Generated Q (1 X DMODEL) (generated on the fly with QKVProj so as not to store all Q values) is the multiplied with K.T (DMODEL x  N) (pre computed). Max is taken along the last dimension in the process.
- This gives a max index value in a vector of size 1 x N. This is then used to generate the corresponding index V to get the output (1 x DMODEL) on the fly.

### attention.cpp
Puts all the above functions together to do the attention operation. Populates the memory.
Takes input of size N x DMODEL and outputs N x DMODEL.




