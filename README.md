# CSE237C-Project

This project processes an input of size N x DMODEL and outputs a result of the same size, N x DMODEL.

The code is divided into different branches.

## Main branches are:

1. **main_with_tests**
Main codebase with all the files and test cases for N and DMODEL = 16, 32, and 64 with hls stream implemented internally
2. **streaming_for_PYNQ**
Codebase with axis stream implementation at top level for PYNQ demo at N, DMODEL = 16. No HLS streaming internally. .hwh, .bit and .ipynb files are checked in here.


## Process Overview:

1. **K Projection**: K is projected using weights of size DMODEL x DMODEL to obtain K.T (DMODEL x N). This is done once and the result is stored in memory.
2. **On-the-fly Computations for Q and V**: Q and V are computed on the fly and not stored in memory.
3. **Max Index Calculation**: Each Q (1 X DMODEL) is multiplied with K.T (DMODEL x N) and then the max is taken along the last dimension, yielding a max index value in a vector of size 1 x N.
4. **V Generation and Output**: The max index is used to generate V (1 x DMODEL) at the max index to obtain the output (1 x DMODEL).

*Note: Softmax is done with max*. Please refer to `softmaxVsMax.ipynb` for a simulation of softmax vs max. We observe that there is a ~8.5% error in the output using max compared to softmax. The error is not higher due to 4-bit quantization.

This process is repeated for all N inputs, resulting in an output of size N x DMODEL.

## Note on Memory/Resource 

- Input of size N x DMODEL
- 3 weights of size DMODEL x DMODEL
- K.T of size DMODEL x N
- Output of size N x DMODEL

If N == DMODEL, then our memory requirement is `3 x DMODEL^2 + DMODEL^2 + DMODEL^2 + DMODEL^2 = 6 x DMODEL^2`.

- At DMODEL = 16, we need 6 x 16^2 = 1536 values.
- At DMODEL = 32, we need 6 x 32^2 = 6144 values.
- At DMODEL = 64, we need 6 x 64^2 = 24576 values.

## Code Structure:

### attention.h
Header file for the attention mechanism.

### weights*.h
Contains the weights of size DMODEL x DMODEL for Q, K, V, with each file corresponding to different DMODEL values.

*Note: The weights are pre transposed- no transpose required*


### dotProd.cpp
Provides an efficient implementation of row x row multiplication.

### QKVProj.cpp
Takes input tokens (1 X DMODEL) and projects them to Q, K, V (1 X DMODEL) using weights of size DMODEL x DMODEL. Separate functions are used for Q, K, V projections.

### QKV.cpp
Processes a TOKEN (1 X DMODEL) and K (N x DMODEL) by doing the following:
 - Generates Q (1 X DMODEL) on the fly using QKVProj (to avoid storing all Q values) and multiplies it with the precomputed K (N x DMODEL). The max is then taken along the last dimension.
 - This yields a max index value in a vector of size 1 x N, which is then used to generate the corresponding index V to get the output (1 x DMODEL) on the fly.

### attention.cpp
Integrates all the above functions to perform the attention operation and populates the memory. It takes an input of size N x DMODEL and outputs N x DMODEL.

## Other files

### softmaxVsMax.ipynb
Estimates error of using max instead of softmax with quantized values

### create_testcases.ipynb
Generate .txt expected input output tokens with softmax and normalization for use in test2.cpp

### test_*.cpp
Sanity checks for each function

### test.cpp
Sanity checks for attention.cpp

### test2.cpp
Error check for attention.cpp

