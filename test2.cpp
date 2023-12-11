#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm> // Include for max_element
#include <fstream>
#include "attention.h"
#include "weights16.h"

using data_t_test = double; // Data type for computation

// Initialize a matrix with random values
void initialize_matrix_random(data_t matrix[N][DMODEL])
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < DMODEL; j++)
        {
            matrix[i][j] = dis(gen) * 16;
        }
    }
}

// Print a matrix
void print_matrix1(const char *name, data_t matrix[N][DMODEL])
{
    std::cout << name << ":\n";
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < DMODEL; j++)
        {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << "\n";
    }
}

void print_matrix2(const char *name, double matrix[N][DMODEL])
{
    std::cout << name << ":\n";
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < DMODEL; j++)
        {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << "\n";
    }
}

// Softmax function with numerical stability checks
void softmax(data_t_test matrix[N][DMODEL])
{
    for (int i = 0; i < N; i++)
    {
        double max_val = *std::max_element(matrix[i], matrix[i] + DMODEL);
        double sum = 0.0;

        for (int j = 0; j < DMODEL; j++)
        {
            matrix[i][j] = exp(matrix[i][j] - max_val); // Shift by max for numerical stability
            sum += matrix[i][j];
        }

        if (sum != 0) // Check for zero division
        {
            for (int j = 0; j < DMODEL; j++)
            {
                matrix[i][j] /= sum;
            }
        }
    }
}

// Scaled Dot-Product Attention
void scaled_dot_product_attention(
    data_t tokens[N][DMODEL],
    data_t Q_W[N][DMODEL],
    data_t K_W[N][DMODEL],
    data_t V_W[N][DMODEL],
    data_t_test output[N][DMODEL])
{

    data_t_test Q[N][DMODEL] = {0};
    data_t_test K[N][DMODEL] = {0};
    data_t_test V[N][DMODEL] = {0};

    // Dot product with weights
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < DMODEL; j++)
        {
            for (int k = 0; k < DMODEL; k++)
            {
                Q[i][j] += tokens[i][k] * Q_W[k][j];
                K[i][j] += tokens[i][k] * K_W[k][j];
                V[i][j] += tokens[i][k] * V_W[k][j];
            }
        }
    }

    data_t_test temp[N][DMODEL] = {0};

    // Calculate Q*K^T
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < DMODEL; k++)
            {
                temp[i][j] += Q[i][k] * K[j][k];
            }
            temp[i][j] /= sqrt(DMODEL); // Scale
        }
    }

    // Apply softmax
    softmax(temp);

    // Calculate temp*V
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < DMODEL; j++)
        {
            output[i][j] = 0;
            for (int k = 0; k < N; k++)
            {
                output[i][j] += temp[i][k] * V[k][j];
            }
        }
    }
}

// Compute Mean Squared Error
double root_mean_squared_error(data_t output[N][DMODEL], data_t expected[N][DMODEL])
{
    double error = 0.0;
    double data_t_double[N][DMODEL];
    double expected_double[N][DMODEL];

    // Convert to double
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < DMODEL; j++)
        {
            data_t_double[i][j] = (double)output[i][j];
            expected_double[i][j] = (double)expected[i][j];
        }
    }

    // normalize across rows to 0-1
    for (int i = 0; i < N; i++)
    {
        double max_val = *std::max_element(data_t_double[i], data_t_double[i] + DMODEL);
        double min_val = *std::min_element(data_t_double[i], data_t_double[i] + DMODEL);
        double diff = max_val - min_val;

        if (diff != 0) // Check for zero division
        {
            for (int j = 0; j < DMODEL; j++)
            {
                data_t_double[i][j] = (data_t_double[i][j] - min_val) / diff;
            }
        }
    }

    // normalize across rows to 0-1 for expected
    for (int i = 0; i < N; i++)
    {
        double max_val = *std::max_element(expected_double[i], expected_double[i] + DMODEL);
        double min_val = *std::min_element(expected_double[i], expected_double[i] + DMODEL);
        double diff = max_val - min_val;

        if (diff != 0) // Check for zero division
        {
            for (int j = 0; j < DMODEL; j++)
            {
                expected_double[i][j] = (expected_double[i][j] - min_val) / diff;
            }
        }
    }



    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < DMODEL; j++)
        {
            double diff = data_t_double[i][j] - expected_double[i][j];
            error += diff * diff;
        }
    }
    return sqrt(error / (N * DMODEL));
}


int main()
{
    data_t tokens[N][DMODEL];
    data_t weightsQ[N][DMODEL], weightsK[N][DMODEL], weightsV[N][DMODEL];
    data_t output[N][DMODEL];
    data_t expected_output[N][DMODEL];

    std::cout << "N: " << N << std::endl;
    std::cout << "DMODEL: " << DMODEL << std::endl;


    // Open test input file
    std::ifstream input_file("generate_tests_input.txt");
    if (!input_file)
    {
        std::cout << "Failed to open generate_tests_input.txt" << std::endl;
        return 1;
    }

    // print here
    std::cout << "Finished opening file" << std::endl;

    // Open test output file
    std::ifstream output_file("generate_tests_output.txt");
    if (!output_file)
    {
        std::cout << "Failed to open generate_tests_output.txt" << std::endl;
        return 1;
    }

    // print here
    std::cout << "Finished opening file" << std::endl;

    // Initialize matrices with random values
    for (int loop = 0; loop < 10; loop++)
    {
        // read test in tokens
        // each line is N*DMODEL tokens
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < DMODEL; j++)
            {
                input_file >> tokens[i][j];
            }
        }

        static hls::stream<data_t> tokens_stream("tokens_stream");
        #pragma HLS STREAM variable=tokens_stream depth=N*DMODEL

        for (int i=N; i > 0; --i) {
        	for (int j=DMODEL; j>0; --j) {
        		tokens_stream << tokens[i-1][j-1];
        	}
        }

        // read output
        // each line is N*DMODEL tokens
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < DMODEL; j++)
            {
                output_file >> expected_output[i][j];
            }
        }


        for (int i = 0; i < DMODEL; i++)
        {
            for (int j = 0; j < DMODEL; j++)
            {
                weightsQ[i][j] = Q_W[i][j];
                weightsK[i][j] = K_W[i][j];
                weightsV[i][j] = V_W[i][j];
            }
        }

        static hls::stream<data_t> output_stream("output_stream");
        #pragma HLS STREAM variable=output_stream depth=DMODEL
        attention(tokens_stream, weightsQ, weightsK, weightsV, output_stream);
        for (int i=0; i<N; i++) {
        	for (int j=0; j<DMODEL; j++) {
        		output[i][j] = output_stream.read();
        	}
        }

        // Print matrices
        print_matrix1("Tokens", tokens);
        print_matrix1("WeightsQ", weightsQ);
        print_matrix1("WeightsK", weightsK);
        print_matrix1("WeightsV", weightsV);
        print_matrix1("Output", output);
        print_matrix1("Expected Output", expected_output);

        // normalize 

        // Calculate and print the error
        double mse = root_mean_squared_error(output, expected_output);
        std::cout << "Normalized RMSE: " << mse << std::endl;
    }

    // Close files
    input_file.close();
    output_file.close();

    return 0;
}
