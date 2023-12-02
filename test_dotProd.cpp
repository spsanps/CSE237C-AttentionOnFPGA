#include <iostream>
#include "dotprod.h"


int main3() {
    // Initialize test data
    data_t row1[DMODEL];
    data_t row2[DMODEL];
    data3_t result;

    // Fill the test data with some values
    for (int i = 0; i < DMODEL; ++i) {
        row1[i] = i;  // Example values
        row2[i] = DMODEL - i - 5;  // Example values
    }

    // Call the function under test
    dotProd(row1, row2, result);

    // Print the result
    std::cout << "Dot Product Result: " << result.to_uint() << std::endl;

    // Verify the result (optional)
    data3_t expected_result = 0;
    for (int i = 0; i < DMODEL; ++i) {
        expected_result += row1[i] * row2[i];
    }

    if (result == expected_result) {
        std::cout << "Test Passed." << std::endl;
    } else {
        std::cout << "Test Failed. Expected: " << expected_result.to_uint() << std::endl;
    }

    return 0;
}
