import numpy as np
import pandas as pd

def generate_random_weights(dmodel, bitwidth):
    """
    Generates a DMODEL x DMODEL array with random weights.
    Each weight is an integer between 0 and 2**BITWIDTH - 1.
    """
    max_value = 2**bitwidth - 1
    return np.random.randint(0, max_value + 1, size=(3*dmodel, dmodel))

def save_to_csv(weights, file_name):
    """
    Saves the given weights array to a CSV file.
    """
    df = pd.DataFrame(weights)
    df.to_csv(file_name, index=False, header=False)

# Example usage
DMODEL = 64  # Replace with desired value
BITWIDTH = 4  # Replace with desired value

weights = generate_random_weights(DMODEL, BITWIDTH)
save_to_csv(weights, 'weights.csv')

print(f"Weights saved")
