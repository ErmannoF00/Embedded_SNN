# File: scripts/convert_weights.py

import numpy as np

def convert_weights(input_weights_file, output_weights_file, output_file_path):
    """
    Convert weights from a numpy .npy format to a C++-compatible text format.
    
    Args:
    - input_weights_file (str): Path to the input .npy file (weights)
    - output_weights_file (str): Path to the output text file where weights will be saved
    - output_file_path (str): Output file for weight matrix conversion
    """
    # Load the weight matrix from the .npy file
    weights = np.load(input_weights_file)

    # Save as text format, row by row
    with open(output_file_path, 'w') as f:
        for row in weights:
            f.write(' '.join(map(str, row)) + '\n')
    
    print(f"Weights saved to {output_file_path}")

if __name__ == "__main__":
    # Convert weights from .npy format to .txt for input-hidden layer
    convert_weights('scripts/weights_ih.npy', 'weights_input_hidden.txt', 'cpp_inference/weights_input_hidden.txt')

    # Convert weights from .npy format to .txt for hidden-output layer
    convert_weights('scripts/weights_ho.npy', 'weights_hidden_output.txt', 'cpp_inference/weights_hidden_output.txt')
