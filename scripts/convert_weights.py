# convert_weights.py 
import numpy as np
import os

def convert_weights(input_weights_file, output_file_path):
    weights = np.load(input_weights_file)
    with open(output_file_path, 'w') as f:
        for row in weights:
            f.write(' '.join(map(str, row)) + '\n')
    print(f"Weights saved to {output_file_path}")

if __name__ == "__main__":
    os.makedirs("cpp_inference", exist_ok=True)
    convert_weights('scripts/weights_ih.npy', 'cpp_inference/weights_input_hidden.txt')
    convert_weights('scripts/weights_ho.npy', 'cpp_inference/weights_hidden_output.txt')