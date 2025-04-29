import numpy as np
import os

def save_weights(weights, filename):
    with open(filename, 'w') as f:
        for w in weights:
            f.write(str(w) + '\n')

os.makedirs("deployment/weights", exist_ok=True)

save_weights(np.load('weights/weights_ih.npy'), 'deployment/weights/weights_input_hidden.txt')
save_weights(np.load('weights/weights_ho.npy'), 'deployment/weights/weights_hidden_output.txt')
print("✅ Weights exported")