from brian2 import *
import numpy as np

# Settings
input_size = 8 * 250
hidden_size = 128
output_size = 2

# Load EEG segments
def load_simulate_input(filename):
    X, y = [], []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            label = int(parts[0])
            data = [float(x) for x in parts[1:]]
            X.append(data)
            y.append(label)
    return np.array(X), np.array(y)

X, y = load_simulate_input('data/simulate_input.txt')

# Model params
duration = 1000*ms

# Adaptive Exponential Integrate-and-Fire model
adex_eqs = '''
dv/dt = (-(v - EL) + DeltaT*exp((v - VT)/DeltaT) - w + I) / tau_m : 1

dw/dt = (a*(v - EL) - w) / tau_w : 1
I : 1
'''

EL = 0.0
VT = 1.0
DeltaT = 0.5

tau_m = 20*ms
tau_w = 100*ms
a = 0.01

# Layers
G_input = NeuronGroup(input_size, 'v : 1', threshold='False', reset='')
G_hidden = NeuronGroup(hidden_size, adex_eqs, threshold='v>1', reset='v=0; w+=0.1', method='euler')
G_output = NeuronGroup(output_size, adex_eqs, threshold='v>1', reset='v=0; w+=0.1', method='euler')

# Synapses
S_ih = Synapses(G_input, G_hidden, 'w_syn : 1', on_pre='I_post += w_syn')
S_ih.connect(p=0.1)
S_ih.w = 'rand()'

S_ho = Synapses(G_hidden, G_output, 'w_syn : 1', on_pre='I_post += w_syn')
S_ho.connect(p=0.1)
S_ho.w = 'rand()'

# Monitors
spikes_output = SpikeMonitor(G_output)

# Training loop
for xi, target in zip(X, y):
    G_input.v = xi + 0.01 * np.random.randn(input_size)
    G_hidden.v = 0
    G_hidden.w = 0
    G_output.v = 0
    G_output.w = 0
    run(duration)

# Save weights
np.save('training/weights_ih.npy', S_ih.w_syn[:])
np.save('training/weights_ho.npy', S_ho.w_syn[:])
print("Training done.")