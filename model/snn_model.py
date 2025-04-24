# File: model/snn_model.brian2.py
from brian2 import *
import numpy as np

# Define the neural network model parameters
input_size = 10  # Number of input neurons
hidden_size = 20  # Number of hidden neurons
output_size = 5  # Number of output neurons
duration = 1000 * ms  # Simulation duration
tau = 10 * ms  # Membrane time constant

# Create the neuron groups
input_neurons = NeuronGroup(input_size, 'dv/dt = -v/tau + I : 1', threshold='v>1', reset='v=0', method='exact')
hidden_neurons = NeuronGroup(hidden_size, 'dv/dt = -v/tau + I : 1', threshold='v>1', reset='v=0', method='exact')
output_neurons = NeuronGroup(output_size, 'dv/dt = -v/tau + I : 1', threshold='v>1', reset='v=0', method='exact')

# Create synapses
input_to_hidden = Synapses(input_neurons, hidden_neurons, 'w : 1', on_pre='v_post += w')
hidden_to_output = Synapses(hidden_neurons, output_neurons, 'w : 1', on_pre='v_post += w')

# Randomly initialize synaptic weights
input_to_hidden.w = 'rand()'  # Random weights between 0 and 1
hidden_to_output.w = 'rand()'

# Create spike generators for input neurons (simulating DVS spikes or random input)
input_spikes = PoissonGroup(input_size, rates='20*Hz')  # Random spike trains at 20 Hz
input_neurons.I = 'input_spikes * 0.2'

# Record the spikes of neurons
spike_monitor_input = SpikeMonitor(input_neurons)
spike_monitor_hidden = SpikeMonitor(hidden_neurons)
spike_monitor_output = SpikeMonitor(output_neurons)

# Run the simulation
run(duration)

# Save the weight matrices and spike data
weights_input_hidden = input_to_hidden.w[:]
weights_hidden_output = hidden_to_output.w[:]

np.save('scripts/weights_ih.npy', weights_input_hidden)
np.save('scripts/weights_ho.npy', weights_hidden_output)

# Save connection indices for sparse matrix representation
connections_ih_i, connections_ih_j = input_to_hidden.i, input_to_hidden.j
connections_ho_i, connections_ho_j = hidden_to_output.i, hidden_to_output.j

np.save('scripts/connections_ih.npy', connections_ih_i)
np.save('scripts/connections_ih_j.npy', connections_ih_j)
np.save('scripts/connections_ho.npy', connections_ho_i)
np.save('scripts/connections_ho_j.npy', connections_ho_j)

# Output the spike times for later analysis
print(f"Input neurons spiked at: {spike_monitor_input.t[:10]}")
print(f"Hidden neurons spiked at: {spike_monitor_hidden.t[:10]}")
print(f"Output neurons spiked at: {spike_monitor_output.t[:10]}")
