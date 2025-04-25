import nest
import numpy as np

def create_snn():
    # Reset NEST kernel
    nest.ResetKernel()

    # Create neuron populations
    input_neurons = nest.Create('poisson_generator', 1, {'rate': 800.0})
    excitatory_neurons = nest.Create('iaf_psc_alpha', 100)
    inhibitory_neurons = nest.Create('iaf_psc_alpha', 25)
    output_neurons = nest.Create('iaf_psc_alpha', 2)  # Assuming binary classification

    # Create spike detectors
    spike_detector = nest.Create('spike_recorder')

    # Connect neurons
    nest.Connect(input_neurons, excitatory_neurons, syn_spec={'weight': 1.0})
    nest.Connect(excitatory_neurons, inhibitory_neurons, syn_spec={'weight': 1.0})
    nest.Connect(inhibitory_neurons, excitatory_neurons, syn_spec={'weight': -1.0})
    nest.Connect(excitatory_neurons, output_neurons, syn_spec={'weight': 1.0})
    nest.Connect(output_neurons, spike_detector)

    return {
        'input': input_neurons,
        'excitatory': excitatory_neurons,
        'inhibitory': inhibitory_neurons,
        'output': output_neurons,
        'spike_detector': spike_detector
    }
