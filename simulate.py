import nest_asyncio as nest
import numpy as np
from model.model import create_snn

network = create_snn()
simulation_time = 1000.0

nest.Simulate(simulation_time)

spike_events = nest.GetStatus(network['spike_detector'], 'events')[0]
spike_times = spike_events['times']
spike_senders = spike_events['senders']

print("Spike times (first 10):", spike_times[:10])
print("Spike senders (first 10):", spike_senders[:10])
