"""Adaptive leaky-integrate-and-fire classifier with surrogate gradients.

Identical recurrence is implemented in deployment/src/snn_runtime.cpp.
"""
import torch
from torch import nn

N_INPUT = 32
N_HIDDEN = 48
N_OUTPUT = 2
STEPS = 20
LEAK = 0.85
ADAPT_DECAY = 0.90
ADAPT_GAIN = 0.20
THRESHOLD = 1.0
READOUT_LEAK = 0.90


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return (value >= 0.0).to(value.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (value,) = ctx.saved_tensors
        # SuperSpike-like gradient; forward pass remains hard-threshold.
        return grad_output / (1.0 + value.abs()).square()


class AdaptiveLIF(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(N_INPUT, N_HIDDEN)
        self.readout_proj = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        batch = x.shape[0]
        v = x.new_zeros((batch, N_HIDDEN))
        adaptation = x.new_zeros((batch, N_HIDDEN))
        output_v = x.new_zeros((batch, N_OUTPUT))
        current = self.input_proj(x)
        spike_total = x.new_zeros((batch, N_HIDDEN))
        for _ in range(STEPS):
            v = LEAK * v + current - ADAPT_GAIN * adaptation
            spike = SurrogateSpike.apply(v - THRESHOLD)
            v = v - spike * THRESHOLD
            adaptation = ADAPT_DECAY * adaptation + spike
            output_v = READOUT_LEAK * output_v + self.readout_proj(spike)
            spike_total = spike_total + spike
        return output_v, spike_total
