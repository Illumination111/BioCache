"""SpikingJelly PLIF with explicit external state and physical-time parameters."""
import torch
from torch import nn
from torch.func import functional_call
from spikingjelly.activation_based import neuron, surrogate


class SpikingJellyLIF(nn.Module):
    """Use the unmodified library charge/fire/reset operators.

    Normalize voltage by the learned threshold and map physical tau to the
    library's sigmoid decay parameter. No state survives in the library node.
    """

    def __init__(self):
        super().__init__()
        self.node = neuron.ParametricLIFNode(
            init_tau=2.0, decay_input=False, v_threshold=1.0, v_reset=None,
            surrogate_function=surrogate.ATan(alpha=2.0), detach_reset=False,
            step_mode="s", backend="torch", store_v_seq=False,
        )
        # Physical log_tau in LIFArray is the trainable source of decay.
        # functional_call supplies a derived tensor w for each microstep.
        self.node.w.requires_grad_(False)

    def forward(self, voltage, current, tau, threshold, dt):
        ratio = dt / tau
        # sigmoid(w) = 1 - exp(-dt/tau); stable even for very small dt/tau.
        w = torch.log(-torch.expm1(-ratio)) + ratio
        self.node.v = voltage / threshold
        try:
            spikes = functional_call(self.node, {"w": w}, (current / threshold,))
            updated_voltage = self.node.v * threshold
        finally:
            # Avoid a hidden second state, graph retention, and stream leakage.
            self.node.reset()
        return updated_voltage, spikes
