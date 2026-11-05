"""One source of truth for synchronous adjacent-column LIF updates."""
from dataclasses import dataclass, fields
import math
import torch
from torch import nn
from biocache.config import ArrayConfig
from biocache.memory.lif import SpikingJellyLIF


@dataclass(frozen=True)
class LIFState:
    u: torch.Tensor                 # [B,L,C,N], FP32 by default
    spikes: torch.Tensor            # same shape, float for surrogate gradient
    time: torch.Tensor              # [B,L], event time, NOT microstep index
    events: torch.Tensor            # [B,L], integer event count
    valid: torch.Tensor             # [B,L], whether a write has occurred

    def detach(self):
        return LIFState(**{f.name: getattr(self, f.name).detach() for f in fields(self)})

    def nbytes(self):
        return sum(getattr(self, f.name).numel() * getattr(self, f.name).element_size()
                   for f in fields(self))


class LIFArray(nn.Module):
    def __init__(self, config: ArrayConfig, mode="lif"):
        super().__init__()
        if mode not in ("lif", "ema"):
            raise ValueError("mode must be lif or ema")
        self.config, self.mode = config, mode
        self.lif = SpikingJellyLIF()
        l, c, n = config.layers, config.columns, config.neurons_per_column
        # Full dense matrices. Diagonal initialization encourages a propagation path.
        weights = torch.eye(n).expand(l, c - 1, n, n).clone() * 0.9
        weights += torch.randn_like(weights) * (0.02 / math.sqrt(n))
        self.W = nn.Parameter(weights)
        self.log_tau = nn.Parameter(torch.tensor(config.tau).log().view(1, c, 1).expand(l, c, n).clone())
        self.log_threshold = nn.Parameter(torch.full((l, c, n), math.log(config.threshold)))
        self.register_buffer("connection_mask", torch.ones(l, c - 1, 1, 1))

    def initial_state(self, batch_size):
        shape = (batch_size, self.config.layers, self.config.columns, self.config.neurons_per_column)
        u = self.W.new_zeros(shape)
        return LIFState(u, u.clone(), self.W.new_zeros(shape[:2]),
                        torch.zeros(shape[:2], device=u.device, dtype=torch.int64),
                        torch.zeros(shape[:2], device=u.device, dtype=torch.bool))

    def microstep(self, state, current, dt, active=None):
        """All columns read old spikes; active permits independent layer updates."""
        b, l, _, n = state.u.shape
        if current.shape != (b, l, n) or dt <= 0:
            raise ValueError("current must be [B,L,N] and dt must be positive")
        if active is None:
            active = torch.ones((b, l), device=current.device, dtype=torch.bool)
        if active.shape != (b, l) or active.dtype != torch.bool:
            raise ValueError("active must be bool [B,L]")
        # j = output neuron, i = input neuron. No connections across L.
        synaptic = torch.einsum("lcji,blci->blcj", self.W * self.connection_mask,
                               state.spikes[:, :, :-1])
        drive = torch.cat((current.unsqueeze(2), synaptic), dim=2)
        tau = self.log_tau.exp().clamp_min(1e-4)
        if self.mode == "lif":
            threshold = self.log_threshold.exp().clamp_min(1e-4)
            u, spikes = self.lif(state.u, drive, tau, threshold, dt)
        else:
            decay = torch.exp(-dt / tau)
            # Linear cascade / EMA control: same allocated state bytes; no threshold/reset.
            # Both buffers are reported, although spikes here duplicates the current EMA.
            u = decay * state.u + (1 - decay) * drive
            spikes = u
        mask = active[:, :, None, None]
        return LIFState(torch.where(mask, u, state.u), torch.where(mask, spikes, state.spikes),
                        state.time + dt * active, state.events, state.valid)

    def event(self, state, current, active=None, write=True):
        """Exactly S microsteps and fixed event_duration. No implicit flush."""
        if active is None:
            active = torch.ones_like(state.valid)
        dt = self.config.event_duration / self.config.microsteps_per_token
        for step in range(self.config.microsteps_per_token):
            state = self.microstep(state, current if step == 0 else torch.zeros_like(current), dt, active)
        return LIFState(state.u, state.spikes, state.time,
                        state.events + active.to(torch.int64), state.valid | (active & write))
