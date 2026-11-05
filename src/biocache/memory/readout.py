"""Signed KV encoder and fixed-slot query-conditioned attention readout."""
import torch
from torch import nn
from biocache.memory.layered_array import LIFArray, LIFState


class LatentReadout(nn.Module):
    def __init__(self, state_dim, heads, dim, slots=4, hidden=64):
        super().__init__()
        self.heads, self.dim, self.slots = heads, dim, slots
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2 * heads * slots * dim))

    def forward(self, q, features):
        b = q.shape[0]
        k, v = self.net(features).reshape(b, 2, self.heads, self.slots, self.dim).unbind(1)
        # Latent keys live directly in query space. No second RoPE application.
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


class KVMemory(nn.Module):
    def __init__(self, config, kv_heads=1, query_heads=1, head_dim=8, slots=4, hidden=64, mode="lif"):
        super().__init__()
        if query_heads % kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads")
        self.array = LIFArray(config, mode)
        self.kv_heads, self.query_heads, self.head_dim = kv_heads, query_heads, head_dim
        n = config.neurons_per_column
        self.encoders = nn.ModuleList([
            nn.Linear(2 * kv_heads * head_dim + 2, n) for _ in range(config.layers)])
        self.readers = nn.ModuleDict()
        self.probes = nn.ModuleDict()
        for view in ("first_column", "last_column", "all_columns"):
            width = 2 * n * (config.columns if view == "all_columns" else 1)
            self.readers[view] = nn.ModuleList([
                LatentReadout(width, query_heads, head_dim, slots, hidden) for _ in range(config.layers)])
            self.probes[view] = nn.ModuleList([
                nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, head_dim))
                for _ in range(config.layers)])

    def write(self, state, k, v, position):
        """k/v [B,L,Hkv,D]; position and state time are causal scalar metadata."""
        expected = (state.u.shape[0], self.array.config.layers, self.kv_heads, self.head_dim)
        if k.shape != expected or v.shape != expected or position.shape != state.time.shape:
            raise ValueError("invalid KV or metadata shape")
        # Bounded metadata encoding, no future labels or future queries.
        meta = torch.stack((torch.sin(position / 16), torch.sin(state.time / 16)), dim=-1)
        raw = torch.cat((k.flatten(2), v.flatten(2), meta), dim=-1)
        current = torch.stack([encoder(raw[:, l]) for l, encoder in enumerate(self.encoders)], dim=1)
        return self.array.event(state, current)

    def wait(self, state):
        current = state.u.new_zeros(state.u.shape[0], self.array.config.layers,
                                    self.array.config.neurons_per_column)
        return self.array.event(state, current, write=False)

    @staticmethod
    def features(state, layer, view):
        u, s = state.u[:, layer], state.spikes[:, layer]
        if view == "first_column":
            u, s = u[:, 0], s[:, 0]
        elif view == "last_column":
            u, s = u[:, -1], s[:, -1]
        elif view != "all_columns":
            raise ValueError(view)
        return torch.cat((u.flatten(1), s.flatten(1)), dim=-1)

    def read(self, state, q, layer=0, view="last_column"):
        if q.ndim != 4 or q.shape[1] != self.query_heads or q.shape[-1] != self.head_dim:
            raise ValueError("q must be [B,Hq,Nq,D]")
        result = self.readers[view][layer](q, self.features(state, layer, view))
        return result * state.valid[:, layer, None, None, None]

    def probe(self, state, layer=0, view="last_column"):
        return self.probes[view][layer](self.features(state, layer, view))

    def export_state(self, state):
        from dataclasses import fields
        return {"version": 1, "array": self.array.config.asdict(), "mode": self.array.mode,
                "kv_heads": self.kv_heads, "query_heads": self.query_heads, "head_dim": self.head_dim,
                "state": {f.name: getattr(state, f.name).detach().clone() for f in fields(state)}}

    def import_state(self, payload):
        expected = self.export_state(self.array.initial_state(1))
        for name in ("version", "array", "mode", "kv_heads", "query_heads", "head_dim"):
            if payload[name] != expected[name]:
                raise ValueError(f"incompatible state: {name}")
        tensors = payload["state"]
        ref = self.array.initial_state(tensors["u"].shape[0])
        for name, tensor in tensors.items():
            target = getattr(ref, name)
            if tensor.shape != target.shape or tensor.dtype != target.dtype:
                raise ValueError(f"incompatible tensor: {name}")
        return LIFState(**{name: value.to(self.array.W.device).clone() for name, value in tensors.items()})
