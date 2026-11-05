"""Topology and propagation: isolated input, synchronous propagation, layer isolation, broken chain."""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from biocache.config import ArrayConfig, load_config
from biocache.memory import LIFArray
from biocache.artifacts import start_run, write_json


@torch.no_grad()
def run(config, output):
    torch.manual_seed(config["seed"])
    cfg = ArrayConfig(**config["array"])
    array = LIFArray(cfg)
    array.W.fill_(0.9 / cfg.neurons_per_column)
    path = start_run(output, config)
    u, s = [], []
    state = array.initial_state(1)
    for step in range(config["microsteps_observed"]):
        x = state.u.new_zeros(1, cfg.layers, cfg.neurons_per_column)
        if step == 0:
            x[:, 0] = 0.75
        state = array.microstep(state, x, cfg.event_duration / cfg.microsteps_per_token)
        u.append(state.u[0, 0].clone())
        s.append(state.spikes[0, 0].clone())
        assert not state.u[:, 1:].any(), "cross-layer contamination"
    u, s = torch.stack(u).numpy(), torch.stack(s).numpy()
    first = [int(np.flatnonzero(s[:, c].any(axis=1))[0]) + 1
             if s[:, c].any() else None for c in range(cfg.columns)]
    assert first == list(range(1, cfg.columns + 1)), first
    assert np.isfinite(u).all() and np.isfinite(s).all()
    np.testing.assert_allclose(u[0, 0], 0.25, atol=1e-6)
    # Break the first edge: all subsequent columns remain exactly at zero.
    if cfg.columns > 1:
        array.connection_mask[:, 0] = 0
        broken = array.initial_state(1)
        for step in range(config["microsteps_observed"]):
            x = broken.u.new_zeros(1, cfg.layers, cfg.neurons_per_column)
            if step == 0:
                x[:, 0] = 0.75
            broken = array.microstep(broken, x, cfg.event_duration / cfg.microsteps_per_token)
            assert not broken.u[:, :, 1:].any()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    for ax, values, title in zip(axes, (u, s), ("Membrane potential", "Spike raster")):
        im = ax.imshow(values.reshape(len(u), -1).T, aspect="auto", origin="lower")
        ax.set(title=title, xlabel="Microstep (zero based)", ylabel="Column x neuron")
        fig.colorbar(im, ax=ax)
    fig.savefig(path / "propagation.png", dpi=150)
    plt.close(fig)
    np.savez_compressed(path / "trajectory.npz", u=u, spikes=s)
    result = {"test": "topology_and_propagation", "status": "passed", "first_spike_microstep_1based": first,
              "activity_per_column": s.mean(axis=(0, 2)).tolist(),
              "state_bytes": state.nbytes(), "layer_isolation": True, "broken_chain": True}
    write_json(path / "metrics.json", result)
    print(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="experiments/lif_kv_feasibility/configs/propagation.yaml")
    p.add_argument("--output", default="outputs/propagation_spikingjelly")
    args = p.parse_args()
    run(load_config(args.config), args.output)
