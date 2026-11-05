"""Synthetic KV storage and delayed readout: synthetic-only storage probes, query readout, controls and byte accounting."""
import argparse
from dataclasses import replace
import math
import resource
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from biocache.config import ArrayConfig, load_config
from biocache.data import make_sequence, teacher_attention
from biocache.memory import KVMemory
from biocache.artifacts import start_run, write_json

VIEWS = ("first_column", "last_column", "all_columns")


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def rollout(model, seq, delay, delay_mode, seed, tbptt=0):
    state = model.array.initial_state(seq.k.shape[0])
    distractor = None
    if delay and delay_mode == "interference":
        distractor = make_sequence(seq.k.shape[0], max(2, delay), seq.k.shape[-1], seed,
                                   "random_association", seq.k.device)
    total = seq.k.shape[1] + delay
    for t in range(total):
        if tbptt and t and t % tbptt == 0:
            state = state.detach()
        if t < seq.k.shape[1]:
            k, v = seq.k[:, t], seq.v[:, t]
        elif distractor is not None:
            k, v = distractor.k[:, t - seq.k.shape[1]], distractor.v[:, t - seq.k.shape[1]]
        else:
            state = model.wait(state)
            continue
        state = model.write(state, k, v, torch.full_like(state.time, float(t)))
    return state, distractor


def state_control(state, kind):
    if kind == "zero":
        # Keep validity true: tests readout priors instead of the empty-memory mask.
        return replace(state, u=torch.zeros_like(state.u), spikes=torch.zeros_like(state.spikes))
    index = torch.arange(state.u.shape[0], device=state.u.device).roll(1)
    return replace(state, u=state.u[index], spikes=state.spikes[index])


@torch.no_grad()
def evaluate(model, config, seed, full=True):
    data, ev = config["data"], config["evaluation"]
    rows = []
    delays = ev["delays"] if full else [0, 16]
    for pi, pattern in enumerate(data["patterns"]):
        seq = make_sequence(data["test_size"], data["history"], data["head_dim"],
                            seed + 100000 + pi, pattern, config["device"])
        teacher = teacher_attention(seq)
        norm = teacher.norm().item()
        for delay_mode in ev["delay_modes"]:
            for delay in delays:
                sync(config["device"])
                start = time.perf_counter()
                state, distractor = rollout(model, seq, delay, delay_mode, seed + 200000 + pi)
                sync(config["device"])
                write_ms = (time.perf_counter() - start) * 1000 / (data["history"] + delay)
                full_teacher = teacher_attention(seq, distractor)
                window = (torch.nn.functional.scaled_dot_product_attention(
                    seq.q, distractor.k[:, -data["history"]:, 0, 0][:, None],
                    distractor.v[:, -data["history"]:, 0, 0][:, None])
                    if distractor is not None else teacher)
                for view in VIEWS:
                    sync(config["device"])
                    start = time.perf_counter()
                    pred = model.read(state, seq.q, view=view)
                    sync(config["device"])
                    read_ms = (time.perf_counter() - start) * 1000
                    zero = model.read(state_control(state, "zero"), seq.q, view=view)
                    shuffled = model.read(state_control(state, "shuffle"), seq.q, view=view)
                    mse = lambda a, b: (a - b).square().mean().item()
                    probe = model.probe(state, view=view)
                    rows.append({"pattern": pattern, "delay_mode": delay_mode, "delay": delay,
                        "view": view, "mse_remote": mse(pred, teacher),
                        "mse_full": mse(pred, full_teacher), "zero_mse": mse(zero, teacher),
                        "shuffle_mse": mse(shuffled, teacher),
                        "window_remote_mse": mse(window, teacher),
                        "probe_mse": mse(probe, seq.event_value),
                        "probe_zero_mse": mse(model.probe(state_control(state, "zero"), view=view), seq.event_value),
                        "probe_shuffle_mse": mse(model.probe(state_control(state, "shuffle"), view=view), seq.event_value),
                        "relative_l2": (pred - teacher).norm().item() / max(norm, 1e-8),
                        "teacher_norm": norm, "write_ms_per_event_batch": write_ms,
                        "read_ms_batch": read_ms, "state_bytes_batch": state.nbytes(),
                        "elapsed_event_time": float(state.time[0, 0]),
                        "microsteps": (data["history"] + delay) * model.array.config.microsteps_per_token,
                        "activity": float(state.spikes.mean()),
                        "voltage_min": float(state.u.min()), "voltage_max": float(state.u.max())})
    return rows


def build_model(config, seed, variant):
    torch.manual_seed(seed)
    r, d = config["readout"], config["data"]
    model = KVMemory(ArrayConfig(**config["array"]), head_dim=d["head_dim"],
                     slots=r["latent_slots"], hidden=r["hidden"], mode="ema" if variant == "ema" else "lif")
    if variant == "disconnected":
        model.array.connection_mask[:, 0] = 0
    if variant == "frozen":
        model.array.W.requires_grad_(False)
    return model.to(config["device"])


def train_one(config, path, seed, variant):
    model = build_model(config, seed, variant)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                           lr=config["training"]["learning_rate"])
    initial = evaluate(model, config, seed, full=False)
    d, t = config["data"], config["training"]
    history = []
    grad_max = {}
    for step in range(t["steps"]):
        pattern = d["patterns"][step % len(d["patterns"])]
        # No training sequence seed overlaps the evaluation offsets.
        seq = make_sequence(d["batch_size"], d["history"], d["head_dim"],
                            seed * 1000000 + step, pattern, config["device"])
        delay = t["train_delays"][step % len(t["train_delays"])]
        mode = "wait" if (step // len(d["patterns"])) % 2 == 0 else "interference"
        state, _ = rollout(model, seq, delay, mode, seed * 1000000 + 50000 + step, t["tbptt_steps"])
        target = teacher_attention(seq)
        losses = [(model.read(state, seq.q, view=view) - target).square().mean()
                  + (model.probe(state, view=view) - seq.event_value).square().mean() for view in VIEWS]
        loss = sum(losses) / len(losses)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm().item()
                if not math.isfinite(norm):
                    raise RuntimeError(f"nonfinite gradient {name}")
                grad_max[name] = max(norm, grad_max.get(name, 0.))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history.append(float(loss.detach()))
    model.eval()
    metrics = evaluate(model, config, seed)
    # Checkpoint includes optimizer, config and explicit state. Restore must match.
    seq = make_sequence(2, d["history"], d["head_dim"], seed + 900000, "unique_event", config["device"])
    with torch.no_grad():
        state, _ = rollout(model, seq, 0, "wait", seed)
    checkpoint = path / f"{variant}_{seed}.pt"
    torch.save({"config": config, "model": model.state_dict(), "optimizer": opt.state_dict(),
                "seed": seed, "variant": variant, "step": t["steps"],
                "memory": model.export_state(state)}, checkpoint)
    restored = build_model(config, seed, variant)
    saved = torch.load(checkpoint, map_location=config["device"], weights_only=False)
    restored.load_state_dict(saved["model"])
    with torch.no_grad():
        restored_state = restored.import_state(saved["memory"])
        torch.testing.assert_close(model.read(state, seq.q), restored.read(restored_state, seq.q), rtol=0, atol=0)
        advanced, advanced_restored = model.wait(state), restored.wait(restored_state)
        torch.testing.assert_close(advanced.u, advanced_restored.u, rtol=0, atol=0)
        torch.testing.assert_close(model.read(advanced, seq.q), restored.read(advanced_restored, seq.q), rtol=0, atol=0)
    parameters = {"total": sum(p.numel() for p in model.parameters()),
                  "bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                  "buffer_bytes": sum(b.numel() * b.element_size() for b in model.buffers()),
                  "synapses": model.array.W.numel(),
                  "readout": {v: sum(p.numel() for p in model.readers[v].parameters()) for v in VIEWS},
                  "probe": {v: sum(p.numel() for p in model.probes[v].parameters()) for v in VIEWS}}
    result = {"seed": seed, "variant": variant, "initial": initial, "metrics": metrics,
              "training_loss": history, "max_gradient_norm": grad_max,
              "parameters": parameters, "checkpoint_restored_exactly": True}
    write_json(path / f"{variant}_{seed}.json", result)
    print(f"finished {variant} seed={seed}, loss={history[-1]:.4f}", flush=True)
    return result


@torch.no_grad()
def bounded_memory(config):
    model = build_model(config, config["seeds"][0], "lif")
    state = model.array.initial_state(1)
    g = torch.Generator(device=config["device"]).manual_seed(100)
    reports = []
    times = []
    for t in range(1, max(config["evaluation"]["memory_events"]) + 1):
        k = torch.randn((1, 1, 1, config["data"]["head_dim"]), generator=g, device=config["device"])
        v = torch.randn(k.shape, generator=g, device=config["device"])
        sync(config["device"])
        start = time.perf_counter()
        state = model.write(state, k, v, torch.full_like(state.time, t - 1))
        sync(config["device"])
        if t <= 256:
            times.append((time.perf_counter() - start) * 1000)
        if t in config["evaluation"]["memory_events"]:
            assert state.u.grad_fn is None and state.spikes.grad_fn is None
            assert torch.isfinite(state.u).all()
            full_bytes = 2 * t * config["data"]["head_dim"] * 4
            param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
            reports.append({"events": t, "state_bytes": state.nbytes(), "full_kv_bytes": full_bytes,
                "cache_compression_ratio": full_bytes / state.nbytes(),
                "student_static_plus_state_bytes": param_bytes + buffer_bytes + state.nbytes(),
                "graph_retained": False, "runtime_tensor_count": 5})
    assert len({r["state_bytes"] for r in reports}) == 1
    return {"samples": reports, "write_ms_p50": float(np.percentile(times, 50)),
            "write_ms_p95": float(np.percentile(times, 95)),
            "scope": "Inference state+static modules; includes all diagnostic readouts/probes. "
                     "CPU process RSS includes teacher/training history; not deployment peak."}


def summarize(results, config):
    rows = []
    ev = config["evaluation"]
    # Do not hide weak patterns by pooling: each pattern/delay/mode must be visible.
    for variant in config["variants"]:
        runs = [r for r in results if r["variant"] == variant]
        for pattern in config["data"]["patterns"]:
            for mode in ev["delay_modes"]:
                for delay in ev["delays"]:
                    values = [next(m for m in r["metrics"] if m["view"] == "last_column"
                              and m["pattern"] == pattern and m["delay_mode"] == mode and m["delay"] == delay) for r in runs]
                    row = {"variant": variant, "pattern": pattern, "delay_mode": mode, "delay": delay}
                    for key in ("mse_remote", "zero_mse", "shuffle_mse", "probe_mse"):
                        samples = [v[key] for v in values]
                        row[key + "_mean"] = float(np.mean(samples))
                        row[key + "_std"] = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.
                    def better(x, ref):
                        return x <= ((1 - ev["improvement_fraction"]) * ref
                                     if ref > ev["absolute_tolerance"] else ev["absolute_tolerance"])
                    row["passes_20pct_vs_both"] = all(better(row["mse_remote_mean"], row[k + "_mean"])
                                                       for k in ("zero_mse", "shuffle_mse"))
                    rows.append(row)
    return rows


def plot(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for mode, ax in zip(("wait", "interference"), axes):
        for variant in sorted({r["variant"] for r in results}):
            runs = [r for r in results if r["variant"] == variant]
            for view in VIEWS if variant == "lif" else ("last_column",):
                delays = sorted({m["delay"] for m in runs[0]["metrics"]})
                ys = [np.mean([m["mse_remote"] for r in runs for m in r["metrics"]
                               if m["delay"] == delay and m["delay_mode"] == mode and m["view"] == view]) for delay in delays]
                ax.plot(delays, ys, marker="o", label=f"{variant}/{view}")
        ax.set(xlabel="Delay (external events)", ylabel="Remote-history attention MSE", title=mode)
        ax.legend(fontsize=7)
    fig.savefig(path / "retention.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="experiments/lif_kv_feasibility/configs/synthetic_memory.yaml")
    p.add_argument("--output", default="outputs/synthetic_memory")
    p.add_argument("--device", help="Override config device, e.g. cpu or cuda:1")
    args = p.parse_args()
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    if config["array"]["layers"] != 1:
        raise ValueError("synthetic storage runner currently tests one layer; the topology test covers layer isolation")
    torch.set_num_threads(config["threads"])
    path = start_run(args.output, config)
    if config["device"].startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(config["device"])
    results = [train_one(config, path, seed, variant)
               for variant in config["variants"] for seed in config["seeds"]]
    summary = summarize(results, config)
    memory = bounded_memory(config)
    memory["process_peak_rss_bytes_including_training"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    memory["cuda_peak_allocated_bytes_including_training"] = (
        torch.cuda.max_memory_allocated(config["device"]) if config["device"].startswith("cuda") else None)
    write_json(path / "summary.json", summary)
    write_json(path / "memory.json", memory)
    plot(results, path)
    checks = [r for r in summary if r["variant"] == "lif" and r["delay"] > 0]
    passed = sum(r["passes_20pct_vs_both"] for r in checks)
    (path / "report.md").write_text(
        f"# Synthetic KV storage and delayed readout\n\nLast-column delay checks passing the predeclared criterion: {passed}/{len(checks)}.\n\n"
        f"Synthetic KV storage and delayed readout: {'passed' if passed == len(checks) else 'not passed across all conditions'}. "
        "Three seeds are required by the default config; see per-seed JSON, initial/final validation errors, "
        "parameter counts, probes and summary mean/std.\n\n"
        "Attention training uses remote original history targets; distractors are additionally included only in "
        "the reported full-history metric. Test delays 64/256 exceed training delays. No real VLM results.\n\n"
        "Window baseline uses up to history-length most recent distractors (no full-history quality claim); "
        "wait leaves the exact history unchanged. EMA has the same two allocated float buffers, one redundant. "
        "Readout/probe capacities differ for all-column input and are explicitly counted.\n\n"
        "Real VLM trace collection, real-layer distillation and model replacement: not executed. No task accuracy or end-to-end VLM compression claims.\n",
        encoding="utf-8")
    print(f"Synthetic storage/readout delay criterion: {passed}/{len(checks)}; artifacts: {path}", flush=True)


if __name__ == "__main__":
    main()
