# Testing methods

Validation separates implementation correctness, signal propagation, and useful
information retention. Correct dynamics and propagation do not establish useful
KV storage or readout quality.
Current experiments use synthetic KV tensors without a pretrained VLM.

## Implementation tests

Run `python -m pytest -q` from the repository root in the `biocache` environment.
[test_memory.py](../tests/test_memory.py) contains 17 cases, including five
parameterized invalid-configuration cases.

| Area | Method |
|---|---|
| Dynamics | Compare tensor updates with a scalar reference and zero-input decay with the analytical exponential. Check soft reset and elapsed time. |
| Topology | Verify that a pulse advances at most one column per microstep; cut a connection and check downstream silence. |
| Isolation | Update one layer and verify an inactive layer with existing state remains unchanged. |
| Reading | Check zero output for empty memory, read-only behavior, and fresh-state reset. |
| Gradients | Require finite nonzero gradients for encoder, W, tau, and threshold; compare training and inference hard-spike forward results. |
| Restoration | Export/import state, reject incompatible versions, and compare the next update exactly. |
| Causality | Change future KV and require earlier reads to remain identical. |
| Memory | Stream 100 events without autograd, check fixed tensor bytes, and verify detach cuts the training graph. |
| Contracts | Reject invalid topology, dimensions, and head ratios; compare teacher attention with explicit softmax attention. |
| CUDA | Run a small GPU write/read/backward pass and check finite output and a W gradient. |

The local run recorded **17 passed**, including CUDA. Without CUDA that case is
skipped. GitHub CI uses CPU PyTorch and runs pytest plus the propagation test;
it does not run the full synthetic storage experiment.

## Topology and propagation test

[run_propagation.py](../experiments/lif_kv_feasibility/run_propagation.py) starts
from zero state with two independent layers, four columns, and 16 neurons per
column. Adjacent-column weights are `0.9 / 16`, threshold is `0.5`.

Inject current `0.75` into layer 0, column 0, only at the first microstep. Observe
16 microsteps without further external input. Assertions check first spikes at
microsteps **1, 2, 3, 4**, finite states, initial soft-reset voltage `0.25`, and
zero activity in other layers. Repeat after cutting the first connection:
downstream columns must remain zero. Save voltage heatmaps, spike raster,
trajectory, and metrics. The recorded propagation run passed.

## Synthetic KV storage and delayed readout test

Entry: [run_synthetic.py](../experiments/lif_kv_feasibility/run_synthetic.py).
Settings: [synthetic_memory.yaml](../experiments/lif_kv_feasibility/configs/synthetic_memory.yaml).
State-to-output computation is documented in [LIF readout](readout.md).

### Training procedure

1. Generate eight-token KV histories with one KV head and head dimension 8.
   Use four distributions: low-rank background, random associations, a unique
   event amid repeated background, and similar keys with different values.
2. Construct a query from a selected historical key plus noise. The query is
   used only for readout and supervision, never for historical writes.
3. Write KV sequentially into four columns of 64 neurons. Each event executes
   four synchronous microsteps; only the first column's first microstep receives
   external current. There is no additional flush between events.
4. Delay reading by 0/4/16 events during training, using either zero-input waiting
   or interfering random KV writes. Waiting still advances the dynamics.
5. The teacher retains the original history and computes
   `Y = softmax(Q K^T / sqrt(8)) V`. The student generates four latent KV slots
   from state and attends to them with Q. Train separate first/last/all-column
   readers and query-free Value probes. The probe recovers the marked event's
   Value for the unique-event distribution, or the final token's Value otherwise.
6. Minimize `attention MSE + probe MSE`, averaged over the three reading locations.
   Use Adam, learning rate 0.002, batch size 32, and 100 steps with fresh sequences.
   TBPTT is configured at 32 events; default training histories plus delay are
   at most 24 events, so these runs do not cross an internal truncation boundary.

Run seeds **11, 22, 33** for each variant: trainable LIF, linear EMA, disconnected
LIF, and LIF with frozen W: **12 training runs**. The student state contains no
original historical KV. The recorded experiment runs used CPU, one thread.

### Held-out evaluation

For each seed and distribution, use 64 held-out sequences with seeds separate
from training. Evaluate before training at delays 0/16 and after training at
**0/16/64/256**, for both waiting and interference. Delays 64/256 therefore test
extrapolation. Each event advances time by 1 and executes four microsteps.

Measure attention MSE, relative L2, Value-probe MSE, and first/last/all-column
results. The main target uses only the original history (`mse_remote`). A
separate metric includes interference in the teacher history (`mse_full`);
that is not the training target.

Controls include zero state, states cyclically shifted between samples, EMA,
cut connections, frozen W, and a recent exact window. Zero-state control preserves
the valid flag to measure learned readout priors. All-column readers have more
parameters; EMA allocates the same state bytes with a redundant second buffer.
These comparisons are diagnostic and do not fully match effective model capacity.

The predeclared acceptance rule requires **last-column attention MSE** to be at
least 20% below both zero-state and shifted-state controls in the three-seed
mean; use absolute tolerance `1e-6` for near-zero controls. Judge each distribution,
nonzero delay, and mode separately: `4 x 3 x 2 = 24 conditions`. Report means
and sample standard deviations. No statistical significance test is performed;
probe scores are diagnostic and are not part of the acceptance rule.

The recorded run passed **3/24 conditions**: similar keys under zero-input
waiting at delays 16/64/256. The storage/readout test has not passed overall. All three LIF seeds
improved held-out MSE after training, which alone does not meet this criterion.

### State, resources, and restoration

A separate inference run uses a newly initialized LIF model and generates then
discards KV pairs under `torch.no_grad()`. At 1,000 and 10,000 events, compare
state tensor bytes, require finite voltage, and check for absent autograd graphs.
Both checkpoints used **2,061 bytes**. This verifies bounded tensor state, not a
comprehensive allocator or Python heap leak audit. Static parameters and buffers
are reported separately; CPU peak RSS includes the teacher and training process.

Every training run saves model, optimizer, configuration, and memory state.
Restore model/state and require exact equality of current readout, next-event
voltage, and next-event readout. Record gradient norms, voltage ranges, state
activity, batch write/read latency, and parameter counts alongside errors.

## Reproduce and inspect

Use the commands in [README](../README.md). Existing run directories are rejected;
choose fresh output paths for reruns. For custom paths, the report builder accepts
`--propagation PATH --synthetic PATH --output PATH`.

- `outputs/propagation/`: full topology and propagation artifacts.
- `outputs/synthetic_memory/`: per-seed metrics, checkpoints, summary, and memory data.
- [Recorded evidence](../experiments/lif_kv_feasibility/results/): committed JSON, CSV, configurations, and plots.
- [Report](../experiments/lif_kv_feasibility/report.md): observed results and limitations.

Real model traces, online model replacement, and video QA have not been tested.
