# BioCache

A PyTorch prototype for storing KV information in bounded, layered LIF arrays.
Includes topology/propagation tests and synthetic KV storage/delayed-readout
tests. Real VLM integration is pending. LIF dynamics use native PyTorch, with
custom surrogate gradients; SpikingJelly is not used.

## Setup

```bash
conda env create -f environment.yml
conda activate biocache
```

Existing environment: `/data/user25101366/.conda/envs/biocache`.
Uses Python 3.11 and PyTorch 2.7.1 with CUDA 12.8. Exact dependencies are in
`requirements-lock.txt` and `conda-explicit.txt`.

## Run

From the repository root, with unused output directories:

```bash
python -m pytest -q
python experiments/lif_kv_feasibility/run_propagation.py --output outputs/propagation
python experiments/lif_kv_feasibility/run_synthetic.py --output outputs/synthetic_memory
python experiments/lif_kv_feasibility/build_report.py
```

Experiments default to CPU and require no model downloads. For GPU execution,
add `--device cuda:1 --output outputs/synthetic_cuda` to the synthetic command.
Configs are in `experiments/lif_kv_feasibility/configs/`.

## Validation

- **Unit tests:** dynamics, propagation timing, layer isolation, causality,
  gradients, state restoration, fixed state size, and CUDA execution.
- **Topology and propagation:** inject an isolated current and verify column-by-column propagation.
- **Synthetic KV storage and delayed readout:** train on synthetic KV streams; evaluate delayed Query readout against
  full-KV attention, with state controls and memory measurements.

Recorded results: **17 unit tests passed; propagation passed; storage/readout
met its criterion in 3/24 nonzero-delay conditions and has not passed overall.** Persistent state remained
2,061 bytes at both 1,000 and 10,000 events.

Readout: concatenate membrane voltage and current spikes, map them to fixed
latent K/V slots with an MLP, then apply Query attention. The default reads the
last column; first/all-column readers are controls. See [readout details](docs/readout.md).

See [testing methods](docs/testing.md), [experimental results](experiments/lif_kv_feasibility/report.md),
and [architecture](docs/architecture.md). Raw runs and checkpoints stay in
`outputs/`; recorded evidence is in `experiments/lif_kv_feasibility/results/`.
