# BioCache

A PyTorch prototype for storing KV information in bounded, layered LIF arrays.
Implements propagation (E0) and synthetic memory experiments (E1). Real VLM
integration is pending.

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
- **E0:** inject an isolated current and verify column-by-column propagation.
- **E1:** train on synthetic KV streams; evaluate delayed Query readout against
  full-KV attention, with state controls and memory measurements.

Recorded results: **17 tests passed; E0 passed; E1 met its criterion in 3/24
nonzero-delay conditions and has not passed overall.** Persistent state remained
2,061 bytes at both 1,000 and 10,000 events.

See [testing methods](docs/testing.md), [experimental results](experiments/lif_kv_feasibility/report.md),
and [architecture](docs/architecture.md). Raw runs and checkpoints stay in
`outputs/`; recorded evidence is in `experiments/lif_kv_feasibility/results/`.
