# LIF implementation and readout

## LIF operators

The implementation uses **SpikingJelly 0.0.0.0.14**:
`activation_based.neuron.ParametricLIFNode`, with `surrogate.ATan(alpha=2.0)`,
`decay_input=False`, `v_reset=None`, `detach_reset=False`, and the single-step
`torch` backend. See the [official neuron implementation](https://spikingjelly.readthedocs.io/zh-cn/0.0.0.0.14/_modules/spikingjelly/activation_based/neuron.html).

[lif.py](../src/biocache/memory/lif.py) adapts physical parameters and external
state to the unmodified library operators. The library performs charge, firing,
and soft reset; there is no custom autograd spike function.
Adjacent-column currents still use old spikes and `torch.einsum` in
[layered_array.py](../src/biocache/memory/layered_array.py).

```text
Physical dynamics:
  u_pre = exp(-dt / tau) * u_old + current
  spikes = H(u_pre - threshold)
  u_new = u_pre - threshold * spikes

Mapping to the library:
  v_old = u_old / threshold
  x = current / threshold
  sigmoid(w) = 1 - exp(-dt / tau)
  v_new, spikes = ParametricLIFNode(v_old, x, w)
  u_new = threshold * v_new
```

`torch.func.functional_call` supplies the derived per-neuron w tensor from
trainable physical tau for each microstep. The library's original scalar w is
frozen; it is not an additional trained time constant. This keeps per-neuron
tau and threshold trainable and respects variable microstep duration without
subclassing or changing the library charge/fire/reset implementations.
The mapping is equivalent in exact arithmetic; FP32 rounding can differ slightly.

The surrogate is the library's ATan derivative in normalized voltage coordinates,
not the previous custom fast-sigmoid gradient. Training must therefore be rerun.
The library node is reset after each call, including error paths, so all retained
memory stays in the explicit `LIFState`; no duplicate voltage or graph is hidden
in the node. Checkpoint state version 2 records this dynamics implementation and
rejects old version-1 states. The torch backend supports CPU and CUDA; this does
not claim a custom CuPy/CUDA kernel or event-driven acceleration.

## Reading the state

Code: [KVMemory.features, KVMemory.read, and LatentReadout](../src/biocache/memory/readout.py).
The stored state contains membrane voltage `u` and current spikes with shape
`[B, L, C, N]`. It does not retain the original historical KV or a spike history.
For a selected layer, the main reader uses its last column:

```text
z = concat(u[:, layer, -1, :], spikes[:, layer, -1, :])
z.shape = [B, 2*N]
```

Both residual voltage and the latest spikes are available to the reader.
Time, event counters, and labels are not concatenated into these readout features.
The valid flag is used only to mask the final output for an empty state.

### State to latent KV

A learned two-layer MLP transforms z into a fixed set of latent keys and values:

```text
h = SiLU(Linear(z))
packed = Linear(h)
K_latent, V_latent = reshape(packed, [B, 2, Hq, M, Dh]).unbind(dim=1)
```

With the default single-head configuration, `N=64`, hidden width is 64,
`M=4`, and `Dh=8`. The last-column path is therefore:

```text
128 state features -> 64 hidden units -> 64 outputs
                   -> 4 keys of size 8 + 4 values of size 8
```

The MLP depends on the state, not on Q. Each layer and reading location has its
own learned parameters. Latent keys are defined directly in Query space; there
is no extra RoPE operation in this synthetic experiment.

### Query attention

The Query selects information from these latent slots using standard scaled
dot-product attention:

```text
Y_hat = softmax(Q @ K_latent.transpose(-2, -1) / sqrt(Dh)) @ V_latent
Q.shape = Y_hat.shape = [B, Hq, Nq, Dh]
```

The code calls `torch.nn.functional.scaled_dot_product_attention`. Reading leaves
all state tensors and time unchanged. If the layer has no valid history, the
result is masked to zero.

These four slots are learned summaries, not recovered historical tokens.
The objective is to approximate the teacher's attention output; the code does
not guarantee exact reconstruction or retrieval of arbitrary original KV pairs.
Query-conditioned attention chooses among summaries after compression; Query
does not affect past writes or the state-to-latent-KV MLP.

## Reading-location controls and Value probes

Three separately trained readers use:

| Reader | Features | Default feature width |
|---|---|---:|
| First column | First-column voltage and current spikes | 128 |
| Last column, primary | Last-column voltage and current spikes | 128 |
| All columns | Flattened voltage and current spikes from all columns | 512 |

The all-column reader has more parameters because its input is larger.
Every reader generates the same four latent slots.

A separate lightweight MLP probe maps the same features directly to an
8-dimensional Value, without a Query or latent attention. It diagnoses retained
information and is not the main readout path. Training averages attention-output
MSE plus probe MSE over all three locations. See [testing methods](testing.md)
for targets, delays, controls, and observed results.
