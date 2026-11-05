from dataclasses import replace
import math
import pytest
import torch
from biocache.config import ArrayConfig
from biocache.memory import LIFArray, KVMemory
from biocache.memory.layered_array import HardSpike
from biocache.data import make_sequence, teacher_attention

torch.set_num_threads(1)


def cfg(**kwargs):
    return ArrayConfig(layers=2, columns=3, neurons_per_column=4,
                       microsteps_per_token=3, tau=(4., 16., 64.), **kwargs)


def test_sync_update_matches_scalar_reference():
    torch.manual_seed(5)
    a = LIFArray(cfg())
    s = a.initial_state(2)
    s = replace(s, u=torch.randn_like(s.u), spikes=torch.randint(0, 2, s.u.shape).float())
    x = torch.randn(2, 2, 4)
    out = a.microstep(s, x, 0.25)
    for b in range(2):
        for l in range(2):
            for c in range(3):
                for n in range(4):
                    drive = x[b, l, n] if c == 0 else (a.W[l, c-1, n] * s.spikes[b, l, c-1]).sum()
                    pre = math.exp(-0.25 / a.log_tau[l, c, n].exp().item()) * s.u[b, l, c, n] + drive
                    spike = float(pre >= 0.5)
                    torch.testing.assert_close(out.u[b, l, c, n], pre - 0.5 * spike)
                    assert out.spikes[b, l, c, n] == spike


def test_no_same_step_jump_and_broken_chain():
    a = LIFArray(cfg())
    with torch.no_grad():
        a.W.fill_(0.9 / 4)
        for broken in (False, True):
            s = a.initial_state(1)
            if broken:
                a.connection_mask[:, 0] = 0
            first = []
            for t in range(3):
                x = torch.full((1, 2, 4), 0.75) if t == 0 else torch.zeros(1, 2, 4)
                s = a.microstep(s, x, 1/3)
                first.append(s.spikes[0, 0].any(dim=-1).tolist())
                if t == 0:
                    torch.testing.assert_close(s.u[:, :, 0], torch.full((1, 2, 4), 0.25))
            assert first[0] == [True, False, False]
            assert first[1] == [False, not broken, False]
            assert first[2] == [False, False, not broken]


def test_analytical_decay_signed_voltage_and_time():
    a = LIFArray(cfg())
    s = a.initial_state(1)
    s = replace(s, u=torch.full_like(s.u, -2.))
    with torch.no_grad():
        out = a.event(s, torch.zeros(1, 2, 4), write=False)
    torch.testing.assert_close(out.u, -2 * torch.exp(-1/a.log_tau.exp()).unsqueeze(0))
    assert not out.spikes.any() and not out.valid.any()
    torch.testing.assert_close(out.time, torch.ones(1, 2))
    assert torch.all(out.events == 1)


def test_layer_isolation_even_with_existing_state():
    a = LIFArray(cfg())
    s = a.initial_state(1)
    s = replace(s, u=torch.randn_like(s.u), spikes=torch.ones_like(s.spikes))
    out = a.event(s, torch.randn(1, 2, 4), active=torch.tensor([[True, False]]))
    for name in ("u", "spikes", "time", "events", "valid"):
        torch.testing.assert_close(getattr(out, name)[:, 1], getattr(s, name)[:, 1])


def test_read_purity_empty_state_and_reset():
    m = KVMemory(cfg(), kv_heads=1, query_heads=2, head_dim=4)
    s = m.array.initial_state(2)
    q = torch.randn(2, 2, 3, 4)
    before = m.export_state(s)
    assert torch.equal(m.read(s, q), torch.zeros_like(q))
    for name, value in before["state"].items():
        torch.testing.assert_close(value, getattr(s, name))
    s = m.write(s, torch.randn(2, 2, 1, 4), torch.randn(2, 2, 1, 4), torch.zeros(2, 2))
    assert s.valid.all()
    fresh = m.array.initial_state(2)
    assert not fresh.valid.any() and not fresh.u.any() and not fresh.events.any()


def test_gradients_and_hard_forward_identical():
    torch.manual_seed(51)
    m = KVMemory(cfg(), head_dim=4)
    k, v = torch.randn(3, 2, 1, 4), torch.randn(3, 2, 1, 4)
    s = m.write(m.array.initial_state(3), k, v, torch.zeros(3, 2))
    # The signal reaches the last column only at the final microstep of the
    # first event. Advance again so last-column tau participates in the loss.
    s = m.wait(s)
    q = torch.randn(3, 1, 1, 4)
    loss = m.read(s, q).square().mean()
    loss.backward()
    for p in (m.array.W, m.array.log_tau, m.array.log_threshold, m.encoders[0].weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    with torch.no_grad():
        infer = m.write(m.array.initial_state(3), k, v, torch.zeros(3, 2))
        infer = m.wait(infer)
    torch.testing.assert_close(infer.u, s.u, rtol=0, atol=0)
    torch.testing.assert_close(infer.spikes, s.spikes, rtol=0, atol=0)
    assert torch.equal(HardSpike.apply(torch.tensor([-1., 0., 1.])), torch.tensor([0., 1., 1.]))


def test_serialization_continuation_and_incompatible_state():
    m = KVMemory(cfg(), head_dim=4)
    with torch.no_grad():
        s = m.write(m.array.initial_state(1), torch.randn(1, 2, 1, 4),
                    torch.randn(1, 2, 1, 4), torch.zeros(1, 2))
        loaded = m.import_state(m.export_state(s))
        torch.testing.assert_close(m.wait(s).u, m.wait(loaded).u, rtol=0, atol=0)
        assert loaded.u.data_ptr() != s.u.data_ptr()
    invalid = m.export_state(s)
    invalid["version"] = 100
    with pytest.raises(ValueError):
        m.import_state(invalid)


def test_causal_prefix_unchanged_by_future_input():
    m = KVMemory(cfg(), head_dim=4)
    k, v = torch.randn(2, 5, 2, 1, 4), torch.randn(2, 5, 2, 1, 4)
    def stream(keys):
        s, outputs = m.array.initial_state(2), []
        for t in range(5):
            outputs.append(m.read(s, torch.ones(2, 1, 1, 4)))
            s = m.write(s, keys[:, t], v[:, t], torch.full((2, 2), float(t)))
        return outputs
    with torch.no_grad():
        original = stream(k)
        changed = k.clone()
        changed[:, 3:] *= 10
        other = stream(changed)
    for t in range(4):
        torch.testing.assert_close(original[t], other[t], rtol=0, atol=0)


def test_tbptt_cuts_graph_and_inference_state_size_fixed():
    m = KVMemory(cfg(), head_dim=4)
    s = m.array.initial_state(1)
    with torch.no_grad():
        size = s.nbytes()
        for _ in range(100):
            s = m.write(s, torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4), s.time)
        assert s.nbytes() == size and s.u.grad_fn is None
    s = m.write(s, torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4), s.time)
    assert s.u.grad_fn is not None and s.detach().u.grad_fn is None


@pytest.mark.parametrize("kwargs", [dict(tau=(1.,)), dict(cross_layer_connections=True),
    dict(reset="hard"), dict(threshold=-1.), dict(event_duration=float("nan"))])
def test_config_rejects_invalid_contract(kwargs):
    with pytest.raises(ValueError):
        ArrayConfig(**kwargs)


def test_invalid_head_contract_and_kv_shape():
    with pytest.raises(ValueError):
        KVMemory(cfg(), query_heads=3, kv_heads=2)
    m = KVMemory(cfg(), head_dim=4)
    with pytest.raises(ValueError):
        m.write(m.array.initial_state(1), torch.zeros(1, 2, 4), torch.zeros(1, 2, 4), torch.zeros(1, 2))


def test_synthetic_split_and_teacher_attention_reference():
    seq = make_sequence(4, 8, 4, 123, "random_association")
    other = make_sequence(4, 8, 4, 124, "random_association")
    assert not torch.equal(seq.v, other.v)
    k, v = seq.k[:, :, 0, 0], seq.v[:, :, 0, 0]
    score = seq.q[:, 0] @ k.transpose(1, 2) / 2
    reference = (score.softmax(-1) @ v)[:, None]
    torch.testing.assert_close(teacher_attention(seq), reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_write_read_backward():
    # GPU 1 has substantially more free memory on the development machine.
    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    m = KVMemory(cfg(), head_dim=4).to(device)
    s = m.write(m.array.initial_state(2), torch.randn(2, 2, 1, 4, device=device),
                torch.randn(2, 2, 1, 4, device=device), torch.zeros(2, 2, device=device))
    out = m.read(s, torch.randn(2, 1, 1, 4, device=device))
    out.square().mean().backward()
    assert torch.isfinite(out).all() and m.array.W.grad is not None
