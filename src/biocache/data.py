"""Independent synthetic sequences. Teacher history never enters the readout."""
from dataclasses import dataclass
import math
import torch

PATTERNS = ("low_rank", "random_association", "unique_event", "similar_keys")


@dataclass
class Sequence:
    k: torch.Tensor       # [B,T,1,1,D]
    v: torch.Tensor
    q: torch.Tensor       # [B,1,1,D], future query
    event_value: torch.Tensor
    pattern: str


def make_sequence(batch, length, dim, seed, pattern, device="cpu"):
    if pattern not in PATTERNS or length < 2 or dim < 2:
        raise ValueError("unknown pattern or insufficient length/dimension")
    g = torch.Generator().manual_seed(seed)
    def rand(*shape):
        return torch.randn(*shape, generator=g)
    k, v = rand(batch, length, dim), rand(batch, length, dim)
    index = torch.randint(length, (batch,), generator=g)
    if pattern == "low_rank":
        basis = rand(2, dim)
        k = rand(batch, length, 2) @ basis / math.sqrt(2)
        # Sequence-specific low-rank values: no fixed query-to-value shortcut.
        v = rand(batch, length, 2) @ rand(batch, 2, dim) / math.sqrt(2)
    elif pattern == "unique_event":
        k = rand(batch, 1, dim).expand(-1, length, -1).clone() * 0.15
        v = rand(batch, 1, dim).expand(-1, length, -1).clone() * 0.15
        k[torch.arange(batch), index] = rand(batch, dim) * 2
        v[torch.arange(batch), index] = rand(batch, dim)
    elif pattern == "similar_keys":
        k = rand(batch, 1, dim) + 0.03 * rand(batch, length, dim)
    q = k[torch.arange(batch), index] * 2 + 0.02 * rand(batch, dim)
    # A query-free probe needs a predetermined address. Probe the final value,
    # except for unique_event, where the strong key identifies the marked event.
    probe_value = v[torch.arange(batch), index] if pattern == "unique_event" else v[:, -1]
    return Sequence(k[:, :, None, None].to(device), v[:, :, None, None].to(device),
                    q[:, None, None].to(device), probe_value.to(device), pattern)


def teacher_attention(sequence, distractor=None):
    k, v = sequence.k[:, :, 0, 0], sequence.v[:, :, 0, 0]
    if distractor is not None:
        k = torch.cat((k, distractor.k[:, :, 0, 0]), dim=1)
        v = torch.cat((v, distractor.v[:, :, 0, 0]), dim=1)
    return torch.nn.functional.scaled_dot_product_attention(sequence.q, k[:, None], v[:, None])
