"""Validated configuration; experiment criteria are recorded before training."""
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import yaml


@dataclass(frozen=True)
class ArrayConfig:
    layers: int = 1
    columns: int = 4
    neurons_per_column: int = 64
    microsteps_per_token: int = 4
    event_duration: float = 1.0
    tau: tuple[float, ...] = (16., 64., 256., 1024.)
    threshold: float = 0.5
    topology: str = "adjacent_columns_dense"
    cross_layer_connections: bool = False
    update_mode: str = "synchronous_old_state"
    external_input: str = "first_column_first_microstep_only"
    reset: str = "soft"

    def __post_init__(self):
        for name in ("layers", "columns", "neurons_per_column", "microsteps_per_token"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if len(self.tau) != self.columns or any(not math.isfinite(t) or t <= 0 for t in self.tau):
            raise ValueError("tau must specify one positive time constant per column")
        if any(not math.isfinite(v) or v <= 0 for v in (self.threshold, self.event_duration)):
            raise ValueError("threshold and event_duration must be positive")
        if (self.topology != "adjacent_columns_dense" or self.cross_layer_connections
                or self.update_mode != "synchronous_old_state"
                or self.external_input != "first_column_first_microstep_only"
                or self.reset != "soft"):
            raise ValueError("unsupported topology or dynamics")

    def asdict(self):
        return asdict(self)


def load_config(path):
    with Path(path).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    ArrayConfig(**config["array"])
    return config
