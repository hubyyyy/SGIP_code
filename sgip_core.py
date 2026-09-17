from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


Array = np.ndarray
InitialSampler = Callable[[np.random.Generator, int, np.dtype], Array]
VelocityField = Callable[[Array, float], Array]


@dataclass
class SGIPConfig:
    dimension: int
    particles: int
    bins: int
    half_width: float = 60.0
    dt: float = 0.5
    final_time: float = 20.0
    diffusion: float = 1.0
    reaction: str = "fkpp"
    reaction_parameter: float = 0.5
    resampling: str = "manuscript"
    backend: str = "gpu"
    seed: int = 2025
    dtype: str = "float32"
    show_progress: bool = True

    def validate(self) -> None:
        if self.dimension not in (1, 2, 3):
            raise ValueError("dimension must be 1, 2, or 3")
        if self.particles <= 0 or self.bins <= 1:
            raise ValueError("particles must be positive and bins must exceed one")
        if self.half_width <= 0 or self.dt <= 0 or self.final_time <= 0:
            raise ValueError("half_width, dt, and final_time must be positive")
        if self.diffusion < 0:
            raise ValueError("diffusion must be non-negative")
        if self.resampling not in ("piecewise", "manuscript"):
            raise ValueError("resampling must be 'piecewise' or 'manuscript'")
        if self.backend != "gpu":
            raise ValueError("the released SGIP implementation requires CUDA")
        steps = self.final_time / self.dt
        if not np.isclose(steps, round(steps), rtol=0.0, atol=1.0e-10):
            raise ValueError("final_time must be an integer multiple of dt")


@dataclass
class SGIPResult:
    centers: Array
    snapshots: Dict[float, Array]
    mass_history: Array
    step_times: Array
    elapsed: float
    config: SGIPConfig


def default_save_times(final_time: float, dt: float) -> Tuple[float, ...]:
    preferred = (0.0, 5.0, 10.0, 15.0, 20.0)
    times = [time for time in preferred if time <= final_time + 1.0e-12]
    if not times or not np.isclose(times[-1], final_time):
        times.append(final_time)
    return tuple(sorted(set(round(time / dt) * dt for time in times)))
