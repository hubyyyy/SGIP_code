from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def run_fdm_1d(
    half_width: float,
    final_time: float,
    diffusion: float,
    reaction: str,
    reaction_parameter: float,
    save_times: Sequence[float],
    initial_left: float = 0.0,
    initial_right: float = 1.0,
    dx: float = 0.05,
    requested_dt: float = 1.0e-3,
) -> Tuple[np.ndarray, Dict[float, np.ndarray]]:
    from sgip_gpu import _import_cupy

    cp = _import_cupy()
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("The 1D FDM reference requires a CUDA GPU.")
    from cupyx.scipy.fft import dctn, idctn

    cells = int(round(2.0 * half_width / dx))
    if cells < 4:
        raise ValueError("the FDM grid must contain at least four cells")
    dx = 2.0 * half_width / cells
    x = np.linspace(-half_width, half_width, cells, endpoint=False)
    x += 0.5 * dx
    steps = int(np.ceil(final_time / requested_dt))
    dt = final_time / steps
    save_steps = {
        int(round(time / dt)): time
        for time in save_times
        if 0.0 <= time <= final_time
    }

    values = cp.asarray(((x >= initial_left) & (x <= initial_right)).astype(np.float64))
    snapshots: Dict[float, np.ndarray] = {}
    if 0 in save_steps:
        snapshots[save_steps[0]] = cp.asnumpy(values)

    modes = cp.arange(cells, dtype=cp.float64)
    eigenvalues = -4.0 * cp.sin(0.5 * cp.pi * modes / cells) ** 2 / dx**2
    diffusion_denominator = 1.0 - dt * diffusion * eigenvalues

    for step in range(1, steps + 1):
        transformed = dctn(values, type=2, norm="ortho", overwrite_x=True)
        transformed /= diffusion_denominator
        values = idctn(transformed, type=2, norm="ortho", overwrite_x=True)
        if reaction == "fkpp":
            exponential = cp.exp(dt)
            values = values * exponential / (1.0 + values * (exponential - 1.0))
        elif reaction == "cubic":
            values += dt * values**2 * (1.0 - values)
        elif reaction == "bistable":
            values += dt * values * (1.0 - values) * (values - reaction_parameter)
        elif reaction == "arrhenius":
            safe = cp.maximum(values, 1.0e-12)
            values += dt * cp.where(
                values > 0.0,
                cp.exp(-reaction_parameter / safe) * (1.0 - values),
                0.0,
            )
        else:
            raise ValueError(f"unknown reaction: {reaction}")
        values = cp.clip(values, 0.0, 1.0)
        if step in save_steps:
            snapshots[save_steps[step]] = cp.asnumpy(values)
    return x, snapshots
