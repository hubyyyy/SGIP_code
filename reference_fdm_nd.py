from __future__ import annotations

from time import perf_counter
from typing import Dict, Sequence, Tuple

import numpy as np

def _backend(name: str):
    if name == "gpu":
        try:
            import cupy as cp
            from cupyx.scipy.fft import dctn, idctn
        except ImportError as error:
            raise RuntimeError(
                "The GPU FDM backend requires a CUDA-matched CuPy package."
            ) from error
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy cannot access a CUDA device.")
        return cp, cp.asnumpy, cp.cuda.Stream.null.synchronize, dctn, idctn
    raise ValueError("the released FDM solver supports only the GPU backend")


def _velocity_2d(xp, coordinates, flow: str):
    x = coordinates[:, None]
    y = coordinates[None, :]
    if flow == "none":
        return xp.zeros((1, 1)), xp.zeros((1, 1))
    if flow == "shear":
        return xp.sin(y), xp.zeros((1, 1))
    if flow == "cellular":
        return -xp.sin(x) * xp.cos(y), xp.cos(x) * xp.sin(y)
    if flow == "cat_eye":
        return (
            -xp.sin(x) * xp.cos(y) + 2.0 * xp.cos(x) * xp.sin(y),
            xp.cos(x) * xp.sin(y) - 2.0 * xp.sin(x) * xp.cos(y),
        )
    raise ValueError(f"unknown 2D flow: {flow}")


def _velocity_3d(xp, coordinates, flow: str):
    if flow != "abc":
        raise ValueError("the 3D reference solver supports only the ABC flow")
    x = coordinates[:, None, None]
    y = coordinates[None, :, None]
    z = coordinates[None, None, :]
    coefficient_a = 1.0
    coefficient_b = np.sqrt(2.0 / 3.0)
    coefficient_c = np.sqrt(1.0 / 3.0)
    return (
        coefficient_a * xp.sin(z) + coefficient_c * xp.cos(y),
        coefficient_b * xp.sin(x) + coefficient_a * xp.cos(z),
        coefficient_c * xp.sin(y) + coefficient_b * xp.cos(x),
    )


def _shift_neumann(xp, values, axis: int, direction: int):
    shifted = xp.roll(values, direction, axis=axis)
    boundary = [slice(None)] * values.ndim
    source = [slice(None)] * values.ndim
    if direction > 0:
        boundary[axis] = 0
        source[axis] = 0
    else:
        boundary[axis] = -1
        source[axis] = -1
    shifted[tuple(boundary)] = values[tuple(source)]
    return shifted


def _reaction(xp, values, name: str, parameter: float):
    if name == "fkpp":
        return values * (1.0 - values)
    if name == "cubic":
        return values**2 * (1.0 - values)
    if name == "bistable":
        return values * (1.0 - values) * (values - parameter)
    if name == "arrhenius":
        safe = xp.maximum(values, 1.0e-12)
        return xp.where(
            values > 0.0,
            xp.exp(-parameter / safe) * (1.0 - values),
            0.0,
        )
    raise ValueError(f"unknown reaction: {name}")


def _initial_values(
    xp,
    centers,
    dimension: int,
    initial_kind: str,
    initial_radius: float,
):
    if initial_kind == "square":
        if dimension != 2:
            raise ValueError("square initial data is only defined in 2D")
        mask = (centers >= 0.0) & (centers <= 1.0)
        values = xp.zeros((centers.size, centers.size), dtype=xp.float32)
        values[xp.ix_(mask, mask)] = 1.0
        return values
    if initial_kind == "ball":
        if dimension != 3:
            raise ValueError("ball initial data is only defined in 3D")
        radius_squared = (
            centers[:, None, None] ** 2
            + centers[None, :, None] ** 2
            + centers[None, None, :] ** 2
        )
        return (radius_squared <= initial_radius**2).astype(xp.float32)
    raise ValueError(f"unknown initial condition: {initial_kind}")


def run_fdm_nd(
    *,
    dimension: int,
    half_width: float,
    cells_per_axis: int,
    final_time: float,
    requested_dt: float,
    diffusion: float,
    flow: str,
    reaction: str,
    reaction_parameter: float = 0.5,
    initial_kind: str,
    initial_radius: float = 1.0,
    save_times: Sequence[float] = (20.0,),
    backend: str = "gpu",
    max_cells: int | None = 16_000_000,
) -> Tuple[np.ndarray, Dict[float, np.ndarray], float]:
    """IMEX finite-difference solver used for the paper comparisons.

    The equation is written as u_t = D Delta u + w dot grad(u) + r(u).
    Advection uses first-order upwinding, diffusion is treated implicitly
    through a DCT diagonalization of the Neumann Laplacian, and reaction is
    advanced pointwise on a CUDA GPU.
    """
    if dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3")
    if cells_per_axis < 4:
        raise ValueError("cells_per_axis must be at least four")
    if requested_dt <= 0.0 or final_time <= 0.0:
        raise ValueError("time parameters must be positive")
    total_cells = cells_per_axis**dimension
    if max_cells is not None and total_cells > max_cells:
        gib = total_cells * np.dtype(np.float32).itemsize / 1024**3
        raise RuntimeError(
            f"the requested grid has {total_cells:,} cells "
            f"({gib:.2f} GiB per float32 array); pass --allow-large-fdm "
            "only on a machine with adequate memory"
        )

    xp, to_numpy, synchronize, dctn, idctn = _backend(backend)
    dx = 2.0 * half_width / cells_per_axis
    host_centers = np.linspace(
        -half_width, half_width, cells_per_axis, endpoint=False
    )
    host_centers += 0.5 * dx
    centers = xp.asarray(host_centers, dtype=xp.float32)
    values = _initial_values(
        xp, centers, dimension, initial_kind, initial_radius
    )
    velocities = (
        _velocity_2d(xp, centers, flow)
        if dimension == 2
        else _velocity_3d(xp, centers, flow)
    )

    steps = int(np.ceil(final_time / requested_dt))
    dt = final_time / steps
    save_steps = {
        int(round(float(time) / dt)): round(float(time), 12)
        for time in save_times
        if 0.0 <= float(time) <= final_time
    }
    snapshots: Dict[float, np.ndarray] = {}
    if 0 in save_steps:
        snapshots[save_steps[0]] = to_numpy(values)

    modes = xp.arange(cells_per_axis, dtype=xp.float32)
    eigenvalues_1d = -4.0 * xp.sin(
        0.5 * xp.pi * modes / cells_per_axis
    ) ** 2 / dx**2
    diffusion_eigenvalues = eigenvalues_1d[:, None] + eigenvalues_1d[None, :]
    if dimension == 3:
        diffusion_eigenvalues = (
            diffusion_eigenvalues[:, :, None]
            + eigenvalues_1d[None, None, :]
        )
    diffusion_denominator = 1.0 - dt * diffusion * diffusion_eigenvalues

    synchronize()
    start = perf_counter()
    for step in range(1, steps + 1):
        old = values
        advection = xp.zeros_like(old)
        for axis, velocity in enumerate(velocities):
            lower = _shift_neumann(xp, old, axis, 1)
            upper = _shift_neumann(xp, old, axis, -1)
            forward = (upper - old) / dx
            backward = (old - lower) / dx
            advection += velocity * xp.where(
                velocity >= 0.0, forward, backward
            )
        rhs = old + dt * advection
        transformed = dctn(rhs, type=2, norm="ortho", overwrite_x=True)
        transformed /= diffusion_denominator
        values = idctn(
            transformed, type=2, norm="ortho", overwrite_x=True
        )
        if reaction == "fkpp":
            exponential = xp.exp(dt)
            values = values * exponential / (
                1.0 + values * (exponential - 1.0)
            )
        else:
            values = values + dt * _reaction(
                xp, values, reaction, reaction_parameter
            )
        values = xp.clip(values, 0.0, 1.0)
        if step in save_steps:
            snapshots[save_steps[step]] = to_numpy(values)

    synchronize()
    return host_centers, snapshots, perf_counter() - start
