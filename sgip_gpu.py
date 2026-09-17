from __future__ import annotations

from time import perf_counter
from typing import Optional, Sequence, Tuple

import numpy as np

from sgip_core import (
    InitialSampler,
    SGIPConfig,
    SGIPResult,
    VelocityField,
    default_save_times,
)


def _import_cupy():
    try:
        import cupy as cp
    except ImportError as error:
        raise RuntimeError(
            "The GPU backend requires CuPy. Install the CuPy package that "
            "matches the CUDA version on this machine, for example "
            "'pip install cupy-cuda12x'."
        ) from error
    return cp


def cupy_available() -> bool:
    try:
        cp = _import_cupy()
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _reconstruct_density_gpu(
    cp,
    particles,
    total_mass,
    config: SGIPConfig,
):
    width = 2.0 * config.half_width / config.bins
    scaled = cp.floor((particles + config.half_width) / width).astype(
        cp.int32, copy=False
    )
    valid = cp.all((scaled >= 0) & (scaled < config.bins), axis=1)
    valid_indices = scaled[valid]

    flat_ids = cp.zeros(valid_indices.shape[0], dtype=cp.int64)
    for axis in range(config.dimension):
        flat_ids *= config.bins
        flat_ids += valid_indices[:, axis]

    number_of_bins = config.bins ** config.dimension
    counts = cp.bincount(flat_ids, minlength=number_of_bins)
    particle_mass = total_mass / config.particles
    cell_volume = width ** config.dimension
    density = counts.astype(cp.float64) * (particle_mass / cell_volume)
    density = density.reshape((config.bins,) * config.dimension)
    return density, flat_ids, valid


def _reaction_value_gpu(cp, values, name: str, parameter: float):
    if name == "fkpp":
        return values * (1.0 - values)
    if name == "cubic":
        return values**2 * (1.0 - values)
    if name == "bistable":
        return values * (1.0 - values) * (values - parameter)
    if name == "arrhenius":
        safe = cp.maximum(values, 1.0e-12)
        result = cp.exp(-parameter / safe) * (1.0 - values)
        return cp.where(values > 0.0, result, 0.0)
    raise ValueError(f"unknown reaction: {name}")


def _reaction_derivative_gpu(cp, values, name: str, parameter: float):
    if name == "cubic":
        return 2.0 * values - 3.0 * values**2
    if name == "bistable":
        return (
            -3.0 * values**2
            + 2.0 * (1.0 + parameter) * values
            - parameter
        )
    if name == "arrhenius":
        safe = cp.maximum(values, 1.0e-12)
        exponential = cp.exp(-parameter / safe)
        derivative = exponential * (
            parameter * (1.0 - values) / safe**2 - 1.0
        )
        return cp.where(values > 0.0, derivative, 0.0)
    raise ValueError(f"no implicit derivative implemented for: {name}")


def _apply_reaction_gpu(cp, density, config: SGIPConfig):
    result = density.copy()
    active = result > 0.0
    old = result[active]
    if config.reaction == "fkpp":
        exponential = cp.exp(config.dt)
        result[active] = (
            old * exponential
            / (1.0 + old * (exponential - 1.0))
        )
        return cp.clip(result, 0.0, 1.0)

    updated = cp.clip(
        old
        + config.dt
        * _reaction_value_gpu(
            cp, old, config.reaction, config.reaction_parameter
        ),
        0.0,
        1.0,
    )
    for _ in range(10):
        residual = (
            updated
            - old
            - config.dt
            * _reaction_value_gpu(
                cp,
                updated,
                config.reaction,
                config.reaction_parameter,
            )
        )
        jacobian = (
            1.0
            - config.dt
            * _reaction_derivative_gpu(
                cp,
                updated,
                config.reaction,
                config.reaction_parameter,
            )
        )
        correction = residual / cp.where(
            cp.abs(jacobian) > 1.0e-12, jacobian, 1.0
        )
        updated = cp.clip(updated - correction, 0.0, 1.0)
    result[active] = updated
    return result


def _velocity_gpu(cp, particles, velocity: VelocityField):
    flow_name = getattr(velocity, "sgip_flow_name", None)
    drift_sign = getattr(velocity, "sgip_drift_sign", None)
    if flow_name is None or drift_sign is None:
        raise ValueError(
            "The GPU backend supports velocity fields returned by "
            "sgip_models.make_velocity()."
        )

    result = cp.zeros_like(particles)
    if flow_name == "shear":
        result[:, 0] = cp.sin(particles[:, 1])
    elif flow_name == "cellular":
        x = particles[:, 0]
        y = particles[:, 1]
        result[:, 0] = -cp.sin(x) * cp.cos(y)
        result[:, 1] = cp.cos(x) * cp.sin(y)
    elif flow_name == "cat_eye":
        x = particles[:, 0]
        y = particles[:, 1]
        result[:, 0] = (
            -cp.sin(x) * cp.cos(y)
            + 2.0 * cp.cos(x) * cp.sin(y)
        )
        result[:, 1] = (
            cp.cos(x) * cp.sin(y)
            - 2.0 * cp.sin(x) * cp.cos(y)
        )
    elif flow_name == "abc":
        coefficient_a = 1.0
        coefficient_b = np.sqrt(2.0 / 3.0)
        coefficient_c = np.sqrt(1.0 / 3.0)
        x = particles[:, 0]
        y = particles[:, 1]
        z = particles[:, 2]
        result[:, 0] = (
            coefficient_a * cp.sin(z)
            + coefficient_c * cp.cos(y)
        )
        result[:, 1] = (
            coefficient_b * cp.sin(x)
            + coefficient_a * cp.cos(z)
        )
        result[:, 2] = (
            coefficient_c * cp.sin(y)
            + coefficient_b * cp.cos(x)
        )
    else:
        raise ValueError(f"unsupported GPU flow: {flow_name}")
    return drift_sign * result


def _draw_target_bins_gpu(cp, density, particle_count: int, rng):
    flat_density = density.ravel()
    active = cp.flatnonzero(flat_density > 0.0)
    probabilities = flat_density[active].astype(cp.float64, copy=True)
    probabilities /= probabilities.sum()
    cumulative = cp.cumsum(probabilities)
    cumulative[-1] = 1.0
    uniforms = rng.random_sample(particle_count).astype(cp.float64)
    active_indices = cp.searchsorted(
        cumulative, uniforms, side="right"
    )
    active_indices = cp.minimum(active_indices, active.size - 1)
    return active[active_indices]


def _sample_uniform_in_bins_gpu(
    cp,
    flat_bins,
    config: SGIPConfig,
    rng,
    dtype,
):
    count = flat_bins.size
    particles = cp.empty((count, config.dimension), dtype=dtype)
    work = flat_bins.copy()
    width = 2.0 * config.half_width / config.bins
    for axis in range(config.dimension - 1, -1, -1):
        indices = work % config.bins
        work //= config.bins
        offsets = rng.random_sample(count).astype(dtype)
        particles[:, axis] = (
            -config.half_width
            + (indices.astype(dtype) + offsets) * width
        )
    return particles


def _resample_piecewise_gpu(
    cp, density, config: SGIPConfig, rng, dtype
):
    sampled_bins = _draw_target_bins_gpu(
        cp, density, config.particles, rng
    )
    return _sample_uniform_in_bins_gpu(
        cp, sampled_bins, config, rng, dtype
    )


def _resample_manuscript_gpu(
    cp,
    particles,
    density,
    flat_ids,
    valid,
    config: SGIPConfig,
    rng,
    dtype,
):
    sampled_bins = _draw_target_bins_gpu(
        cp, density, config.particles, rng
    )
    sampled_bins.sort()
    new_particles = cp.empty(
        (config.particles, config.dimension), dtype=dtype
    )

    number_of_bins = config.bins ** config.dimension
    current_counts = cp.bincount(flat_ids, minlength=number_of_bins)
    target_counts = cp.bincount(
        sampled_bins, minlength=number_of_bins
    )
    clone_mask = current_counts[sampled_bins] > 0
    empty_mask = ~clone_mask
    new_particles[empty_mask] = _sample_uniform_in_bins_gpu(
        cp, sampled_bins[empty_mask], config, rng, dtype
    )

    valid_particle_indices = cp.flatnonzero(valid)
    random_keys = rng.random_sample(flat_ids.size).astype(cp.float32)
    # CuPy lexsort requires a homogeneous 2D array; two stable sorts preserve
    # integer bin IDs and randomize the order within each bin.
    random_order = cp.argsort(random_keys)
    grouped_order = random_order[
        cp.argsort(flat_ids[random_order])
    ]
    grouped_sources = valid_particle_indices[grouped_order]
    current_starts = cp.cumsum(current_counts) - current_counts
    target_starts = cp.cumsum(target_counts) - target_counts

    within_target_rank = cp.arange(
        config.particles, dtype=cp.int64
    ) - target_starts[sampled_bins]
    target_count_per_sample = target_counts[sampled_bins]
    current_count_per_sample = current_counts[sampled_bins]
    without_replacement = (
        clone_mask & (target_count_per_sample <= current_count_per_sample)
    )

    replacement_offsets = (
        rng.random_sample(config.particles)
        * cp.maximum(current_count_per_sample, 1)
    ).astype(cp.int64)
    source_offsets = cp.where(
        without_replacement,
        within_target_rank,
        replacement_offsets,
    )
    grouped_positions = (
        current_starts[sampled_bins[clone_mask]]
        + source_offsets[clone_mask]
    )
    source_indices = grouped_sources[grouped_positions]
    new_particles[clone_mask] = particles[source_indices]
    return new_particles


def simulate_sgip_gpu(
    config: SGIPConfig,
    initial_sampler: InitialSampler,
    initial_mass: float,
    velocity: Optional[VelocityField] = None,
    save_times: Optional[Sequence[float]] = None,
) -> SGIPResult:
    cp = _import_cupy()
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("No CUDA-capable GPU was detected by CuPy.")
    config.validate()

    dtype = cp.dtype(config.dtype)
    host_dtype = np.dtype(config.dtype)
    host_rng = np.random.default_rng(config.seed)
    host_particles = initial_sampler(
        host_rng, config.particles, host_dtype
    )
    expected_shape = (config.particles, config.dimension)
    if host_particles.shape != expected_shape:
        raise ValueError(
            f"initial_sampler returned {host_particles.shape}, "
            f"expected {expected_shape}"
        )
    particles = cp.asarray(host_particles, dtype=dtype)
    del host_particles
    rng = cp.random.RandomState(config.seed)

    number_of_steps = int(round(config.final_time / config.dt))
    if save_times is None:
        save_times = default_save_times(config.final_time, config.dt)
    save_steps = {
        int(round(time / config.dt)): round(time, 12)
        for time in save_times
        if 0.0 <= time <= config.final_time + 1.0e-12
    }
    save_steps[0] = 0.0

    centers = np.linspace(
        -config.half_width,
        config.half_width,
        config.bins,
        endpoint=False,
        dtype=np.float64,
    )
    centers += config.half_width / config.bins
    snapshots = {}
    mass_history_device = cp.empty(
        number_of_steps + 1, dtype=cp.float64
    )
    total_mass = cp.asarray(initial_mass, dtype=cp.float64)

    density, flat_ids, valid = _reconstruct_density_gpu(
        cp, particles, total_mass, config
    )
    snapshots[0.0] = cp.asnumpy(density.astype(cp.float32))
    cell_volume = (
        2.0 * config.half_width / config.bins
    ) ** config.dimension
    mass_history_device[0] = density.sum() * cell_volume

    progress_stride = max(1, number_of_steps // 10)
    timing_events = []
    cp.cuda.Stream.null.synchronize()
    start = perf_counter()
    for step in range(1, number_of_steps + 1):
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        start_event.record()

        if velocity is not None:
            particles += config.dt * _velocity_gpu(
                cp, particles, velocity
            )
        if config.diffusion > 0.0:
            noise = rng.standard_normal(particles.shape).astype(dtype)
            particles += np.sqrt(
                2.0 * config.diffusion * config.dt
            ) * noise

        density, flat_ids, valid = _reconstruct_density_gpu(
            cp, particles, total_mass, config
        )
        density = _apply_reaction_gpu(cp, density, config)
        total_mass = density.sum() * cell_volume
        mass_history_device[step] = total_mass

        if step in save_steps:
            snapshots[save_steps[step]] = cp.asnumpy(
                density.astype(cp.float32)
            )

        if config.resampling == "piecewise":
            particles = _resample_piecewise_gpu(
                cp, density, config, rng, dtype
            )
        else:
            particles = _resample_manuscript_gpu(
                cp,
                particles,
                density,
                flat_ids,
                valid,
                config,
                rng,
                dtype,
            )

        end_event.record()
        timing_events.append((start_event, end_event))
        if config.show_progress and (
            step == 1
            or step == number_of_steps
            or step % progress_stride == 0
        ):
            end_event.synchronize()
            step_seconds = (
                cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0
            )
            print(
                f"GPU step {step:>3}/{number_of_steps}, "
                f"t={step * config.dt:>5.1f}, "
                f"mass={float(total_mass.item()):.6g}, "
                f"step_time={step_seconds:.3f}s"
            )

    cp.cuda.Stream.null.synchronize()
    elapsed = perf_counter() - start
    step_times = np.array(
        [
            cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0
            for start_event, end_event in timing_events
        ],
        dtype=np.float64,
    )
    mass_history = cp.asnumpy(mass_history_device)
    return SGIPResult(
        centers=centers,
        snapshots=snapshots,
        mass_history=mass_history,
        step_times=step_times,
        elapsed=elapsed,
        config=config,
    )
