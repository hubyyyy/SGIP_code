from __future__ import annotations

import argparse
from typing import Dict

from sgip_core import SGIPConfig


REACTIONS = ("fkpp", "cubic", "arrhenius", "bistable")
RESAMPLING_METHODS = ("piecewise", "manuscript")


def add_common_arguments(
    parser: argparse.ArgumentParser,
    dimension: int,
    quick_particles: int,
    quick_bins: int,
    paper_particles: int,
    paper_bins: int,
) -> None:
    parser.set_defaults(
        dimension=dimension,
        quick_particles=quick_particles,
        quick_bins=quick_bins,
        paper_particles=paper_particles,
        paper_bins=paper_bins,
    )
    parser.add_argument(
        "--preset", choices=("quick", "paper"), default="quick"
    )
    parser.add_argument("--particles", type=int)
    parser.add_argument("--bins", type=int)
    parser.add_argument("--half-width", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--final-time", type=float, default=20.0)
    parser.add_argument("--diffusion", type=float, default=1.0)
    parser.add_argument("--reaction", choices=REACTIONS, default="fkpp")
    parser.add_argument("--reaction-parameter", type=float, default=0.5)
    parser.add_argument(
        "--resampling", choices=RESAMPLING_METHODS, default=None
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--quiet", action="store_true")


def config_from_arguments(arguments: argparse.Namespace) -> SGIPConfig:
    paper = arguments.preset == "paper"
    particles = arguments.particles
    if particles is None:
        particles = (
            arguments.paper_particles if paper else arguments.quick_particles
        )
    bins = arguments.bins
    if bins is None:
        bins = arguments.paper_bins if paper else arguments.quick_bins
    resampling = arguments.resampling
    if resampling is None:
        resampling = "manuscript"
    backend = resolve_backend("gpu")

    return SGIPConfig(
        dimension=arguments.dimension,
        particles=particles,
        bins=bins,
        half_width=arguments.half_width,
        dt=arguments.dt,
        final_time=arguments.final_time,
        diffusion=arguments.diffusion,
        reaction=arguments.reaction,
        reaction_parameter=arguments.reaction_parameter,
        resampling=resampling,
        backend=backend,
        seed=arguments.seed,
        show_progress=not arguments.quiet,
    )


def resolve_backend(requested: str) -> str:
    if requested != "gpu":
        raise ValueError("the released code supports only the GPU backend")
    from sgip_gpu import cupy_available

    if not cupy_available():
        raise RuntimeError(
            "CuPy could not access a CUDA GPU. Install a CUDA-matched "
            "CuPy package and run on a CUDA-capable device."
        )
    return "gpu"


def simulate_with_backend(
    config,
    initial_sampler,
    initial_mass,
    velocity=None,
    save_times=None,
):
    from sgip_gpu import simulate_sgip_gpu

    return simulate_sgip_gpu(
        config,
        initial_sampler,
        initial_mass,
        velocity=velocity,
        save_times=save_times,
    )


def print_run_summary(result, artifacts: Dict[str, object]) -> None:
    print(
        f"SGIP ({result.config.backend}) completed in {result.elapsed:.3f}s "
        f"({result.step_times.mean():.3f}s per step)."
    )
    for label, path in artifacts.items():
        print(f"{label}: {path}")
