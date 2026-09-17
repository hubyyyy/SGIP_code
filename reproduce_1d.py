from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cli_common import resolve_backend, simulate_with_backend
from reference_fdm import run_fdm_1d
from sgip_core import SGIPConfig
from sgip_models import interval_initial_condition


CASES = (
    ("figure01_fkpp", "fkpp", 0.5, 0.0, 1.0),
    ("figure02_cubic", "cubic", 0.5, 0.0, 1.0),
    ("figure03_arrhenius", "arrhenius", 0.5, 0.0, 1.0),
    ("figure05_bistable_narrow", "bistable", 0.25, 0.0, 1.0),
    ("figure06_bistable_wide", "bistable", 0.25, -10.0, 10.0),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the one-dimensional results in Figures 1--6."
    )
    parser.add_argument("--preset", choices=("quick", "paper"), default="quick")
    parser.add_argument("--output-dir", default="output/paper/figures01-06")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def front_position(x: np.ndarray, values: np.ndarray, threshold: float = 0.5) -> float:
    indices = np.flatnonzero(values >= threshold)
    return float(x[indices[-1]]) if indices.size else float("nan")


def comparison_figure(
    name: str,
    reaction: str,
    times: tuple[float, ...],
    sgip_x: np.ndarray,
    sgip: dict[float, np.ndarray],
    fdm_x: np.ndarray,
    fdm: dict[float, np.ndarray],
    output: Path,
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), constrained_layout=True)
    for axis, time in zip(axes.flat, times):
        axis.plot(sgip_x, sgip[time], label="SGIP", linewidth=1.8)
        axis.plot(fdm_x, fdm[time], "--", label="FDM", linewidth=1.5)
        axis.set_title(f"t = {time:g}")
        axis.set_xlim(-50.0, 50.0)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("x")
        axis.set_ylabel("u")
        axis.grid(alpha=0.2)
    axes.flat[0].legend()
    figure.suptitle(f"1D comparison: {reaction}")
    path = output / f"{name}.pdf"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    arguments = parse_arguments()
    paper = arguments.preset == "paper"
    backend = resolve_backend("gpu")
    output = Path(arguments.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    particles = 1_000_000 if paper else 30_000
    bins = 150 if paper else 60
    final_time = 20.0 if paper else 2.0
    dt = 0.5 if paper else 0.25
    fdm_dx = 1.0e-3 if paper else 0.2
    fdm_dt = 1.0e-3 if paper else 2.0e-2
    all_times = tuple(np.arange(0.0, final_time + 0.5 * dt, dt))
    panel_times = tuple(
        time for time in (5.0, 10.0, 15.0, 20.0) if time <= final_time
    )
    if not panel_times:
        panel_times = tuple(all_times[-min(4, len(all_times) - 1):])

    summaries = []
    position_results = {}
    for case_index, (name, reaction, parameter, left, right) in enumerate(CASES):
        config = SGIPConfig(
            dimension=1,
            particles=particles,
            bins=bins,
            half_width=60.0,
            dt=dt,
            final_time=final_time,
            diffusion=1.0,
            reaction=reaction,
            reaction_parameter=parameter,
            resampling="manuscript",
            backend=backend,
            seed=arguments.seed + case_index,
            show_progress=not arguments.quiet,
        )
        sampler, initial_mass = interval_initial_condition(left, right)
        result = simulate_with_backend(
            config,
            sampler,
            initial_mass,
            velocity=None,
            save_times=all_times,
        )
        fdm_x, fdm = run_fdm_1d(
            half_width=60.0,
            final_time=final_time,
            diffusion=1.0,
            reaction=reaction,
            reaction_parameter=parameter,
            save_times=all_times,
            initial_left=left,
            initial_right=right,
            dx=fdm_dx,
            requested_dt=fdm_dt,
        )
        figure = comparison_figure(
            name,
            reaction,
            panel_times,
            result.centers,
            result.snapshots,
            fdm_x,
            fdm,
            output,
        )
        np.savez_compressed(
            output / f"{name}.npz",
            sgip_x=result.centers,
            fdm_x=fdm_x,
            times=np.asarray(all_times),
            sgip=np.stack([result.snapshots[t] for t in all_times]),
            fdm=np.stack([fdm[t] for t in all_times]),
        )
        summaries.append(
            {
                "case": name,
                "reaction": reaction,
                "initial_interval": [left, right],
                "figure": str(figure),
                "sgip_elapsed_seconds": result.elapsed,
            }
        )
        if reaction in ("fkpp", "cubic", "arrhenius"):
            position_results[reaction] = {
                "times": np.asarray(all_times),
                "sgip": np.asarray(
                    [front_position(result.centers, result.snapshots[t]) for t in all_times]
                ),
                "fdm": np.asarray([front_position(fdm_x, fdm[t]) for t in all_times]),
            }

    for reaction, positions in position_results.items():
        figure, axis = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
        axis.plot(positions["times"], positions["sgip"], "o-", label="SGIP")
        axis.plot(positions["times"], positions["fdm"], "--", label="FDM")
        axis.set_xlabel("t")
        axis.set_ylabel("rightmost u = 0.5 position")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.savefig(output / f"figure04_{reaction}_front_position.pdf")
        plt.close(figure)

    (output / "summary.json").write_text(
        json.dumps(
            {
                "preset": arguments.preset,
                "backend": backend,
                "parameters": {
                    "particles": particles,
                    "bins": bins,
                    "sgip_dt": dt,
                    "fdm_dx": fdm_dx,
                    "fdm_dt": fdm_dt,
                    "final_time": final_time,
                },
                "cases": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
