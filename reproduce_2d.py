from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cli_common import resolve_backend, simulate_with_backend
from reference_fdm_nd import run_fdm_nd
from sgip_core import SGIPConfig
from sgip_models import make_velocity, square_initial_condition


FIGURES = {
    7: {"flow": "none", "diffusion": 1.0, "times": (5.0, 10.0, 15.0, 20.0)},
    8: {"flow": "shear", "diffusion": 1.0, "times": (20.0,)},
    9: {"flow": "cellular", "diffusion": 1.0, "times": (20.0,)},
    10: {"flow": "cat_eye", "diffusion": (0.5, 1.0), "times": (20.0,)},
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the two-dimensional results in Figures 7--10."
    )
    parser.add_argument("--figures", default="7,8,9,10")
    parser.add_argument("--preset", choices=("quick", "paper"), default="quick")
    parser.add_argument("--allow-large-fdm", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--output-dir", default="output/paper/figures07-10")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def parse_figures(text: str) -> list[int]:
    if text.strip().lower() == "all":
        return sorted(FIGURES)
    figures = sorted({int(item) for item in text.split(",") if item.strip()})
    invalid = [figure for figure in figures if figure not in FIGURES]
    if invalid:
        raise ValueError(f"unsupported 2D figure numbers: {invalid}")
    return figures


def save_field(
    density: np.ndarray,
    centers: np.ndarray,
    path: Path,
    title: str,
    contours: bool = False,
) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    if contours:
        image = axis.contour(
            centers,
            centers,
            density.T,
            levels=(0.1, 0.5, 0.9),
            linewidths=1.5,
        )
        axis.clabel(image, inline=True, fontsize=8)
    else:
        image = axis.pcolormesh(
            centers,
            centers,
            density.T,
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        figure.colorbar(image, ax=axis, label="u")
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    selected = parse_figures(arguments.figures)
    paper = arguments.preset == "paper"
    backend = resolve_backend("gpu")
    fdm_backend = "gpu"
    output = Path(arguments.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    particles = 3_000_000 if paper else 30_000
    bins = 100 if paper else 24
    final_time = 20.0 if paper else 1.0
    sgip_dt = 0.5 if paper else 0.25
    fdm_cells = 12_000 if paper else 32
    fdm_dt = 1.0e-3 if paper else 2.0e-2
    sampler, initial_mass = square_initial_condition()
    run_records = []

    for figure_number in selected:
        specification = FIGURES[figure_number]
        diffusions = specification["diffusion"]
        if not isinstance(diffusions, tuple):
            diffusions = (diffusions,)
        requested_times = tuple(
            time for time in specification["times"] if time <= final_time
        ) or (final_time,)
        for diffusion_index, diffusion in enumerate(diffusions):
            flow = str(specification["flow"])
            config = SGIPConfig(
                dimension=2,
                particles=particles,
                bins=bins,
                half_width=60.0,
                dt=sgip_dt,
                final_time=final_time,
                diffusion=float(diffusion),
                reaction="fkpp",
                reaction_parameter=0.5,
                resampling="manuscript",
                backend=backend,
                seed=arguments.seed + 10 * figure_number + diffusion_index,
                show_progress=not arguments.quiet,
            )
            result = simulate_with_backend(
                config,
                sampler,
                initial_mass,
                velocity=make_velocity(flow, drift_sign=-1.0),
                save_times=requested_times,
            )
            fdm_centers, fdm_snapshots, fdm_elapsed = run_fdm_nd(
                dimension=2,
                half_width=60.0,
                cells_per_axis=fdm_cells,
                final_time=final_time,
                requested_dt=fdm_dt,
                diffusion=float(diffusion),
                flow=flow,
                reaction="fkpp",
                initial_kind="square",
                save_times=requested_times,
                backend=fdm_backend,
                max_cells=None if arguments.allow_large_fdm else 16_000_000,
            )

            tag = f"figure{figure_number:02d}_{flow}_D{float(diffusion):g}"
            for time in requested_times:
                contour_mode = figure_number == 7
                save_field(
                    result.snapshots[time],
                    result.centers,
                    output / f"{tag}_t{time:g}_sgip.pdf",
                    f"SGIP: t = {time:g}",
                    contours=contour_mode,
                )
                save_field(
                    fdm_snapshots[time],
                    fdm_centers,
                    output / f"{tag}_t{time:g}_fdm.pdf",
                    f"FDM: t = {time:g}",
                    contours=contour_mode,
                )
            if arguments.save_data:
                np.savez_compressed(
                    output / f"{tag}.npz",
                    sgip_centers=result.centers,
                    fdm_centers=fdm_centers,
                    times=np.asarray(requested_times),
                    sgip=np.stack([result.snapshots[t] for t in requested_times]),
                    fdm=np.stack([fdm_snapshots[t] for t in requested_times]),
                )
            run_records.append(
                {
                    "figure": figure_number,
                    "flow": flow,
                    "diffusion": diffusion,
                    "times": requested_times,
                    "sgip_elapsed_seconds": result.elapsed,
                    "fdm_elapsed_seconds": fdm_elapsed,
                }
            )

    (output / "summary.json").write_text(
        json.dumps(
            {
                "preset": arguments.preset,
                "sgip_backend": backend,
                "fdm_backend": fdm_backend,
                "parameters": {
                    "particles": particles,
                    "bins_per_axis": bins,
                    "sgip_dt": sgip_dt,
                    "fdm_cells_per_axis": fdm_cells,
                    "fdm_dt": fdm_dt,
                    "final_time": final_time,
                },
                "runs": run_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
