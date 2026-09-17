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
from sgip_models import ball_initial_condition, make_velocity


FIGURES = {
    11: {"diffusion": 1.0, "reaction": "fkpp", "radius": 1.0, "fdm_grid": 200, "fdm_dt": 5.0e-3},
    12: {"diffusion": 0.5, "reaction": "fkpp", "radius": 1.0, "fdm_grid": 300, "fdm_dt": 1.0e-4},
    13: {"diffusion": 0.1, "reaction": "fkpp", "radius": 1.0, "fdm_grid": None, "fdm_dt": None},
    14: {"diffusion": 1.0, "reaction": "cubic", "radius": 5.0, "fdm_grid": 200, "fdm_dt": 1.0e-3},
    15: {"diffusion": 1.0, "reaction": "arrhenius", "radius": 5.0, "fdm_grid": 200, "fdm_dt": 1.0e-3},
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the three-dimensional results in Figures 11--15."
    )
    parser.add_argument("--figures", default="11,12,13,14,15")
    parser.add_argument("--preset", choices=("quick", "paper"), default="quick")
    parser.add_argument("--allow-large-fdm", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--output-dir", default="output/paper/figures11-15")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def parse_figures(text: str) -> list[int]:
    if text.strip().lower() == "all":
        return sorted(FIGURES)
    figures = sorted({int(item) for item in text.split(",") if item.strip()})
    invalid = [figure for figure in figures if figure not in FIGURES]
    if invalid:
        raise ValueError(f"unsupported 3D figure numbers: {invalid}")
    return figures


def central_indices(centers: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isclose(np.abs(centers), np.min(np.abs(centers))))


def slice_fields(centers: np.ndarray, density: np.ndarray):
    middle = central_indices(centers)
    return {
        "x0": np.mean(density[middle, :, :], axis=0).T,
        "y0": np.mean(density[:, middle, :], axis=1).T,
        "z0": np.mean(density[:, :, middle], axis=2).T,
    }


def save_slice(
    centers: np.ndarray,
    density_slice: np.ndarray,
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    image = axis.pcolormesh(
        centers,
        centers,
        density_slice,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_aspect("equal")
    figure.colorbar(image, ax=axis, label="u")
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
    particles = 5_000_000 if paper else 30_000
    bins = 100 if paper else 20
    final_time = 20.0 if paper else 1.0
    sgip_dt = 0.5 if paper else 0.25
    records = []

    for figure_number in selected:
        specification = FIGURES[figure_number]
        reaction = str(specification["reaction"])
        diffusion = float(specification["diffusion"])
        radius = float(specification["radius"])
        sampler, initial_mass = ball_initial_condition(3, radius)
        config = SGIPConfig(
            dimension=3,
            particles=particles,
            bins=bins,
            half_width=60.0,
            dt=sgip_dt,
            final_time=final_time,
            diffusion=diffusion,
            reaction=reaction,
            reaction_parameter=0.5,
            resampling="manuscript",
            backend=backend,
            seed=arguments.seed + figure_number,
            show_progress=not arguments.quiet,
        )
        result = simulate_with_backend(
            config,
            sampler,
            initial_mass,
            velocity=make_velocity("abc", drift_sign=-1.0),
            save_times=(final_time,),
        )
        sgip_density = result.snapshots[final_time]
        sgip_slices = slice_fields(result.centers, sgip_density)
        planes = ("x0", "y0", "z0") if figure_number == 13 else ("y0",)
        axis_labels = {"x0": ("y", "z"), "y0": ("x", "z"), "z0": ("x", "y")}
        for plane in planes:
            x_label, y_label = axis_labels[plane]
            save_slice(
                result.centers,
                sgip_slices[plane],
                output / f"figure{figure_number:02d}_sgip_{plane}.pdf",
                f"SGIP: {plane}, t = {final_time:g}",
                x_label,
                y_label,
            )

        fdm_elapsed = None
        fdm_centers = None
        fdm_density = None
        paper_grid = specification["fdm_grid"]
        if paper_grid is not None:
            fdm_grid = int(paper_grid) if paper else 24
            fdm_dt = float(specification["fdm_dt"]) if paper else 2.0e-2
            fdm_centers, fdm_snapshots, fdm_elapsed = run_fdm_nd(
                dimension=3,
                half_width=60.0,
                cells_per_axis=fdm_grid,
                final_time=final_time,
                requested_dt=fdm_dt,
                diffusion=diffusion,
                flow="abc",
                reaction=reaction,
                reaction_parameter=0.5,
                initial_kind="ball",
                initial_radius=radius,
                save_times=(final_time,),
                backend=fdm_backend,
                max_cells=None if arguments.allow_large_fdm else 16_000_000,
            )
            fdm_density = fdm_snapshots[final_time]
            fdm_y0 = slice_fields(fdm_centers, fdm_density)["y0"]
            save_slice(
                fdm_centers,
                fdm_y0,
                output / f"figure{figure_number:02d}_fdm_y0.pdf",
                f"FDM: y0, t = {final_time:g}",
                "x",
                "z",
            )
        if arguments.save_data:
            arrays = {
                "sgip_centers": result.centers,
                "sgip_density": sgip_density,
            }
            if fdm_centers is not None and fdm_density is not None:
                arrays["fdm_centers"] = fdm_centers
                arrays["fdm_density"] = fdm_density
            np.savez_compressed(output / f"figure{figure_number:02d}.npz", **arrays)
        records.append(
            {
                "figure": figure_number,
                "diffusion": diffusion,
                "reaction": reaction,
                "initial_radius": radius,
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
                "sgip_parameters": {
                    "particles": particles,
                    "bins_per_axis": bins,
                    "dt": sgip_dt,
                    "final_time": final_time,
                },
                "runs": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
