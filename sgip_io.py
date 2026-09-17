from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sgip_core import SGIPResult


def ensure_output_dir(path: str) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_summary(
    result: SGIPResult,
    output_dir: Path,
    extra: Optional[Dict[str, object]] = None,
) -> Path:
    summary = {
        "config": asdict(result.config),
        "elapsed_seconds": result.elapsed,
        "mean_step_seconds": float(np.mean(result.step_times)),
        "max_step_seconds": float(np.max(result.step_times)),
        "final_mass": float(result.mass_history[-1]),
        "saved_times": sorted(result.snapshots),
    }
    if extra:
        summary.update(extra)
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def save_result(result: SGIPResult, output_dir: Path) -> Path:
    arrays = {
        "centers": result.centers,
        "mass_history": result.mass_history,
        "step_times": result.step_times,
    }
    for time, density in result.snapshots.items():
        arrays[f"density_t_{time:g}"] = density
    path = output_dir / "densities.npz"
    np.savez_compressed(path, **arrays)
    return path


def plot_1d(
    result: SGIPResult,
    output_dir: Path,
    fdm_snapshots: Optional[Dict[float, np.ndarray]] = None,
    fdm_x: Optional[np.ndarray] = None,
) -> Path:
    times = [time for time in sorted(result.snapshots) if time > 0.0]
    columns = 2
    rows = int(np.ceil(len(times) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(12, 4.2 * rows), squeeze=False
    )
    for axis, time in zip(axes.flat, times):
        axis.plot(
            result.centers,
            result.snapshots[time],
            color="#1769aa",
            linewidth=2.0,
            label="SGIP",
        )
        if (
            fdm_snapshots is not None
            and fdm_x is not None
            and time in fdm_snapshots
        ):
            axis.plot(
                fdm_x,
                fdm_snapshots[time],
                color="#c62828",
                linestyle="--",
                linewidth=1.6,
                label="FDM",
            )
        axis.set_title(f"t = {time:g}")
        axis.set_xlabel("x")
        axis.set_ylabel("u")
        axis.set_xlim(-result.config.half_width, result.config.half_width)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
        axis.legend()
    for axis in axes.flat[len(times):]:
        axis.set_visible(False)
    figure.suptitle(f"1D SGIP: {result.config.reaction} reaction")
    figure.tight_layout()
    path = output_dir / "sgip_1d.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_2d(
    result: SGIPResult,
    output_dir: Path,
    flow_name: str,
) -> Path:
    times = [time for time in sorted(result.snapshots) if time > 0.0]
    columns = 2
    rows = int(np.ceil(len(times) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(11, 4.8 * rows), squeeze=False
    )
    image = None
    for axis, time in zip(axes.flat, times):
        image = axis.pcolormesh(
            result.centers,
            result.centers,
            result.snapshots[time].T,
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        axis.set_title(f"t = {time:g}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")
    for axis in axes.flat[len(times):]:
        axis.set_visible(False)
    figure.suptitle(f"2D SGIP: {flow_name} flow")
    figure.tight_layout(rect=(0.0, 0.0, 0.9, 0.95))
    if image is not None:
        color_axis = figure.add_axes((0.92, 0.12, 0.018, 0.76))
        figure.colorbar(image, cax=color_axis, label="u")
    path = output_dir / f"sgip_2d_{flow_name}.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_3d_slices(
    result: SGIPResult,
    output_dir: Path,
) -> Path:
    final_time = max(result.snapshots)
    density = result.snapshots[final_time]
    middle = result.config.bins // 2
    slices = (
        (density[middle, :, :].T, "x = 0", "y", "z"),
        (density[:, middle, :].T, "y = 0", "x", "z"),
        (density[:, :, middle].T, "z = 0", "x", "y"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    image = None
    for axis, (plane, title, xlabel, ylabel) in zip(axes, slices):
        image = axis.pcolormesh(
            result.centers,
            result.centers,
            plane,
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_aspect("equal")
    figure.suptitle(f"3D ABC-flow SGIP at t = {final_time:g}")
    figure.tight_layout(rect=(0.0, 0.0, 0.93, 0.9))
    if image is not None:
        color_axis = figure.add_axes((0.95, 0.15, 0.012, 0.68))
        figure.colorbar(image, cax=color_axis, label="u")
    path = output_dir / "sgip_3d_abc_slices.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path
