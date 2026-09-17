from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from cli_common import resolve_backend
from reference_fdm import run_fdm_1d
from reference_fdm_nd import run_fdm_nd
from sgip_core import SGIPConfig
from sgip_gpu import cupy_available, simulate_sgip_gpu
from sgip_models import (
    ball_initial_condition,
    interval_initial_condition,
    make_velocity,
    square_initial_condition,
)


class PackageChecks(unittest.TestCase):
    def test_manifest_covers_all_figures(self) -> None:
        path = Path(__file__).with_name("paper_experiments.json")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            [item["figure"] for item in manifest["figures"]],
            list(range(1, 16)),
        )

    def test_cpu_backend_is_not_available(self) -> None:
        with self.assertRaises(ValueError):
            resolve_backend("cpu")
        with self.assertRaises(ValueError):
            SGIPConfig(dimension=1, particles=100, bins=10, backend="cpu").validate()


@unittest.skipUnless(cupy_available(), "requires a CUDA GPU")
class CUDASmokeTests(unittest.TestCase):
    def run_dimension(self, dimension: int) -> None:
        if dimension == 1:
            sampler, mass = interval_initial_condition()
            velocity = None
        elif dimension == 2:
            sampler, mass = square_initial_condition()
            velocity = make_velocity("cellular")
        else:
            sampler, mass = ball_initial_condition(3, 1.0)
            velocity = make_velocity("abc")

        config = SGIPConfig(
            dimension=dimension,
            particles=3000,
            bins=12,
            final_time=1.0,
            dt=0.5,
            show_progress=False,
        )
        result = simulate_sgip_gpu(config, sampler, mass, velocity=velocity)
        self.assertEqual(result.snapshots[1.0].shape, (config.bins,) * dimension)
        self.assertTrue(np.all(np.isfinite(result.snapshots[1.0])))
        self.assertGreater(result.mass_history[-1], 0.0)

    def test_sgip_1d(self) -> None:
        self.run_dimension(1)

    def test_sgip_2d(self) -> None:
        self.run_dimension(2)

    def test_sgip_3d(self) -> None:
        self.run_dimension(3)

    def test_fdm_1d(self) -> None:
        centers, snapshots = run_fdm_1d(
            half_width=4.0,
            final_time=0.1,
            diffusion=1.0,
            reaction="fkpp",
            reaction_parameter=0.5,
            save_times=(0.1,),
            dx=0.5,
            requested_dt=0.02,
        )
        self.assertEqual(snapshots[0.1].shape, (centers.size,))
        self.assertTrue(np.all(np.isfinite(snapshots[0.1])))

    def test_fdm_2d(self) -> None:
        centers, snapshots, elapsed = run_fdm_nd(
            dimension=2,
            half_width=4.0,
            cells_per_axis=12,
            final_time=0.1,
            requested_dt=0.02,
            diffusion=0.5,
            flow="cat_eye",
            reaction="fkpp",
            initial_kind="square",
            save_times=(0.1,),
        )
        self.assertEqual(snapshots[0.1].shape, (centers.size,) * 2)
        self.assertTrue(np.all(np.isfinite(snapshots[0.1])))
        self.assertGreaterEqual(elapsed, 0.0)

    def test_fdm_3d(self) -> None:
        centers, snapshots, _ = run_fdm_nd(
            dimension=3,
            half_width=4.0,
            cells_per_axis=10,
            final_time=0.1,
            requested_dt=0.02,
            diffusion=1.0,
            flow="abc",
            reaction="cubic",
            initial_kind="ball",
            initial_radius=1.0,
            save_times=(0.1,),
        )
        self.assertEqual(snapshots[0.1].shape, (centers.size,) * 3)
        self.assertTrue(np.all(np.isfinite(snapshots[0.1])))


if __name__ == "__main__":
    unittest.main()
