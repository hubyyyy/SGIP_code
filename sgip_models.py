from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


Array = np.ndarray
VelocityField = Callable[[Array, float], Array]


def _tag_velocity(
    velocity: VelocityField, name: str, drift_sign: float
) -> VelocityField:
    velocity.sgip_flow_name = name
    velocity.sgip_drift_sign = drift_sign
    return velocity


def interval_initial_condition(
    left: float = 0.0, right: float = 1.0
) -> Tuple[Callable, float]:
    if right <= left:
        raise ValueError("the initial interval must satisfy right > left")

    def sample(
        rng: np.random.Generator, count: int, dtype: np.dtype
    ) -> Array:
        return rng.uniform(left, right, size=(count, 1)).astype(dtype)

    return sample, right - left


def square_initial_condition() -> Tuple[Callable, float]:
    def sample(
        rng: np.random.Generator, count: int, dtype: np.dtype
    ) -> Array:
        return rng.uniform(0.0, 1.0, size=(count, 2)).astype(dtype)

    return sample, 1.0


def ball_initial_condition(
    dimension: int, radius: float
) -> Tuple[Callable, float]:
    if dimension not in (2, 3):
        raise ValueError("ball initial condition supports dimensions 2 and 3")

    def sample(
        rng: np.random.Generator, count: int, dtype: np.dtype
    ) -> Array:
        directions = rng.standard_normal((count, dimension), dtype=dtype)
        norms = np.linalg.norm(directions, axis=1)
        directions /= norms[:, None]
        radii = radius * rng.random(count, dtype=dtype) ** (1.0 / dimension)
        return directions * radii[:, None]

    if dimension == 2:
        volume = np.pi * radius**2
    else:
        volume = (4.0 / 3.0) * np.pi * radius**3
    return sample, volume


def make_velocity(
    name: str, drift_sign: float = -1.0
) -> Optional[VelocityField]:
    """Return the SDE drift for the flow formulas used in the paper.

    The numerical examples write the PDE with +w dot grad(u), so the
    corresponding particle drift is -w. Set drift_sign=+1 to use w itself.
    """
    normalized = name.lower().replace("-", "_")
    if normalized in ("none", "zero"):
        return None

    if normalized == "shear":
        def shear(positions: Array, _: float) -> Array:
            result = np.zeros_like(positions)
            result[:, 0] = np.sin(positions[:, 1])
            return drift_sign * result

        return _tag_velocity(shear, "shear", drift_sign)

    if normalized == "cellular":
        def cellular(positions: Array, _: float) -> Array:
            x = positions[:, 0]
            y = positions[:, 1]
            result = np.empty_like(positions)
            result[:, 0] = -np.sin(x) * np.cos(y)
            result[:, 1] = np.cos(x) * np.sin(y)
            return drift_sign * result

        return _tag_velocity(cellular, "cellular", drift_sign)

    if normalized in ("cat_eye", "cateye"):
        def cat_eye(positions: Array, _: float) -> Array:
            x = positions[:, 0]
            y = positions[:, 1]
            result = np.empty_like(positions)
            result[:, 0] = (
                -np.sin(x) * np.cos(y)
                + 2.0 * np.cos(x) * np.sin(y)
            )
            result[:, 1] = (
                np.cos(x) * np.sin(y)
                - 2.0 * np.sin(x) * np.cos(y)
            )
            return drift_sign * result

        return _tag_velocity(cat_eye, "cat_eye", drift_sign)

    if normalized == "abc":
        coefficient_a = 1.0
        coefficient_b = np.sqrt(2.0 / 3.0)
        coefficient_c = np.sqrt(1.0 / 3.0)

        def abc(positions: Array, _: float) -> Array:
            x = positions[:, 0]
            y = positions[:, 1]
            z = positions[:, 2]
            result = np.empty_like(positions)
            result[:, 0] = (
                coefficient_a * np.sin(z)
                + coefficient_c * np.cos(y)
            )
            result[:, 1] = (
                coefficient_b * np.sin(x)
                + coefficient_a * np.cos(z)
            )
            result[:, 2] = (
                coefficient_c * np.sin(y)
                + coefficient_b * np.cos(x)
            )
            return drift_sign * result

        return _tag_velocity(abc, "abc", drift_sign)

    raise ValueError(f"unknown flow: {name}")
