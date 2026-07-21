"""Shared SO(2) actions and quotient minimization for tree metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize_scalar


_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class SO2Minimum:
    """Minimum of a scalar objective over rotations around a fixed axis."""

    value: float
    angle_rad: float
    grid_value: float
    grid_angle_rad: float
    evaluations: int


def _unit_axis(axis: Sequence[float]) -> np.ndarray:
    arr = np.asarray(axis, dtype=np.float64).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"axis must contain exactly three values, got shape {arr.shape}")
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("axis must have a finite, non-zero norm")
    return arr / norm


def rotation_matrix_about_axis(
    angle_rad: float,
    axis: Sequence[float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Return a right-handed 3D rotation matrix using Rodrigues' formula."""
    u = _unit_axis(axis)
    x, y, z = u
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    one_minus_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )


def rotate_points_about_axis(
    points: np.ndarray,
    angle_rad: float,
    axis: Sequence[float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Rotate an ``(N, 3)`` point array around an axis through the origin."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {arr.shape}")
    rotation = rotation_matrix_about_axis(angle_rad, axis)
    # ``np.dot`` avoids spurious floating-point warnings emitted by some
    # Accelerate-backed NumPy builds for otherwise finite small matrix products.
    return np.dot(arr, rotation.T)


def _canonical_angle(angle_rad: float) -> float:
    angle = float(angle_rad) % _TWO_PI
    if math.isclose(angle, _TWO_PI, abs_tol=1e-12):
        return 0.0
    return angle


def minimize_over_so2(
    objective: Callable[[float], float],
    *,
    grid_size: int = 72,
    refine: bool = True,
    refinement_tolerance: float = 1e-8,
) -> SO2Minimum:
    """Minimize a periodic scalar objective over one complete SO(2) orbit.

    A deterministic uniform grid provides the global search. The optional local
    bounded refinement searches one grid cell on either side of the best angle;
    the wrapped objective makes refinement safe across the ``0``/``2π`` seam.
    """
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3")
    if refinement_tolerance <= 0:
        raise ValueError("refinement_tolerance must be positive")

    evaluations = 0

    def evaluate(angle: float) -> float:
        nonlocal evaluations
        value = float(objective(_canonical_angle(angle)))
        evaluations += 1
        return value

    angles = np.linspace(0.0, _TWO_PI, num=grid_size, endpoint=False, dtype=np.float64)
    values = np.asarray([evaluate(float(angle)) for angle in angles], dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("SO(2) objective returned no finite values on the search grid")

    comparable = np.where(finite, values, np.inf)
    best_index = int(np.argmin(comparable))
    grid_angle = float(angles[best_index])
    grid_value = float(values[best_index])
    best_angle = grid_angle
    best_value = grid_value

    if refine:
        step = _TWO_PI / float(grid_size)
        result = minimize_scalar(
            lambda delta: evaluate(grid_angle + float(delta)),
            bounds=(-step, step),
            method="bounded",
            options={"xatol": float(refinement_tolerance)},
        )
        candidate_value = float(result.fun)
        if result.success and np.isfinite(candidate_value) and candidate_value <= best_value:
            best_value = candidate_value
            best_angle = _canonical_angle(grid_angle + float(result.x))

    return SO2Minimum(
        value=best_value,
        angle_rad=_canonical_angle(best_angle),
        grid_value=grid_value,
        grid_angle_rad=_canonical_angle(grid_angle),
        evaluations=evaluations,
    )
