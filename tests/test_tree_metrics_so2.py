import math

import numpy as np

from metrics.so2 import minimize_over_so2, rotate_points_about_axis


def _angular_error(a: float, b: float) -> float:
    delta = (a - b) % (2.0 * math.pi)
    return min(delta, 2.0 * math.pi - delta)


def test_minimize_over_so2_recovers_periodic_minimum() -> None:
    target = 2.0 * math.pi - 0.037
    result = minimize_over_so2(
        lambda angle: 1.0 - math.cos(angle - target),
        grid_size=24,
        refine=True,
    )
    assert result.value < 1e-9
    assert _angular_error(result.angle_rad, target) < 1e-4


def test_rotate_points_about_z_preserves_z_and_norms() -> None:
    points = np.asarray([[1.0, 2.0, 3.0], [-2.0, 0.5, -1.0]])
    rotated = rotate_points_about_axis(points, math.pi / 3.0)
    np.testing.assert_allclose(rotated[:, 2], points[:, 2])
    np.testing.assert_allclose(np.linalg.norm(rotated, axis=1), np.linalg.norm(points, axis=1))
