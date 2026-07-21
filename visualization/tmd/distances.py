"""Reusable distances between persistence diagrams.

The functions in this module deliberately operate on diagrams rather than on
trees.  Tree-specific filtration and normalization choices belong in the
calling metric wrapper.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


GroundNorm = Literal["euclidean", "chebyshev"]
NonfinitePolicy = Literal["raise", "drop"]


def persistence_diagram_wasserstein_distance(
    diagram_a: object,
    diagram_b: object,
    *,
    order: float = 1,
    ground_norm: GroundNorm = "euclidean",
    nonfinite_policy: NonfinitePolicy = "raise",
) -> float:
    """Return the Wasserstein distance between two persistence diagrams.

    Each diagram may be an ``(n, 2)`` birth/death array or an object exposing
    an ``as_pairs()`` method. Reversed interval endpoints are canonicalized,
    zero-persistence points are omitted, and unmatched points may be assigned
    to the diagonal. Non-finite points are rejected by default because silently
    deleting essential bars can change a comparison into zero; ``"drop"`` is
    available only as an explicit finite-bar analysis policy.

    Args:
        diagram_a: First persistence diagram.
        diagram_b: Second persistence diagram.
        order: Wasserstein aggregation order, which must be at least one.
        ground_norm: Pointwise norm in the birth/death plane.  ``"euclidean"``
            uses the L2 norm and ``"chebyshev"`` uses the L-infinity norm.
        nonfinite_policy: Reject non-finite bars, or explicitly drop them.
    """
    try:
        order = float(order)
    except (TypeError, ValueError) as exc:
        raise ValueError("Wasserstein order must be a finite number >= 1.") from exc
    if not np.isfinite(order) or order < 1:
        raise ValueError("Wasserstein order must be a finite number >= 1.")
    if ground_norm not in {"euclidean", "chebyshev"}:
        raise ValueError(
            "ground_norm must be either 'euclidean' or 'chebyshev', "
            f"got {ground_norm!r}."
        )
    if nonfinite_policy not in {"raise", "drop"}:
        raise ValueError("nonfinite_policy must be either 'raise' or 'drop'.")

    pairs_a = _canonical_persistent_pairs(
        _diagram_pairs(diagram_a), nonfinite_policy=nonfinite_policy
    )
    pairs_b = _canonical_persistent_pairs(
        _diagram_pairs(diagram_b), nonfinite_policy=nonfinite_policy
    )
    n_a = pairs_a.shape[0]
    n_b = pairs_b.shape[0]

    if n_a == 0 and n_b == 0:
        return 0.0

    diagonal_a = _diagonal_distances(pairs_a, ground_norm=ground_norm)
    diagonal_b = _diagonal_distances(pairs_b, ground_norm=ground_norm)
    if n_a == 0:
        return float(np.sum(diagonal_b**order) ** (1.0 / order))
    if n_b == 0:
        return float(np.sum(diagonal_a**order) ** (1.0 / order))

    from scipy.optimize import linear_sum_assignment

    # The augmented assignment contains one diagonal slot for every point in
    # the opposite diagram.  Diagonal slots are interchangeable, so repeating
    # a point's diagonal cost across its block is equivalent to the usual
    # diagonal-only construction while avoiding artificial infinities.
    cost = np.zeros((n_a + n_b, n_b + n_a), dtype=np.float64)
    cost[:n_a, :n_b] = (
        _pairwise_ground_distances(pairs_a, pairs_b, ground_norm=ground_norm) ** order
    )
    cost[:n_a, n_b:] = (diagonal_a**order)[:, None]
    cost[n_a:, :n_b] = (diagonal_b**order)[None, :]

    row_ind, col_ind = linear_sum_assignment(cost)
    total = float(cost[row_ind, col_ind].sum())
    return float(total ** (1.0 / order))


def _diagram_pairs(diagram: object) -> np.ndarray:
    """Convert a supported diagram representation to an ``(n, 2)`` array."""
    if diagram is None:
        # ``None`` has historically represented a missing/empty diagram in the
        # visualization pipeline, so retain that behavior here.
        return np.zeros((0, 2), dtype=np.float64)

    as_pairs = getattr(diagram, "as_pairs", None)
    values = as_pairs() if callable(as_pairs) else diagram
    pairs = np.asarray(values, dtype=np.float64)
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(
            "A persistence diagram must be an (n, 2) birth/death array or "
            "expose as_pairs() returning one."
        )
    return pairs


def _canonical_persistent_pairs(
    pairs: np.ndarray,
    *,
    nonfinite_policy: NonfinitePolicy,
) -> np.ndarray:
    """Return finite, positive-persistence pairs with ordered endpoints."""
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    finite = np.isfinite(pairs).all(axis=1)
    if not np.all(finite) and nonfinite_policy == "raise":
        raise ValueError(
            "Persistence diagram contains non-finite bars; pass "
            "nonfinite_policy='drop' only for an explicit finite-bar comparison."
        )
    finite_pairs = pairs[finite]
    if finite_pairs.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    lo = np.minimum(finite_pairs[:, 0], finite_pairs[:, 1])
    hi = np.maximum(finite_pairs[:, 0], finite_pairs[:, 1])
    canonical = np.column_stack([lo, hi])
    return canonical[(canonical[:, 1] - canonical[:, 0]) > 1e-12]


def _diagonal_distances(pairs: np.ndarray, *, ground_norm: GroundNorm) -> np.ndarray:
    """Return each point's distance to the persistence diagonal."""
    persistence = np.abs(pairs[:, 1] - pairs[:, 0])
    if ground_norm == "euclidean":
        return persistence / np.sqrt(2.0)
    return persistence / 2.0


def _pairwise_ground_distances(
    a: np.ndarray,
    b: np.ndarray,
    *,
    ground_norm: GroundNorm,
) -> np.ndarray:
    """Return all pairwise birth/death-plane distances."""
    difference = np.abs(a[:, None, :] - b[None, :, :])
    if ground_norm == "euclidean":
        return np.linalg.norm(difference, axis=2)
    return np.max(difference, axis=2)


__all__ = [
    "GroundNorm",
    "NonfinitePolicy",
    "persistence_diagram_wasserstein_distance",
]
