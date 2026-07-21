"""Point-cloud Chamfer dissimilarities for embedded trees.

The core nearest-neighbour calculation follows the earlier implementation in
``validation/chamfer.py``. Sampling is deliberately implemented here instead
of importing that large evaluation script. It samples maximal degree-2 paths
by arc length, so inserting collinear SWC support points does not change the
representation merely by adding extra point mass.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable, Literal, Sequence

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from .so2 import minimize_over_so2, rotate_points_about_axis


Reduction = Literal["sum", "mean"]


@dataclass(frozen=True)
class ChamferResult:
    """Symmetric Chamfer result plus alignment diagnostics."""

    value: float
    a_to_b: float
    b_to_a: float
    angle_rad: float
    point_count_a: int
    point_count_b: int
    spacing: float
    squared: bool
    reduction: Reduction
    quotient_so2: bool
    grid_size: int
    refine: bool
    refinement_tolerance: float
    objective_evaluations: int


def _position(graph: nx.Graph, node: Hashable) -> np.ndarray:
    if "pos" not in graph.nodes[node]:
        raise ValueError(f"Node {node!r} is missing the required 'pos' attribute")
    pos = np.asarray(graph.nodes[node]["pos"], dtype=np.float64).reshape(-1)
    if pos.shape != (3,) or not np.all(np.isfinite(pos)):
        raise ValueError(f"Node {node!r} must have a finite 3D position, got {pos!r}")
    return pos


def _resolve_root(graph: nx.Graph, root: Hashable | None) -> Hashable:
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot sample an empty graph")
    if not nx.is_tree(graph):
        raise ValueError("Expected a connected, acyclic tree")
    resolved = graph.graph.get("root") if root is None else root
    if resolved not in graph:
        raise ValueError("A valid root must be passed or stored in graph.graph['root']")
    return resolved


def _maximal_degree_two_paths(graph: nx.Graph, root: Hashable) -> list[list[Hashable]]:
    critical = {node for node in graph if node == root or graph.degree(node) != 2}
    visited: set[frozenset[Hashable]] = set()
    paths: list[list[Hashable]] = []

    for start in graph:
        if start not in critical:
            continue
        for neighbor in graph.neighbors(start):
            edge = frozenset((start, neighbor))
            if edge in visited:
                continue
            visited.add(edge)
            path = [start, neighbor]
            previous, current = start, neighbor
            while current not in critical:
                next_nodes = [node for node in graph.neighbors(current) if node != previous]
                if len(next_nodes) != 1:
                    raise ValueError("Invalid degree-2 path while sampling tree")
                following = next_nodes[0]
                visited.add(frozenset((current, following)))
                path.append(following)
                previous, current = current, following
            paths.append(path)
    return paths


def _sample_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    segment_vectors = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    total_length = float(segment_lengths.sum())
    if total_length <= 1e-12:
        return np.zeros((0, 3), dtype=np.float64)

    count = max(1, int(math.ceil(total_length / spacing)))
    targets = (np.arange(count, dtype=np.float64) + 0.5) * (total_length / count)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    segment_indices = np.searchsorted(cumulative[1:], targets, side="right")
    segment_indices = np.minimum(segment_indices, len(segment_lengths) - 1)
    local = targets - cumulative[segment_indices]
    fractions = np.divide(
        local,
        segment_lengths[segment_indices],
        out=np.zeros_like(local),
        where=segment_lengths[segment_indices] > 1e-12,
    )
    return points[segment_indices] + fractions[:, None] * segment_vectors[segment_indices]


def sample_tree_points(
    graph: nx.Graph,
    *,
    spacing: float = 1.0,
    root: Hashable | None = None,
    center_root: bool = True,
) -> np.ndarray:
    """Sample a tree approximately uniformly along maximal branch polylines."""
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"spacing must be finite and positive, got {spacing}")
    resolved_root = _resolve_root(graph, root)
    origin = _position(graph, resolved_root) if center_root else np.zeros(3, dtype=np.float64)

    if graph.number_of_edges() == 0:
        return (_position(graph, resolved_root) - origin).reshape(1, 3)

    samples: list[np.ndarray] = []
    for path in _maximal_degree_two_paths(graph, resolved_root):
        polyline = np.stack([_position(graph, node) - origin for node in path], axis=0)
        branch_samples = _sample_polyline(polyline, spacing)
        if branch_samples.size:
            samples.append(branch_samples)

    if not samples:
        return (_position(graph, resolved_root) - origin).reshape(1, 3)
    return np.concatenate(samples, axis=0)


def _point_array(points: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite coordinates")
    return arr


def point_chamfer_components(
    points_a: np.ndarray,
    points_b: np.ndarray,
    *,
    squared: bool = False,
) -> tuple[float, float]:
    """Return directional mean nearest-neighbour distances ``a→b`` and ``b→a``."""
    a = _point_array(points_a, "points_a")
    b = _point_array(points_b, "points_b")
    if len(a) == 0 and len(b) == 0:
        return 0.0, 0.0
    if len(a) == 0:
        return math.inf, 0.0
    if len(b) == 0:
        return 0.0, math.inf

    dist_a = cKDTree(b).query(a, k=1)[0]
    dist_b = cKDTree(a).query(b, k=1)[0]
    if squared:
        dist_a = dist_a**2
        dist_b = dist_b**2
    return float(np.mean(dist_a)), float(np.mean(dist_b))


def point_chamfer_distance(
    points_a: np.ndarray,
    points_b: np.ndarray,
    *,
    squared: bool = False,
    reduction: Reduction = "sum",
) -> float:
    """Return a symmetric Chamfer dissimilarity from its directional components."""
    if reduction not in ("sum", "mean"):
        raise ValueError("reduction must be 'sum' or 'mean'")
    a_to_b, b_to_a = point_chamfer_components(points_a, points_b, squared=squared)
    total = a_to_b + b_to_a
    return float(total if reduction == "sum" else 0.5 * total)


def tree_chamfer_distance(
    graph_a: nx.Graph,
    graph_b: nx.Graph,
    *,
    spacing: float = 1.0,
    root_a: Hashable | None = None,
    root_b: Hashable | None = None,
    squared: bool = False,
    reduction: Reduction = "sum",
    quotient_so2: bool = True,
    axis: Sequence[float] = (0.0, 0.0, 1.0),
    grid_size: int = 72,
    refine: bool = True,
    refinement_tolerance: float = 1e-8,
) -> ChamferResult:
    """Compare two rooted trees, optionally minimizing over relative SO(2) rotation."""
    points_a = sample_tree_points(graph_a, spacing=spacing, root=root_a, center_root=True)
    points_b = sample_tree_points(graph_b, spacing=spacing, root=root_b, center_root=True)

    def objective(angle: float) -> float:
        rotated_b = rotate_points_about_axis(points_b, angle, axis)
        return point_chamfer_distance(
            points_a,
            rotated_b,
            squared=squared,
            reduction=reduction,
        )

    if quotient_so2:
        minimum = minimize_over_so2(
            objective,
            grid_size=grid_size,
            refine=refine,
            refinement_tolerance=refinement_tolerance,
        )
        angle = minimum.angle_rad
        objective_evaluations = minimum.evaluations
    else:
        angle = 0.0
        objective_evaluations = 0

    aligned_b = rotate_points_about_axis(points_b, angle, axis)
    a_to_b, b_to_a = point_chamfer_components(points_a, aligned_b, squared=squared)
    total = a_to_b + b_to_a
    value = total if reduction == "sum" else 0.5 * total
    return ChamferResult(
        value=float(value),
        a_to_b=float(a_to_b),
        b_to_a=float(b_to_a),
        angle_rad=float(angle),
        point_count_a=len(points_a),
        point_count_b=len(points_b),
        spacing=float(spacing),
        squared=bool(squared),
        reduction=reduction,
        quotient_so2=bool(quotient_so2),
        grid_size=int(grid_size),
        refine=bool(refine),
        refinement_tolerance=float(refinement_tolerance),
        objective_evaluations=int(objective_evaluations),
    )
