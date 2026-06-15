"""Radius synthesis for visualization-only cylinder tree models.

This module is inspired by the path-wise correction idea used in rTwig, but it
does not try to reproduce rTwig's measured-QSM workflow. It creates plausible
render radii for bare skeleton graphs by combining a simple pipe-model prior
with monotone smoothing along every root-to-tip path.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx


SYNTHESIZED_RADIUS_ATTR = "_visualization_radius"
_EPS = 1e-12


@dataclass(frozen=True)
class _RootedTree:
    roots: list[Hashable]
    parent: dict[Hashable, Hashable | None]
    children: dict[Hashable, list[Hashable]]
    order: list[Hashable]


def _pos_to_xyz(value: object) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size), mode="constant", constant_values=0.0)
    return arr[:3]


def _graph_positions(graph: "nx.Graph") -> dict[Hashable, np.ndarray]:
    return {
        node: _pos_to_xyz(graph.nodes[node].get("pos", np.zeros(3)))
        for node in graph.nodes
    }


def _finite_positive(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 0.0:
        return None
    return number


def _choose_root(
    graph: "nx.Graph",
    positions: dict[Hashable, np.ndarray],
    root: Hashable | None,
) -> Hashable:
    if root is not None:
        if root not in graph:
            raise ValueError(f"Requested root {root!r} is not in the graph.")
        return root

    graph_root = graph.graph.get("root")
    if graph_root in graph:
        return graph_root
    if 0 in graph:
        return 0
    if 1 in graph:
        return 1
    if positions:
        return min(positions, key=lambda node: (float(positions[node][2]), str(node)))
    raise ValueError("Cannot choose a root for an empty graph.")


def _root_graph(
    graph: "nx.Graph",
    positions: dict[Hashable, np.ndarray],
    *,
    root: Hashable | None,
) -> _RootedTree:
    start = _choose_root(graph, positions, root)
    parent: dict[Hashable, Hashable | None] = {}
    children: dict[Hashable, list[Hashable]] = {node: [] for node in graph.nodes}
    order: list[Hashable] = []
    roots: list[Hashable] = []

    def visit_component(component_root: Hashable) -> None:
        roots.append(component_root)
        parent[component_root] = None
        order.append(component_root)
        queue = [component_root]
        for node in queue:
            for neighbor in graph.neighbors(node):
                if neighbor in parent:
                    continue
                parent[neighbor] = node
                children[node].append(neighbor)
                order.append(neighbor)
                queue.append(neighbor)

    visit_component(start)
    for node in graph.nodes:
        if node not in parent:
            visit_component(node)

    return _RootedTree(roots=roots, parent=parent, children=children, order=order)


def _edge_length(positions: dict[Hashable, np.ndarray], u: Hashable, v: Hashable) -> float:
    return max(float(np.linalg.norm(positions[u] - positions[v])), _EPS)


def _root_to_tip_paths(rooted: _RootedTree) -> list[list[Hashable]]:
    paths: list[list[Hashable]] = []
    tips = [node for node in rooted.order if not rooted.children[node]]
    for tip in tips:
        path: list[Hashable] = []
        node: Hashable | None = tip
        while node is not None:
            path.append(node)
            node = rooted.parent[node]
        paths.append(list(reversed(path)))
    return paths


def _path_length(path: list[Hashable], positions: dict[Hashable, np.ndarray]) -> float:
    if len(path) < 2:
        return 0.0
    return sum(_edge_length(positions, u, v) for u, v in zip(path[:-1], path[1:]))


def _default_twig_radius(
    positions: dict[Hashable, np.ndarray],
    *,
    twig_radius_scale: float,
) -> float:
    if not positions:
        return float(twig_radius_scale)
    pts = np.stack(list(positions.values()), axis=0)
    diagonal = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    if diagonal <= _EPS:
        return float(twig_radius_scale)
    return max(diagonal * float(twig_radius_scale), _EPS)


def _pava_nondecreasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    block_values: list[float] = []
    block_weights: list[float] = []
    block_counts: list[int] = []

    for value, weight in zip(values, weights):
        block_values.append(float(value))
        block_weights.append(float(weight))
        block_counts.append(1)
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            total_weight = block_weights[-2] + block_weights[-1]
            merged_value = (
                block_values[-2] * block_weights[-2] + block_values[-1] * block_weights[-1]
            ) / max(total_weight, _EPS)
            block_values[-2] = merged_value
            block_weights[-2] = total_weight
            block_counts[-2] += block_counts[-1]
            block_values.pop()
            block_weights.pop()
            block_counts.pop()

    fitted: list[float] = []
    for value, count in zip(block_values, block_counts):
        fitted.extend([value] * count)
    return np.asarray(fitted, dtype=float)


def _monotone_nonincreasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return -_pava_nondecreasing(-values, weights)


def _subtree_metrics(
    rooted: _RootedTree,
    positions: dict[Hashable, np.ndarray],
) -> tuple[dict[Hashable, int], dict[Hashable, float]]:
    tip_count: dict[Hashable, int] = {}
    downstream_length: dict[Hashable, float] = {}
    for node in reversed(rooted.order):
        children = rooted.children[node]
        if not children:
            tip_count[node] = 1
            downstream_length[node] = 0.0
            continue
        tip_count[node] = sum(tip_count[child] for child in children)
        downstream_length[node] = sum(
            _edge_length(positions, node, child) + downstream_length[child]
            for child in children
        )
    return tip_count, downstream_length


def _initial_radius_prior(
    rooted: _RootedTree,
    positions: dict[Hashable, np.ndarray],
    paths: list[list[Hashable]],
    *,
    twig_radius: float,
    pipe_exponent: float,
    length_exponent: float,
) -> dict[Hashable, float]:
    tip_count, downstream_length = _subtree_metrics(rooted, positions)
    positive_path_lengths = [_path_length(path, positions) for path in paths if len(path) > 1]
    valid_path_lengths = [length for length in positive_path_lengths if length > _EPS]
    path_ref = float(np.median(valid_path_lengths)) if valid_path_lengths else 1.0
    path_ref = max(path_ref, _EPS)

    prior: dict[Hashable, float] = {}
    for node in rooted.order:
        pipe_factor = float(tip_count[node]) ** float(pipe_exponent)
        length_factor = (1.0 + downstream_length[node] / path_ref) ** float(length_exponent)
        prior[node] = max(float(twig_radius) * pipe_factor * length_factor, float(twig_radius))
    return prior


def _smooth_path_radii(
    path: list[Hashable],
    prior: dict[Hashable, float],
    *,
    twig_radius: float,
    terminal_weight: float,
) -> list[float]:
    values = np.asarray([prior[node] for node in path], dtype=float)
    weights = np.ones_like(values, dtype=float)
    if values.size:
        values[-1] = float(twig_radius)
        weights[-1] = max(float(terminal_weight), 1.0)
    fitted = _monotone_nonincreasing(values, weights)
    fitted = np.maximum(fitted, float(twig_radius))
    if fitted.size:
        fitted[-1] = float(twig_radius)
        for idx in range(fitted.size - 2, -1, -1):
            fitted[idx] = max(float(fitted[idx]), float(fitted[idx + 1]))
    return [float(value) for value in fitted]


def _merge_path_predictions(
    paths: list[list[Hashable]],
    path_radii: list[list[float]],
    fallback: dict[Hashable, float],
) -> dict[Hashable, float]:
    sum_radius = defaultdict(float)
    sum_radius_sq = defaultdict(float)
    for path, radii in zip(paths, path_radii):
        for node, radius in zip(path, radii):
            sum_radius[node] += radius
            sum_radius_sq[node] += radius * radius

    merged: dict[Hashable, float] = {}
    for node, fallback_radius in fallback.items():
        denominator = sum_radius[node]
        if denominator <= _EPS:
            merged[node] = float(fallback_radius)
        else:
            merged[node] = float(sum_radius_sq[node] / denominator)
    return merged


def _enforce_parent_child_monotonicity(rooted: _RootedTree, radii: dict[Hashable, float]) -> None:
    for node in reversed(rooted.order):
        child_radii = [radii[child] for child in rooted.children[node]]
        if child_radii:
            radii[node] = max(radii[node], max(child_radii))


def synthesize_radii(
    graph: "nx.Graph",
    *,
    root: Hashable | None = None,
    twig_radius: float | None = None,
    twig_radius_scale: float = 0.002,
    pipe_exponent: float = 0.35,
    length_exponent: float = 0.12,
    terminal_weight: float = 100.0,
    smoothing_passes: int = 1,
) -> dict[Hashable, float]:
    """Return rTwig-inspired render radii for a skeleton graph.

    The returned values are node radii intended for visualization. They are not
    measured biological radii and should not be used as ground-truth geometry.
    """
    if graph.number_of_nodes() == 0:
        return {}

    positions = _graph_positions(graph)
    rooted = _root_graph(graph, positions, root=root)

    if twig_radius_scale <= 0.0:
        raise ValueError("twig_radius_scale must be positive.")
    if pipe_exponent < 0.0:
        raise ValueError("pipe_exponent must be non-negative.")
    if length_exponent < 0.0:
        raise ValueError("length_exponent must be non-negative.")

    resolved_twig_radius = _finite_positive(twig_radius)
    if resolved_twig_radius is None:
        resolved_twig_radius = _default_twig_radius(
            positions,
            twig_radius_scale=twig_radius_scale,
        )
    if resolved_twig_radius <= 0.0:
        raise ValueError("twig_radius must be positive.")

    paths = _root_to_tip_paths(rooted)
    prior = _initial_radius_prior(
        rooted,
        positions,
        paths,
        twig_radius=resolved_twig_radius,
        pipe_exponent=pipe_exponent,
        length_exponent=length_exponent,
    )
    smoothed_paths = [
        _smooth_path_radii(
            path,
            prior,
            twig_radius=resolved_twig_radius,
            terminal_weight=terminal_weight,
        )
        for path in paths
    ]
    radii = _merge_path_predictions(paths, smoothed_paths, prior)

    for _ in range(max(int(smoothing_passes), 0)):
        _enforce_parent_child_monotonicity(rooted, radii)

    return {node: max(float(radius), _EPS) for node, radius in radii.items()}


def with_synthesized_radii(
    graph: "nx.Graph",
    *,
    radius_attr: str = SYNTHESIZED_RADIUS_ATTR,
    **kwargs,
) -> "nx.Graph":
    """Return a graph copy with synthesized radii stored on every node."""
    graph_copy = graph.copy()
    radii = synthesize_radii(graph_copy, **kwargs)
    for node, radius in radii.items():
        graph_copy.nodes[node][radius_attr] = radius
    return graph_copy
