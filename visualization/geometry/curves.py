"""Endpoint-preserving branch curve helpers for visualization."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
import hashlib
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


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


def _graph_positions(graph: nx.Graph) -> dict[Hashable, np.ndarray]:
    return {
        node: _pos_to_xyz(graph.nodes[node].get("pos", np.zeros(3)))
        for node in graph.nodes
    }


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _choose_root(
    graph: nx.Graph,
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
    graph: nx.Graph,
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


def _branch_paths(rooted: _RootedTree) -> list[list[Hashable]]:
    critical = {
        node
        for node in rooted.order
        if node in rooted.roots or len(rooted.children[node]) != 1
    }
    paths: list[list[Hashable]] = []
    for start in rooted.order:
        if start not in critical:
            continue
        for child in rooted.children[start]:
            path = [start, child]
            current = child
            while current not in critical:
                current = rooted.children[current][0]
                path.append(current)
            paths.append(path)
    return paths


def _path_positions(
    path: Sequence[Hashable],
    positions: dict[Hashable, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float]:
    pts = np.stack([positions[node] for node in path], axis=0)
    if len(pts) == 1:
        return pts, np.array([0.0]), 0.0
    lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    lengths = np.maximum(lengths, _EPS)
    distances = np.concatenate([[0.0], np.cumsum(lengths)])
    return pts, distances, float(distances[-1])


def _sample_polyline(
    pts: np.ndarray,
    distances: np.ndarray,
    sample_distances: np.ndarray,
) -> np.ndarray:
    if len(pts) == 1:
        return np.repeat(pts, repeats=len(sample_distances), axis=0)

    sampled = []
    for distance in sample_distances:
        idx = int(np.searchsorted(distances, distance, side="right") - 1)
        idx = min(max(idx, 0), len(pts) - 2)
        segment_start = distances[idx]
        segment_length = max(float(distances[idx + 1] - segment_start), _EPS)
        alpha = float((distance - segment_start) / segment_length)
        sampled.append((1.0 - alpha) * pts[idx] + alpha * pts[idx + 1])
    return np.asarray(sampled, dtype=float)


def _perpendicular_frame(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(direction, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= _EPS:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    axis = axis / axis_norm
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u = u / max(float(np.linalg.norm(u)), _EPS)
    v = np.cross(axis, u)
    v = v / max(float(np.linalg.norm(v)), _EPS)
    return u, v


def _rng_for_path(seed: int, path_index: int, path: Sequence[Hashable]) -> np.random.Generator:
    key = f"{seed}:{path_index}:{repr(tuple(path))}".encode("utf8")
    digest = hashlib.sha256(key).digest()
    path_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(path_seed)


def _momentum_bridge_offsets(
    count: int,
    *,
    rng: np.random.Generator,
    amplitude: float,
    momentum: float,
) -> np.ndarray:
    if count <= 2 or amplitude <= 0.0:
        return np.zeros((count, 2), dtype=float)

    offsets = np.zeros((count, 2), dtype=float)
    velocity = np.zeros(2, dtype=float)
    memory = float(np.clip(momentum, 0.0, 0.98))
    innovation = max(1.0 - memory, 0.02)
    for idx in range(1, count):
        velocity = memory * velocity + innovation * rng.normal(size=2)
        offsets[idx] = offsets[idx - 1] + velocity

    s = np.linspace(0.0, 1.0, count)
    bridge = (1.0 - s[:, None]) * offsets[0] + s[:, None] * offsets[-1]
    offsets = offsets - bridge
    offsets *= np.sin(np.pi * s)[:, None]

    rms = float(np.sqrt(np.mean(np.sum(offsets * offsets, axis=1))))
    if rms <= _EPS:
        return np.zeros((count, 2), dtype=float)
    offsets *= float(amplitude) / rms

    norms = np.linalg.norm(offsets, axis=1)
    limit = max(float(amplitude) * 2.5, _EPS)
    too_large = norms > limit
    if np.any(too_large):
        offsets[too_large] *= (limit / norms[too_large])[:, None]
    offsets[0] = 0.0
    offsets[-1] = 0.0
    return offsets


def _curved_path_positions(
    path_index: int,
    path: Sequence[Hashable],
    positions: dict[Hashable, np.ndarray],
    *,
    subsegments: int,
    wiggle_scale: float,
    momentum: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts, distances, total_length = _path_positions(path, positions)
    samples_per_edge = max(int(subsegments), 1)
    count = max((len(path) - 1) * samples_per_edge + 1, 2)
    sample_distances = np.linspace(0.0, total_length, count)
    base = _sample_polyline(pts, distances, sample_distances)

    if total_length <= _EPS or wiggle_scale <= 0.0:
        return base, sample_distances, distances

    rng = _rng_for_path(seed, path_index, path)
    offsets_2d = _momentum_bridge_offsets(
        count,
        rng=rng,
        amplitude=float(wiggle_scale) * total_length,
        momentum=momentum,
    )
    u, v = _perpendicular_frame(pts[-1] - pts[0])
    curved = base + offsets_2d[:, 0, None] * u + offsets_2d[:, 1, None] * v
    curved[0] = pts[0]
    curved[-1] = pts[-1]
    return curved, sample_distances, distances


def _interpolate_numeric_attr(
    graph: nx.Graph,
    path: Sequence[Hashable],
    distances: np.ndarray,
    sample_distance: float,
    attr: str,
) -> float | None:
    values = [_finite_float(graph.nodes[node].get(attr)) for node in path]
    if any(value is None for value in values):
        return None
    return float(np.interp(sample_distance, distances, values))


def with_curved_branches(
    graph: nx.Graph,
    *,
    subsegments: int = 5,
    wiggle_scale: float = 0.02,
    momentum: float = 0.75,
    seed: int = 0,
    radius_attrs: Sequence[str] = ("radius",),
) -> nx.Graph:
    """Return a graph copy whose branch paths have smooth random centerlines.

    The transform preserves original branch endpoints exactly and inserts
    intermediate nodes along each branch path. It is intended as a
    visualization-only pre-processing step before radius synthesis/rendering.
    """
    if graph.number_of_nodes() <= 1 or graph.number_of_edges() == 0:
        return graph.copy()

    positions = _graph_positions(graph)
    rooted = _root_graph(graph, positions, root=None)
    paths = _branch_paths(rooted)
    if not paths:
        return graph.copy()

    curved = nx.Graph()
    curved.graph.update(graph.graph)

    def add_original_node(node: Hashable) -> None:
        if node in curved:
            return
        attrs = dict(graph.nodes[node])
        attrs["pos"] = positions[node].copy()
        curved.add_node(node, **attrs)

    for path_index, path in enumerate(paths):
        add_original_node(path[0])
        add_original_node(path[-1])
        curved_positions, sample_distances, original_distances = _curved_path_positions(
            path_index,
            path,
            positions,
            subsegments=subsegments,
            wiggle_scale=wiggle_scale,
            momentum=momentum,
            seed=seed,
        )

        previous: Hashable = path[0]
        for sample_idx in range(1, len(curved_positions) - 1):
            node = ("curve", path_index, sample_idx)
            attrs = {"pos": curved_positions[sample_idx]}
            for attr in radius_attrs:
                if not attr:
                    continue
                value = _interpolate_numeric_attr(
                    graph,
                    path,
                    original_distances,
                    float(sample_distances[sample_idx]),
                    attr,
                )
                if value is not None:
                    attrs[attr] = value
            curved.add_node(node, **attrs)
            curved.add_edge(previous, node)
            previous = node

        curved.add_edge(previous, path[-1])

    if graph.graph.get("root") in curved:
        curved.graph["root"] = graph.graph["root"]
    return curved
