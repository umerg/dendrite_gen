"""Tree-level morphometric descriptors and reference-standardized distances.

This module intentionally duplicates the 16-component descriptor implemented in
``validation/dist_metrics.py`` and the supporting extractors in
``validation/structural_metrics.py``.  The duplication keeps the reusable
``metrics`` package independent of the training-time validation stack; a parity
test protects the two implementations from silently drifting apart.

The descriptor assumes a rooted critical skeleton: stored nodes are branch or
termination points, so graph edges represent maximal branch segments.  It is not
invariant to arbitrary edge subdivision.  All components are translation
invariant and invariant to rotations about ``axis``.  Several components discard
even more orientation information.

Raw Euclidean distance is deliberately not exposed as the primary comparison:
the 16 components have incompatible units and scales.  Fit one
``MorphometricReference`` on a fixed ground-truth cohort, then compare z-scored
vectors.  The resulting Euclidean distance is a metric in descriptor space and a
pseudometric on trees because different trees can share the same summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal, Mapping, Sequence

import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist


MORPHOMETRIC_FEATURES: tuple[str, ...] = (
    "node_count",
    "leaf_count",
    "bifurcation_count",
    "axial_extent",
    "radial_span",
    "total_extent",
    "strahler",
    "partition_asymmetry",
    "mean_branch_length",
    "mean_bifurcation_angle",
    "mean_path_to_root",
    "mean_radial_to_root",
    "mean_contraction",
    "sholl_peak",
    "sholl_critical_radius",
    "sholl_auc",
)
DEFAULT_SHOLL_SHELLS = 32
NonfinitePolicy = Literal["raise", "reference_mean"]


def _position(graph: nx.Graph, node: object) -> np.ndarray:
    values = np.asarray(graph.nodes[node].get("pos", np.zeros(3)), dtype=np.float64)
    values = values.reshape(-1)
    if values.size < 3:
        values = np.pad(values, (0, 3 - values.size), mode="constant")
    return values[:3]


def _unit_axis(axis: Sequence[float]) -> np.ndarray:
    values = np.asarray(axis, dtype=np.float64).reshape(-1)
    if values.size != 3 or not np.all(np.isfinite(values)):
        raise ValueError("axis must contain three finite values")
    norm = float(np.linalg.norm(values))
    if norm <= 1e-12:
        raise ValueError("axis must be nonzero")
    return values / norm


def _root(graph: nx.Graph) -> object | None:
    root = graph.graph.get("root")
    return root if root in graph else None


def _rooted_children(
    graph: nx.Graph,
    root: object,
) -> dict[object, list[object]]:
    children: dict[object, list[object]] = {node: [] for node in graph.nodes}
    seen = {root}
    stack = [root]
    while stack:
        parent = stack.pop()
        for child in graph.neighbors(parent):
            if child in seen:
                continue
            seen.add(child)
            children[parent].append(child)
            stack.append(child)
    return children


def _edge_length(graph: nx.Graph, node_a: object, node_b: object) -> float:
    return float(np.linalg.norm(_position(graph, node_a) - _position(graph, node_b)))


def _finite_mean(values: Iterable[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def _branch_lengths(graph: nx.Graph) -> np.ndarray:
    return np.asarray(
        [_edge_length(graph, node_a, node_b) for node_a, node_b in graph.edges],
        dtype=np.float64,
    )


def _bifurcation_angles(graph: nx.Graph) -> np.ndarray:
    root = _root(graph)
    if root is None:
        return np.zeros(0, dtype=np.float64)
    children = _rooted_children(graph, root)
    angles: list[float] = []
    for parent, child_nodes in children.items():
        vectors = [
            _position(graph, child) - _position(graph, parent)
            for child in child_nodes
        ]
        vectors = [vector for vector in vectors if np.linalg.norm(vector) > 1e-12]
        for index_a in range(len(vectors)):
            for index_b in range(index_a + 1, len(vectors)):
                vector_a = vectors[index_a]
                vector_b = vectors[index_b]
                denominator = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
                cosine = float(
                    np.clip(np.dot(vector_a, vector_b) / denominator, -1.0, 1.0)
                )
                angles.append(float(math.degrees(math.acos(cosine))))
    return np.asarray(angles, dtype=np.float64)


def _root_path_lengths(graph: nx.Graph) -> np.ndarray:
    root = _root(graph)
    if root is None or graph.number_of_nodes() < 2:
        return np.zeros(0, dtype=np.float64)
    lengths = nx.single_source_dijkstra_path_length(
        graph,
        root,
        weight=lambda node_a, node_b, _data: _edge_length(
            graph, node_a, node_b
        ),
    )
    return np.asarray(
        [float(lengths[node]) for node in graph.nodes if node != root and node in lengths],
        dtype=np.float64,
    )


def _root_euclidean_distances(graph: nx.Graph) -> np.ndarray:
    root = _root(graph)
    if root is None or graph.number_of_nodes() < 2:
        return np.zeros(0, dtype=np.float64)
    root_position = _position(graph, root)
    return np.asarray(
        [
            float(np.linalg.norm(_position(graph, node) - root_position))
            for node in graph.nodes
            if node != root
        ],
        dtype=np.float64,
    )


def _contraction_ratios(graph: nx.Graph) -> np.ndarray:
    root = _root(graph)
    if root is None or graph.number_of_nodes() < 2:
        return np.zeros(0, dtype=np.float64)
    children = _rooted_children(graph, root)
    leaves = [
        node for node, node_children in children.items() if node != root and not node_children
    ]
    if not leaves:
        return np.zeros(0, dtype=np.float64)
    path_lengths = nx.single_source_dijkstra_path_length(
        graph,
        root,
        weight=lambda node_a, node_b, _data: _edge_length(
            graph, node_a, node_b
        ),
    )
    root_position = _position(graph, root)
    ratios: list[float] = []
    for leaf in leaves:
        path_length = float(path_lengths.get(leaf, 0.0))
        if path_length <= 1e-12:
            continue
        chord_length = float(np.linalg.norm(_position(graph, leaf) - root_position))
        ratios.append(min(chord_length / path_length, 1.0))
    return np.asarray(ratios, dtype=np.float64)


def _subtree_statistics(
    graph: nx.Graph,
    root: object,
) -> tuple[dict[object, list[object]], dict[object, int], dict[object, int]]:
    children = _rooted_children(graph, root)
    preorder: list[object] = []
    stack = [root]
    while stack:
        node = stack.pop()
        preorder.append(node)
        stack.extend(children[node])

    leaf_counts: dict[object, int] = {}
    strahler: dict[object, int] = {}
    for node in reversed(preorder):
        node_children = children[node]
        if not node_children:
            leaf_counts[node] = 1
            strahler[node] = 1
            continue
        leaf_counts[node] = sum(leaf_counts[child] for child in node_children)
        child_orders = [strahler[child] for child in node_children]
        maximum = max(child_orders)
        strahler[node] = maximum + 1 if child_orders.count(maximum) >= 2 else maximum
    return children, leaf_counts, strahler


def _strahler_number(graph: nx.Graph) -> float:
    root = _root(graph)
    if root is None or graph.number_of_nodes() == 0:
        return float("nan")
    _children, _leaf_counts, orders = _subtree_statistics(graph, root)
    return float(orders[root])


def _partition_asymmetry(graph: nx.Graph) -> float:
    root = _root(graph)
    if root is None or graph.number_of_nodes() == 0:
        return float("nan")
    children, leaf_counts, _orders = _subtree_statistics(graph, root)
    node_values: list[float] = []
    for child_nodes in children.values():
        if len(child_nodes) < 2:
            continue
        counts = [leaf_counts[child] for child in child_nodes]
        pair_values: list[float] = []
        for index_a in range(len(counts)):
            for index_b in range(index_a + 1, len(counts)):
                count_a, count_b = counts[index_a], counts[index_b]
                denominator = count_a + count_b - 2
                pair_values.append(
                    0.0
                    if denominator <= 0
                    else abs(count_a - count_b) / float(denominator)
                )
        if pair_values:
            node_values.append(float(np.mean(pair_values)))
    return float(np.mean(node_values)) if node_values else float("nan")


def fit_shared_sholl_radii(
    graphs: Iterable[nx.Graph],
    *,
    n_shells: int = DEFAULT_SHOLL_SHELLS,
) -> np.ndarray:
    """Fit shared shell radii over ``(0, max reference root radius]``."""

    if isinstance(n_shells, bool) or not isinstance(n_shells, int) or n_shells <= 0:
        raise ValueError("n_shells must be a positive integer")
    maximum = 0.0
    for graph in graphs:
        values = _root_euclidean_distances(graph)
        if values.size:
            maximum = max(maximum, float(np.max(values)))
    if maximum <= 0.0:
        return np.zeros(0, dtype=np.float64)
    return np.linspace(0.0, maximum, n_shells + 1, dtype=np.float64)[1:]


def _sholl_summary(
    graph: nx.Graph,
    radii: np.ndarray | None,
) -> tuple[float, float, float]:
    root = _root(graph)
    if root is None or graph.number_of_edges() == 0:
        return float("nan"), float("nan"), float("nan")
    root_position = _position(graph, root)
    distances = {
        node: float(np.linalg.norm(_position(graph, node) - root_position))
        for node in graph.nodes
    }
    if radii is None:
        maximum = max(distances.values(), default=0.0)
        if maximum <= 0.0:
            return float("nan"), float("nan"), float("nan")
        radii = np.linspace(
            0.0,
            maximum,
            DEFAULT_SHOLL_SHELLS + 1,
            dtype=np.float64,
        )[1:]
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    if radii.size == 0:
        return float("nan"), float("nan"), float("nan")
    counts = np.zeros(radii.shape, dtype=np.float64)
    for node_a, node_b in graph.edges:
        lower, upper = sorted((distances[node_a], distances[node_b]))
        counts += ((radii > lower) & (radii <= upper)).astype(np.float64)
    if float(np.max(counts)) <= 0.0:
        return float("nan"), float("nan"), float("nan")
    peak_index = int(np.argmax(counts))
    maximum_radius = float(np.max(radii))
    critical_radius = (
        float(radii[peak_index] / maximum_radius)
        if maximum_radius > 0.0
        else float("nan")
    )
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return (
        float(np.max(counts)),
        critical_radius,
        float(trapezoid(counts, radii)),
    )


def tree_morphometric_vector(
    graph: nx.Graph,
    *,
    axis: Sequence[float] = (0.0, 0.0, 1.0),
    sholl_radii: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Return the fixed-order 16-component descriptor for one rooted tree."""

    unit_axis = _unit_axis(axis)
    root = _root(graph)
    if graph.number_of_nodes() == 0:
        positions = np.zeros((0, 3), dtype=np.float64)
    else:
        positions = np.stack([_position(graph, node) for node in graph.nodes], axis=0)

    if positions.size:
        axial_coordinates = positions @ unit_axis
        axial_extent = float(np.max(axial_coordinates) - np.min(axial_coordinates))
        perpendicular = positions - np.outer(axial_coordinates, unit_axis)
        radial_span = float(np.max(pdist(perpendicular))) if len(positions) >= 2 else 0.0
        total_extent = float(np.max(pdist(positions))) if len(positions) >= 2 else 0.0
    else:
        axial_extent = radial_span = total_extent = float("nan")

    if root is None:
        leaf_count = bifurcation_count = float("nan")
    else:
        children = _rooted_children(graph, root)
        leaf_count = float(sum(not values for values in children.values()))
        bifurcation_count = float(sum(len(values) >= 2 for values in children.values()))

    sholl_peak, sholl_critical_radius, sholl_auc = _sholl_summary(
        graph,
        (
            None
            if sholl_radii is None
            else np.asarray(sholl_radii, dtype=np.float64)
        ),
    )
    values = np.asarray(
        [
            float(graph.number_of_nodes()),
            leaf_count,
            bifurcation_count,
            axial_extent,
            radial_span,
            total_extent,
            _strahler_number(graph),
            _partition_asymmetry(graph),
            _finite_mean(_branch_lengths(graph)),
            _finite_mean(_bifurcation_angles(graph)),
            _finite_mean(_root_path_lengths(graph)),
            _finite_mean(_root_euclidean_distances(graph)),
            _finite_mean(_contraction_ratios(graph)),
            sholl_peak,
            sholl_critical_radius,
            sholl_auc,
        ],
        dtype=np.float64,
    )
    if values.shape != (len(MORPHOMETRIC_FEATURES),):
        raise RuntimeError("Internal morphometric descriptor length mismatch")
    return values


@dataclass(frozen=True)
class MorphometricReference:
    """Reference-cohort parameters defining standardized descriptor space."""

    axis: tuple[float, float, float]
    sholl_radii: tuple[float, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    reference_tree_count: int
    nonfinite_policy: NonfinitePolicy = "raise"
    scale_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        _unit_axis(self.axis)
        feature_count = len(MORPHOMETRIC_FEATURES)
        if len(self.mean) != feature_count or len(self.scale) != feature_count:
            raise ValueError(f"mean and scale must each have {feature_count} values")
        if not np.all(np.isfinite(self.mean)):
            raise ValueError("reference means must be finite")
        if not np.all(np.isfinite(self.scale)) or np.any(np.asarray(self.scale) <= 0.0):
            raise ValueError("reference scales must be finite and positive")
        if not np.all(np.isfinite(self.sholl_radii)):
            raise ValueError("Sholl radii must be finite")
        if self.reference_tree_count <= 0:
            raise ValueError("reference_tree_count must be positive")
        if not math.isfinite(self.scale_epsilon) or self.scale_epsilon <= 0.0:
            raise ValueError("scale_epsilon must be finite and positive")
        if self.nonfinite_policy not in {"raise", "reference_mean"}:
            raise ValueError(f"Unknown nonfinite policy: {self.nonfinite_policy!r}")

    @property
    def configuration(self) -> Mapping[str, object]:
        return {
            "descriptor": "validation_morphometric_vector_v1",
            "feature_names": list(MORPHOMETRIC_FEATURES),
            "axis": list(self.axis),
            "sholl_radii": list(self.sholl_radii),
            "reference_mean": list(self.mean),
            "reference_scale": list(self.scale),
            "reference_tree_count": self.reference_tree_count,
            "standardization": "population_zscore",
            "standard_deviation_ddof": 0,
            "scale_epsilon": self.scale_epsilon,
            "zero_variance_rule": "replace_scale_with_one",
            "nonfinite_policy": self.nonfinite_policy,
        }


def _finite_column_mean_std(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feature_count = matrix.shape[1]
    means = np.zeros(feature_count, dtype=np.float64)
    scales = np.ones(feature_count, dtype=np.float64)
    for feature_index in range(feature_count):
        values = matrix[:, feature_index]
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        means[feature_index] = float(np.mean(values))
        scales[feature_index] = float(np.std(values, ddof=0))
    return means, scales


def fit_morphometric_reference(
    graphs: Sequence[nx.Graph],
    *,
    axis: Sequence[float] = (0.0, 0.0, 1.0),
    n_shells: int = DEFAULT_SHOLL_SHELLS,
    nonfinite_policy: NonfinitePolicy = "raise",
    scale_epsilon: float = 1e-8,
) -> MorphometricReference:
    """Fit shared Sholl radii and per-feature population z-score parameters."""

    reference_graphs = tuple(graphs)
    if not reference_graphs:
        raise ValueError("At least one reference tree is required")
    if nonfinite_policy not in {"raise", "reference_mean"}:
        raise ValueError(f"Unknown nonfinite policy: {nonfinite_policy!r}")
    if not math.isfinite(scale_epsilon) or scale_epsilon <= 0.0:
        raise ValueError("scale_epsilon must be finite and positive")
    unit_axis = _unit_axis(axis)
    radii = fit_shared_sholl_radii(reference_graphs, n_shells=n_shells)
    matrix = np.stack(
        [
            tree_morphometric_vector(
                graph,
                axis=unit_axis,
                sholl_radii=radii,
            )
            for graph in reference_graphs
        ],
        axis=0,
    )
    if nonfinite_policy == "raise" and not np.all(np.isfinite(matrix)):
        rows, columns = np.where(~np.isfinite(matrix))
        examples = [
            f"tree {int(row)}: {MORPHOMETRIC_FEATURES[int(column)]}"
            for row, column in zip(rows[:5], columns[:5], strict=True)
        ]
        raise ValueError(
            "Reference cohort contains undefined morphometric components ("
            + "; ".join(examples)
            + ")"
        )
    means, scales = _finite_column_mean_std(matrix)
    scales = np.where(scales < scale_epsilon, 1.0, scales)
    return MorphometricReference(
        axis=tuple(float(value) for value in unit_axis),  # type: ignore[arg-type]
        sholl_radii=tuple(float(value) for value in radii),
        mean=tuple(float(value) for value in means),
        scale=tuple(float(value) for value in scales),
        reference_tree_count=len(reference_graphs),
        nonfinite_policy=nonfinite_policy,
        scale_epsilon=float(scale_epsilon),
    )


def standardize_morphometric_vector(
    vector: Sequence[float] | np.ndarray,
    reference: MorphometricReference,
) -> np.ndarray:
    """Transform one raw vector into the fitted descriptor space."""

    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if values.shape != (len(MORPHOMETRIC_FEATURES),):
        raise ValueError(
            f"vector must contain {len(MORPHOMETRIC_FEATURES)} components"
        )
    nonfinite = ~np.isfinite(values)
    if np.any(nonfinite):
        if reference.nonfinite_policy == "raise":
            missing = [
                MORPHOMETRIC_FEATURES[index]
                for index in np.flatnonzero(nonfinite)
            ]
            raise ValueError(f"Undefined morphometric components: {missing!r}")
        values = values.copy()
        values[nonfinite] = np.asarray(reference.mean)[nonfinite]
    return (values - np.asarray(reference.mean)) / np.asarray(reference.scale)


def prepare_morphometric_tree(
    graph: nx.Graph,
    reference: MorphometricReference,
) -> np.ndarray:
    """Extract and standardize one tree using a fixed reference cohort."""

    raw = tree_morphometric_vector(
        graph,
        axis=reference.axis,
        sholl_radii=reference.sholl_radii,
    )
    return standardize_morphometric_vector(raw, reference)


def morphometric_euclidean_distance_prepared(
    prepared_a: Sequence[float] | np.ndarray,
    prepared_b: Sequence[float] | np.ndarray,
) -> float:
    """Euclidean distance between two standardized descriptor vectors."""

    vector_a = np.asarray(prepared_a, dtype=np.float64).reshape(-1)
    vector_b = np.asarray(prepared_b, dtype=np.float64).reshape(-1)
    expected = (len(MORPHOMETRIC_FEATURES),)
    if vector_a.shape != expected or vector_b.shape != expected:
        raise ValueError(f"Prepared vectors must have shape {expected}")
    if not np.all(np.isfinite(vector_a)) or not np.all(np.isfinite(vector_b)):
        raise ValueError("Prepared vectors must be finite")
    return float(np.linalg.norm(vector_a - vector_b))


def morphometric_euclidean_distance(
    tree_a: nx.Graph,
    tree_b: nx.Graph,
    *,
    reference: MorphometricReference,
) -> float:
    """Compare two trees in reference-standardized morphometric space."""

    return morphometric_euclidean_distance_prepared(
        prepare_morphometric_tree(tree_a, reference),
        prepare_morphometric_tree(tree_b, reference),
    )


__all__ = [
    "DEFAULT_SHOLL_SHELLS",
    "MORPHOMETRIC_FEATURES",
    "MorphometricReference",
    "NonfinitePolicy",
    "fit_morphometric_reference",
    "fit_shared_sholl_radii",
    "morphometric_euclidean_distance",
    "morphometric_euclidean_distance_prepared",
    "prepare_morphometric_tree",
    "standardize_morphometric_vector",
    "tree_morphometric_vector",
]
