r"""Distribution-based Wasserstein dissimilarities for rooted geometric trees.

This module is intentionally standalone: it does not depend on the validation
or visualization packages.  Every public entry point validates that its input
is a non-empty, undirected NetworkX tree with an explicit root and finite 3-D
``pos`` attributes.

The canonical distribution names are deliberately explicit:

``critical_branch_cable_length``
    Cable lengths of maximal paths whose internal nodes have one child (degree
    two in the unrooted tree).  These are morphological branches rather than
    raw SWC edges.
``critical_branch_chord_sibling_angle_deg``
    Pairwise angles between sibling branch chords at every branching critical
    node.  A chord joins the branch point to the next critical node.
``critical_node_root_path_length``
    Cable distance from the root to every non-root critical node.
``uniform_cable_radial_xy``
    Root-relative :math:`\sqrt{x^2+y^2}` sampled along cable.
``uniform_cable_height_z``
    Signed root-relative height sampled along cable.
``uniform_cable_root_euclidean``
    Root-relative Euclidean distance sampled along cable.
``critical_node_branch_order``
    Centrifugal order of every non-root critical node.  Primary branches have
    order one and the order increases after each downstream bifurcation.

The three ``uniform_cable_*`` distributions use midpoint quadrature on every
edge.  Each sample is weighted by the amount of cable it represents, so raw
SWC edge subdivision does not assign extra probability mass to densely traced
regions.  ``sample_spacing`` controls quadrature resolution, not biological
normalization.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Literal

import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance


CRITICAL_BRANCH_CABLE_LENGTH = "critical_branch_cable_length"
CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG = (
    "critical_branch_chord_sibling_angle_deg"
)
CRITICAL_NODE_ROOT_PATH_LENGTH = "critical_node_root_path_length"
UNIFORM_CABLE_RADIAL_XY = "uniform_cable_radial_xy"
UNIFORM_CABLE_HEIGHT_Z = "uniform_cable_height_z"
UNIFORM_CABLE_ROOT_EUCLIDEAN = "uniform_cable_root_euclidean"
CRITICAL_NODE_BRANCH_ORDER = "critical_node_branch_order"

DEFAULT_DISTRIBUTIONS: tuple[str, ...] = (
    CRITICAL_BRANCH_CABLE_LENGTH,
    CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG,
    CRITICAL_NODE_ROOT_PATH_LENGTH,
    UNIFORM_CABLE_RADIAL_XY,
    UNIFORM_CABLE_HEIGHT_Z,
    UNIFORM_CABLE_ROOT_EUCLIDEAN,
    CRITICAL_NODE_BRANCH_ORDER,
)
"""Canonical distribution names used by the default comparison panel."""

DISTRIBUTION_NAMES = DEFAULT_DISTRIBUTIONS
"""All distribution names accepted by :func:`tree_distribution`."""

EmptyPolicy = Literal["nan", "raise"]
DistributionStatus = Literal["ok", "both_empty", "undefined_one_empty"]

_CABLE_DISTRIBUTIONS = frozenset(
    {
        UNIFORM_CABLE_RADIAL_XY,
        UNIFORM_CABLE_HEIGHT_Z,
        UNIFORM_CABLE_ROOT_EUCLIDEAN,
    }
)
_LENGTH_EPS = 1e-12


@dataclass(frozen=True)
class EmpiricalTreeDistribution:
    """One empirical distribution and optional probability-mass weights.

    ``weights`` is ``None`` for equally weighted structural observations.  For
    cable-sampled distributions, each weight is the physical cable length
    represented by the corresponding midpoint sample.  SciPy normalizes these
    positive weights when evaluating the 1-Wasserstein distance.
    """

    name: str
    values: np.ndarray
    weights: np.ndarray | None = None


@dataclass(frozen=True)
class DistributionWassersteinResult:
    """One distribution comparison with explicit empty-feature diagnostics."""

    name: str
    value: float
    status: DistributionStatus
    sample_count_a: int
    sample_count_b: int
    empty_a: bool
    empty_b: bool


@dataclass(frozen=True)
class _RootedGeometry:
    graph: nx.Graph
    root: Hashable
    positions: dict[Hashable, np.ndarray]
    parent: dict[Hashable, Hashable | None]
    children: dict[Hashable, tuple[Hashable, ...]]
    traversal: tuple[Hashable, ...]


def _validate_sample_spacing(sample_spacing: float) -> float:
    try:
        spacing = float(sample_spacing)
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_spacing must be a finite positive number.") from exc
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("sample_spacing must be a finite positive number.")
    return spacing


def _validate_empty_policy(empty_policy: str) -> EmptyPolicy:
    if empty_policy not in {"nan", "raise"}:
        raise ValueError("empty_policy must be either 'nan' or 'raise'.")
    return empty_policy  # type: ignore[return-value]


def _position_xyz(graph: nx.Graph, node: Hashable) -> np.ndarray:
    if "pos" not in graph.nodes[node]:
        raise ValueError(f"Node {node!r} is missing its required 3-D 'pos' attribute.")
    try:
        position = np.asarray(graph.nodes[node]["pos"], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Node {node!r} has a non-numeric 'pos' attribute.") from exc
    if position.size < 3:
        raise ValueError(
            f"Node {node!r} has a 'pos' attribute with fewer than three coordinates."
        )
    xyz = position[:3].copy()
    if not np.all(np.isfinite(xyz)):
        raise ValueError(f"Node {node!r} has a non-finite 3-D position.")
    return xyz


def _validated_rooted_geometry(
    graph: nx.Graph,
    *,
    root: Hashable | None,
) -> _RootedGeometry:
    if not isinstance(graph, nx.Graph):
        raise TypeError("Expected a NetworkX Graph containing one rooted tree.")
    if graph.is_directed():
        raise ValueError("Tree distributions require an undirected NetworkX graph.")
    if graph.is_multigraph():
        raise ValueError("Tree distributions do not accept multigraphs.")
    if graph.number_of_nodes() == 0:
        raise ValueError("Tree distributions do not accept an empty graph.")
    if not nx.is_tree(graph):
        raise ValueError("Expected a connected, acyclic NetworkX tree.")

    resolved_root = graph.graph.get("root") if root is None else root
    if resolved_root is None or resolved_root not in graph:
        raise ValueError(
            "A valid root is required; pass root=... or set graph.graph['root']."
        )

    positions = {node: _position_xyz(graph, node) for node in graph.nodes}

    parent: dict[Hashable, Hashable | None] = {resolved_root: None}
    mutable_children: dict[Hashable, list[Hashable]] = {
        node: [] for node in graph.nodes
    }
    traversal: list[Hashable] = []
    queue: deque[Hashable] = deque([resolved_root])
    while queue:
        node = queue.popleft()
        traversal.append(node)
        for neighbor in graph.neighbors(node):
            if neighbor in parent:
                continue
            parent[neighbor] = node
            mutable_children[node].append(neighbor)
            queue.append(neighbor)

    children = {node: tuple(nodes) for node, nodes in mutable_children.items()}
    return _RootedGeometry(
        graph=graph,
        root=resolved_root,
        positions=positions,
        parent=parent,
        children=children,
        traversal=tuple(traversal),
    )


def _edge_length(tree: _RootedGeometry, u: Hashable, v: Hashable) -> float:
    return float(np.linalg.norm(tree.positions[v] - tree.positions[u]))


def _critical_nodes(tree: _RootedGeometry) -> set[Hashable]:
    return {
        node
        for node in tree.traversal
        if node == tree.root or len(tree.children[node]) != 1
    }


def _critical_branches(
    tree: _RootedGeometry,
) -> list[tuple[Hashable, Hashable, float]]:
    """Return ``(start, end, cable_length)`` for maximal critical branches."""
    critical = _critical_nodes(tree)
    branches: list[tuple[Hashable, Hashable, float]] = []
    for start in tree.traversal:
        if start not in critical:
            continue
        for first_child in tree.children[start]:
            previous = start
            current = first_child
            cable_length = _edge_length(tree, previous, current)
            while current not in critical:
                next_node = tree.children[current][0]
                cable_length += _edge_length(tree, current, next_node)
                previous, current = current, next_node
            branches.append((start, current, cable_length))
    return branches


def _critical_branch_cable_lengths(tree: _RootedGeometry) -> np.ndarray:
    return np.asarray(
        [length for _start, _end, length in _critical_branches(tree)],
        dtype=np.float64,
    )


def _critical_branch_chord_sibling_angles_deg(
    tree: _RootedGeometry,
) -> np.ndarray:
    endpoints_by_start: dict[Hashable, list[Hashable]] = {}
    for start, end, _length in _critical_branches(tree):
        endpoints_by_start.setdefault(start, []).append(end)

    angles: list[float] = []
    for start, endpoints in endpoints_by_start.items():
        if len(endpoints) < 2:
            continue
        origin = tree.positions[start]
        chords = [tree.positions[end] - origin for end in endpoints]
        chords = [chord for chord in chords if np.linalg.norm(chord) > _LENGTH_EPS]
        for index, first in enumerate(chords):
            first_norm = float(np.linalg.norm(first))
            for second in chords[index + 1 :]:
                denominator = first_norm * float(np.linalg.norm(second))
                cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
                angles.append(float(np.degrees(np.arccos(cosine))))
    return np.asarray(angles, dtype=np.float64)


def _root_path_lengths(tree: _RootedGeometry) -> dict[Hashable, float]:
    distances: dict[Hashable, float] = {tree.root: 0.0}
    for node in tree.traversal[1:]:
        parent = tree.parent[node]
        if parent is None:  # pragma: no cover - impossible after tree validation
            raise RuntimeError("Encountered a non-root node without a parent.")
        distances[node] = distances[parent] + _edge_length(tree, parent, node)
    return distances


def _critical_node_root_path_lengths(tree: _RootedGeometry) -> np.ndarray:
    critical = _critical_nodes(tree)
    path_lengths = _root_path_lengths(tree)
    return np.asarray(
        [
            path_lengths[node]
            for node in tree.traversal
            if node != tree.root and node in critical
        ],
        dtype=np.float64,
    )


def _critical_node_branch_orders(tree: _RootedGeometry) -> np.ndarray:
    orders: dict[Hashable, int] = {tree.root: 0}
    for node in tree.traversal:
        for child in tree.children[node]:
            if node == tree.root:
                orders[child] = 1
            else:
                orders[child] = orders[node] + int(len(tree.children[node]) >= 2)

    critical = _critical_nodes(tree)
    return np.asarray(
        [
            float(orders[node])
            for node in tree.traversal
            if node != tree.root and node in critical
        ],
        dtype=np.float64,
    )


def _stable_bin_count(length: float, spacing: float) -> int:
    # The small relative tolerance prevents an exact spacing boundary from
    # changing bin count under roundoff introduced by an SO(2) rotation.
    ratio = length / spacing
    adjusted = ratio - 1e-12 * max(1.0, abs(ratio))
    return max(1, int(np.ceil(adjusted)))


def _uniform_cable_samples(
    tree: _RootedGeometry,
    *,
    distribution: str,
    sample_spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    root_position = tree.positions[tree.root]
    value_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []

    for u, v in tree.graph.edges:
        start = tree.positions[u]
        end = tree.positions[v]
        length = float(np.linalg.norm(end - start))
        if length <= _LENGTH_EPS:
            continue

        count = _stable_bin_count(length, sample_spacing)
        alpha = (np.arange(count, dtype=np.float64) + 0.5) / count
        samples = start[None, :] + alpha[:, None] * (end - start)[None, :]
        relative = samples - root_position[None, :]

        if distribution == UNIFORM_CABLE_RADIAL_XY:
            values = np.linalg.norm(relative[:, :2], axis=1)
        elif distribution == UNIFORM_CABLE_HEIGHT_Z:
            values = relative[:, 2]
        elif distribution == UNIFORM_CABLE_ROOT_EUCLIDEAN:
            values = np.linalg.norm(relative, axis=1)
        else:  # pragma: no cover - guarded by the caller
            raise ValueError(f"Unsupported cable distribution: {distribution!r}")

        value_parts.append(np.asarray(values, dtype=np.float64))
        weight_parts.append(np.full(count, length / count, dtype=np.float64))

    if not value_parts:
        empty = np.zeros((0,), dtype=np.float64)
        return empty, empty.copy()
    return np.concatenate(value_parts), np.concatenate(weight_parts)


def tree_distribution(
    graph: nx.Graph,
    name: str,
    *,
    root: Hashable | None = None,
    sample_spacing: float = 1.0,
) -> EmpiricalTreeDistribution:
    """Extract one named empirical distribution from a rooted geometric tree.

    Parameters
    ----------
    graph:
        A connected, acyclic, undirected NetworkX graph.  Every node must have
        a finite three-dimensional ``pos`` attribute.
    name:
        One of :data:`DISTRIBUTION_NAMES`.
    root:
        Root node identifier.  If omitted, ``graph.graph['root']`` is required.
    sample_spacing:
        Maximum midpoint-quadrature bin length for ``uniform_cable_*``
        distributions, in the same physical units as node positions.
    """
    if name not in DISTRIBUTION_NAMES:
        supported = ", ".join(DISTRIBUTION_NAMES)
        raise ValueError(f"Unknown tree distribution {name!r}. Supported: {supported}.")
    spacing = _validate_sample_spacing(sample_spacing)
    tree = _validated_rooted_geometry(graph, root=root)

    if name == CRITICAL_BRANCH_CABLE_LENGTH:
        values = _critical_branch_cable_lengths(tree)
    elif name == CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG:
        values = _critical_branch_chord_sibling_angles_deg(tree)
    elif name == CRITICAL_NODE_ROOT_PATH_LENGTH:
        values = _critical_node_root_path_lengths(tree)
    elif name == CRITICAL_NODE_BRANCH_ORDER:
        values = _critical_node_branch_orders(tree)
    elif name in _CABLE_DISTRIBUTIONS:
        values, weights = _uniform_cable_samples(
            tree,
            distribution=name,
            sample_spacing=spacing,
        )
        return EmpiricalTreeDistribution(name=name, values=values, weights=weights)
    else:  # pragma: no cover - name was checked against the exhaustive constant
        raise AssertionError(f"Unhandled tree distribution: {name!r}")

    return EmpiricalTreeDistribution(name=name, values=values, weights=None)


def distribution_wasserstein_distance(
    graph_a: nx.Graph,
    graph_b: nx.Graph,
    name: str,
    *,
    root_a: Hashable | None = None,
    root_b: Hashable | None = None,
    sample_spacing: float = 1.0,
    empty_policy: EmptyPolicy = "nan",
) -> float:
    """Return the 1-Wasserstein distance between one named tree distribution.

    Two empty distributions are identical and return ``0.0``.  If exactly one
    is empty, ``empty_policy='nan'`` returns ``NaN`` and
    ``empty_policy='raise'`` raises ``ValueError``.  This avoids silently
    inventing a biological ground cost for a missing feature family (for
    example, comparing a tree without bifurcations to one with bifurcations).
    """
    return distribution_wasserstein_result(
        graph_a,
        graph_b,
        name,
        root_a=root_a,
        root_b=root_b,
        sample_spacing=sample_spacing,
        empty_policy=empty_policy,
    ).value


def distribution_wasserstein_result(
    graph_a: nx.Graph,
    graph_b: nx.Graph,
    name: str,
    *,
    root_a: Hashable | None = None,
    root_b: Hashable | None = None,
    sample_spacing: float = 1.0,
    empty_policy: EmptyPolicy = "nan",
) -> DistributionWassersteinResult:
    """Return a 1-Wasserstein value plus explicit empty-feature status."""
    policy = _validate_empty_policy(empty_policy)
    distribution_a = tree_distribution(
        graph_a,
        name,
        root=root_a,
        sample_spacing=sample_spacing,
    )
    distribution_b = tree_distribution(
        graph_b,
        name,
        root=root_b,
        sample_spacing=sample_spacing,
    )

    empty_a = distribution_a.values.size == 0
    empty_b = distribution_b.values.size == 0
    if empty_a and empty_b:
        return DistributionWassersteinResult(
            name=name,
            value=0.0,
            status="both_empty",
            sample_count_a=0,
            sample_count_b=0,
            empty_a=True,
            empty_b=True,
        )
    if empty_a or empty_b:
        if policy == "nan":
            return DistributionWassersteinResult(
                name=name,
                value=float("nan"),
                status="undefined_one_empty",
                sample_count_a=int(distribution_a.values.size),
                sample_count_b=int(distribution_b.values.size),
                empty_a=bool(empty_a),
                empty_b=bool(empty_b),
            )
        empty_side = "first" if empty_a else "second"
        raise ValueError(
            f"The {empty_side} tree has an empty {name!r} distribution."
        )

    value = float(
        wasserstein_distance(
            distribution_a.values,
            distribution_b.values,
            u_weights=distribution_a.weights,
            v_weights=distribution_b.weights,
        )
    )
    return DistributionWassersteinResult(
        name=name,
        value=value,
        status="ok",
        sample_count_a=int(distribution_a.values.size),
        sample_count_b=int(distribution_b.values.size),
        empty_a=False,
        empty_b=False,
    )


def all_default_distribution_wasserstein_distances(
    graph_a: nx.Graph,
    graph_b: nx.Graph,
    *,
    root_a: Hashable | None = None,
    root_b: Hashable | None = None,
    sample_spacing: float = 1.0,
    empty_policy: EmptyPolicy = "nan",
) -> dict[str, float]:
    """Return every default named distribution Wasserstein distance."""
    return {
        name: distribution_wasserstein_distance(
            graph_a,
            graph_b,
            name,
            root_a=root_a,
            root_b=root_b,
            sample_spacing=sample_spacing,
            empty_policy=empty_policy,
        )
        for name in DEFAULT_DISTRIBUTIONS
    }


__all__ = [
    "CRITICAL_BRANCH_CABLE_LENGTH",
    "CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG",
    "CRITICAL_NODE_BRANCH_ORDER",
    "CRITICAL_NODE_ROOT_PATH_LENGTH",
    "DEFAULT_DISTRIBUTIONS",
    "DISTRIBUTION_NAMES",
    "DistributionStatus",
    "DistributionWassersteinResult",
    "EmpiricalTreeDistribution",
    "UNIFORM_CABLE_HEIGHT_Z",
    "UNIFORM_CABLE_RADIAL_XY",
    "UNIFORM_CABLE_ROOT_EUCLIDEAN",
    "all_default_distribution_wasserstein_distances",
    "distribution_wasserstein_distance",
    "distribution_wasserstein_result",
    "tree_distribution",
]
