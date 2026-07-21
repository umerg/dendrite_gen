"""Fused Gromov--Wasserstein distances between rooted geometric trees.

This module deliberately has no dependency on the validation or visualization
packages.  POT is an optional dependency and is imported only when the metric
is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Literal

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

from .so2 import minimize_over_so2, rotate_points_about_axis


FeatureMode = Literal["axis", "xyz"]
MassMode = Literal["cable_length", "uniform_nodes"]


@dataclass(frozen=True)
class FusedGWResult:
    """Result of a pairwise Fused Gromov--Wasserstein comparison."""

    value: float
    feature_mode: FeatureMode
    alpha: float
    mass_mode: MassMode
    normalize: bool
    quotient_so2: bool
    angle_rad: float
    grid_size: int
    refine: bool
    max_iter: int
    tolerance: float
    n_nodes_1: int
    n_nodes_2: int


def _load_pot():
    """Import POT lazily so the rest of :mod:`metrics` remains lightweight."""

    try:
        return import_module("ot")
    except ImportError as exc:
        raise ImportError(
            "Fused Gromov-Wasserstein distance requires the optional POT "
            "package. Install it with `pip install POT` or "
            "`conda install -c conda-forge pot`."
        ) from exc


def _validate_tree(graph: nx.Graph, *, name: str) -> tuple[list[object], object, np.ndarray]:
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"{name} must be a NetworkX graph.")
    if graph.is_multigraph():
        raise ValueError(f"{name} must be a simple tree, not a multigraph.")
    if graph.number_of_nodes() == 0:
        raise ValueError(f"{name} must contain at least one node.")

    undirected = graph.to_undirected(as_view=True) if graph.is_directed() else graph
    if not nx.is_tree(undirected):
        raise ValueError(f"{name} must be connected and acyclic.")

    root = graph.graph.get("root")
    if root not in graph:
        raise ValueError(
            f"{name} must declare an existing root in graph.graph['root']."
        )

    nodes = list(graph.nodes)
    positions = np.empty((len(nodes), 3), dtype=np.float64)
    for index, node in enumerate(nodes):
        if "pos" not in graph.nodes[node]:
            raise ValueError(f"{name} node {node!r} is missing the 'pos' attribute.")
        position = np.asarray(graph.nodes[node]["pos"], dtype=np.float64).reshape(-1)
        if position.size < 3:
            raise ValueError(
                f"{name} node {node!r} has a 'pos' attribute with fewer than 3 values."
            )
        positions[index] = position[:3]

    if not np.all(np.isfinite(positions)):
        raise ValueError(f"{name} node positions must contain only finite values.")

    return nodes, root, positions


def _root_centered_positions(
    nodes: list[object], root: object, positions: np.ndarray
) -> np.ndarray:
    root_index = nodes.index(root)
    return positions - positions[root_index]


def _tree_distance_matrix(
    graph: nx.Graph, nodes: list[object], positions: np.ndarray
) -> np.ndarray:
    """Return path distances using Euclidean edge lengths as weights."""

    node_index = {node: index for index, node in enumerate(nodes)}
    weighted = nx.Graph()
    weighted.add_nodes_from(nodes)
    for u, v in graph.edges:
        length = float(
            np.linalg.norm(positions[node_index[u]] - positions[node_index[v]])
        )
        weighted.add_edge(u, v, length=length)

    distances = np.empty((len(nodes), len(nodes)), dtype=np.float64)
    for row, source in enumerate(nodes):
        lengths = nx.single_source_dijkstra_path_length(
            weighted, source, weight="length"
        )
        distances[row] = [lengths[target] for target in nodes]
    return distances


def _node_masses(
    graph: nx.Graph,
    nodes: list[object],
    positions: np.ndarray,
    *,
    mode: MassMode,
) -> np.ndarray:
    """Return probability mass under an explicit SWC discretization model."""
    if mode == "uniform_nodes":
        return np.full(len(nodes), 1.0 / len(nodes), dtype=np.float64)
    if mode != "cable_length":
        raise ValueError("mass_mode must be either 'cable_length' or 'uniform_nodes'.")

    node_index = {node: index for index, node in enumerate(nodes)}
    mass = np.zeros(len(nodes), dtype=np.float64)
    for u, v in graph.edges:
        length = float(
            np.linalg.norm(positions[node_index[u]] - positions[node_index[v]])
        )
        mass[node_index[u]] += 0.5 * length
        mass[node_index[v]] += 0.5 * length
    total = float(mass.sum())
    if total <= 1e-12:
        return np.full(len(nodes), 1.0 / len(nodes), dtype=np.float64)
    return mass / total


def _node_features(centered_positions: np.ndarray, mode: FeatureMode) -> np.ndarray:
    if mode == "xyz":
        return centered_positions
    if mode == "axis":
        radial_xy = np.linalg.norm(centered_positions[:, :2], axis=1)
        return np.column_stack((centered_positions[:, 2], radial_xy))
    raise ValueError("feature_mode must be either 'axis' or 'xyz'.")


def _normalize_inputs(
    structure_1: np.ndarray,
    structure_2: np.ndarray,
    features_1: np.ndarray,
    features_2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize a pair with symmetric, SO(2)-invariant scale factors."""

    structure_scale = max(
        float(np.max(structure_1, initial=0.0)),
        float(np.max(structure_2, initial=0.0)),
    )
    if structure_scale > 0.0:
        structure_1 = structure_1 / structure_scale
        structure_2 = structure_2 / structure_scale

    feature_scale = max(
        float(np.max(np.linalg.norm(features_1, axis=1), initial=0.0)),
        float(np.max(np.linalg.norm(features_2, axis=1), initial=0.0)),
    )
    if feature_scale > 0.0:
        features_1 = features_1 / feature_scale
        features_2 = features_2 / feature_scale

    return structure_1, structure_2, features_1, features_2


def _fgw_objective(
    ot,
    features_1: np.ndarray,
    features_2: np.ndarray,
    structure_1: np.ndarray,
    structure_2: np.ndarray,
    mass_1: np.ndarray,
    mass_2: np.ndarray,
    *,
    alpha: float,
    max_iter: int,
    tol: float,
) -> float:
    feature_cost = cdist(features_1, features_2, metric="sqeuclidean")
    value = ot.gromov.fused_gromov_wasserstein2(
        feature_cost,
        structure_1,
        structure_2,
        mass_1,
        mass_2,
        loss_fun="square_loss",
        alpha=alpha,
        max_iter=max_iter,
        tol_rel=tol,
        tol_abs=tol,
    )
    result = float(np.asarray(value))
    if not np.isfinite(result):
        raise RuntimeError("POT returned a non-finite Fused Gromov-Wasserstein value.")
    # Numerical solvers can return a tiny negative residual for a zero distance.
    return max(result, 0.0)


def fused_gromov_wasserstein_distance(
    tree_1: nx.Graph,
    tree_2: nx.Graph,
    *,
    feature_mode: FeatureMode = "xyz",
    alpha: float = 0.5,
    mass_mode: MassMode = "cable_length",
    normalize: bool = True,
    quotient_so2: bool = True,
    grid_size: int = 72,
    refine: bool = True,
    max_iter: int = 1_000,
    tol: float = 1e-9,
) -> FusedGWResult:
    """Compute a pairwise Fused Gromov--Wasserstein tree distance.

    The structural costs are shortest-path distances on each tree, with every
    edge weighted by its Euclidean length.  Node-feature costs are squared
    Euclidean distances. The default node mass assigns half of every incident
    edge's cable length to each endpoint, reducing sensitivity to uneven SWC
    tracing density; ``uniform_nodes`` remains available as an explicit
    baseline. The default ``xyz`` features retain relative azimuth and form
    the required shape quotient by
    minimizing over a relative z rotation of ``tree_2``.  The optional
    ``axis`` features ``(z, sqrt(x**2 + y**2))`` are cheaper and already
    invariant, but discard azimuthal information beyond what remains in tree
    path lengths.

    When ``normalize`` is true, both structural matrices share one maximum-path
    scale and both feature sets share one maximum-radius scale.  The common
    scales make normalization symmetric in the input pair and independent of a
    relative SO(2) rotation.
    """

    if feature_mode not in ("axis", "xyz"):
        raise ValueError("feature_mode must be either 'axis' or 'xyz'.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in the closed interval [0, 1].")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")

    nodes_1, root_1, positions_1 = _validate_tree(tree_1, name="tree_1")
    nodes_2, root_2, positions_2 = _validate_tree(tree_2, name="tree_2")
    centered_1 = _root_centered_positions(nodes_1, root_1, positions_1)
    centered_2 = _root_centered_positions(nodes_2, root_2, positions_2)

    structure_1 = _tree_distance_matrix(tree_1, nodes_1, positions_1)
    structure_2 = _tree_distance_matrix(tree_2, nodes_2, positions_2)
    features_1 = _node_features(centered_1, feature_mode)
    features_2 = _node_features(centered_2, feature_mode)

    if normalize:
        structure_1, structure_2, features_1, features_2 = _normalize_inputs(
            structure_1, structure_2, features_1, features_2
        )

    mass_1 = _node_masses(tree_1, nodes_1, positions_1, mode=mass_mode)
    mass_2 = _node_masses(tree_2, nodes_2, positions_2, mode=mass_mode)
    ot = _load_pot()

    def objective(candidate_features_2: np.ndarray) -> float:
        return _fgw_objective(
            ot,
            features_1,
            candidate_features_2,
            structure_1,
            structure_2,
            mass_1,
            mass_2,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
        )

    angle_rad = 0.0
    effective_quotient = quotient_so2 and feature_mode == "xyz" and alpha < 1.0
    if effective_quotient:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        minimization = minimize_over_so2(
            lambda angle: objective(
                rotate_points_about_axis(features_2, angle, z_axis)
            ),
            grid_size=grid_size,
            refine=refine,
        )
        value = float(minimization.value)
        angle_rad = float(minimization.angle_rad)
    else:
        # Axis features are invariant, and alpha=1 ignores features entirely.
        value = objective(features_2)

    return FusedGWResult(
        value=max(value, 0.0),
        feature_mode=feature_mode,
        alpha=float(alpha),
        mass_mode=mass_mode,
        normalize=bool(normalize),
        quotient_so2=bool(effective_quotient),
        angle_rad=angle_rad,
        grid_size=int(grid_size),
        refine=bool(refine),
        max_iter=int(max_iter),
        tolerance=float(tol),
        n_nodes_1=len(nodes_1),
        n_nodes_2=len(nodes_2),
    )


# Concise aliases for callers that already use the standard FGW abbreviation.
fused_gw_distance = fused_gromov_wasserstein_distance
fgw_distance = fused_gromov_wasserstein_distance


__all__ = [
    "FeatureMode",
    "FusedGWResult",
    "MassMode",
    "fgw_distance",
    "fused_gromov_wasserstein_distance",
    "fused_gw_distance",
]
