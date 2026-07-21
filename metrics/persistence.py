"""Tree-pair distances based on paper-style TMD persistence diagrams.

This module is intentionally a thin adapter: diagram construction remains in
``utils.tmd`` and the reusable diagram distance remains in
``visualization.tmd.distances``.
"""

from __future__ import annotations

from typing import Literal, Sequence

import networkx as nx
import numpy as np

try:
    # Package execution, e.g. ``python -m dendrite_gen...``.
    from dendrite_gen.utils.tmd import compute_tmd_barcode_diagram
    from dendrite_gen.visualization.tmd.distances import (
        GroundNorm,
        persistence_diagram_wasserstein_distance,
    )
except ModuleNotFoundError as exc:
    if exc.name != "dendrite_gen":
        raise
    # Repo-root execution, e.g. importing ``metrics.persistence``.
    from utils.tmd import compute_tmd_barcode_diagram  # type: ignore
    from visualization.tmd.distances import (  # type: ignore
        GroundNorm,
        persistence_diagram_wasserstein_distance,
    )


FiltrationName = Literal["path", "height", "rho"]
NormalizeMode = Literal["minmax", "max", "none"]
DEFAULT_FILTRATIONS: tuple[FiltrationName, ...] = ("path", "height", "rho")


def _root_centered_copy(tree: nx.Graph) -> nx.Graph:
    """Validate and copy a rooted geometric tree with its root at the origin."""
    if not isinstance(tree, nx.Graph):
        raise TypeError("tree must be a NetworkX graph.")
    if tree.number_of_nodes() == 0:
        raise ValueError("Persistence diagrams require a non-empty tree.")
    if not nx.is_tree(tree):
        raise ValueError("Persistence diagrams require a connected, acyclic tree.")
    root = tree.graph.get("root")
    if root not in tree:
        raise ValueError("tree.graph['root'] must name an existing root node.")

    positions: dict[object, np.ndarray] = {}
    for node in tree.nodes:
        if "pos" not in tree.nodes[node]:
            raise ValueError(f"Node {node!r} is missing the required 'pos' attribute.")
        position = np.asarray(tree.nodes[node]["pos"], dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"Node {node!r} must have a finite 3-D position.")
        positions[node] = position

    origin = positions[root]
    centered = tree.copy()
    for node, position in positions.items():
        centered.nodes[node]["pos"] = position - origin
    return centered


def compute_tmd_diagrams(
    tree: nx.Graph,
    *,
    normalize_mode: NormalizeMode,
    filtrations: Sequence[FiltrationName] = DEFAULT_FILTRATIONS,
    weight_edges_by_euclidean: bool = True,
    simplify_to_critical_tree: bool = True,
) -> dict[str, object]:
    """Compute one paper-style TMD diagram for each requested filtration.

    ``normalize_mode`` is required deliberately: normalization changes the
    scientific meaning and scale of the resulting distance and should never be
    selected implicitly by this wrapper. Coordinates are root-centered on a
    copy before any filtration is computed, so direct programmatic calls have
    the same translation behavior as SWCs loaded through the standard loader.
    """
    centered_tree = _root_centered_copy(tree)
    diagrams: dict[str, object] = {}
    for filtration in filtrations:
        _, diagram = compute_tmd_barcode_diagram(
            centered_tree,
            filtration=filtration,
            normalize_mode=normalize_mode,
            weight_edges_by_euclidean=weight_edges_by_euclidean,
            simplify_to_critical_tree=simplify_to_critical_tree,
        )
        diagrams[filtration] = diagram
    return diagrams


def tmd_persistence_distances(
    tree_a: nx.Graph,
    tree_b: nx.Graph,
    *,
    normalize_mode: NormalizeMode,
    filtrations: Sequence[FiltrationName] = DEFAULT_FILTRATIONS,
    order: float = 1,
    ground_norm: GroundNorm = "chebyshev",
    weight_edges_by_euclidean: bool = True,
    simplify_to_critical_tree: bool = True,
) -> dict[str, float]:
    """Return the TMD diagram distance for each requested filtration.

    The result keeps path length, height, and radial-XY (``rho``) distances
    separate so their different units and sensitivities remain visible.
    """
    diagrams_a = compute_tmd_diagrams(
        tree_a,
        normalize_mode=normalize_mode,
        filtrations=filtrations,
        weight_edges_by_euclidean=weight_edges_by_euclidean,
        simplify_to_critical_tree=simplify_to_critical_tree,
    )
    diagrams_b = compute_tmd_diagrams(
        tree_b,
        normalize_mode=normalize_mode,
        filtrations=filtrations,
        weight_edges_by_euclidean=weight_edges_by_euclidean,
        simplify_to_critical_tree=simplify_to_critical_tree,
    )

    return {
        filtration: persistence_diagram_wasserstein_distance(
            diagrams_a[filtration],
            diagrams_b[filtration],
            order=order,
            ground_norm=ground_norm,
        )
        for filtration in filtrations
    }


__all__ = [
    "DEFAULT_FILTRATIONS",
    "FiltrationName",
    "GroundNorm",
    "NormalizeMode",
    "compute_tmd_diagrams",
    "tmd_persistence_distances",
]
