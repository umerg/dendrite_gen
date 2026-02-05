"""
Structural metrics for rooted tree graphs.

Assumes graphs are already "critical" skeletons (branch/termination points only),
so per-edge lengths correspond to branch segment lengths.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import math
import networkx as nx
import numpy as np
from persim import bottleneck as _bottleneck
try:
    from zss import distance as _zss_distance
except ModuleNotFoundError:  # optional dependency
    _zss_distance = None


def _pos_to_xyz(pos) -> np.ndarray:
    arr = np.asarray(pos, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size), mode="constant", constant_values=0.0)
    return arr[:3]


def mean_branch_length(G: nx.Graph) -> float:
    """
    Mean branch length computed as the mean Euclidean length of edges.

    Returns NaN if the graph has no edges.
    """
    if G.number_of_edges() == 0:
        return float("nan")
    lengths: List[float] = []
    for u, v in G.edges():
        pu = _pos_to_xyz(G.nodes[u].get("pos", np.zeros(3)))
        pv = _pos_to_xyz(G.nodes[v].get("pos", np.zeros(3)))
        lengths.append(float(np.linalg.norm(pu - pv)))
    return float(np.mean(lengths)) if lengths else float("nan")


def branch_length_values(G: nx.Graph) -> np.ndarray:
    """
    Per-edge Euclidean branch lengths.

    Returns an empty array if the graph has no edges.
    """
    if G.number_of_edges() == 0:
        return np.zeros((0,), dtype=np.float64)
    lengths: list[float] = []
    for u, v in G.edges():
        pu = _pos_to_xyz(G.nodes[u].get("pos", np.zeros(3)))
        pv = _pos_to_xyz(G.nodes[v].get("pos", np.zeros(3)))
        lengths.append(float(np.linalg.norm(pu - pv)))
    return np.asarray(lengths, dtype=np.float64)


def _root_tree(G: nx.Graph, root: int) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    """Return parent and children maps for an undirected tree rooted at root."""
    parent: Dict[int, int] = {}
    children: Dict[int, List[int]] = {n: [] for n in G.nodes}
    stack: List[int] = [root]
    seen = {root}
    while stack:
        u = stack.pop()
        for v in G.neighbors(u):
            if v in seen:
                continue
            seen.add(v)
            parent[v] = u
            children[u].append(v)
            stack.append(v)
    return parent, children


def mean_branch_amplitude(
    G: nx.Graph,
    *,
    root: int | None = None,
    degrees: bool = True,
    eps: float = 1e-12,
) -> float:
    """
    Mean sibling-branch angle at each bifurcation (parent) node.

    We compute, for each node with >=2 children, the mean of all pairwise
    angles between vectors (child_pos - parent_pos). Then average across
    such parent nodes.

    Returns NaN if no bifurcations are present.
    """
    if G.number_of_nodes() == 0:
        return float("nan")

    if root is None:
        root = G.graph.get("root")
    if root is None or root not in G.nodes:
        raise ValueError("Root node is required for branch amplitude computation.")

    _parent, children = _root_tree(G, root)
    node_means: List[float] = []

    for u, ch in children.items():
        if len(ch) < 2:
            continue
        pu = _pos_to_xyz(G.nodes[u].get("pos", np.zeros(3)))
        vecs: List[np.ndarray] = []
        for c in ch:
            pc = _pos_to_xyz(G.nodes[c].get("pos", np.zeros(3)))
            v = pc - pu
            if float(np.linalg.norm(v)) > eps:
                vecs.append(v)
        if len(vecs) < 2:
            continue

        angles: List[float] = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                v1 = vecs[i]
                v2 = vecs[j]
                denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
                if denom <= eps:
                    continue
                cos = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
                ang = float(math.acos(cos))
                if degrees:
                    ang = float(math.degrees(ang))
                angles.append(ang)
        if angles:
            node_means.append(float(np.mean(angles)))

    return float(np.mean(node_means)) if node_means else float("nan")


def bifurcation_angle_values(
    G: nx.Graph,
    *,
    root: int | None = None,
    degrees: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Collect all pairwise sibling-branch angles at bifurcation nodes.

    Returns an empty array if no bifurcations are present.
    """
    if G.number_of_nodes() == 0:
        return np.zeros((0,), dtype=np.float64)

    if root is None:
        root = G.graph.get("root")
    if root is None or root not in G.nodes:
        raise ValueError("Root node is required for bifurcation angle computation.")

    _parent, children = _root_tree(G, root)
    angles: list[float] = []

    for u, ch in children.items():
        if len(ch) < 2:
            continue
        pu = _pos_to_xyz(G.nodes[u].get("pos", np.zeros(3)))
        vecs: list[np.ndarray] = []
        for c in ch:
            pc = _pos_to_xyz(G.nodes[c].get("pos", np.zeros(3)))
            v = pc - pu
            if float(np.linalg.norm(v)) > eps:
                vecs.append(v)
        if len(vecs) < 2:
            continue

        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                v1 = vecs[i]
                v2 = vecs[j]
                denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
                if denom <= eps:
                    continue
                cos = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
                ang = float(math.acos(cos))
                if degrees:
                    ang = float(math.degrees(ang))
                angles.append(ang)

    return np.asarray(angles, dtype=np.float64)


def _diagram_pairs(diagram) -> np.ndarray:
    if diagram is None:
        return np.zeros((0, 2), dtype=np.float64)
    if hasattr(diagram, "as_pairs"):
        pairs = diagram.as_pairs()
        return np.asarray(pairs, dtype=np.float64).reshape(-1, 2) if pairs.size else np.zeros((0, 2), dtype=np.float64)
    arr = np.asarray(diagram, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return arr.reshape(-1, 2)


def _canonicalize_pairs(pairs: np.ndarray) -> np.ndarray:
    if pairs.size == 0:
        return pairs
    lo = np.minimum(pairs[:, 0], pairs[:, 1])
    hi = np.maximum(pairs[:, 0], pairs[:, 1])
    return np.stack([lo, hi], axis=1)


def bottleneck_distance(
    diagram_a,
    diagram_b,
    *,
    canonicalize: bool = False,
) -> float:
    """
    Bottleneck distance between two persistence diagrams.

    Accepts PersistenceDiagram-like objects (with .as_pairs) or array-like (M,2).
    If canonicalize=True, pairs are converted to (min, max).
    """
    dgm1 = _diagram_pairs(diagram_a)
    dgm2 = _diagram_pairs(diagram_b)
    if canonicalize:
        dgm1 = _canonicalize_pairs(dgm1)
        dgm2 = _canonicalize_pairs(dgm2)

    if dgm1.size == 0 and dgm2.size == 0:
        return 0.0
    if dgm1.size == 0:
        return float(np.max(np.abs(dgm2[:, 1] - dgm2[:, 0])) / 2.0)
    if dgm2.size == 0:
        return float(np.max(np.abs(dgm1[:, 1] - dgm1[:, 0])) / 2.0)

    return float(_bottleneck(dgm1, dgm2))


class _ZssNode:
    __slots__ = ("children",)

    def __init__(self) -> None:
        self.children: list["_ZssNode"] = []


def _resolve_root(G: nx.Graph, root: int | None) -> int:
    if root is None:
        root = G.graph.get("root")
    if root is None or root not in G.nodes:
        raise ValueError("Root node is required for tree edit distance.")
    return int(root)


def _build_zss_tree(
    G: nx.Graph,
    *,
    root: int | None,
    unordered: bool,
) -> _ZssNode:
    root = _resolve_root(G, root)
    _parent, children = _root_tree(G, root)

    sig_cache: dict[int, tuple[int, tuple]] = {}

    def _signature(u: int) -> tuple[int, tuple]:
        if u in sig_cache:
            return sig_cache[u]
        child_sigs = [_signature(c) for c in children[u]]
        child_sigs_sorted = tuple(sorted(child_sigs))
        size = 1 + sum(s[0] for s in child_sigs_sorted)
        sig = (size, child_sigs_sorted)
        sig_cache[u] = sig
        return sig

    if unordered:
        for u in children:
            _signature(u)

    def _build(u: int) -> _ZssNode:
        node = _ZssNode()
        child_ids = children[u]
        if unordered and child_ids:
            child_ids = sorted(child_ids, key=lambda c: sig_cache[c])
        for c in child_ids:
            node.children.append(_build(c))
        return node

    return _build(root)


def graph_edit_distance_topology(
    G1: nx.Graph,
    G2: nx.Graph,
    *,
    normalize: bool = False,
    normalization: str = "nodes_edges",
    timeout: float | None = None,
    unordered: bool = True,
    root1: int | None = None,
    root2: int | None = None,
) -> float | None:
    """
    Topology-only tree edit distance using zss (ordered TED).

    Substitution costs are zero (unlabeled topology), and insert/delete costs
    are 1 for nodes. zss is ordered; when unordered=True we canonicalize child
    order by subtree signatures (approximate for unordered TED).
    """
    if _zss_distance is None:
        raise ModuleNotFoundError(
            "zss is required for tree edit distance. Install with: pip install zss"
        )
    if timeout is not None:
        _ = timeout  # zss has no built-in timeout; kept for API compatibility

    t1 = _build_zss_tree(G1, root=root1, unordered=unordered)
    t2 = _build_zss_tree(G2, root=root2, unordered=unordered)
    ged_val = float(
        _zss_distance(
            t1,
            t2,
            get_children=lambda n: n.children,
            insert_cost=lambda _n: 1.0,
            remove_cost=lambda _n: 1.0,
            update_cost=lambda _a, _b: 0.0,
        )
    )
    if not normalize:
        return ged_val

    if normalization == "nodes":
        denom = max(G1.number_of_nodes(), G2.number_of_nodes())
    elif normalization == "edges":
        denom = max(G1.number_of_edges(), G2.number_of_edges())
    else:
        denom = max(
            G1.number_of_nodes() + G1.number_of_edges(),
            G2.number_of_nodes() + G2.number_of_edges(),
        )
    if denom <= 0:
        return 0.0

    return ged_val / float(denom)
