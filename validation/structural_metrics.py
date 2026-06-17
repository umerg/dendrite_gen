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
try:
    from persim import bottleneck as _bottleneck
except ImportError:  # optional dependency (only needed for bottleneck_distance)
    _bottleneck = None
try:
    from zss import distance as _zss_distance
except ImportError:  # optional dependency (only needed for graph_edit_distance_topology)
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


# --- additional morphometrics (root-anchored geometry + topology) -------------------
#
# These reuse the shared helpers above (_root_tree, _pos_to_xyz) and the rooting
# conventions in visualization/stats (path/radial distance, branch order). All are
# nan/empty-safe: a missing/degenerate root yields an empty array (pooled metrics)
# or nan (per-tree scalars) rather than raising, so they can be pooled across a
# whole generated set without guarding each call.


def _resolve_root_or_none(G: nx.Graph, root: int | None) -> int | None:
    if root is None:
        root = G.graph.get("root")
    if root is None or root not in G.nodes:
        return None
    return int(root)


def _edge_length(G: nx.Graph, u: int, v: int) -> float:
    pu = _pos_to_xyz(G.nodes[u].get("pos", np.zeros(3)))
    pv = _pos_to_xyz(G.nodes[v].get("pos", np.zeros(3)))
    return float(np.linalg.norm(pu - pv))


def path_length_to_root_values(G: nx.Graph, *, root: int | None = None) -> np.ndarray:
    """Per-non-root-node path length from root along the tree (Euclidean-weighted)."""
    root = _resolve_root_or_none(G, root)
    if root is None or G.number_of_nodes() < 2:
        return np.zeros((0,), dtype=np.float64)
    path_map = nx.single_source_dijkstra_path_length(
        G, root, weight=lambda u, v, _d: _edge_length(G, u, v)
    )
    return np.asarray(
        [float(path_map[n]) for n in G.nodes() if n != root and n in path_map],
        dtype=np.float64,
    )


def radial_distance_to_root_values(G: nx.Graph, *, root: int | None = None) -> np.ndarray:
    """Per-non-root-node straight-line (Euclidean) distance from the root position."""
    root = _resolve_root_or_none(G, root)
    if root is None or G.number_of_nodes() < 2:
        return np.zeros((0,), dtype=np.float64)
    root_pos = _pos_to_xyz(G.nodes[root].get("pos", np.zeros(3)))
    return np.asarray(
        [
            float(np.linalg.norm(_pos_to_xyz(G.nodes[n].get("pos", np.zeros(3))) - root_pos))
            for n in G.nodes()
            if n != root
        ],
        dtype=np.float64,
    )


def contraction_ratio_values(
    G: nx.Graph, *, root: int | None = None, eps: float = 1e-12
) -> np.ndarray:
    """
    Per-leaf contraction = radial(root->leaf) / path(root->leaf), in (0, 1].

    A robust "tortuosity" surrogate for critical (branch-point-only) trees, where
    per-branch geometric tortuosity is ~1 by construction. 1 means a straight reach
    from the root; smaller means the dendrite wanders before terminating.
    """
    root = _resolve_root_or_none(G, root)
    if root is None or G.number_of_nodes() < 2:
        return np.zeros((0,), dtype=np.float64)
    _parent, children = _root_tree(G, root)
    leaves = [n for n, ch in children.items() if len(ch) == 0 and n != root]
    if not leaves:
        return np.zeros((0,), dtype=np.float64)
    path_map = nx.single_source_dijkstra_path_length(
        G, root, weight=lambda u, v, _d: _edge_length(G, u, v)
    )
    root_pos = _pos_to_xyz(G.nodes[root].get("pos", np.zeros(3)))
    out: list[float] = []
    for n in leaves:
        path = float(path_map.get(n, 0.0))
        if path <= eps:
            continue
        radial = float(np.linalg.norm(_pos_to_xyz(G.nodes[n].get("pos", np.zeros(3))) - root_pos))
        out.append(min(radial / path, 1.0))
    return np.asarray(out, dtype=np.float64)


def branch_order_values(G: nx.Graph, *, root: int | None = None) -> np.ndarray:
    """
    Per-non-root-node branch order (number of bifurcations on the root->node path).

    Matches the convention in visualization/stats/_graph.branch_order_map: order
    increments only when passing through a node of degree >= 3 (a true branch point).
    """
    root = _resolve_root_or_none(G, root)
    if root is None or G.number_of_nodes() < 2:
        return np.zeros((0,), dtype=np.float64)
    order = {root: 0}
    stack = [root]
    seen = {root}
    while stack:
        u = stack.pop()
        parent_order = order[u]
        for v in G.neighbors(u):
            if v in seen:
                continue
            seen.add(v)
            order[v] = parent_order + 1 if (G.degree(u) >= 3 and u != root) else parent_order
            stack.append(v)
    return np.asarray([float(order[n]) for n in G.nodes() if n != root], dtype=np.float64)


def _postorder_subtree_stats(
    G: nx.Graph, root: int
) -> tuple[dict[int, list[int]], dict[int, int], dict[int, int]]:
    """
    Bottom-up subtree leaf counts and Strahler numbers for a rooted tree.

    Returns (children_map, subtree_leaves, strahler). Processed in reverse
    descendants-after-ancestors order (iterative) to avoid recursion limits on
    deep path-like trees.
    """
    _parent, children = _root_tree(G, root)
    # Ancestors-before-descendants order via stack DFS; reverse gives valid post-order.
    pre: list[int] = []
    stack = [root]
    while stack:
        u = stack.pop()
        pre.append(u)
        for c in children[u]:
            stack.append(c)
    subtree_leaves: dict[int, int] = {}
    strahler: dict[int, int] = {}
    for u in reversed(pre):
        ch = children[u]
        if not ch:
            subtree_leaves[u] = 1
            strahler[u] = 1
            continue
        subtree_leaves[u] = sum(subtree_leaves[c] for c in ch)
        child_orders = [strahler[c] for c in ch]
        m = max(child_orders)
        strahler[u] = m + 1 if child_orders.count(m) >= 2 else m
    return children, subtree_leaves, strahler


def strahler_number(G: nx.Graph, *, root: int | None = None) -> float:
    """Horton-Strahler order of the whole rooted tree (per-tree scalar). nan if empty/unrooted."""
    root = _resolve_root_or_none(G, root)
    if root is None or G.number_of_nodes() == 0:
        return float("nan")
    _children, _leaves, strahler = _postorder_subtree_stats(G, root)
    return float(strahler[root])


def partition_asymmetry(
    G: nx.Graph, *, root: int | None = None, eps: float = 1e-12
) -> float:
    """
    Van Pelt tree asymmetry index: mean over branch points of the local partition
    asymmetry |r-s|/(r+s-2) of the subtree leaf counts (r,s); a partition with
    r+s==2 contributes 0. For multifurcations, averaged over all child pairs.

    Per-tree scalar in [0, 1]; nan if there is no qualifying branch point.
    """
    root = _resolve_root_or_none(G, root)
    if root is None or G.number_of_nodes() == 0:
        return float("nan")
    children, subtree_leaves, _strahler = _postorder_subtree_stats(G, root)
    node_vals: list[float] = []
    for u, ch in children.items():
        if len(ch) < 2:
            continue
        counts = [subtree_leaves[c] for c in ch]
        pair_vals: list[float] = []
        for i in range(len(counts)):
            for j in range(i + 1, len(counts)):
                r, s = counts[i], counts[j]
                denom = r + s - 2
                pair_vals.append(0.0 if denom <= 0 else abs(r - s) / float(denom))
        if pair_vals:
            node_vals.append(float(np.mean(pair_vals)))
    return float(np.mean(node_vals)) if node_vals else float("nan")


def sholl_intersection_profile(
    G: nx.Graph,
    *,
    root: int | None = None,
    radii: np.ndarray | None = None,
    n_shells: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sholl analysis: number of edges crossing each concentric sphere centred at the
    root. An edge (u,v) crosses radius r iff min(d_u,d_v) < r <= max(d_u,d_v), where
    d_* is the node's radial distance from the root.

    Returns (radii, counts). If ``radii`` is None, uses ``n_shells`` evenly spaced
    radii over (0, max radial extent]; pass shared ``radii`` (cached from GT) so the
    gen/GT/floor profiles are directly comparable. Empty arrays on degenerate trees.
    """
    root = _resolve_root_or_none(G, root)
    if root is None or G.number_of_edges() == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    root_pos = _pos_to_xyz(G.nodes[root].get("pos", np.zeros(3)))
    dist = {
        n: float(np.linalg.norm(_pos_to_xyz(G.nodes[n].get("pos", np.zeros(3))) - root_pos))
        for n in G.nodes()
    }
    if radii is None:
        max_r = max(dist.values()) if dist else 0.0
        if max_r <= 0.0:
            return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        radii = np.linspace(0.0, max_r, int(n_shells) + 1, dtype=np.float64)[1:]
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    counts = np.zeros(radii.shape, dtype=np.float64)
    for u, v in G.edges():
        lo, hi = sorted((dist[u], dist[v]))
        counts += ((radii > lo) & (radii <= hi)).astype(np.float64)
    return radii, counts


def sholl_summary(
    G: nx.Graph,
    *,
    root: int | None = None,
    radii: np.ndarray | None = None,
    n_shells: int = 32,
) -> dict[str, float]:
    """
    Reduce a Sholl profile to three per-tree scalars:
      - sholl_peak            : maximum intersection count
      - sholl_critical_radius : radius of the peak, normalised by max radial extent
      - sholl_auc             : area under the profile (trapezoid)
    nan-filled on degenerate trees.
    """
    out = {
        "sholl_peak": float("nan"),
        "sholl_critical_radius": float("nan"),
        "sholl_auc": float("nan"),
    }
    r, counts = sholl_intersection_profile(G, root=root, radii=radii, n_shells=n_shells)
    if r.size == 0 or counts.size == 0 or float(counts.max()) <= 0.0:
        return out
    peak_idx = int(np.argmax(counts))
    rmax = float(r.max())
    out["sholl_peak"] = float(counts.max())
    out["sholl_critical_radius"] = float(r[peak_idx] / rmax) if rmax > 0 else float("nan")
    _trapezoid = getattr(np, "trapezoid", np.trapz)
    out["sholl_auc"] = float(_trapezoid(counts, r))
    return out


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
    if _bottleneck is None:
        raise ModuleNotFoundError(
            "persim is required for bottleneck_distance. Install with: pip install persim"
        )
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
