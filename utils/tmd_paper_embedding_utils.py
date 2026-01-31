"""
tmd_paper_embedding_utils.py

Implementation-oriented utilities for the **TMD** (Topological Morphology Descriptor)
pipeline described in *A Topological Representation of Branching Neuronal Morphologies*.

This module is designed to be a drop-in companion to `tmd_conditioning_utils.py`:

- Same expected NetworkX input format:
    - G is a tree (nx.is_tree(G) == True)
    - G.graph["root"] is the root node id
    - G.nodes[nid]["pos"] is (3,) xyz array
- Same filtration names ("path", "height", "rho") and normalization modes.

What this adds (paper-style TMD):
--------------------------------
1) **TMD barcode** computation on a rooted tree, derived from a scalar function f on nodes.
   The barcode is a multiset of intervals (b_i, d_i) produced by killing all-but-the-oldest
   sibling component at each branching event (older = maximal v, where v(node) is the max f
   over leaves in that subtree).

2) A **1D density profile embedding** of the barcode: a fixed-length vector h where
   h[k] is the number of barcode intervals covering bin-center x_k.

3) Optional: convert the barcode to a persistence-diagram-like representation and then
   reuse `persistence_image(...)` from `tmd_conditioning_utils.py` to get a fixed-size
   vector image embedding (unweighted / weighted).

Notes:
------
- The paper defines TMD on the *critical tree* (root, branch points, leaves), effectively
  ignoring degree-2 points along branches. We include an optional simplification step
  that contracts "chain" nodes (nodes with exactly one child, except the root).

- Unlike standard PH filtrations (where birth <= death), the scalar f used for morphology
  (e.g., radial distance) is not necessarily monotone along root-to-leaf paths. Therefore,
  some intervals can have "death < birth". For embeddings we canonicalize each interval by
  sorting endpoints (lo=min, hi=max).

Dependencies:
-------------
- networkx, numpy
- imports `tmd_conditioning_utils` for filtrations/normalization/persistence-image helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import networkx as nx

# Reuse your existing helpers for:
# - graph validation
# - filtrations
# - normalization
# - persistence image
try:
    # When imported as part of the utils package (e.g., from notebooks with repo root on sys.path)
    from utils.tmd_conditioning_utils import (  # type: ignore
        FiltrationName,
        PersistenceDiagram0D,
        assert_rooted_tree_graph,
        filtration_height_z,
        filtration_path_length_from_root,
        filtration_radial_rho,
        normalize_filtration_values,
        persistence_image,
    )
except ModuleNotFoundError:
    # Fallback for direct module execution/import when utils is already on sys.path
    from tmd_conditioning_utils import (  # type: ignore
        FiltrationName,
        PersistenceDiagram0D,
        assert_rooted_tree_graph,
        filtration_height_z,
        filtration_path_length_from_root,
        filtration_radial_rho,
        normalize_filtration_values,
        persistence_image,
    )

# -----------------------------
# Rooting + critical-tree utils
# -----------------------------

@dataclass(frozen=True)
class RootedTree:
    """Rooted representation extracted from an undirected NX tree."""
    root: int
    parent: Dict[int, int]         # node -> parent (root excluded)
    children: Dict[int, List[int]] # node -> children (possibly empty)


def root_undirected_tree(G: nx.Graph, root: int) -> RootedTree:
    """Compute parent/children maps by DFS from `root` on an undirected tree."""
    parent: Dict[int, int] = {}
    children: Dict[int, List[int]] = {n: [] for n in G.nodes}

    stack: List[int] = [root]
    order: List[int] = [root]
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
            order.append(v)

    # Tree sanity: all nodes should be reached
    if len(seen) != G.number_of_nodes():
        raise ValueError("Graph is a tree but not connected from the provided root?")

    return RootedTree(root=root, parent=parent, children=children)


def is_critical_node(node: int, rooted: RootedTree) -> bool:
    """Critical nodes = root, branching points (>=2 children), and leaves (0 children)."""
    if node == rooted.root:
        return True
    k = len(rooted.children[node])
    return (k == 0) or (k >= 2)


def build_critical_tree(rooted: RootedTree) -> RootedTree:
    """
    Contract degree-2 chain nodes (nodes with exactly one child), producing a rooted tree
    on critical nodes only.
    """
    root = rooted.root
    critical = {n for n in rooted.children.keys() if is_critical_node(n, rooted)}

    # parent / children for critical nodes
    c_parent: Dict[int, int] = {}
    c_children: Dict[int, List[int]] = {n: [] for n in critical}

    def next_critical(desc: int) -> int:
        """Walk down single-child chains until a critical node is reached."""
        cur = desc
        while cur not in critical:
            ch = rooted.children[cur]
            if len(ch) != 1:
                # Should not happen if critical set defined correctly
                break
            cur = ch[0]
        return cur

    # Traverse critical nodes and connect to next critical descendants
    stack = [root]
    seen = {root}
    while stack:
        u = stack.pop()
        for v in rooted.children[u]:
            w = next_critical(v)
            if w == u:
                continue
            c_parent[w] = u
            c_children[u].append(w)
            if w not in seen:
                seen.add(w)
                stack.append(w)

    # Ensure lists are deterministic (stable ordering)
    for u in c_children:
        c_children[u].sort(key=lambda x: int(x) if isinstance(x, (int, np.integer)) else str(x))

    return RootedTree(root=root, parent=c_parent, children=c_children)


# -----------------------------
# TMD barcode (paper algorithm)
# -----------------------------

def compute_tmd_barcode(
    rooted: RootedTree,
    f: Dict[int, float],
) -> np.ndarray:
    """
    Compute the TMD barcode for a rooted tree given a scalar function f on nodes.

    Returns:
        barcode: (M, 2) float array of (birth, death) pairs.

    Algorithm (binary + multi-ary):
        - For each leaf l: v(l) = f(l)
        - For each internal node p:
            v(p) = max_{child c of p} v(c)
            For every other child c != argmax: add interval (v(c), f(p))
        - Finally, add interval (v(root), f(root))

    Tie-breaking for argmax(v(c)):
        - Choose the child with maximum v; if ties, choose the smallest child id (stable).
    """
    root = rooted.root
    children = rooted.children

    barcode: List[Tuple[float, float]] = []

    def dfs(u: int) -> float:
        ch = children.get(u, [])
        if len(ch) == 0:
            # leaf
            return float(f[u])

        child_vs: List[Tuple[int, float]] = [(c, dfs(c)) for c in ch]

        # pick "oldest" child with maximal v; stable tie-break
        max_v = max(v for _c, v in child_vs)
        max_children = [c for c, v in child_vs if v == max_v]
        max_child = min(max_children, key=lambda x: int(x) if isinstance(x, (int, np.integer)) else str(x))

        # intervals for killed components
        for c, v_c in child_vs:
            if c != max_child:
                barcode.append((float(v_c), float(f[u])))

        # propagate v up
        return float(max_v)

    v_root = dfs(root)
    barcode.append((float(v_root), float(f[root])))

    return np.asarray(barcode, dtype=np.float64)


def barcode_to_diagram(barcode: np.ndarray) -> PersistenceDiagram0D:
    """
    Canonicalize barcode endpoints so birth <= death for downstream diagram/image use.
    """
    if barcode.size == 0:
        return PersistenceDiagram0D(births=np.zeros((0,), dtype=np.float64), deaths=np.zeros((0,), dtype=np.float64))
    b = np.minimum(barcode[:, 0], barcode[:, 1]).astype(np.float64)
    d = np.maximum(barcode[:, 0], barcode[:, 1]).astype(np.float64)
    return PersistenceDiagram0D(births=b, deaths=d)


# -----------------------------
# 1D density-profile embedding
# -----------------------------

def barcode_density_profile(
    barcode: np.ndarray,
    *,
    n_bins: int = 64,
    x_range: Tuple[float, float] = (0.0, 1.0),
    normalize: Literal["none", "max", "l1"] = "l1",
) -> np.ndarray:
    """
    Convert a barcode (M,2) into a 1D density profile vector h of length n_bins.

    We sample at bin centers x_k in [x_range[0], x_range[1]] and set:
        h(x_k) = #{ intervals [lo_i, hi_i] that contain x_k }

    This matches the "histogram / density profile" intuition in the paper.

    normalize:
        - "none": raw counts
        - "max": divide by max(h) (if >0)
        - "l1": divide by sum(h) (if >0)
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    x0, x1 = x_range
    if not (x1 > x0):
        raise ValueError("x_range must satisfy x_range[1] > x_range[0].")

    if barcode.size == 0:
        return np.zeros((n_bins,), dtype=np.float64)

    lo = np.minimum(barcode[:, 0], barcode[:, 1])
    hi = np.maximum(barcode[:, 0], barcode[:, 1])

    edges = np.linspace(x0, x1, n_bins + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # sweep-line diff array: O(M + n_bins)
    diff = np.zeros((n_bins + 1,), dtype=np.float64)

    # clip intervals to range for stability
    lo = np.clip(lo, x0, x1)
    hi = np.clip(hi, x0, x1)

    for a, b in zip(lo, hi):
        # bins whose centers are within [a,b]
        i0 = int(np.searchsorted(centers, a, side="left"))
        i1 = int(np.searchsorted(centers, b, side="right") - 1)
        if i1 < 0 or i0 >= n_bins:
            continue
        i0 = max(i0, 0)
        i1 = min(i1, n_bins - 1)
        diff[i0] += 1.0
        diff[i1 + 1] -= 1.0

    h = np.cumsum(diff[:-1])
    if normalize == "max":
        m = float(h.max())
        if m > 0:
            h = h / m
    elif normalize == "l1":
        s = float(h.sum())
        if s > 0:
            h = h / s
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalize={normalize!r}")
    return h


# -----------------------------
# Global embedding (drop-in API)
# -----------------------------

def compute_tmd_global_embedding_paper(
    G: nx.Graph,
    *,
    filtrations: Sequence[FiltrationName] = ("path", "height", "rho"),
    n_bins: int = 64,
    sigma: float = 0.05,
    normalize_mode: Literal["minmax", "max"] = "minmax",
    weighting: Literal["none", "persistence"] = "none",
    weight_edges_by_euclidean: bool = True,
    simplify_to_critical_tree: bool = True,
    embedding: Literal["density", "pi", "density+pi"] = "density",
    density_normalize: Literal["none", "max", "l1"] = "l1",
) -> np.ndarray:
    """
    Compute a concatenated global embedding using the *paper-style* TMD barcode.

    This is designed to accept the same core inputs as `compute_tmd_global_embedding` in
    `tmd_conditioning_utils.py` (filtrations, n_bins, sigma, normalize_mode, weighting,
    weight_edges_by_euclidean), while producing embeddings from the paper-style barcode.

    embedding:
        - "density": concatenated 1D density profiles, shape (len(filtrations)*n_bins,)
        - "pi": concatenated persistence images, shape (len(filtrations)*n_bins*n_bins,)
        - "density+pi": concatenation of both

    Notes:
        - sigma/weighting only affect the "pi" branch.
        - We normalize each filtration to ~[0,1] by default for comparability.
    """
    assert_rooted_tree_graph(G)
    root = G.graph["root"]

    rooted = root_undirected_tree(G, root)
    if simplify_to_critical_tree:
        rooted_use = build_critical_tree(rooted)
    else:
        rooted_use = rooted

    out_parts: List[np.ndarray] = []

    for name in filtrations:
        if name == "path":
            f_full = filtration_path_length_from_root(G, weight_edges_by_euclidean=weight_edges_by_euclidean)
        elif name == "height":
            f_full = filtration_height_z(G)
        elif name == "rho":
            f_full = filtration_radial_rho(G)
        else:
            raise ValueError(f"Unknown filtration name: {name!r}")

        f_full = normalize_filtration_values(f_full, mode=normalize_mode)
        # restrict f to the nodes in the rooted_use tree
        f = {nid: float(f_full[nid]) for nid in rooted_use.children.keys()}

        barcode = compute_tmd_barcode(rooted_use, f)

        if embedding in ("density", "density+pi"):
            h = barcode_density_profile(
                barcode,
                n_bins=n_bins,
                x_range=(0.0, 1.0),
                normalize=density_normalize,
            )
            out_parts.append(h.astype(np.float64))

        if embedding in ("pi", "density+pi"):
            diag = barcode_to_diagram(barcode)
            pi = persistence_image(
                diag,
                n_bins=n_bins,
                sigma=sigma,
                birth_range=(0.0, 1.0),
                pers_range=(0.0, 1.0),
                weighting=weighting,
            )
            out_parts.append(pi.astype(np.float64))

    return np.concatenate(out_parts, axis=0).astype(np.float32)
