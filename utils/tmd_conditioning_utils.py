"""
tmd_conditioning_utils.py

Utilities to compute *global* topological-morphological conditioning vectors (TMD-style)
for rooted tree graphs, using simple 0D persistence on a scalar filtration.

Designed for your workflow:
SWC -> cleaned rooted binary tree -> NetworkX graph -> PyG

Expected NetworkX format
------------------------
- G is a NetworkX *tree* (nx.is_tree(G) == True).
- Each node has attribute:  G.nodes[nid]["pos"]  = np.ndarray shape (3,) (x,y,z)
- Root node id stored at:   G.graph["root"]

What you get
------------
- 0D persistence diagram (birth, death) from a chosen filtration f(v)
- Persistence image embedding (fixed-size vector)
- A concatenated global embedding for 3 filtrations:
    (path length from root), (height z), (radial distance rho)
- Helper to attach embedding to graph and broadcast in PyG

No hard dependency on external TDA libs:
- If `tmd` is installed, you *can* optionally use it, but this file includes a fast
  union-find implementation for 0D persistence on graphs, which is sufficient for trees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import math
import numpy as np
import networkx as nx


# -----------------------------
# Validation / graph utilities
# -----------------------------

def assert_rooted_tree_graph(G: nx.Graph) -> None:
    """Raise a helpful error if G doesn't match the expected rooted-tree format."""
    if not nx.is_tree(G):
        raise ValueError("Expected G to be a tree (nx.is_tree(G) == True).")
    if "root" not in G.graph:
        raise ValueError('Expected G.graph["root"] to be set (root node id).')
    root = G.graph["root"]
    if root not in G.nodes:
        raise ValueError(f'G.graph["root"]={root!r} is not a node in the graph.')
    # Check node positions exist and have correct shape
    for nid in G.nodes:
        if "pos" not in G.nodes[nid]:
            raise ValueError(f'Node {nid!r} missing required attribute G.nodes[nid]["pos"].')
        pos = np.asarray(G.nodes[nid]["pos"])
        if pos.shape != (3,):
            raise ValueError(f'Node {nid!r} has pos with shape {pos.shape}; expected (3,).')


def get_node_positions(G: nx.Graph, nodelist: Optional[List[int]] = None) -> np.ndarray:
    """Return Nx3 float64 array of positions in the provided node order."""
    if nodelist is None:
        nodelist = list(G.nodes)
    xyz = np.stack([np.asarray(G.nodes[nid]["pos"], dtype=np.float64) for nid in nodelist], axis=0)
    return xyz


# -----------------------------
# Filtrations (scalar functions)
# -----------------------------

def filtration_path_length_from_root(G: nx.Graph, *, weight_edges_by_euclidean: bool = True) -> Dict[int, float]:
    """
    Filtration: path length from root along the tree.

    For botanical trees, weighting edges by Euclidean length is usually preferable.
    """
    assert_rooted_tree_graph(G)
    root = G.graph["root"]

    if not weight_edges_by_euclidean:
        # unweighted hop-distance
        lengths = nx.single_source_shortest_path_length(G, root)
        return {nid: float(d) for nid, d in lengths.items()}

    # Build per-edge weights = Euclidean distance between node positions.
    def edge_weight(u: int, v: int, _attr: dict) -> float:
        pu = np.asarray(G.nodes[u]["pos"], dtype=np.float64)
        pv = np.asarray(G.nodes[v]["pos"], dtype=np.float64)
        return float(np.linalg.norm(pu - pv))

    # Dijkstra on tree is cheap; for a tree it is O(N log N).
    lengths = nx.single_source_dijkstra_path_length(G, root, weight=edge_weight)
    return {nid: float(d) for nid, d in lengths.items()}


def filtration_height_z(G: nx.Graph) -> Dict[int, float]:
    """Filtration: node height (z coordinate)."""
    assert_rooted_tree_graph(G)
    return {nid: float(np.asarray(G.nodes[nid]["pos"], dtype=np.float64)[2]) for nid in G.nodes}


def filtration_radial_rho(G: nx.Graph) -> Dict[int, float]:
    """Filtration: radial distance rho = sqrt(x^2 + y^2)."""
    assert_rooted_tree_graph(G)
    out: Dict[int, float] = {}
    for nid in G.nodes:
        x, y, _z = np.asarray(G.nodes[nid]["pos"], dtype=np.float64)
        out[nid] = float(math.sqrt(x * x + y * y))
    return out


def normalize_filtration_values(
    f: Dict[int, float],
    *,
    mode: Literal["minmax", "max"] = "minmax",
    eps: float = 1e-12,
) -> Dict[int, float]:
    """
    Normalize filtration values into a consistent range.

    - minmax: map values to [0,1] using (v-min)/(max-min)
    - max: divide by max(|v|) (keeps sign; not recommended for z if negative values exist)

    For persistence images, minmax is generally simplest and stable.
    """
    vals = np.asarray(list(f.values()), dtype=np.float64)
    if vals.size == 0:
        return f

    if mode == "minmax":
        vmin = float(vals.min())
        vmax = float(vals.max())
        denom = max(vmax - vmin, eps)
        return {k: float((v - vmin) / denom) for k, v in f.items()}

    if mode == "max":
        denom = float(np.max(np.abs(vals)))
        denom = max(denom, eps)
        return {k: float(v / denom) for k, v in f.items()}

    raise ValueError(f"Unknown normalization mode: {mode!r}")


# -----------------------------
# 0D persistence on a graph (union-find)
# -----------------------------

@dataclass
class PersistenceDiagram0D:
    """
    Stores a 0D persistence diagram as arrays of birth and death times.

    births, deaths: shape (M,)
    """
    births: np.ndarray  # float64
    deaths: np.ndarray  # float64

    def as_pairs(self) -> np.ndarray:
        """Return (M,2) array of (birth, death)."""
        return np.stack([self.births, self.deaths], axis=1)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int64)
        # track "birth time" of the component representative
        self.birth = np.zeros(n, dtype=np.float64)

    def find(self, a: int) -> int:
        p = self.parent[a]
        if p != a:
            self.parent[a] = self.find(p)
        return self.parent[a]

    def union(self, a: int, b: int) -> Tuple[int, int]:
        """
        Union by rank. Returns (new_root, old_root) where old_root got attached.
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra, rb
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra, rb


def compute_0d_persistence_diagram(
    G: nx.Graph,
    f: Dict[int, float],
    *,
    include_infinite_bar: bool = False,
) -> PersistenceDiagram0D:
    """
    Compute 0D persistent homology of a graph under a lower-star filtration induced by vertex values f(v).

    Filtration rule:
    - vertex v appears at time f(v)
    - edge (u,v) appears at time max(f(u), f(v))

    For 0D persistence:
    - each vertex births a component at its appearance time
    - when an edge appears, it merges components; the component with *later birth*
      is killed at the edge time (elder rule: earlier birth survives)

    Returns only finite bars by default (recommended for persistence images).
    """
    assert_rooted_tree_graph(G)

    # fixed node order for union-find
    nodes = list(G.nodes)
    idx = {nid: i for i, nid in enumerate(nodes)}
    n = len(nodes)

    # vertex times
    vtime = np.asarray([float(f[nid]) for nid in nodes], dtype=np.float64)

    # edges with filtration time
    edges = []
    for u, v in G.edges:
        tu = float(f[u]); tv = float(f[v])
        edges.append((max(tu, tv), idx[u], idx[v]))
    edges.sort(key=lambda x: x[0])

    # process vertices in increasing time: but union-find can be initialized with births immediately
    uf = _UnionFind(n)
    uf.birth[:] = vtime

    # We'll record bars when merges happen.
    births: List[float] = []
    deaths: List[float] = []

    for t, iu, iv in edges:
        ru, rv = uf.find(iu), uf.find(iv)
        if ru == rv:
            continue

        # elder rule: earlier birth survives, later birth dies
        bu, bv = uf.birth[ru], uf.birth[rv]
        if bu <= bv:
            survivor, dead = ru, rv
            dead_birth = bv
        else:
            survivor, dead = rv, ru
            dead_birth = bu

        # union: ensure survivor becomes root
        # union() doesn't guarantee chosen root; so manually attach
        uf.parent[dead] = survivor
        births.append(float(dead_birth))
        deaths.append(float(t))

    if include_infinite_bar:
        # The surviving component persists to +inf. For images, usually exclude it.
        # If included, we set death=1.0 (assuming normalized filtration) to keep finite.
        root_rep = uf.find(0)
        births.append(float(uf.birth[root_rep]))
        deaths.append(float(np.inf))

    if len(births) == 0:
        return PersistenceDiagram0D(births=np.zeros((0,), dtype=np.float64),
                                    deaths=np.zeros((0,), dtype=np.float64))

    b = np.asarray(births, dtype=np.float64)
    d = np.asarray(deaths, dtype=np.float64)

    # keep only finite by default
    finite = np.isfinite(d)
    return PersistenceDiagram0D(births=b[finite], deaths=d[finite])


# -----------------------------
# Persistence image embedding
# -----------------------------

def persistence_image(
    diagram: PersistenceDiagram0D,
    *,
    n_bins: int = 16,
    sigma: float = 0.05,
    birth_range: Tuple[float, float] = (0.0, 1.0),
    pers_range: Tuple[float, float] = (0.0, 1.0),
    weighting: Literal["none", "persistence"] = "persistence",
) -> np.ndarray:
    """
    Convert a (birth, death) 0D diagram into a persistence image (n_bins x n_bins), returned flattened.

    - Uses coordinates (birth, persistence=death-birth)
    - Adds isotropic Gaussians with std=sigma (in the normalized filtration coordinate system)
    - weighting:
        - "none": all points weight 1
        - "persistence": weight = persistence (common default)
    """
    b = diagram.births
    d = diagram.deaths
    if b.size == 0:
        return np.zeros((n_bins * n_bins,), dtype=np.float32)

    p = d - b
    # remove non-positive persistence (numerical)
    keep = p > 1e-12
    b = b[keep]
    p = p[keep]
    if b.size == 0:
        return np.zeros((n_bins * n_bins,), dtype=np.float32)

    # grid centers
    b0, b1 = birth_range
    p0, p1 = pers_range
    bx = np.linspace(b0, b1, n_bins, dtype=np.float64)
    py = np.linspace(p0, p1, n_bins, dtype=np.float64)

    # precompute for vectorization
    B = bx[None, :, None]          # (1, n_bins, 1)
    P = py[None, None, :]          # (1, 1, n_bins)

    # points
    bp = b[:, None, None]          # (M,1,1)
    pp = p[:, None, None]          # (M,1,1)

    # weights
    if weighting == "none":
        w = np.ones((b.size,), dtype=np.float64)
    elif weighting == "persistence":
        w = p.astype(np.float64)
    else:
        raise ValueError(f"Unknown weighting: {weighting!r}")

    w = w[:, None, None]           # (M,1,1)

    # Gaussian contributions
    denom = 2.0 * (sigma ** 2)
    img = np.sum(w * np.exp(-((B - bp) ** 2 + (P - pp) ** 2) / denom), axis=0)  # (n_bins, n_bins)

    # flatten
    return img.astype(np.float32).reshape(-1)


# -----------------------------
# High-level API for your workflow
# -----------------------------

FiltrationName = Literal["path", "height", "rho"]


def compute_tmd_global_embedding(
    G: nx.Graph,
    *,
    filtrations: Sequence[FiltrationName] = ("path", "height", "rho"),
    n_bins: int = 16,
    sigma: float = 0.05,
    normalize_mode: Literal["minmax", "max"] = "minmax",
    weighting: Literal["none", "persistence"] = "persistence",
    weight_edges_by_euclidean: bool = True,
) -> np.ndarray:
    """
    Compute a concatenated global conditioning vector for a rooted tree graph.

    Returns:
        e: np.ndarray shape (len(filtrations) * n_bins * n_bins,), dtype float32

    Notes:
    - For stability across trees, we normalize each filtration to ~[0,1] by default.
    - Uses 0D persistence on the *graph* induced by vertex filtration values.
    """
    assert_rooted_tree_graph(G)

    emb_list: List[np.ndarray] = []
    for name in filtrations:
        if name == "path":
            f = filtration_path_length_from_root(G, weight_edges_by_euclidean=weight_edges_by_euclidean)
        elif name == "height":
            f = filtration_height_z(G)
        elif name == "rho":
            f = filtration_radial_rho(G)
        else:
            raise ValueError(f"Unknown filtration name: {name!r}")

        f = normalize_filtration_values(f, mode=normalize_mode)
        diag = compute_0d_persistence_diagram(G, f, include_infinite_bar=False)
        pi = persistence_image(
            diag,
            n_bins=n_bins,
            sigma=sigma,
            birth_range=(0.0, 1.0),
            pers_range=(0.0, 1.0),
            weighting=weighting,
        )
        emb_list.append(pi)

    e = np.concatenate(emb_list, axis=0).astype(np.float32)
    return e


def attach_tmd_global_to_nx_graph(
    G: nx.Graph,
    *,
    key: str = "tmd_global",
    **kwargs,
) -> np.ndarray:
    """
    Compute and store the global embedding in G.graph[key]. Returns the embedding.

    Example:
        e = attach_tmd_global_to_nx_graph(G, n_bins=16, sigma=0.05)
        # later in PyG conversion: e = G.graph["tmd_global"]
    """
    e = compute_tmd_global_embedding(G, **kwargs)
    G.graph[key] = e
    return e


def broadcast_global_to_node_features(
    x: np.ndarray,
    global_vec: np.ndarray,
) -> np.ndarray:
    """
    Broadcast a global vector to every node and concatenate to node features.

    Args:
        x: (N, F) node features (numpy)
        global_vec: (D,) global conditioning vector

    Returns:
        x_cat: (N, F + D)
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (N,F). Got shape={x.shape}")
    if global_vec.ndim != 1:
        raise ValueError(f"global_vec must be 1D (D,). Got shape={global_vec.shape}")
    N = x.shape[0]
    g = np.repeat(global_vec[None, :], repeats=N, axis=0)
    return np.concatenate([x, g], axis=1)


def torch_broadcast_and_concat(x, global_vec):
    """
    Torch version of broadcast_global_to_node_features for use in PyG.

    Args:
        x: torch.Tensor shape (N,F)
        global_vec: torch.Tensor shape (D,) or (1,D)

    Returns:
        x_cat: torch.Tensor shape (N, F+D)
    """
    import torch
    if global_vec.dim() == 1:
        global_vec = global_vec[None, :]
    if global_vec.size(0) != 1:
        raise ValueError("global_vec must be shape (D,) or (1,D) in torch_broadcast_and_concat.")
    N = x.size(0)
    g = global_vec.expand(N, -1)
    return torch.cat([x, g], dim=-1)
