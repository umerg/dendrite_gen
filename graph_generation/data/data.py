# data.py
import numpy as np
import scipy as sp
import torch as th
import networkx as nx
from torch_geometric.data import Data
from torch_sparse import SparseTensor

class ReducedGraphData(Data):
    """
    Required fields:
      - adj:               adjacency of the current reduced graph (SparseTensor)
      - pos:               node positions (FloatTensor shape [N,3])
      - leaf_idx:          indices of leaf nodes in this graph (LongTensor shape [L])
      - leaf_mask:         boolean mask for leaf nodes (BoolTensor shape [N])
      - leaf_expansion:    labels for those leaves (LongTensor in {1,2}, shape [L])
      - parent_idx_1b:     parent index for every node, 1-based (LongTensor shape [N]; roots have 0).
                           Recover conventional parent indices (root=-1) via: parent_idx = parent_idx_1b - 1
      - reduction_level:   current level (int)
      - target_size:       n (node count of this graph), for bookkeeping
    Optional fields we now supply:
      - sibling_order:     LongTensor shape [N] with -1 for roots, else 0..k-1
      - total_tree_size:   scalar int giving the node count of the original tree
      - new_leaf_idx_from_next: LongTensor indices of nodes considered "new leaves" when expanding from next level
      - new_leaf_mask_from_next: BoolTensor mask aligned with nodes for the above
      - num_root_children: scalar int, branching factor of the root node (k)
      - cell_class: per-graph int64 cell-type label, shape (1,). A graph-level field
                    (constant across reduction levels) carried like `tmd`; it is
                    intentionally NOT in the __inc__ offset tuple below, so PyG
                    batching concatenates it to (B,) without node-index offsetting.
    """
    def __init__(self, **kwargs):
        super().__init__()
        if not kwargs:
            return

        # Use position matrix directly as x, EGNN expects positions as node features
        pos = kwargs.get("pos", None)
        if isinstance(pos, np.ndarray):
            x = th.from_numpy(pos).to(th.float32)
        elif isinstance(pos, th.Tensor):
            x = pos if pos.dtype.is_floating_point else pos.float()
        else:
            raise ValueError("pos must be a numpy array or torch tensor")
        super().__init__(x=x)

        for key, value in kwargs.items():
            if value is None:
                continue
            if isinstance(value, (int, np.integer)):
                value = th.tensor(int(value), dtype=th.long)
            elif isinstance(value, np.ndarray):
                # keep booleans as bool tensors, floats as float32, others as long
                if value.dtype == np.bool_:
                    value = th.from_numpy(value).to(th.bool)
                else:
                    value = th.from_numpy(value).to(th.float32 if value.dtype.kind == "f" else th.long)
            elif sp.sparse.issparse(value) or isinstance(value, sp.sparse.sparray):
                value = SparseTensor.from_scipy(value).to(
                    th.float32 if np.issubdtype(value.dtype, np.floating) else th.long
                )
            elif isinstance(value, th.Tensor) or isinstance(value, SparseTensor):
                pass
            else:
                raise ValueError(f"Unsupported type {type(value)} for key {key}")
            setattr(self, key, value)

    def __cat_dim__(self, key, value, *args, **kwargs):
        # Keep block-diagonal concatenation for sparse tensors
        if isinstance(value, SparseTensor):
            return (0, 1)
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        # Offset indices correctly when batching
        if key in ("leaf_idx", "parent_idx_1b", "new_leaf_idx_from_next"):
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


def generate_tree_graphs(
    num_graphs: int,
    min_size: int,
    max_size: int,
    seed: int | None = None,
) -> list[nx.Graph]:
    """Generate a list of random binary tree graphs with 3D positions.

    The returned graphs are plain ``networkx.Graph`` objects whose nodes each have
    a ``pos`` attribute: a length-3 ``numpy.ndarray`` of dtype ``float32``.
    This matches the geometric requirement enforced in ``Trainer.evaluate``.

    Binary tree constraint: Each internal node has exactly 2 children (degree 3),
    and leaf nodes have degree 1. Only the root has degree 2 if it has children.

    Args:
        num_graphs: Number of tree graphs to generate.
        min_size: Minimum number of nodes per tree (inclusive).
        max_size: Maximum number of nodes per tree (inclusive).
        seed: Optional RNG seed for reproducibility. If provided, generation is
            deterministic for the given (num_graphs, min_size, max_size, seed).

    Returns:
        A list of ``networkx.Graph`` objects. For each node ``u`` in each graph
        ``G``, ``G.nodes[u]['pos']`` is a 3D coordinate ``np.ndarray``.

    Notes:
        * Sizes are sampled uniformly from the integer range [min_size, max_size].
        * Binary tree topology: internal nodes have exactly 2 children.
                * 3D positions are assigned via a spring layout (``nx.spring_layout``)
                    with dimension=3; then translated so the root is at the origin and
                    scaled (divide by max node norm) for stability.
        * All graphs share a single master RNG so that calls are reproducible.
    """
    assert min_size > 0 and max_size >= min_size, "Invalid size bounds"
    rng = np.random.default_rng(seed)
    graphs: list[nx.Graph] = []
    
    for i in range(num_graphs):
        n = int(rng.integers(min_size, max_size + 1))
        
        # Generate binary tree using recursive splitting
        G = nx.Graph()
        
        if n == 1:
            G.add_node(0)
        else:
            # For strict binary tree: every internal node has exactly 2 children
            # This means we can only have certain tree sizes: 1, 3, 5, 7, 9, etc. (1 + 2k)
            # Adjust n to nearest valid size if needed
            if n % 2 == 0:
                n = n + 1  # Force odd number for full binary tree
                
            # Simple approach: create complete binary tree structure
            # Number internal nodes = (n-1)/2, number of leaves = (n+1)/2
            # Ensure node 0 is always the root; shuffle remaining nodes only
            root = 0
            ordered_nodes = [root] + list(range(1, n))
            rng.shuffle(ordered_nodes[1:])  # shuffle non-root nodes for variety

            # Create tree: root first, then alternate levels
            G.add_node(root)
            
            if n >= 3:
                # Build tree level by level ensuring binary property
                level_nodes = [root]  # current level contains only the root initially
                node_idx = 1
                
                while node_idx < n and level_nodes:
                    next_level = []
                    
                    for parent in level_nodes:
                        # Give this parent exactly 2 children if possible
                        for _ in range(2):
                            if node_idx >= n:
                                break
                            child = ordered_nodes[node_idx]
                            G.add_node(child)
                            G.add_edge(parent, child)
                            next_level.append(child)
                            node_idx += 1
                    
                    level_nodes = next_level

        # Spring layout in 3D
        layout_seed = int(rng.integers(0, 2**32 - 1))
        pos_dict = nx.spring_layout(G, dim=3, seed=layout_seed)
        # Convert to numpy arrays (float32) and (optionally) normalize.
        coords = np.vstack([pos_dict[u] for u in G.nodes()]).astype(np.float32)
        # Translate so root (node 0) is at origin, then scale by max norm.
        root_pos = coords[0].copy()
        coords -= root_pos
        max_norm = np.max(np.linalg.norm(coords, axis=1))
        if max_norm > 0:
            coords /= max_norm
        # Assign back
        for idx, u in enumerate(G.nodes()):
            G.nodes[u]['pos'] = coords[idx]

        graphs.append(G)
    return graphs


# ---------------------------------------------------------------------------
# Deterministic synthetic trees (zero conditional entropy)
# ---------------------------------------------------------------------------
# Built to isolate "can the framework learn at all" from "real neuron data is
# intrinsically stochastic". Each child's local-frame offset C_0 and each
# expansion decision is a DETERMINISTIC function of features the model observes
# (depth, child ordinal, local turn-history, lateral eccentricity), so the
# achievable ceiling is R^2 ~= 1 and ~100% expansion accuracy.
#
# Topology: root has 4 arms; every interior node is binary (0 or 2 children) --
# a hard framework constraint (the expansion label is {1,2} = spawn 0-or-2).
# Growth/symmetry axis is +y (uhat = [0,1,0]), matching the neuron runs.
#
# Per-tree variation comes from a single observable latent: the 4 arm
# base-depths d_r, drawn per tree and ENCODED in each root child's y-height.
# The 4 direct root children are therefore the only entropy source; everything
# below them is zero-entropy (and `is_root_child` masks them in diagnostics).

_Y_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float64)


def _proj_perp_y(v: np.ndarray) -> np.ndarray:
    """Component of v in the plane perpendicular to the y growth-axis."""
    return v - float(v @ _Y_AXIS) * _Y_AXIS


def _parent_frame(incoming: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local (forward, sideways) frame for a node's children.

    Matches the pipeline convention (helpers._compute_tree_directions /
    compute_local_bases): forward = normalize(proj_perp_y(parent - grandparent)),
    sideways = cross(y, forward). Placing children in this frame makes the
    recovered C_0 equal the injected (forward, sideways, axial) offset exactly.
    """
    f = _proj_perp_y(incoming)
    n = np.linalg.norm(f)
    if n < 1e-8:  # purely axial incoming edge -> degenerate; not hit by our placements
        f = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        f = f / n
    s = np.cross(_Y_AXIS, f)
    s = s / (np.linalg.norm(s) + 1e-12)
    return f, s


def _run_len(path: list[int], cap: int) -> int:
    """Length of the maximal run of identical turns ending at path[-1], capped."""
    if not path:
        return 0
    last = path[-1]
    r = 0
    for x in reversed(path):
        if x == last:
            r += 1
        else:
            break
    return min(r, cap)


def _switch_count(path: list[int], window: int) -> int:
    """Number of left/right direction changes over the last `window` ordinals."""
    seg = path[-window:] if window > 0 else path
    return sum(1 for i in range(1, len(seg)) if seg[i] != seg[i - 1])


def generate_deterministic_trees(
    num_graphs: int,
    seed: int | None = None,
    *,
    max_depth: int = 12,
    n_arms: int = 4,
    # per-tree latent: arm base-depths drawn from [d_lo, d_hi] (inclusive)
    d_lo: int = 4,
    d_hi: int = 9,
    min_T: int = 3,
    # --- length schedule:  base(d) = L0 * gamma^d * (1 + amp*sin(omega*d + phase_ord))
    L0: float = 1.0,
    gamma: float = 0.92,
    amp: float = 0.25,
    omega: float = 1.3,
    # --- eccentricity taper on axial:  axial = base(d) * (1 - lam*tanh(ecc/e0))
    lam: float = 0.4,
    e0: float = 3.0,
    # --- path-momentum splay:  sideways = sign(ord) * S0 * gamma^d * (1 + kappa*run)
    S0: float = 0.6,
    kappa: float = 0.35,
    # --- sibling asymmetry: forward persistence by ordinal (left != right)
    F_left: float = 0.7,
    F_right: float = 0.45,
    # --- root children: y-height encodes the arm latent d_r; fanned at fixed azimuths
    A0: float = 1.0,
    A_step: float = 0.35,
    R_root: float = 1.0,
    # --- expansion budget:  terminate iff depth >= clamp(d_r + run_w*run - switch_w*switches - floor(ecc/e1))
    e1: float = 1.0,
    run_w: int = 1,
    switch_w: int = 2,
    run_cap: int = 3,
    switch_window: int = 4,
) -> list[nx.Graph]:
    """Generate `num_graphs` deterministic binary trees with zero conditional entropy.

    The only randomness is the per-tree arm-depth latent ``d_r`` (which selects
    *which* deterministic tree, not any offset). Returns plain ``networkx.Graph``
    objects with a float32 ``pos`` attribute per node and ``G.graph['root']=0``.

    Each child additionally stores ``G.nodes[c]['c0_inject']`` -- the local-frame
    offset (forward, sideways, axial) that was injected -- so the pre-flight check
    can verify the pipeline recovers exactly this as ``C_0`` (root children excepted).
    """
    rng = np.random.default_rng(seed)
    graphs: list[nx.Graph] = []

    def grow(G, node, parent, depth, d_r, path, nid):
        """Decide whether `node` expands; if so place 2 children and recurse.

        Returns the next free node id.
        """
        run = _run_len(path, run_cap)
        switches = _switch_count(path, switch_window)
        ecc = float(np.linalg.norm(_proj_perp_y(G.nodes[node]["pos"].astype(np.float64))))
        T = d_r + run_w * run - switch_w * switches - int(ecc // e1)
        T = max(min_T, min(max_depth, T))
        if depth >= T:
            return nid  # leaf -> expansion label 1

        incoming = (
            G.nodes[node]["pos"].astype(np.float64)
            - G.nodes[parent]["pos"].astype(np.float64)
        )
        fwd_hat, side_hat = _parent_frame(incoming)
        node_pos = G.nodes[node]["pos"].astype(np.float64)
        dd = depth + 1  # child depth
        decay = gamma ** dd
        axial_scale = 1.0 - lam * np.tanh(ecc / e0)

        for ordn in (0, 1):
            cpath = path + [ordn]
            crun = _run_len(cpath, run_cap)
            phase = 0.0 if ordn == 0 else np.pi / 2.0
            base = L0 * decay * (1.0 + amp * np.sin(omega * dd + phase))
            axial = base * axial_scale
            sign = -1.0 if ordn == 0 else 1.0
            sideways = sign * S0 * decay * (1.0 + kappa * crun)
            forward = (F_left if ordn == 0 else F_right) * decay
            offset = forward * fwd_hat + sideways * side_hat + axial * _Y_AXIS
            cid = nid
            G.add_node(cid, pos=(node_pos + offset).astype(np.float32))
            G.nodes[cid]["c0_inject"] = np.array(
                [forward, sideways, axial], dtype=np.float64
            )
            G.add_edge(node, cid)
            nid = grow(G, cid, node, dd, d_r, cpath, cid + 1)
        return nid

    for _ in range(num_graphs):
        arm_depths = rng.integers(d_lo, d_hi + 1, size=n_arms)
        G = nx.Graph()
        G.add_node(0, pos=np.zeros(3, dtype=np.float32))
        G.graph["root"] = 0
        nid = 1
        root_children = []
        for r in range(n_arms):
            az = 2.0 * np.pi * r / n_arms  # 0, 90, 180, 270 deg in the xz-plane
            horiz = R_root * np.array([np.cos(az), 0.0, np.sin(az)], dtype=np.float64)
            height = (A0 + float(arm_depths[r]) * A_step) * _Y_AXIS  # encodes d_r
            cid = nid
            G.add_node(cid, pos=(horiz + height).astype(np.float32))
            G.add_edge(0, cid)
            root_children.append((cid, int(arm_depths[r])))
            nid += 1
        for cid, d_r in root_children:
            nid = grow(G, cid, 0, depth=1, d_r=d_r, path=[], nid=nid)
        graphs.append(G)

    return graphs
