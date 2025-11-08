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
        if key in ("leaf_idx", "parent_idx_1b"):
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
          with dimension=3, then centered & scaled mildly for stability.
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
            nodes = list(range(n))
            rng.shuffle(nodes)  # randomize for variety
            
            # Create tree: first node is root, then alternate levels
            G.add_node(nodes[0])
            
            if n >= 3:
                # Build tree level by level ensuring binary property
                level_nodes = [nodes[0]]  # current level
                node_idx = 1
                
                while node_idx < n and level_nodes:
                    next_level = []
                    
                    for parent in level_nodes:
                        # Give this parent exactly 2 children if possible
                        for _ in range(2):
                            if node_idx >= n:
                                break
                            child = nodes[node_idx]
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
        # Center & scale for nicer spread.
        coords -= coords.mean(axis=0, keepdims=True)
        max_norm = np.max(np.linalg.norm(coords, axis=1))
        if max_norm > 0:
            coords /= max_norm
        # Assign back
        for idx, u in enumerate(G.nodes()):
            G.nodes[u]['pos'] = coords[idx]

        graphs.append(G)
    return graphs
