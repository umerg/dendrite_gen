# datasets.py
from abc import ABC
import numpy as np
import scipy as sp
import torch as th
from torch.utils.data import IterableDataset
from torch_geometric.typing import SparseTensor

from .data import ReducedGraphData
from ..reduction import ReductionFactory  # your existing factory (extended to "cherries")

class RandRedDataset(IterableDataset, ABC):
    """
    Tree-focused random reduction dataset.
    Expects ReductionFactory to yield a stateful reducer (e.g., CherryReduction).
    No spectral features; only structural fields needed by the expansion model.
    """
    def __init__(self, adjs, poses, red_factory: ReductionFactory, tmds=None):
        super().__init__()
        self.red_factory = red_factory
        self.adjs = adjs  # list of scipy.sparse adjacency arrays (float64 okay)
        self.poses = poses  # list of np.ndarray, each of shape (n, 3) for 3D positions
        self.tmds = tmds  # list of np.ndarray, each of shape (D,) or (1, D)

    def _build_reduced_graph_data(self, graph, pos, forced_new_leaf_idx=None, tmd=None):
        """Helper: convert reducer state into ReducedGraphData."""
        if graph.leaf_idx is not None:
            leaf_idx = graph.leaf_idx
            leaf_mask = graph.leaf_mask
        else:
            leaf_idx = np.array(sorted(graph._state.leaves - {graph._state.root}), dtype=np.int64)
            leaf_mask = np.zeros(graph.n, dtype=bool)
            if len(leaf_idx) > 0:
                leaf_mask[leaf_idx] = True

        if graph.leaf_expansion is not None:
            leaf_expansion = graph.leaf_expansion
        else:
            leaf_expansion = np.ones_like(leaf_idx, dtype=np.int32)

        parent_idx = np.array([
            (graph._state.parent[u] if graph._state.parent[u] is not None else -1) for u in range(graph.n)
        ], dtype=np.int64)
        parent_idx_1b = parent_idx + 1

        if forced_new_leaf_idx is not None:
            new_leaf_idx = np.asarray(forced_new_leaf_idx, dtype=np.int64)
        else:
            new_leaf_idx = getattr(graph, "new_leaves_from_next", None)
            if new_leaf_idx is None:
                new_leaf_idx = np.empty(0, dtype=np.int64)
            new_leaf_idx = np.asarray(new_leaf_idx, dtype=np.int64)
        new_leaf_mask = np.zeros(graph.n, dtype=bool)
        if new_leaf_idx.size > 0:
            new_leaf_mask[new_leaf_idx] = True

        adj = graph.adj.astype(bool).astype(np.float32) if sp.sparse.issparse(graph.adj) else graph.adj
        if tmd is not None:
            tmd = np.asarray(tmd, dtype=np.float32)
            if tmd.ndim == 1:
                tmd = tmd[None, :]

        return ReducedGraphData(
            target_size=graph.n,
            reduction_level=graph.level,
            adj=adj,
            pos=pos,
            leaf_idx=leaf_idx,
            leaf_mask=leaf_mask,
            leaf_expansion=leaf_expansion,
            parent_idx_1b=parent_idx_1b,
            sibling_order=graph.sibling_order_array,
            total_tree_size=graph.total_nodes,
            new_leaf_idx_from_next=new_leaf_idx,
            new_leaf_mask_from_next=new_leaf_mask,
            tmd=tmd,
        )

    def get_random_reduction_sequence(self, graph, pos, rng, tmd=None):
        """
        Generate one full sequence of (fine -> coarse) steps
        until the reducer stops (n <= 1 or no cherries).
        """
        data = []

        pos = pos.astype(np.float32)
        while True:
            reduced_graph = graph.get_reduced_graph()  # use the reducer's internal RNG

            forced_new = None
            if not reduced_graph.did_contract:
                root = graph._state.root
                children = graph._state.children.get(root, []) if root is not None else []
                if children:
                    forced_new = np.array(children, dtype=np.int64)

            rgd = self._build_reduced_graph_data(graph, pos, forced_new_leaf_idx=forced_new, tmd=tmd)
            data.append(rgd)

            if not reduced_graph.did_contract:  # terminal: smallest graph already recorded
                break

            pos = pos[reduced_graph.survivor_mask]  # update positions to surviving nodes
            graph = reduced_graph  # advance to next level

        return data


class FiniteRandRedDataset(RandRedDataset):
    """
    Precompute K random reduction sequences per input graph.
    """
    def __init__(self, adjs, poses, red_factory: ReductionFactory, num_red_seqs: int, tmds=None):
        super().__init__(adjs, poses, red_factory, tmds=tmds)
        self.num_red_seqs = int(num_red_seqs)

        self.rng = np.random.default_rng(seed=0)
        self.graph_reduced_data = {i: [] for i in range(len(adjs))}

        for i, adj in enumerate(adjs):
            for _ in range(self.num_red_seqs):
                # NEW: fresh reducer per sequence
                graph = red_factory(adj, rng=self.rng)
                pos = self.poses[i]
                tmd = self.tmds[i] if self.tmds is not None else None
                seq = self.get_random_reduction_sequence(graph, pos, self.rng, tmd=tmd)
                if seq:  # guard in case tree is already terminal
                    self.graph_reduced_data[i].extend(seq)

    def __iter__(self):
        # uniform over precomputed steps
        while True:
            i = self.rng.integers(len(self.adjs))
            seq = self.graph_reduced_data[i]
            j = self.rng.integers(len(seq))
            yield seq[j]


class PrecomputedRedDataset(RandRedDataset):
    """
    Precompute all deterministic reduction sequences once.
    One epoch = one pass through all samples (N graphs × M_i levels each).
    Infinite iteration: reshuffles after each epoch.
    """
    def __init__(self, adjs, poses, red_factory: ReductionFactory, tmds=None):
        super().__init__(adjs, poses, red_factory, tmds=tmds)
        self.samples = []
        rng = np.random.default_rng(seed=0)

        for i, adj in enumerate(adjs):
            graph = red_factory(adj, rng=rng)
            pos = self.poses[i].copy()
            tmd = self.tmds[i] if self.tmds is not None else None
            seq = self.get_random_reduction_sequence(graph, pos, rng, tmd=tmd)
            self.samples.extend(seq)

        print(f"Precomputed {len(self.samples)} samples from {len(adjs)} graphs")

    def __iter__(self):
        rng = np.random.default_rng(seed=42)
        indices = np.arange(len(self.samples))
        while True:  # infinite iteration for step-based training
            rng.shuffle(indices)
            for i in indices:
                yield self.samples[int(i)]


class InfiniteRandRedDataset(RandRedDataset):
    """
    Infinite stream: cache one sampled sequence per graph, pop elements randomly;
    when empty, resample a new sequence from the current reducer state.
    """
    def __iter__(self):
        # NEW: keep raw adj list so we can reinit reducers
        base_adjs = [A.copy() for A in self.adjs]
        base_poses = [P.copy() for P in self.poses]
        base_tmds = list(self.tmds) if self.tmds is not None else None

        # worker-specific RNG
        worker_info = th.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = np.random.default_rng(worker_id)

        # warm cache with fresh reducers
        graphs = [self.red_factory(A, rng=rng) for A in base_adjs]

        graph_reduced_data = {}
        for i, g in enumerate(graphs):
            pos = base_poses[i]
            tmd = base_tmds[i] if base_tmds is not None else None
            graph_reduced_data[i] = self.get_random_reduction_sequence(g, pos, rng, tmd=tmd)

        while True:
            i = rng.integers(len(base_adjs))
            if not graph_reduced_data[i]:
                # NEW: reinit reducer and resample a full sequence
                graphs[i] = self.red_factory(base_adjs[i].copy(), rng=rng)
                pos = base_poses[i].copy()
                tmd = base_tmds[i] if base_tmds is not None else None
                seq = self.get_random_reduction_sequence(graphs[i], pos, rng, tmd=tmd)
                if not seq:
                    # Degenerate: nothing to reduce (e.g., single-node tree). Skip this i.
                    continue
                rng.shuffle(seq)
                graph_reduced_data[i] = seq

            yield graph_reduced_data[i].pop()

    @property
    def max_node_expansion(self):
        raise NotImplementedError  # not used in training; intentionally trimmed
