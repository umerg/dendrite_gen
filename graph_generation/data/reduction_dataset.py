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
    def __init__(self, adjs, poses, red_factory: ReductionFactory):
        super().__init__()
        self.red_factory = red_factory
        self.adjs = adjs  # list of scipy.sparse adjacency arrays (float64 okay)
        self.poses = poses  # list of np.ndarray, each of shape (n, 3) for 3D positions

    def get_random_reduction_sequence(self, graph, pos, rng):
        """
        Generate one full sequence of (fine -> coarse) steps
        until the reducer stops (n <= 1 or no cherries).
        """
        data = []

        pos = pos.astype(np.float32)
        # G_0: initial graph; leaves from current state; labels = 1
        leaf0_idx = np.array(sorted(graph._state.leaves - {graph._state.root}), dtype=np.int64)
        leaf0_mask = np.zeros(graph.n, dtype=bool)
        if len(leaf0_idx) > 0:
            leaf0_mask[leaf0_idx] = True
        # parent indices for initial leaves
        # Build parent_idx_1b (length N): 1-based parent indices; roots become 0.
        parent_idx = np.array([
            (graph._state.parent[u] if graph._state.parent[u] is not None else -1) for u in range(graph.n)
        ], dtype=np.int64)
        parent_idx_1b = parent_idx + 1  # shift so root becomes 0
        rgd0 = ReducedGraphData(
            target_size=graph.n,
            reduction_level=graph.level,
            adj=graph.adj.astype(bool).astype(np.float32) if sp.sparse.issparse(graph.adj) else graph.adj,
            pos=pos,  # initial positions
            leaf_idx=leaf0_idx,
            leaf_mask=leaf0_mask,
            leaf_expansion=np.ones_like(leaf0_idx, dtype=np.int32),
            parent_idx_1b=parent_idx_1b,
        )
        data.append(rgd0)

        while True:
            reduced_graph = graph.get_reduced_graph(rng)  # must return same class with updated state

            # Stop if no reduction happened (terminal step)
            if not reduced_graph.did_contract: # this happens before root is added to the sequence
                # hence, ensures lowest graph in sequence is root + children
                break
            
            reduced_pos = pos[reduced_graph.survivor_mask]  # update positions to surviving nodes

            # Parent mapping for the reduced graph state
            parent_idx = np.array([
                (reduced_graph._state.parent[u] if reduced_graph._state.parent[u] is not None else -1)
                for u in range(reduced_graph.n)
            ], dtype=np.int64)
            parent_idx_1b = parent_idx + 1

            rgd = ReducedGraphData(
                target_size=reduced_graph.n,
                reduction_level=reduced_graph.level,
                adj=reduced_graph.adj.astype(bool).astype(np.float32)
                    if sp.sparse.issparse(reduced_graph.adj) else reduced_graph.adj,
                pos=reduced_pos,  # updated positions
                leaf_idx=reduced_graph.leaf_idx,
                leaf_mask=reduced_graph.leaf_mask,
                leaf_expansion=reduced_graph.leaf_expansion,  # {1,2}
                parent_idx_1b=parent_idx_1b,
            )
            data.append(rgd)

            graph = reduced_graph  # advance to next level

        return data


class FiniteRandRedDataset(RandRedDataset):
    """
    Precompute K random reduction sequences per input graph.
    """
    def __init__(self, adjs, poses, red_factory: ReductionFactory, num_red_seqs: int):
        super().__init__(adjs, poses, red_factory)
        self.num_red_seqs = int(num_red_seqs)

        self.rng = np.random.default_rng(seed=0)
        self.graph_reduced_data = {i: [] for i in range(len(adjs))}

        for i, adj in enumerate(adjs):
            for _ in range(self.num_red_seqs):
                # NEW: fresh reducer per sequence
                graph = red_factory(adj)
                pos = self.poses[i]
                seq = self.get_random_reduction_sequence(graph, pos, self.rng)
                if seq:  # guard in case tree is already terminal
                    self.graph_reduced_data[i].extend(seq)

    def __iter__(self):
        # uniform over precomputed steps
        while True:
            i = self.rng.integers(len(self.adjs))
            seq = self.graph_reduced_data[i]
            j = self.rng.integers(len(seq))
            yield seq[j]


class InfiniteRandRedDataset(RandRedDataset):
    """
    Infinite stream: cache one sampled sequence per graph, pop elements randomly;
    when empty, resample a new sequence from the current reducer state.
    """
    def __iter__(self):
        # NEW: keep raw adj list so we can reinit reducers
        base_adjs = [A.copy() for A in self.adjs]
        base_poses = [P.copy() for P in self.poses]

        # worker-specific RNG
        worker_info = th.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = np.random.default_rng(worker_id)

        # warm cache with fresh reducers
        graphs = [self.red_factory(A) for A in base_adjs]

        graph_reduced_data = {}
        for i, g in enumerate(graphs):
            pos = base_poses[i]
            graph_reduced_data[i] = self.get_random_reduction_sequence(g, pos, rng)

        while True:
            i = rng.integers(len(base_adjs))
            if not graph_reduced_data[i]:
                # NEW: reinit reducer and resample a full sequence
                graphs[i] = self.red_factory(base_adjs[i].copy())
                pos = base_poses[i].copy()
                seq = self.get_random_reduction_sequence(graphs[i], pos, rng)
                if not seq:
                    # Degenerate: nothing to reduce (e.g., single-node tree). Skip this i.
                    continue
                rng.shuffle(seq)
                graph_reduced_data[i] = seq

            yield graph_reduced_data[i].pop()

    @property
    def max_node_expansion(self):
        raise NotImplementedError  # not used in training; intentionally trimmed
