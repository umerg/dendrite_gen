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
    def __init__(self, adjs, red_factory: ReductionFactory):
        super().__init__()
        self.red_factory = red_factory
        self.adjs = adjs  # list of scipy.sparse adjacency arrays (float64 okay)

    def get_random_reduction_sequence(self, graph, rng):
        """
        Generate one full sequence of (fine -> coarse) steps
        until the reducer stops (n <= 1 or no cherries).
        """
        data = []
        while True:
            reduced_graph = graph.get_reduced_graph(rng)  # must return same class with updated state

            # Stop if no reduction happened (terminal step)
            if reduced_graph.expansion_matrix is None:
                break

            # Pull fields for this step
            adj_fine = graph.adj
            adj_coarse = reduced_graph.adj
            P_inv = reduced_graph.expansion_matrix  # fine->coarse membership (n x m), binary

            # Compute node_expansion (size per coarse node)
            # scipy sparse: sum over rows -> col sums
            node_expansion = np.asarray(P_inv.sum(axis=0)).ravel().astype(np.int32)

            rgd = ReducedGraphData(
                target_size=graph.n,
                reduction_level=graph.level,
                adj=adj_fine.astype(bool).astype(np.float32) if sp.sparse.issparse(adj_fine) else adj_fine,
                node_expansion=node_expansion,
                adj_reduced=adj_coarse.astype(bool).astype(np.float32) if sp.sparse.issparse(adj_coarse) else adj_coarse,
                expansion_matrix=P_inv,
            )
            data.append(rgd)

            graph = reduced_graph  # advance to next level

        return data


class FiniteRandRedDataset(RandRedDataset):
    """
    Precompute K random reduction sequences per input graph.
    """
    def __init__(self, adjs, red_factory: ReductionFactory, num_red_seqs: int):
        super().__init__(adjs, red_factory)
        self.num_red_seqs = int(num_red_seqs)

        self.rng = np.random.default_rng(seed=0)
        self.graph_reduced_data = {i: [] for i in range(len(adjs))}

        for i, adj in enumerate(adjs):
            for _ in range(self.num_red_seqs):
                # NEW: fresh reducer per sequence
                graph = red_factory(adj)
                seq = self.get_random_reduction_sequence(graph, self.rng)
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

        # worker-specific RNG
        worker_info = th.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = np.random.default_rng(worker_id)

        # warm cache with fresh reducers
        graphs = [self.red_factory(A) for A in base_adjs]
        graph_reduced_data = {i: self.get_random_reduction_sequence(g, rng) for i, g in enumerate(graphs)}

        while True:
            i = rng.integers(len(base_adjs))
            if not graph_reduced_data[i]:
                # NEW: reinit reducer and resample a full sequence
                graphs[i] = self.red_factory(base_adjs[i].copy())
                seq = self.get_random_reduction_sequence(graphs[i], rng)
                if not seq:
                    # Degenerate: nothing to reduce (e.g., single-node tree). Skip this i.
                    continue
                rng.shuffle(seq)
                graph_reduced_data[i] = seq

            yield graph_reduced_data[i].pop()

    @property
    def max_node_expansion(self):
        raise NotImplementedError  # not used in training; intentionally trimmed
