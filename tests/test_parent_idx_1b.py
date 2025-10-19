import sys
import pathlib
import numpy as np
import scipy.sparse as sp
import torch as th
from torch_geometric.loader import DataLoader

# Ensure repository root is on path when running tests directly
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph_generation.reduction import ReductionFactory
from graph_generation.data.reduction_dataset import FiniteRandRedDataset


def make_star_tree(n_leaves: int = 3):
    """Create adjacency for a star rooted at 0 with n_leaves leaves."""
    rows = []
    cols = []
    for i in range(1, n_leaves + 1):
        rows.extend([0, i])
        cols.extend([i, 0])
    data = np.ones(len(rows), dtype=np.float64)
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n_leaves + 1, n_leaves + 1))
    # random positions
    pos = np.random.randn(n_leaves + 1, 3).astype(np.float32)
    return adj, pos


def test_parent_idx_1b_single_graph():
    adj, pos = make_star_tree(3)
    factory = ReductionFactory(mode="deterministic", cherry_p=1.0, ensure_progress=True, root=0)
    ds = FiniteRandRedDataset([adj], [pos], factory, num_red_seqs=1)
    sample = next(iter(ds))

    assert hasattr(sample, "parent_idx_1b"), "parent_idx_1b not present in ReducedGraphData"
    parent_idx_1b = sample.parent_idx_1b
    N = sample.pos.size(0)
    assert parent_idx_1b.numel() == N
    # root should have 0 (1-based shift of -1 sentinel)
    assert int(parent_idx_1b[0].item()) == 0
    parent_idx = parent_idx_1b - 1  # restore -1 sentinel
    # children of root (1..n_leaves) should have parent 0
    for i in range(1, N):
        assert int(parent_idx[i].item()) == 0, f"Node {i} expected parent 0, got {parent_idx[i].item()}"
    # derive leaf_parent_idx dynamically and ensure valid
    leaf_parent_idx = parent_idx[sample.leaf_idx]
    assert (leaf_parent_idx >= 0).all(), "Leaf parent indices should be non-negative (root is not a leaf)."


def test_parent_idx_1b_batching():
    adj1, pos1 = make_star_tree(2)
    adj2, pos2 = make_star_tree(3)
    factory = ReductionFactory(mode="deterministic", cherry_p=1.0, ensure_progress=True, root=0)
    ds = FiniteRandRedDataset([adj1, adj2], [pos1, pos2], factory, num_red_seqs=1)
    loader = DataLoader(list(ds.graph_reduced_data[0]) + list(ds.graph_reduced_data[1]), batch_size=2)
    batch = next(iter(loader))
    # After batching, parent_idx_1b should still be length total nodes and >0 for non-root nodes.
    assert hasattr(batch, "parent_idx_1b")
    parent_idx = batch.parent_idx_1b - 1
    # Construct leaf_parent_idx dynamically; all should be valid indices within batch range
    leaf_parent_idx = parent_idx[batch.leaf_idx]
    assert (leaf_parent_idx >= 0).all(), "Batched leaf parent indices invalid (<0)."
    # Sanity: indexing positions with leaf_parent_idx works
    _ = batch.pos[leaf_parent_idx]
