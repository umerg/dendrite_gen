"""Tests for the axial_extent root-child ordinal: the dataset flags the apical
(deepest -uhat subtree) root child, and it survives every reduction level."""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch as th

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import graph_generation as gg

UHAT = (0.0, 1.0, 0.0)


def _apical_tree():
    """Root 0 with 3 primary children; node 2's subtree reaches deepest along -y (apical).

    indices:
      0 root (0,0,0)
      1 basal  (1, 0.5, 0)     -> 6 (1.5,0.7,0), 7 (1.9,0.9,0)
      2 apical (0,-1,0)        -> 4 (0,-3,0) -> 5 (0,-6,0)   [deepest -y]
      3 basal  (-1,0.5,0)
    """
    edges = [(0, 1), (0, 2), (0, 3), (2, 4), (4, 5), (1, 6), (1, 7)]
    pos = np.array([
        [0, 0, 0], [1, 0.5, 0], [0, -1, 0], [-1, 0.5, 0],
        [0, -3, 0], [0, -6, 0], [1.5, 0.7, 0], [1.9, 0.9, 0],
    ], dtype=np.float32)
    n = pos.shape[0]
    A = sp.lil_matrix((n, n), dtype=np.float64)
    for a, b in edges:
        A[a, b] = 1.0
        A[b, a] = 1.0
    return A.tocsr(), pos


def _factory():
    return gg.depth_reduction.DepthReductionFactory(
        mode="deterministic", cherry_p=1.0, ensure_progress=True, root=0, contract_root=False,
    )


def test_axial_extent_flags_apical():
    A, pos = _apical_tree()
    ds = gg.data.PrecomputedRedDataset(
        adjs=[A], poses=[pos], red_factory=_factory(), tmds=None,
        uhat=UHAT, root_child_order="axial_extent",
    )
    assert len(ds.samples) > 0
    for s in ds.samples:
        flag = s.is_apical_root_child
        assert flag.dtype == th.bool
        assert int(flag.sum()) == 1, "exactly one apical flag must survive at every level"
        parent = s.parent_idx_1b - 1
        flagged = int(flag.nonzero(as_tuple=False).flatten()[0].item())
        assert int(parent[flagged].item()) == 0, "flagged node must be a root child"

    # On the finest level the apical is node 2 (deepest -y subtree).
    lvl0 = max(ds.samples, key=lambda s: int(s.target_size))
    assert int(lvl0.is_apical_root_child.nonzero().flatten()[0].item()) == 2


def test_first_edge_mode_threads_no_flag():
    """Legacy default: no is_apical_root_child attribute is attached."""
    A, pos = _apical_tree()
    ds = gg.data.PrecomputedRedDataset(
        adjs=[A], poses=[pos], red_factory=_factory(), tmds=None,
    )
    for s in ds.samples:
        assert getattr(s, "is_apical_root_child", None) is None
