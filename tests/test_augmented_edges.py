"""Tests for the augmented-edges variant (sibling + parent-anchored neighbour edges).

Covers the paradigm guarantees from AUGMENTED_EDGES_PARADIGM.md:
  * builder edge counts + directionality (neighbours are internal->leaf only;
    parent and actual siblings are excluded from the neighbour set);
  * the neighbour edge *set* never depends on a diffusing leaf's own position
    (the train/sample equivalence property — anchored on the fixed parent);
  * SO(2) invariance of all edge geometry (du, rho, cosphi, sinphi, costheta);
  * parent/child angle rows are byte-identical with vs without augmentation
    (zero regression to the proven branch-angle path), i.e. the bearing formula
    reproduces the existing psi for parent->child edges.
"""

from __future__ import annotations

import torch as th

from graph_generation.method.helpers import (
    build_augmented_edge_index,
    build_directed_edge_index,
    precompute_full_geometry,
)


# Tree:        0 (root)
#            / | \
#           1  2  3
#          / \
#         4   5            (4,5 are children of internal node 1)
PARENT = th.tensor([-1, 0, 0, 0, 1, 1], dtype=th.long)
POS = th.tensor(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.2],
        [0.0, 1.0, 0.3],
        [-1.0, -0.2, 0.1],
        [1.8, 0.4, 0.5],
        [1.4, -0.6, 0.4],
    ],
    dtype=th.float,
)
UHAT = th.tensor([0.0, 0.0, 1.0])
LEAF_IDX = th.tensor([4, 5], dtype=th.long)  # diffusing leaves this step


def _Rz(angle: float) -> th.Tensor:
    c, s = th.cos(th.tensor(angle)), th.sin(th.tensor(angle))
    return th.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _edge_set(edge_index, edge_types, t):
    return {
        (int(s), int(d))
        for s, d, et in zip(edge_index[0], edge_index[1], edge_types)
        if int(et) == t
    }


def test_builder_counts_and_directionality():
    ei, et = build_augmented_edge_index(
        PARENT, POS, LEAF_IDX, neighbour_k=12, neighbour_radius=1e9
    )
    types = sorted(set(et.tolist()))
    assert types == [0, 1, 2, 3]

    # siblings: root {1,2,3} -> 3*2=6 ; node1 {4,5} -> 2 ; total 8
    assert int((et == 2).sum()) == 8

    # neighbours are internal->leaf only; dst is always a diffusing leaf, src never is
    leaves = set(LEAF_IDX.tolist())
    nb = et == 3
    for s, d in zip(ei[0][nb].tolist(), ei[1][nb].tolist()):
        assert d in leaves and s not in leaves
        assert s != PARENT[d].item()          # parent excluded (it has its own edge)
        assert PARENT[s].item() != PARENT[d].item() or s not in leaves  # not a sibling


def test_neighbour_set_excludes_parent_and_siblings():
    ei, et = build_augmented_edge_index(
        PARENT, POS, LEAF_IDX, neighbour_k=12, neighbour_radius=1e9
    )
    nb = _edge_set(ei, et, 3)
    # leaves 4 and 5 share parent 1; candidates are non-diffusing {0,2,3} (1 is the parent)
    assert nb == {(0, 4), (2, 4), (3, 4), (0, 5), (2, 5), (3, 5)}


def test_neighbour_set_is_independent_of_leaf_position():
    """The leak-free guarantee: moving a diffusing leaf must not change the edge SET."""
    ei0, et0 = build_augmented_edge_index(
        PARENT, POS, LEAF_IDX, neighbour_k=12, neighbour_radius=1e9
    )
    pos_moved = POS.clone()
    pos_moved[4] += th.tensor([5.0, -3.0, 2.0])  # move a diffusing leaf far away
    pos_moved[5] += th.tensor([-4.0, 7.0, 1.0])
    ei1, et1 = build_augmented_edge_index(
        PARENT, pos_moved, LEAF_IDX, neighbour_k=12, neighbour_radius=1e9
    )
    for t in (0, 1, 2, 3):
        assert _edge_set(ei0, et0, t) == _edge_set(ei1, et1, t)


def test_neighbour_cap_respected():
    ei, et = build_augmented_edge_index(
        PARENT, POS, LEAF_IDX, neighbour_k=1, neighbour_radius=1e9
    )
    # with k=1, each of the 2 leaves gets exactly 1 neighbour
    assert int((et == 3).sum()) == 2


def test_so2_invariance_of_geometry():
    ei, et = build_augmented_edge_index(
        PARENT, POS, LEAF_IDX, neighbour_k=12, neighbour_radius=1e9
    )
    g0 = precompute_full_geometry(POS, PARENT, ei, UHAT, edge_types=et)
    gR = precompute_full_geometry(POS @ _Rz(0.9).T, PARENT, ei, UHAT, edge_types=et)
    for k in ("du", "rho", "cospsi_edge", "sinpsi_edge", "cos_theta_edge"):
        assert (g0[k] - gR[k]).abs().max().item() < 1e-5, k


def test_parent_child_rows_unchanged_by_augmentation():
    """Bearing fills only sibling/neighbour rows; parent/child rows reproduce existing psi."""
    ei_aug, et_aug = build_augmented_edge_index(
        PARENT, POS, LEAF_IDX, neighbour_k=12, neighbour_radius=1e9
    )
    g_aug = precompute_full_geometry(POS, PARENT, ei_aug, UHAT, edge_types=et_aug)

    ei_pc, _ = build_directed_edge_index(PARENT)
    g_pc = precompute_full_geometry(POS, PARENT, ei_pc, UHAT)  # no edge_types -> psi only

    aug_map = {
        (int(s), int(d)): i for i, (s, d) in enumerate(zip(ei_aug[0], ei_aug[1]))
    }
    for i in range(ei_pc.size(1)):
        s, d = int(ei_pc[0, i]), int(ei_pc[1, i])
        j = aug_map[(s, d)]
        for k in ("cospsi_edge", "sinpsi_edge", "cos_theta_edge"):
            assert (g_pc[k][i] - g_aug[k][j]).abs().item() < 1e-6, (k, s, d)


# --- End-to-end: full model + diffusion patch loop with augment_edges=True ----------

def _e2e_model_and_method(augment: bool):
    import graph_generation as gg
    from graph_generation.diffusion.basic import DenoisingDiffusionModel

    edge_kwargs = (
        dict(edge_embedding_nums=[4], edge_embedding_dims=[4])
        if augment
        else dict()
    )
    model = gg.model.SO2_EGNN_Network(
        n_layers=2, feats_dim=4, pos_dim=3, m_dim=16, dropout=0.0,
        edge_attr_dim=1, **edge_kwargs,
    )
    method = gg.method.Expansion(
        diffusion=DenoisingDiffusionModel(num_steps=2),
        red_threshold=0,
        augment_edges=augment,
        neighbour_k=8,
        neighbour_radius=1e9,
    )
    return model, method


def test_end_to_end_training_and_sampling_with_augment():
    """Augmented edges must flow through model + per-step geometry patch without error."""
    import random
    import numpy as np
    from torch.utils.data import DataLoader
    from torch_geometric.data import Batch
    from torch_geometric.utils import to_edge_index

    from test_forward_pass import _generate_graphs, _build_dataset, _make_minimal_cfg

    th.manual_seed(0); np.random.seed(0); random.seed(0)
    cfg = _make_minimal_cfg()
    graphs = _generate_graphs(num_graphs=8, n_min=30, n_max=60, seed=0)
    loader = _build_dataset(graphs, cfg)

    model, method = _e2e_model_and_method(augment=True)

    batch = next(iter(loader))
    ei_tmp, _ = to_edge_index(batch.adj)
    if ei_tmp.numel() == 0:
        batch = next(iter(loader))

    # Training: get_loss runs the augmented builder + precompute + patch + model.
    loss, metrics = method.get_loss(batch=batch, model=model)
    assert th.isfinite(loss), f"non-finite loss: {loss}"

    # Sampling: exercises expand()'s augmented builder across steps + diffusion patch loop.
    with th.no_grad():
        out_graphs = method.sample_graphs(
            target_size=th.tensor([8, 8], dtype=th.long), model=model,
        )
    assert len(out_graphs) == 2
    for g in out_graphs:
        assert g.number_of_nodes() >= 1
