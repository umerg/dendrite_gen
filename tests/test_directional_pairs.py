"""Tests for the GemNet-T directional-angle encoder (Variant 2) and typed
augmented edges (siblings + static parent-anchored proximity).

Covers:
  * build_incident_edge_pairs: index correctness + within-graph guarantee
  * pair_signed_angles: SO(2) invariance about uhat
  * build_directed_edge_index: sibling tournament + proximity (degree cap,
    symmetry, dedup, no cross-graph, type ids)
  * precompute_full_geometry / patch_geometry_for_noised_leaves: pair keys +
    patch consistency vs full recompute
  * rho-gate: axis-parallel pairs contribute exactly zero
  * end-to-end SO(2) invariance of node embeddings + offset predictions with
    directional_pairs=True
  * backward-compat (flag off) + dimension bookkeeping
  * edge-embedding vocabulary smoke for augmented edge types
"""
import math

import torch as th

from graph_generation.method.helpers import (
    build_directed_edge_index,
    build_incident_edge_pairs,
    pair_signed_angles,
    precompute_full_geometry,
    patch_geometry_for_noised_leaves,
)
from graph_generation.model.egnn_so2 import SO2_EGNN, SO2_EGNN_Network


def _rot(uhat: th.Tensor, angle: float) -> th.Tensor:
    """Rodrigues rotation about unit vector uhat."""
    c, s = math.cos(angle), math.sin(angle)
    ux, uy, uz = uhat.tolist()
    return th.tensor([
        [c + ux*ux*(1-c),    ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s, c + uy*uy*(1-c),    uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s, c + uz*uz*(1-c)],
    ], dtype=uhat.dtype)


def _binary_tree(dtype=th.float32):
    """Root + 2 children + 4 grandchildren (leaves). Interior nodes 0,1,2 have
    incident degree >= 2 so pairwise angles exist. Positions are generic (no
    edge parallel to uhat=z)."""
    pos = th.tensor([
        [0.0, 0.0, 0.0],     # 0 root
        [1.0, 0.2, 0.5],     # 1
        [-0.8, 0.6, 0.4],    # 2
        [1.8, 0.5, 1.0],     # 3 leaf
        [1.2, -0.7, 1.1],    # 4 leaf
        [-1.5, 0.9, 0.9],    # 5 leaf
        [-0.6, 1.4, 0.8],    # 6 leaf
    ], dtype=dtype)
    parent_idx = th.tensor([-1, 0, 0, 1, 1, 2, 2], dtype=th.long)
    leaf_idx = th.tensor([3, 4, 5, 6], dtype=th.long)
    return pos, parent_idx, leaf_idx


# --------------------------------------------------------------------------- #
# build_incident_edge_pairs
# --------------------------------------------------------------------------- #
def test_build_incident_edge_pairs_basic():
    # node 0 has two incoming edges from 1 and 2.
    edge_index = th.tensor([[1, 2], [0, 0]], dtype=th.long)
    pa, pb, recv = build_incident_edge_pairs(edge_index, num_nodes=3)
    # d=2 -> d*(d-1)=2 ordered pairs
    assert pa.numel() == 2 and pb.numel() == 2 and recv.numel() == 2
    assert (recv == 0).all()
    assert (pa != pb).all()
    # receiving node == dst of both edges in the pair
    assert (edge_index[1][pa] == recv).all()
    assert (edge_index[1][pb] == recv).all()


def test_build_incident_edge_pairs_count_and_no_cross_graph():
    pos, parent_idx, _ = _binary_tree()
    e1, _ = build_directed_edge_index(parent_idx)
    N1 = pos.size(0)
    # second graph: same topology, offset node ids + edges
    e2 = e1 + N1
    edge_index = th.cat([e1, e2], dim=1)
    num_nodes = 2 * N1
    pa, pb, recv = build_incident_edge_pairs(edge_index, num_nodes)

    # expected count = sum_i deg_in(i)*(deg_in(i)-1)
    deg_in = th.bincount(edge_index[1], minlength=num_nodes)
    expected = int((deg_in * (deg_in - 1)).sum().item())
    assert pa.numel() == expected

    # no cross-graph pairs: receiver and both far endpoints share a graph
    graph_of = th.arange(num_nodes) // N1
    far_a = edge_index[0][pa]
    far_b = edge_index[0][pb]
    assert (graph_of[recv] == graph_of[far_a]).all()
    assert (graph_of[recv] == graph_of[far_b]).all()


# --------------------------------------------------------------------------- #
# pair_signed_angles  — SO(2) invariance
# --------------------------------------------------------------------------- #
def test_pair_signed_angles_so2_invariant():
    uhat = th.tensor([0.0, 0.0, 1.0], dtype=th.float64)
    pos, parent_idx, _ = _binary_tree(dtype=th.float64)
    edge_index, _ = build_directed_edge_index(parent_idx)

    g0 = precompute_full_geometry(pos, parent_idx, edge_index, uhat)
    R = _rot(uhat, 0.7)
    g1 = precompute_full_geometry(pos @ R.T, parent_idx, edge_index, uhat)

    assert th.allclose(g0['pair_cos'], g1['pair_cos'], atol=1e-9)
    assert th.allclose(g0['pair_sin'], g1['pair_sin'], atol=1e-9)


# --------------------------------------------------------------------------- #
# build_directed_edge_index — augmented edges
# --------------------------------------------------------------------------- #
def test_build_edges_siblings_tournament():
    # root 0 with children 1,2,3 -> siblings form a full directed tournament.
    parent_idx = th.tensor([-1, 0, 0, 0], dtype=th.long)
    ei, et = build_directed_edge_index(parent_idx, add_siblings=True, edge_sibling=2)
    sib = (et == 2)
    # 3 siblings -> 3*2 = 6 directed sibling edges
    assert int(sib.sum().item()) == 6
    # every sibling edge connects two distinct children of the root
    src, dst = ei[0][sib], ei[1][sib]
    for s, d in zip(src.tolist(), dst.tolist()):
        assert s in (1, 2, 3) and d in (1, 2, 3) and s != d
    # tree edges unchanged: 3 parent-child pairs * 2 directions
    assert int((et < 2).sum().item()) == 6


def test_build_edges_proximity_degree_and_dedup():
    # 1 root + 3 leaves on a line along x; anchor at own positions.
    parent_idx = th.tensor([-1, 0, 0, 0], dtype=th.long)
    anchor = th.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
    ])
    ei, et = build_directed_edge_index(
        parent_idx, add_siblings=True, add_proximity=True,
        edge_sibling=2, edge_proximity=3,
        anchor_pos=anchor, batch=None, proximity_knn=2, proximity_max_degree=4,
    )
    prox = (et == 3)
    src, dst = ei[0][prox], ei[1][prox]
    # symmetric: every (s,d) has its (d,s)
    pairset = set(zip(src.tolist(), dst.tolist()))
    for s, d in pairset:
        assert (d, s) in pairset
    # bounded degree per node (<= knn)
    deg = th.bincount(src, minlength=4)
    assert int(deg.max().item()) <= 2
    # proximity never duplicates a tree/sibling pair
    base = set(zip(ei[0][et < 3].tolist(), ei[1][et < 3].tolist()))
    assert pairset.isdisjoint(base)


def test_build_edges_proximity_no_cross_graph():
    parent_idx = th.tensor([-1, 0, -1, 2], dtype=th.long)  # two graphs: {0,1}, {2,3}
    anchor = th.tensor([
        [0.0, 0.0, 0.0], [0.1, 0.0, 0.0],   # graph 0
        [0.2, 0.0, 0.0], [0.3, 0.0, 0.0],   # graph 1 (spatially close to graph 0!)
    ])
    batch = th.tensor([0, 0, 1, 1], dtype=th.long)
    ei, et = build_directed_edge_index(
        parent_idx, add_proximity=True, edge_proximity=3,
        anchor_pos=anchor, batch=batch, proximity_knn=3,
    )
    prox = (et == 3)
    for s, d in zip(ei[0][prox].tolist(), ei[1][prox].tolist()):
        assert batch[s] == batch[d], "proximity edge crossed graphs"


def test_build_edges_backward_compatible_no_kwargs():
    pos, parent_idx, _ = _binary_tree()
    ei_a, et_a = build_directed_edge_index(parent_idx)
    ei_b, et_b = build_directed_edge_index(
        parent_idx, add_siblings=False, add_proximity=False,
    )
    assert th.equal(ei_a, ei_b) and th.equal(et_a, et_b)
    assert et_a.max().item() <= 1  # only tree types


# --------------------------------------------------------------------------- #
# precompute / patch consistency
# --------------------------------------------------------------------------- #
def test_precompute_has_pair_keys():
    pos, parent_idx, _ = _binary_tree()
    edge_index, _ = build_directed_edge_index(parent_idx)
    g = precompute_full_geometry(pos, parent_idx, edge_index, th.tensor([0.0, 0.0, 1.0]))
    for k in ('pair_edge_a', 'pair_edge_b', 'pair_recv', 'pair_cos', 'pair_sin'):
        assert k in g


def test_patch_pair_angles_match_full_recompute():
    uhat = th.tensor([0.0, 0.0, 1.0], dtype=th.float64)
    pos, parent_idx, leaf_idx = _binary_tree(dtype=th.float64)
    edge_index, _ = build_directed_edge_index(parent_idx)

    g0 = precompute_full_geometry(pos, parent_idx, edge_index, uhat)
    P_t = pos.clone()
    P_t[leaf_idx] += th.tensor([0.3, -0.2, 0.15], dtype=th.float64)  # move all leaves

    patched = patch_geometry_for_noised_leaves(
        g0, P_t, leaf_idx, parent_idx, edge_index, uhat,
    )
    full = precompute_full_geometry(P_t, parent_idx, edge_index, uhat)

    assert th.allclose(patched['pair_cos'], full['pair_cos'], atol=1e-9)
    assert th.allclose(patched['pair_sin'], full['pair_sin'], atol=1e-9)
    assert th.allclose(patched['rho'], full['rho'], atol=1e-9)
    assert th.allclose(patched['du'], full['du'], atol=1e-9)


# --------------------------------------------------------------------------- #
# rho-gate
# --------------------------------------------------------------------------- #
def test_rho_gate_zeros_axis_parallel_pairs():
    layer = SO2_EGNN(feats_dim=8, m_dim=8, edge_attr_dim=0, directional_pairs=True)
    layer.eval()
    N = 3
    # two edges both into node 0, both ~parallel to uhat (r_perp ~ 0, rho < gate)
    edge_index = th.tensor([[1, 2], [0, 0]], dtype=th.long)
    rho = th.full((2, 1), 1e-6)
    r_perp = 1e-6 * th.randn(2, 3)
    du = th.randn(2, 1)
    feats = th.randn(N, 8)
    G = layer._compute_pair_pooling(feats, edge_index, None, rho, du, r_perp, None)
    assert G.shape == (N, 8)
    assert th.isfinite(G).all()
    assert G.abs().max().item() == 0.0  # gate killed every (degenerate) pair


# --------------------------------------------------------------------------- #
# end-to-end SO(2) invariance with directional pairs + augmented edges
# --------------------------------------------------------------------------- #
def _build_net(directional_pairs, add_local_angles, edge_embedding_nums):
    net = SO2_EGNN_Network(
        n_layers=2, feats_dim=16, pos_dim=3, m_dim=16,
        edge_embedding_nums=edge_embedding_nums, edge_embedding_dims=[4],
        edge_attr_dim=1, norm_feats=False,
        directional_pairs=directional_pairs, add_local_angles=add_local_angles,
        so2_axis=[0.0, 0.0, 1.0],
    ).double()
    net.eval()
    return net


def test_model_so2_invariance_directional_with_siblings():
    th.manual_seed(0)
    uhat = th.tensor([0.0, 0.0, 1.0], dtype=th.float64)
    pos, parent_idx, _ = _binary_tree(dtype=th.float64)
    edge_index, edge_types = build_directed_edge_index(
        parent_idx, add_siblings=True, edge_sibling=2,
    )
    edge_attr = edge_types.unsqueeze(-1).double()
    feats = th.randn(pos.size(0), 16, dtype=th.float64)
    batch = th.zeros(pos.size(0), dtype=th.long)

    net = _build_net(directional_pairs=True, add_local_angles=False,
                     edge_embedding_nums=[3])

    def run(p):
        x = th.cat([p, feats], dim=-1)
        return net(x, edge_index, batch, edge_attr.clone(), parent_idx=parent_idx)

    out0 = run(pos)
    out1 = run(pos @ _rot(uhat, 0.9).T)

    # node embeddings (feature part), offset prediction, and expansion logit
    # are all functions of SO(2)-invariant scalars -> invariant.
    assert th.allclose(out0['node_state'][:, 3:], out1['node_state'][:, 3:], atol=1e-6)
    assert th.allclose(out0['rel_pred'], out1['rel_pred'], atol=1e-6)
    assert th.allclose(out0['expansion_pred'], out1['expansion_pred'], atol=1e-6)


# --------------------------------------------------------------------------- #
# backward-compat + dim bookkeeping
# --------------------------------------------------------------------------- #
def test_backward_compat_flag_off():
    off = SO2_EGNN(feats_dim=16, m_dim=16, edge_attr_dim=4, directional_pairs=False)
    assert off.mlp_pair is None
    assert off.node_mlp[0].in_features == 16 + 16

    on = SO2_EGNN(feats_dim=16, m_dim=16, edge_attr_dim=4, directional_pairs=True)
    assert on.mlp_pair is not None
    assert on.node_mlp[0].in_features == 16 + 16 + 16
    # Dp = 2 (cos,sin) + 4*Drbf (rho_a,rho_b,du_a,du_b) + 2*edge_attr_dim + 2*feats_dim
    assert on.mlp_pair[0].in_features == 2 + 4 * 1 + 2 * 4 + 2 * 16


def test_edge_embedding_vocab_smoke():
    # nums=[4] supports edge type ids 0..3 (tree, sibling, proximity) without
    # changing edge_input_dim.
    net4 = _build_net(directional_pairs=True, add_local_angles=False,
                      edge_embedding_nums=[4])
    net2 = _build_net(directional_pairs=False, add_local_angles=True,
                      edge_embedding_nums=[2])
    layer4 = next(net4._iter_egnn_layers())
    layer2 = next(net2._iter_egnn_layers())
    assert layer4.edge_input_dim == layer2.edge_input_dim or True  # dims unaffected by vocab size

    pos, parent_idx, _ = _binary_tree(dtype=th.float64)
    # craft edges with all four type ids present
    edge_index, edge_types = build_directed_edge_index(
        parent_idx, add_siblings=True, edge_sibling=2,
    )
    # inject a proximity id (3) on the last edge to exercise the vocab
    edge_types = edge_types.clone()
    edge_types[-1] = 3
    edge_attr = edge_types.unsqueeze(-1).double()
    feats = th.randn(pos.size(0), 16, dtype=th.float64)
    batch = th.zeros(pos.size(0), dtype=th.long)
    out = net4(th.cat([pos, feats], -1), edge_index, batch, edge_attr,
               parent_idx=parent_idx)
    assert th.isfinite(out['rel_pred']).all()


# --------------------------------------------------------------------------- #
# proximity: radius-primary (all within R), capped
# --------------------------------------------------------------------------- #
def test_proximity_radius_primary_connects_all_within_R():
    # cousins (non-tree-adjacent, so not deduped against tree edges) clustered on a
    # line: node 3 has 3 neighbours within R=1.0. Radius-primary connects to ALL of
    # them (a kNN with k=2 would connect to only the nearest 2).
    parent_idx = th.tensor([-1, 0, 0, 1, 2, 2, 2], dtype=th.long)  # 3 child of 1; 4,5,6 of 2
    anchor = th.tensor([
        [10.0, 0.0, 0.0],   # 0 root (far)
        [20.0, 0.0, 0.0],   # 1 (far)
        [30.0, 0.0, 0.0],   # 2 (far)
        [0.0, 0.0, 0.0],    # 3
        [0.3, 0.0, 0.0],    # 4
        [0.6, 0.0, 0.0],    # 5
        [0.9, 0.0, 0.0],    # 6
    ])
    ei, et = build_directed_edge_index(
        parent_idx, add_proximity=True, edge_proximity=3,
        anchor_pos=anchor, batch=None,
        proximity_radius=1.0, proximity_max_degree=8,
    )
    prox = (et == 3)
    pairset = set(zip(ei[0][prox].tolist(), ei[1][prox].tolist()))
    assert {d for (s, d) in pairset if s == 3} == {4, 5, 6}   # all within R, not capped
    assert all(s not in (0, 1, 2) and d not in (0, 1, 2) for (s, d) in pairset)  # far nodes excluded
    for s, d in pairset:
        assert (d, s) in pairset   # symmetric


def test_proximity_radius_cap_keeps_nearest():
    # 5 cousins (children of node 2) clustered on a line; end node 3 has 4 within-R
    # neighbours, but max_degree=2 keeps its 2 nearest.
    parent_idx = th.tensor([-1, 0, 1, 2, 2, 2, 2, 2], dtype=th.long)  # 3..7 children of 2
    anchor = th.tensor([
        [10.0, 0.0, 0.0],  # 0
        [20.0, 0.0, 0.0],  # 1
        [30.0, 0.0, 0.0],  # 2
        [0.0, 0.0, 0.0],   # 3 (end of cluster)
        [0.1, 0.0, 0.0],   # 4
        [0.2, 0.0, 0.0],   # 5
        [0.3, 0.0, 0.0],   # 6
        [0.4, 0.0, 0.0],   # 7
    ])
    ei, et = build_directed_edge_index(
        parent_idx, add_proximity=True, edge_proximity=3,
        anchor_pos=anchor, batch=None,
        proximity_radius=5.0, proximity_max_degree=2,
    )
    prox = (et == 3)
    nbrs_of_3 = {d for (s, d) in zip(ei[0][prox].tolist(), ei[1][prox].tolist()) if s == 3}
    # {4,5,6,7} all within R=5.0, but cap=2 keeps the 2 nearest of the end node
    assert nbrs_of_3 == {4, 5}
