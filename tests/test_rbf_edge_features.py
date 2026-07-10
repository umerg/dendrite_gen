"""Tests for the config-wired, dataset-adapted RBF edge-distance kernel.

The model expands the two SO(2)-invariant edge scalars (rho = in-plane radius,
du = axial component) into `rbf_k` Gaussian channels when rbf_k>0, over ranges
[0, rbf_rho_max] and [-rbf_du_max, rbf_du_max]. rbf_k=0 (default) keeps the raw
scalars, so existing checkpoints are unaffected.

Checks:
  * rbf_k=0 is unchanged: layers have no rbf_rho/rbf_du and edge_input_dim is the
    raw-scalar width.
  * rbf_k>0 grows edge_input_dim by exactly (rbf_k-1)*2 (rho + du each go 1 -> rbf_k).
  * The Gaussian centers honor the configured ranges (mu spans [0,rho_max] / [+-du_max]).
  * A forward pass with RBF on runs and returns rel_pred of shape [N,3] with finite loss.
"""

from __future__ import annotations

import math
import random

import numpy as np
import networkx as nx
import torch as th
from torch_geometric.data import Batch

import graph_generation as gg
from graph_generation.model.egnn_so2 import SO2_EGNN
from utils.data_loading import nx_graph_to_adj_pos


def _make_binary_tree(n: int, seed: int = 0) -> nx.Graph:
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_node(0, pos=np.zeros(3, dtype=np.float32))
    dirs = np.eye(3, dtype=np.float32)
    queue, nid = [0], 1
    while queue and nid < n:
        parent = queue.pop(0)
        for _ in range(rng.choice([1, 2])):
            if nid >= n:
                break
            step = dirs[rng.randint(0, 2)] * rng.uniform(0.5, 2.0)
            pos = G.nodes[parent]["pos"] + step + nrng.normal(0, 0.2, 3).astype(np.float32)
            G.add_node(nid, pos=pos)
            G.add_edge(parent, nid)
            queue.append(nid)
            nid += 1
    return G


def _build_model(rbf_k=0, rbf_rho_max=5.0, rbf_du_max=3.0, feats_dim=4, m_dim=16):
    return gg.model.SO2_EGNN_Network(
        n_layers=2, feats_dim=feats_dim, pos_dim=3, m_dim=m_dim, dropout=0.0,
        edge_attr_dim=1,
        rbf_k=rbf_k, rbf_rho_max=rbf_rho_max, rbf_du_max=rbf_du_max,
    )


def _first_egnn_layer(model) -> SO2_EGNN:
    for layer in model.mpnn_layers:
        if isinstance(layer, SO2_EGNN):
            return layer
        for sub in layer:  # ModuleList([attn, egnn]) on global-attn layers
            if isinstance(sub, SO2_EGNN):
                return sub
    raise AssertionError("no SO2_EGNN layer found")


def test_rbf_off_is_unchanged():
    layer = _first_egnn_layer(_build_model(rbf_k=0))
    assert layer.rbf_k == 0
    assert not hasattr(layer, "rbf_rho") and not hasattr(layer, "rbf_du")
    # raw-scalar edge width: base_scalar_dim = 1*2 (+3 angles)
    assert layer.edge_input_dim == 0 + 1 + (2 + 3) + 4 * 2  # fourier + edge_attr + base + feats*2


def test_rbf_grows_edge_input_dim():
    k = 16
    base = _first_egnn_layer(_build_model(rbf_k=0))
    on = _first_egnn_layer(_build_model(rbf_k=k))
    # rho and du each expand 1 -> k, so the width grows by exactly (k-1)*2.
    assert on.edge_input_dim - base.edge_input_dim == (k - 1) * 2


def test_rbf_centers_honor_configured_ranges():
    k, rho_max, du_max = 16, 4.0, 4.0
    layer = _first_egnn_layer(_build_model(rbf_k=k, rbf_rho_max=rho_max, rbf_du_max=du_max))
    assert layer.rbf_rho.mu.shape == (k,) and layer.rbf_du.mu.shape == (k,)
    assert float(layer.rbf_rho.mu.min()) == 0.0
    assert abs(float(layer.rbf_rho.mu.max()) - rho_max) < 1e-6
    assert abs(float(layer.rbf_du.mu.min()) + du_max) < 1e-6   # -du_max
    assert abs(float(layer.rbf_du.mu.max()) - du_max) < 1e-6


def test_forward_pass_with_rbf_on():
    seed = 7
    th.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    graphs = [_make_binary_tree(random.randint(30, 60), seed=s) for s in range(6)]
    adjs, poses = [], []
    for G in graphs:
        A, P, _ = nx_graph_to_adj_pos(G)
        adjs.append(A); poses.append(P)
    red_factory = gg.depth_reduction.DepthReductionFactory(
        mode="deterministic", cherry_p=1.0, ensure_progress=True, root=0, contract_root=False,
    )
    dataset = gg.data.PrecomputedRedDataset(adjs=adjs, poses=poses, red_factory=red_factory)

    feats_dim = 4
    model = _build_model(rbf_k=16, rbf_rho_max=4.0, rbf_du_max=4.0, feats_dim=feats_dim)
    from graph_generation.diffusion.basic import DenoisingDiffusionModel
    method = gg.method.Expansion(diffusion=DenoisingDiffusionModel(num_steps=1))

    from torch_geometric.utils import to_edge_index as _to_edge_index
    batch = None
    for i in range(0, len(dataset.samples), 2):
        cand = Batch.from_data_list(dataset.samples[i:i + 2])
        ei, _ = _to_edge_index(cand.adj)
        if ei.numel() > 0:
            batch = cand
            break
    assert batch is not None, "no non-degenerate batch found"

    # loss path (invokes model forward internally with RBF-expanded edges)
    loss, metrics = method.get_loss(batch=batch, model=model)
    assert math.isfinite(float(loss.item()))

    # explicit forward to check rel_pred shape (mirrors test_forward_pass)
    parent_idx = batch.parent_idx_1b - 1
    pos_gt = batch.pos
    from graph_generation.method.helpers import build_directed_edge_index
    edge_index, edge_types = build_directed_edge_index(
        parent_idx, edge_parent_to_child=0, edge_child_to_parent=1,
    )
    edge_attr = edge_types.unsqueeze(-1).to(pos_gt.dtype) if edge_types.numel() else pos_gt.new_zeros((0, 1))
    is_leaf = pos_gt.new_zeros((pos_gt.size(0), 1)); is_leaf[batch.leaf_idx] = 1.0
    extra = pos_gt.new_zeros((pos_gt.size(0), feats_dim - 1))
    x_in = th.cat([pos_gt, th.cat([is_leaf, extra], dim=-1)], dim=-1)
    out = model(x=x_in, edge_index=edge_index, batch=batch.batch, edge_attr=edge_attr, parent_idx=parent_idx)
    assert "rel_pred" in out
    assert out["rel_pred"].shape[0] == batch.num_nodes and out["rel_pred"].shape[1] == 3


if __name__ == "__main__":
    test_rbf_off_is_unchanged()
    test_rbf_grows_edge_input_dim()
    test_rbf_centers_honor_configured_ranges()
    test_forward_pass_with_rbf_on()
    print("RBF tests passed.")
