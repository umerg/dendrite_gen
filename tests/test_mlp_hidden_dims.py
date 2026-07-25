"""Tests for the configurable per-layer MLP hidden widths.

Three widths that used to be hardcoded are now constructor args:
  * edge_mlp_hidden  -- SO2_EGNN.edge_mlp hidden (was 2*edge_input_dim, now feats_dim)
  * node_mlp_hidden  -- SO2_EGNN.node_mlp hidden (was 2*feats_dim, now feats_dim)
  * global_linear_attn_ff_hidden -- the ff_x/ff_q hidden in GlobalLinearAttention_Sparse
    (unchanged default: 4*dim, the standard transformer ratio)

edge_mlp is the largest parameter block in the model, so the default change from
2*edge_input_dim down to feats_dim is a deliberate architecture change: it makes the
message MLP contract the concatenated node pair (as in the original Satorras EGNN)
instead of expanding it. Checkpoints trained before this change therefore need the
legacy widths passed explicitly -- test_legacy_widths_reproduce_pre_change_shapes
pins that escape hatch.

Checks:
  * defaults are the contracted form; attention FFN default is still 4*dim
  * each knob moves only its own Linear pair
  * legacy widths reproduce the exact pre-change shapes
  * nn.Sequential indices (the checkpoint key strings) are unchanged
  * non-positive widths raise ValueError
  * a forward pass with narrowed MLPs runs and gives a finite loss
"""

from __future__ import annotations

import math
import random

import numpy as np
import networkx as nx
import pytest
import torch as th
from torch_geometric.data import Batch

import graph_generation as gg
from graph_generation.model.egnn_so2 import SO2_EGNN, GlobalLinearAttention_Sparse
from utils.data_loading import nx_graph_to_adj_pos

from tests.test_rbf_edge_features import _make_binary_tree, _first_egnn_layer

FEATS, MDIM = 4, 16
# feats_dim=4, edge_attr_dim=1, no embeddings, rbf_k=0:
#   edge_input_dim = 0 fourier + 1 edge_attr + (2 raw + 3 angles) + 2*4 = 14
EDGE_IN = 14


def _build_model(feats_dim=FEATS, m_dim=MDIM, n_layers=2, **kwargs):
    return gg.model.SO2_EGNN_Network(
        n_layers=n_layers, feats_dim=feats_dim, pos_dim=3, m_dim=m_dim, dropout=0.0,
        edge_attr_dim=1, **kwargs,
    )


def _first_attn_layer(model) -> GlobalLinearAttention_Sparse:
    for layer in model.mpnn_layers:
        if isinstance(layer, SO2_EGNN):
            continue
        for sub in layer:  # ModuleList([attn, egnn])
            if isinstance(sub, GlobalLinearAttention_Sparse):
                return sub
    raise AssertionError("no GlobalLinearAttention_Sparse layer found")


def test_defaults_are_contracted():
    layer = _first_egnn_layer(_build_model())
    assert layer.edge_input_dim == EDGE_IN
    # both MLP hiddens default to feats_dim, NOT 2*edge_input_dim / 2*feats_dim
    assert layer.edge_mlp_hidden == FEATS
    assert layer.node_mlp_hidden == FEATS
    assert layer.edge_mlp[0].in_features == EDGE_IN
    assert layer.edge_mlp[0].out_features == FEATS
    assert layer.edge_mlp[3].in_features == FEATS
    assert layer.edge_mlp[3].out_features == MDIM
    assert layer.node_mlp[0].in_features == FEATS + MDIM
    assert layer.node_mlp[0].out_features == FEATS
    assert layer.node_mlp[3].in_features == FEATS
    assert layer.node_mlp[3].out_features == FEATS


def test_attention_ff_default_is_4x():
    dim = 8
    attn = _first_attn_layer(_build_model(feats_dim=dim, global_linear_attn_every=1))
    assert attn.ff_hidden == dim * 4
    for ff in (attn.ff_x, attn.ff_q):
        assert ff[0].in_features == dim and ff[0].out_features == dim * 4
        assert ff[2].in_features == dim * 4 and ff[2].out_features == dim


def test_each_knob_moves_only_its_own_linears():
    layer = _first_egnn_layer(_build_model(edge_mlp_hidden=64))
    assert layer.edge_mlp[0].out_features == 64 and layer.edge_mlp[3].in_features == 64
    assert layer.edge_mlp[3].out_features == MDIM          # output still m_dim
    assert layer.node_mlp[0].out_features == FEATS         # untouched

    layer = _first_egnn_layer(_build_model(node_mlp_hidden=32))
    assert layer.node_mlp[0].out_features == 32 and layer.node_mlp[3].in_features == 32
    assert layer.node_mlp[3].out_features == FEATS         # output still feats_dim
    assert layer.edge_mlp[0].out_features == FEATS         # untouched

    dim = 8
    model = _build_model(feats_dim=dim, global_linear_attn_every=1,
                         global_linear_attn_ff_hidden=16)
    attn = _first_attn_layer(model)
    assert attn.ff_hidden == 16
    assert attn.ff_x[0].out_features == 16 and attn.ff_q[0].out_features == 16
    # the EGNN MLPs keep their own defaults
    assert _first_egnn_layer(model).edge_mlp_hidden == dim


def test_all_layers_agree():
    model = _build_model(n_layers=4, edge_mlp_hidden=48, node_mlp_hidden=24,
                         global_linear_attn_every=2)
    layers = list(model._iter_egnn_layers())
    assert len(layers) == 4
    for layer in layers:
        assert layer.edge_mlp_hidden == 48 and layer.node_mlp_hidden == 24
        assert layer.edge_mlp[0].out_features == 48
        assert layer.node_mlp[0].out_features == 24


def test_legacy_widths_reproduce_pre_change_shapes():
    """The escape hatch for checkpoints trained before the default changed."""
    layer = _first_egnn_layer(_build_model(
        edge_mlp_hidden=2 * EDGE_IN,   # legacy 2*edge_input_dim = 28
        node_mlp_hidden=2 * FEATS,     # legacy 2*feats_dim = 8
    ))
    assert tuple(layer.edge_mlp[0].weight.shape) == (28, 14)
    assert tuple(layer.edge_mlp[3].weight.shape) == (16, 28)
    assert tuple(layer.node_mlp[0].weight.shape) == (8, 20)
    assert tuple(layer.node_mlp[3].weight.shape) == (4, 8)
    legacy = sum(p.numel() for p in layer.edge_mlp.parameters()) \
        + sum(p.numel() for p in layer.node_mlp.parameters())
    assert legacy == 1088   # 392+28 + 448+16 + 160+8 + 32+4


def test_sequential_indices_unchanged():
    """edge_mlp.0/.3, node_mlp.0/.3 and ff_x.0/.2 are the checkpoint key strings."""
    model = _build_model(feats_dim=8, global_linear_attn_every=1)
    layer = _first_egnn_layer(model)
    assert isinstance(layer.edge_mlp[0], th.nn.Linear)
    assert isinstance(layer.edge_mlp[3], th.nn.Linear)
    assert isinstance(layer.node_mlp[0], th.nn.Linear)
    assert isinstance(layer.node_mlp[3], th.nn.Linear)
    attn = _first_attn_layer(model)
    assert isinstance(attn.ff_x[0], th.nn.Linear) and isinstance(attn.ff_x[2], th.nn.Linear)
    assert isinstance(attn.ff_q[0], th.nn.Linear) and isinstance(attn.ff_q[2], th.nn.Linear)
    keys = set(model.state_dict())
    for k in ("mpnn_layers.0.1.edge_mlp.0.weight", "mpnn_layers.0.1.edge_mlp.3.weight",
              "mpnn_layers.0.1.node_mlp.0.weight", "mpnn_layers.0.1.node_mlp.3.weight",
              "mpnn_layers.0.0.ff_x.0.weight", "mpnn_layers.0.0.ff_x.2.weight"):
        assert k in keys, f"missing checkpoint key {k}"


def test_changing_a_width_changes_only_shapes_not_keys():
    base = set(_build_model().state_dict())
    narrowed = set(_build_model(edge_mlp_hidden=64, node_mlp_hidden=32).state_dict())
    assert base == narrowed


@pytest.mark.parametrize("kwargs", [
    {"edge_mlp_hidden": 0}, {"edge_mlp_hidden": -1},
    {"node_mlp_hidden": 0}, {"node_mlp_hidden": -8},
    {"global_linear_attn_ff_hidden": 0},
])
def test_non_positive_widths_raise(kwargs):
    with pytest.raises(ValueError):
        _build_model(global_linear_attn_every=1, **kwargs)


def test_forward_pass_with_custom_widths():
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
    model = _build_model(feats_dim=feats_dim, edge_mlp_hidden=48, node_mlp_hidden=24,
                         global_linear_attn_every=2, global_linear_attn_ff_hidden=12)
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

    loss, metrics = method.get_loss(batch=batch, model=model)
    assert math.isfinite(float(loss.item()))

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
    assert out["rel_pred"].shape[0] == batch.num_nodes and out["rel_pred"].shape[1] == 3
    assert th.isfinite(out["rel_pred"]).all()


if __name__ == "__main__":
    test_defaults_are_contracted()
    test_attention_ff_default_is_4x()
    test_each_knob_moves_only_its_own_linears()
    test_all_layers_agree()
    test_legacy_widths_reproduce_pre_change_shapes()
    test_sequential_indices_unchanged()
    test_changing_a_width_changes_only_shapes_not_keys()
    test_forward_pass_with_custom_widths()
    print("MLP hidden-width tests passed.")
