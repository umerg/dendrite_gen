"""Tests for cell-type (class) conditioning wired through the expansion pipeline.

Covers the two invariants that the conditioning depends on:
  1. `cell_class` is a per-graph field: PyG batches it to shape (B,) with the raw
     class ids preserved (NOT node-offset like leaf_idx / parent_idx_1b).
  2. The full training forward (ReducedGraphData -> Batch -> Expansion.get_loss ->
     diffusion.forward -> SO2_EGNN_Network.forward(cell_class=...)) runs end-to-end
     with class_hidden_dim > 0 and produces a finite loss; and it raises clearly when
     the model expects a class but the batch carries none.
"""

from __future__ import annotations

import math
import random

import numpy as np
import networkx as nx
import pytest
import torch as th
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

import graph_generation as gg
from graph_generation.data import ReducedGraphData
from graph_generation.diffusion.basic import DenoisingDiffusionModel
from utils.data_loading import nx_graph_to_adj_pos, CELL_CLASS_NAMES


def _make_binary_tree(n: int, seed: int) -> nx.Graph:
    """Small rooted binary tree (root = node 0) with 3D positions."""
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_node(0, pos=np.zeros(3, dtype=np.float32))
    G.graph["root"] = 0
    dirs = np.eye(3, dtype=np.float32)
    queue = [0]
    nid = 1
    while queue and nid < n:
        parent = queue.pop(0)
        for _ in range(2):
            if nid >= n:
                break
            step = dirs[rng.randint(0, 2)] * rng.uniform(0.5, 2.0)
            G.add_node(nid, pos=(G.nodes[parent]["pos"] + step).astype(np.float32))
            G.add_edge(parent, nid)
            queue.append(nid)
            nid += 1
    return G


def _depth_factory():
    return gg.depth_reduction.DepthReductionFactory(
        mode="deterministic", cherry_p=1.0, ensure_progress=True, root=0, contract_root=False,
    )


def _build_dataset(seeds, classes):
    adjs, poses = [], []
    for s in seeds:
        A, P, _ = nx_graph_to_adj_pos(_make_binary_tree(40, s))
        adjs.append(A)
        poses.append(P)
    return gg.data.PrecomputedRedDataset(
        adjs=adjs, poses=poses, classes=classes, red_factory=_depth_factory(),
    )


def test_cell_class_batches_to_B_not_offset():
    """cell_class must batch to (B,) preserving raw ids -- not incremented by node counts."""
    # Two graphs with distinct classes. PrecomputedRedDataset appends graph 0's whole
    # reduction sequence, then graph 1's -> samples[0] is graph 0, samples[-1] is graph 1.
    ds = _build_dataset(seeds=[1, 2], classes=[3, 5])

    b = Batch.from_data_list([ds.samples[0], ds.samples[-1]])
    assert hasattr(b, "cell_class"), "batch must carry cell_class"
    assert b.cell_class.dtype == th.long
    assert b.cell_class.dim() == 1 and b.cell_class.numel() == 2, "expected shape (B,)"
    # Raw ids preserved. If cell_class were in the __inc__ offset tuple, the second graph's
    # value would be 5 + num_nodes(first) != 5 -- this asserts it is NOT node-offset.
    assert b.cell_class.tolist() == [3, 5]


def test_cell_class_absent_when_unconditional():
    """No classes supplied -> no cell_class attribute on the emitted data (backward compatible)."""
    ds = _build_dataset(seeds=[7], classes=None)
    assert not hasattr(ds.samples[0], "cell_class")


def test_training_forward_with_cell_class():
    """Full teacher-forced loss runs with class_hidden_dim>0 and yields a finite loss."""
    th.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    seeds = list(range(6))
    classes = [i % len(CELL_CLASS_NAMES) for i in seeds]
    ds = _build_dataset(seeds=seeds, classes=classes)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=Batch.from_data_list)

    # feats_dim budget: avail = feats_dim - cond_dim(2) - class_hidden(8) = 22 >= MAX_CHILDREN+4=20
    model = gg.model.SO2_EGNN_Network(
        n_layers=2, feats_dim=32, pos_dim=3, m_dim=16, edge_attr_dim=1,
        num_classes=len(CELL_CLASS_NAMES), class_hidden_dim=8,
    )
    method = gg.method.Expansion(diffusion=DenoisingDiffusionModel(num_steps=1))

    batch = next(iter(loader))
    assert hasattr(batch, "cell_class")
    loss, metrics = method.get_loss(batch=batch, model=model)
    assert math.isfinite(float(loss.item())), "class-conditioned loss must be finite"


def test_teacher_forced_eval_fails_fast_under_class_conditioning():
    """TF eval is intentionally NOT wired for conditioning (its metrics are pooled, not
    class-stratified). build_reduction_batches_from_graphs does not thread cell_class, so a
    class-conditioned model must fail fast in the assembler rather than silently report
    non-stratified TF numbers."""
    from types import SimpleNamespace
    from validation.teacher_forced_eval import build_reduction_batches_from_graphs

    graphs = []
    for i, s in enumerate(range(5)):
        G = _make_binary_tree(40, s)
        G.graph["cell_class"] = i % len(CELL_CLASS_NAMES)  # labelled, but TF drops it on purpose
        graphs.append(G)

    red_cfg = SimpleNamespace(
        type="depth", mode="deterministic", cherry_p=1.0,
        ensure_progress=True, root=0, contract_root=False,
    )
    batches = build_reduction_batches_from_graphs(graphs, red_cfg, batch_size=64, pos_scale_factor=1.0)
    assert batches and not hasattr(batches[0], "cell_class"), "TF batch must NOT carry cell_class"

    model = gg.model.SO2_EGNN_Network(
        n_layers=2, feats_dim=32, pos_dim=3, m_dim=16, edge_attr_dim=1,
        num_classes=len(CELL_CLASS_NAMES), class_hidden_dim=8,
    )
    method = gg.method.Expansion(diffusion=DenoisingDiffusionModel(num_steps=1))
    with pytest.raises(ValueError, match="cell_class"):
        method.teacher_forced_sample(batches[0], model)


def test_missing_cell_class_raises_when_conditioning_on():
    """A class-conditioned model on an unlabelled batch must fail loudly, not silently."""
    ds = _build_dataset(seeds=[0, 1, 2], classes=None)  # no classes
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=Batch.from_data_list)
    model = gg.model.SO2_EGNN_Network(
        n_layers=2, feats_dim=32, pos_dim=3, m_dim=16, edge_attr_dim=1,
        num_classes=len(CELL_CLASS_NAMES), class_hidden_dim=8,
    )
    method = gg.method.Expansion(diffusion=DenoisingDiffusionModel(num_steps=1))
    batch = next(iter(loader))
    with pytest.raises(ValueError, match="cell_class"):
        method.get_loss(batch=batch, model=model)
