"""Forward pass smoke test for SO2_EGNN_Network using
the random reduction dataset pipeline.

This test builds a small synthetic set of tree graphs (acting as dendrite
branch graphs), converts them to adjacency + positions, wraps them in the
InfiniteRandRedDataset, draws one batch, and runs a single forward + loss
computation through Expansion + DenoisingDiffusionModel.

Rationale:
  * Avoid external S3 + skeleton_plot dependencies for a lightweight test.
  * Exercise CherryReducer -> ReducedGraphData -> DataLoader -> Method.get_loss
    chain to ensure shapes & required fields line up for the EGNN.

Assertions:
  * Batch contains 3D positions as x/pos.
  * Model output dict has rel_pred of shape [N,3].
  * Loss is finite.

If you later integrate real skeletons, you can replace the synthetic graph
generator with your actual preprocessing and keep the remainder identical.
"""

from __future__ import annotations

import math
import random
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch as th
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

import graph_generation as gg  # root package (expects __init__.py already present)
from utils.data_loading import nx_graph_to_adj_pos  # canonical converter


def _make_random_binary_tree(n_min: int, n_max: int) -> nx.Graph:
    """Generate a random rooted binary tree with positions.
    Ensures each node has at most 2 children. Root is node 0.
    Positions: breadth-first growth with random directional steps.
    """
    n_target = random.randint(n_min, n_max)
    G = nx.Graph()
    G.add_node(0, pos=np.zeros(3, dtype=np.float32))
    dirs = np.eye(3, dtype=np.float32)
    queue = [0]
    next_id = 1
    while queue and next_id < n_target:
        parent = queue.pop(0)
        # choose 1 or 2 children (never 0 unless nearing n_target)
        remaining = n_target - next_id
        num_children = 1 if remaining == 1 else random.choice([1, 2])
        for _ in range(num_children):
            if next_id >= n_target:
                break
            step_dir = dirs[random.randint(0, 2)] * random.uniform(0.5, 2.0)
            jitter = np.random.randn(3).astype(np.float32) * 0.2
            pos = G.nodes[parent]["pos"] + step_dir + jitter
            G.add_node(next_id, pos=pos)
            G.add_edge(parent, next_id)
            queue.append(next_id)
            next_id += 1
    return G


def _generate_graphs(num_graphs: int, n_min: int = 40, n_max: int = 120, seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    graphs = [_make_random_binary_tree(n_min, n_max) for _ in range(num_graphs)]
    return graphs


def _build_dataset(graphs, cfg):
    """Convert list of networkx trees to (adj, pos) arrays and wrap in InfiniteRandRedDataset."""
    adjs = []
    poses = []
    for G in graphs:
        A, P, _ = nx_graph_to_adj_pos(G)  # returns CSR adjacency, (N,3) pos
        adjs.append(A)
        poses.append(P)

    red_factory = gg.reduction.ReductionFactory(
        mode=cfg.reduction.mode,
        cherry_p=cfg.reduction.cherry_p,
        ensure_progress=cfg.reduction.ensure_progress,
        root=cfg.reduction.root,
        contract_root=cfg.reduction.contract_root,  # ensure smallest graph retains root+children
    )

    dataset = gg.data.InfiniteRandRedDataset(adjs=adjs, poses=poses, red_factory=red_factory)

    # Infinite dataset => indicate with negative num_red_seqs (already in cfg)
    is_mp = cfg.reduction.num_red_seqs < 0
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        pin_memory=False,
        collate_fn=Batch.from_data_list,
        num_workers=min(0, cfg.training.max_num_workers) * is_mp,  # keep workers 0 for test stability
        multiprocessing_context="spawn" if is_mp and cfg.training.max_num_workers > 0 else None,
    )
    return loader


def _init_model_and_method(cfg):
    if cfg.model.name != "egnn":
        raise ValueError("Test currently only set up for cfg.model.name == 'egnn'.")
    model = gg.model.SO2_EGNN_Network(
        n_layers=cfg.model.num_layers,
        feats_dim=cfg.model.feats_dim,
        pos_dim=3,
        m_dim=cfg.model.m_dim,
        dropout=cfg.model.dropout,
        edge_attr_dim=1,  # Expansion builds directed edge types of dim 1
    )
    from graph_generation.diffusion.basic import DenoisingDiffusionModel
    diffusion = DenoisingDiffusionModel(num_steps=1)
    method = gg.method.Expansion(
        diffusion=diffusion,
        red_threshold=cfg.reduction.red_threshold,
    )
    return model, method


def _make_minimal_cfg():
    # Provide only fields that the forward pass pipeline touches.
    return SimpleNamespace(
        model=SimpleNamespace(name="egnn", num_layers=2, feats_dim=4, m_dim=16, dropout=0.0),
        reduction=SimpleNamespace(
            mode="stochastic",
            cherry_p=0.8,
            ensure_progress=True,
            root=0,  # force root to be node 0
            num_red_seqs=-1,  # infinite
            red_threshold=0,
            contract_root=False,
        ),
        training=SimpleNamespace(batch_size=2, max_num_workers=0),
        method=SimpleNamespace(
            deterministic_expansion=False,
        ),
    )


def test_forward_pass():
    """Single forward+loss smoke test."""
    seed = 123
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cfg = _make_minimal_cfg()

    # Build synthetic graphs & dataset
    graphs = _generate_graphs(num_graphs=10, n_min=30, n_max=60, seed=seed)
    loader = _build_dataset(graphs, cfg)

    model, method = _init_model_and_method(cfg)
    device = "cuda" if th.cuda.is_available() else "cpu"
    model = model.to(device)
    method = method.to(device)

    # Draw one batch; if degenerate (single node -> no edges), resample once
    batch = next(iter(loader))
    from torch_geometric.utils import to_edge_index as _to_edge_index
    ei_tmp, _ = _to_edge_index(batch.adj)
    if ei_tmp.numel() == 0:  # try one more sample before failing
        batch = next(iter(loader))
        ei_tmp, _ = _to_edge_index(batch.adj)
    assert ei_tmp.numel() > 0, "Degenerate batch with no edges after two tries; check reduction settings." 
    batch = batch.to(device)

    # Sanity checks on batch structure
    assert hasattr(batch, "adj"), "Batch must have adjacency (SparseTensor)."
    assert hasattr(batch, "leaf_idx"), "Batch must have leaf indices."
    assert hasattr(batch, "parent_idx_1b"), "Batch must include 1-based parent indices."
    assert batch.x.shape[1] == 3, "Expected position-only x for ReducedGraphData."  # current design

    # Run loss (invokes model forward internally)
    loss, metrics = method.get_loss(batch=batch, model=model)

    # Model produces per-node offsets in metrics via Expansion expectations
    # Re-run just the forward part explicitly to inspect shapes
    parent_idx = batch.parent_idx_1b - 1
    pos_gt = batch.pos

    from graph_generation.method.helpers import build_directed_edge_index
    edge_index, edge_types = build_directed_edge_index(
        parent_idx,
        edge_parent_to_child=0,
        edge_child_to_parent=1,
    )
    edge_attr = edge_types.unsqueeze(-1).to(pos_gt.dtype) if edge_types.numel() else pos_gt.new_zeros((0, 1))

    # Minimal feature construction (matches method.get_loss logic)
    is_leaf = pos_gt.new_zeros((pos_gt.size(0), 1))
    is_leaf[batch.leaf_idx] = 1.0
    extra = pos_gt.new_zeros((pos_gt.size(0), cfg.model.feats_dim - 1)) if cfg.model.feats_dim > 1 else None
    node_feats = th.cat([is_leaf, extra], dim=-1) if extra is not None else is_leaf
    x_in = th.cat([pos_gt, node_feats], dim=-1)
    out = model(x=x_in, edge_index=edge_index, batch=batch.batch, edge_attr=edge_attr, parent_idx=parent_idx)
    assert isinstance(out, dict) and "rel_pred" in out, "Model forward must return dict with 'rel_pred'."
    rel_pred = out["rel_pred"]
    assert rel_pred.shape[0] == batch.num_nodes and rel_pred.shape[1] == 3, "rel_pred must be [N,3]."

    # Basic numerical sanity
    assert math.isfinite(float(loss.item())), "Loss must be finite."
    assert metrics["num_leaves"] == batch.leaf_idx.numel()  # consistency

    # (Optional) Print a quick summary to help debugging when running pytest -s
    print(f"Forward pass OK | N={batch.num_nodes} | leaves={batch.leaf_idx.numel()} | loss={loss.item():.4f}")
    print(f"rel_pred mean norm: {rel_pred.norm(dim=-1).mean().item():.4f}")


def test_precomputed_dataset():
    """Verify PrecomputedRedDataset precomputes all samples at init and yields valid batches."""
    seed = 456
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    graphs = _generate_graphs(num_graphs=4, n_min=20, n_max=40, seed=seed)
    adjs, poses = [], []
    for G in graphs:
        A, P, _ = nx_graph_to_adj_pos(G)
        adjs.append(A)
        poses.append(P)

    red_factory = gg.depth_reduction.DepthReductionFactory(
        mode="deterministic",
        cherry_p=1.0,
        ensure_progress=True,
        root=0,
        contract_root=False,
    )

    dataset = gg.data.PrecomputedRedDataset(adjs=adjs, poses=poses, red_factory=red_factory)

    # 1) All samples precomputed at init
    assert len(dataset.samples) > 0, "PrecomputedRedDataset should have precomputed samples"
    print(f"Precomputed {len(dataset.samples)} samples from {len(adjs)} graphs")

    # 2) Each sample is a ReducedGraphData with expected fields
    from graph_generation.data import ReducedGraphData
    for i, s in enumerate(dataset.samples[:5]):
        assert isinstance(s, ReducedGraphData), f"Sample {i} is not ReducedGraphData"
        assert hasattr(s, "adj"), f"Sample {i} missing adj"
        assert hasattr(s, "pos"), f"Sample {i} missing pos"
        assert hasattr(s, "leaf_idx"), f"Sample {i} missing leaf_idx"
        assert hasattr(s, "parent_idx_1b"), f"Sample {i} missing parent_idx_1b"

    # 3) Deterministic: rebuilding gives same sample count
    dataset2 = gg.data.PrecomputedRedDataset(adjs=adjs, poses=poses, red_factory=red_factory)
    assert len(dataset2.samples) == len(dataset.samples), \
        "Deterministic reduction should produce same number of samples"

    # 4) DataLoader integration: draw batches from the infinite iterator
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        pin_memory=False,
        collate_fn=Batch.from_data_list,
        num_workers=0,
    )
    it = iter(loader)
    for _ in range(3):
        batch = next(it)
        assert batch.num_nodes > 0, "Batch should have nodes"
        assert batch.x.shape[1] == 3, "Expected 3D positions"
        assert hasattr(batch, "leaf_idx"), "Batch missing leaf_idx"

    # 5) Forward pass through model works with precomputed data
    cfg = _make_minimal_cfg()
    model, method = _init_model_and_method(cfg)
    device = "cuda" if th.cuda.is_available() else "cpu"
    model = model.to(device)
    method = method.to(device)

    batch = next(it).to(device)
    from torch_geometric.utils import to_edge_index as _to_edge_index
    ei_tmp, _ = _to_edge_index(batch.adj)
    if ei_tmp.numel() > 0:
        loss, metrics = method.get_loss(batch=batch, model=model)
        assert math.isfinite(float(loss.item())), "Loss must be finite"
        print(f"Precomputed forward pass OK | N={batch.num_nodes} | loss={loss.item():.4f}")
    else:
        print("Skipped forward pass (degenerate batch with no edges)")

    print("test_precomputed_dataset PASSED")


if __name__ == "__main__":  # Allow ad-hoc execution
    test_forward_pass()
    test_precomputed_dataset()
    print("Manual run complete.")
