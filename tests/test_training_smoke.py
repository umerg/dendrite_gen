"""Training loop smoke test.

Constructs synthetic binary trees, builds InfiniteRandRedDataset, instantiates
SO2_EGNN_Network + Expansion + DenoisingDiffusionModel + Trainer, and runs a
very short training loop (few steps) with validation disabled.

Goals:
  * Exercise Trainer.run_step path (forward, loss, backward, optimizer, EMA update).
  * Ensure no crashes due to missing config fields.
  * Keep runtime < 2s on CPU.

No metrics / validation invoked (validation.interval=0).
"""
from __future__ import annotations

import random
from types import SimpleNamespace
import sys, os
from pathlib import Path

# Ensure project root on sys.path for 'graph_generation' import when running directly.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import networkx as nx
import torch as th
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

import graph_generation as gg
from utils.data_loading import nx_graph_to_adj_pos

# HydraConfig shim (Trainer expects HydraConfig.get().runtime.output_dir). Provide minimal stub.
import types as _types
import graph_generation.training as _training_mod
if not hasattr(_training_mod, 'HydraConfig'):
    _training_mod.HydraConfig = _types.SimpleNamespace(get=lambda: _types.SimpleNamespace(runtime=_types.SimpleNamespace(output_dir='.' )))


def _make_random_binary_tree(n_min: int, n_max: int) -> nx.Graph:
    n_target = random.randint(n_min, n_max)
    G = nx.Graph()
    G.add_node(0, pos=np.zeros(3, dtype=np.float32))
    dirs = np.eye(3, dtype=np.float32)
    queue = [0]
    next_id = 1
    while queue and next_id < n_target:
        parent = queue.pop(0)
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


def _generate_graphs(num_graphs: int, n_min: int, n_max: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    return [_make_random_binary_tree(n_min, n_max) for _ in range(num_graphs)]


def _build_dataloader(graphs, cfg):
    adjs, poses = [], []
    for G in graphs:
        A, P, _ = nx_graph_to_adj_pos(G)
        adjs.append(A)
        poses.append(P)
    red_factory = gg.reduction.ReductionFactory(
        mode=cfg.reduction.mode,
        cherry_p=cfg.reduction.cherry_p,
        ensure_progress=cfg.reduction.ensure_progress,
        root=cfg.reduction.root,
        contract_root=cfg.reduction.contract_root,
    )
    dataset = gg.data.InfiniteRandRedDataset(adjs=adjs, poses=poses, red_factory=red_factory)
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        pin_memory=False,
        collate_fn=Batch.from_data_list,
        num_workers=0,
    )
    return loader


def _make_cfg():
    return SimpleNamespace(
        name="training_smoke",
        debugging=False,
        model=SimpleNamespace(
            name="egnn",
            num_layers=2,
            feats_dim=4,
            m_dim=16,
            dropout=0.0,
        ),
        method=SimpleNamespace(name="expansion"),
        reduction=SimpleNamespace(mode="stochastic", cherry_p=0.8, ensure_progress=True, root=0, contract_root=False,
                                  num_red_seqs=-1, red_threshold=0),
        training=SimpleNamespace(batch_size=2, lr=1e-3, num_steps=3, log_interval=1, save_checkpoint=False, resume=False, max_num_workers=0),
        validation=SimpleNamespace(interval=0, first_step=0, batch_size=None, per_graph_size=False),
        ema=SimpleNamespace(betas=[1], gamma=1.0, power=1.0),
        wandb=SimpleNamespace(logging=False),
    )


def test_training_smoke():
    seed = 777
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cfg = _make_cfg()
    graphs_train = _generate_graphs(num_graphs=4, n_min=30, n_max=50, seed=seed)
    graphs_val = _generate_graphs(num_graphs=2, n_min=30, n_max=50, seed=seed + 1)
    graphs_test = _generate_graphs(num_graphs=2, n_min=30, n_max=50, seed=seed + 2)

    loader = _build_dataloader(graphs_train, cfg)

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

    # Quick loss path smoke before trainer: ensures expansion state loss present
    sample_batch = next(iter(loader))
    res = method.get_loss(sample_batch, model)
    if res is not None:
        loss, metrics = res
        required_keys = {"leaf_pos_loss", "leaf_expansion_loss", "cumulative_loss"}
        missing = required_keys - set(metrics.keys())
        assert not missing, f"Missing loss metrics: {missing}"
        assert metrics["num_leaves"] >= 0, "Invalid num_leaves metric"
        print(f"Pre-training one-step loss check OK: {metrics}")
    else:
        print("Pre-training one-step loss check skipped (batch contained no leaves).")

    # No metrics & validation disabled
    trainer = gg.training.Trainer(
        model=model,
        method=method,
        train_dataloader=loader,
        train_graphs=graphs_train,
        validation_graphs=graphs_val,
        test_graphs=graphs_test,
        metrics=[],  # empty -> validation/evaluate metrics loops skipped if interval=0
        cfg=cfg,
    )

    trainer.train()
    assert trainer.step == cfg.training.num_steps, "Trainer did not complete expected number of steps"
    print(f"Training smoke complete at step {trainer.step}")


if __name__ == "__main__":
    # shim already applied above
    test_training_smoke()
    print("Manual training smoke test run complete.")
