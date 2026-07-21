"""Integration test for the new validation path: distribution metrics + 3D plots.

Reuses the training-smoke scaffolding to build a working Trainer, but ENABLES
validation (enable_dist_metrics + enable_plots) and drives run_validation()/
evaluate() directly. Also uses a non-z so2_axis to confirm the uhat-aligned 3D
plotting path runs (azimuths orbit the true uhat axis, not hardcoded z).
"""
from __future__ import annotations

import random
from types import SimpleNamespace
import sys
import types as _types
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import networkx as nx
import torch as th
from matplotlib.figure import Figure
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

import graph_generation as gg
from utils.data_loading import nx_graph_to_adj_pos

import graph_generation.training as _training_mod
if not hasattr(_training_mod, "HydraConfig"):
    _training_mod.HydraConfig = _types.SimpleNamespace(
        get=lambda: _types.SimpleNamespace(
            runtime=_types.SimpleNamespace(output_dir=".")
        )
    )

from tests.test_training_smoke import _generate_graphs, _build_dataloader


def _make_cfg():
    return SimpleNamespace(
        name="validation_eval",
        debugging=False,
        model=SimpleNamespace(name="egnn", num_layers=2, feats_dim=4, m_dim=16, dropout=0.0),
        method=SimpleNamespace(name="expansion"),
        reduction=SimpleNamespace(
            mode="stochastic", cherry_p=0.8, ensure_progress=True, root=0,
            contract_root=False, num_red_seqs=-1, red_threshold=0,
        ),
        training=SimpleNamespace(
            batch_size=2, lr=1e-3, num_steps=1, log_interval=1,
            save_checkpoint=True, resume=False, max_num_workers=0,
        ),
        validation=SimpleNamespace(
            interval=1, first_step=0, batch_size=4, per_graph_size=False,
            enable_metrics=False, enable_plots=True, enable_dist_metrics=True,
            ged_enabled=True, ged_timeout=5.0,
            plot_angles=[[20, 30], [20, 120]],
        ),
        ema=SimpleNamespace(betas=[1], gamma=1.0, power=1.0),
        wandb=SimpleNamespace(logging=False),
    )


def _build_trainer(cfg, tmp_path, so2_axis):
    seed = 777
    th.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    graphs_train = _generate_graphs(num_graphs=4, n_min=20, n_max=35, seed=seed)
    graphs_val = _generate_graphs(num_graphs=3, n_min=20, n_max=35, seed=seed + 1)
    graphs_test = _generate_graphs(num_graphs=2, n_min=20, n_max=35, seed=seed + 2)
    # validation/dist_metrics + plots need a root on every graph
    for graphs in (graphs_train, graphs_val, graphs_test):
        for G in graphs:
            G.graph["root"] = 0

    loader = _build_dataloader(graphs_train, cfg)
    model = gg.model.SO2_EGNN_Network(
        n_layers=cfg.model.num_layers, feats_dim=cfg.model.feats_dim,
        pos_dim=3, m_dim=cfg.model.m_dim, dropout=cfg.model.dropout,
        edge_attr_dim=1, so2_axis=so2_axis,
    )
    from graph_generation.diffusion.basic import DenoisingDiffusionModel
    method = gg.method.Expansion(diffusion=DenoisingDiffusionModel(num_steps=1),
                                 red_threshold=cfg.reduction.red_threshold)
    trainer = gg.training.Trainer(
        model=model, method=method, train_dataloader=loader,
        train_graphs=graphs_train, validation_graphs=graphs_val,
        test_graphs=graphs_test, metrics=[], cfg=cfg,
    )
    trainer.output_dir = Path(tmp_path)  # redirect eval_plots / pickles to tmp
    return trainer


def test_evaluate_emits_dist_metrics_and_3d_plots(tmp_path):
    cfg = _make_cfg()
    trainer = _build_trainer(cfg, tmp_path, so2_axis=(0.0, 0.0, 1.0))

    results = trainer.evaluate(trainer.validation_graphs, beta=1)

    # Distribution metrics present and well-formed
    assert "dist" in results
    dist = results["dist"]
    for key in ("branch_length_w1", "tmd_barlen_w1", "node_count_w1", "tree_edit_dist_mean"):
        assert key in dist, f"missing dist key {key}"
        assert isinstance(dist[key], float)

    # 3D figures produced and saved
    assert isinstance(results["examples"], Figure)
    assert isinstance(results["examples_compare"], Figure)
    assert Path(results["examples_path"]).exists()
    assert Path(results["examples_compare_path"]).exists()
    # filenames encode the 3D azimuth grid
    assert "gen3d" in results["examples_path"]
    assert "ref3d" in results["examples_compare_path"]


def test_run_validation_non_z_axis(tmp_path):
    """Full run_validation with a non-z uhat: exercises uhat->z alignment + pickling."""
    cfg = _make_cfg()
    trainer = _build_trainer(cfg, tmp_path, so2_axis=(0.0, 1.0, 0.0))

    # model must expose uhat for the plotting alignment (and generation)
    assert getattr(trainer.ema_models[1], "uhat", None) is not None
    uhat = trainer.ema_models[1].uhat.detach().cpu().numpy().reshape(-1)
    assert np.allclose(uhat, [0, 1, 0], atol=1e-6)

    trainer.step = 1
    trainer.run_validation()  # should not raise; writes plots + validation pickle

    eval_plots = list((Path(tmp_path) / "eval_plots").glob("*.png"))
    assert any("gen3d" in p.name for p in eval_plots)
    assert any("ref3d" in p.name for p in eval_plots)
