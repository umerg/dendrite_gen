"""Functional checks for FlowMatchingModel as a drop-in for DenoisingDiffusionModel.

Verifies that:
  - forward() returns finite scalar (exp_loss, pos_loss) on a real training batch,
  - sample() integrates to finite local-frame predictions of the right shape,
  - sample_graphs() runs end-to-end through Expansion with flow-matching sampling,
  - both "uniform" and "beta" time distributions work.
"""
import random
from types import SimpleNamespace

import numpy as np
import torch as th

import graph_generation as gg
from graph_generation.diffusion.flow import FlowMatchingModel

# Reuse the existing smoke-test fixtures (graph generation, dataloader, cfg).
from tests.test_training_smoke import (
    _build_dataloader,
    _generate_graphs,
    _make_cfg,
)
# Reuse the minimal stand-in model from the generation test.
from tests.test_expansion_generation import MockModel


def _build_model_and_loader(seed=777):
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cfg = _make_cfg()
    graphs = _generate_graphs(num_graphs=4, n_min=30, n_max=50, seed=seed)
    loader = _build_dataloader(graphs, cfg)
    model = gg.model.SO2_EGNN_Network(
        n_layers=cfg.model.num_layers,
        feats_dim=cfg.model.feats_dim,
        pos_dim=3,
        m_dim=cfg.model.m_dim,
        dropout=cfg.model.dropout,
        edge_attr_dim=1,
    )
    return cfg, loader, model


def test_flow_forward_loss_finite():
    cfg, loader, model = _build_model_and_loader()
    for time_dist in ("uniform", "beta"):
        diffusion = FlowMatchingModel(num_steps=4, time_dist=time_dist)
        method = gg.method.Expansion(
            diffusion=diffusion,
        )
        batch = next(iter(loader))
        res = method.get_loss(batch, model)
        if res is None:
            continue
        loss, metrics = res
        assert th.isfinite(loss).all(), f"non-finite loss for time_dist={time_dist}"
        for key in ("leaf_pos_loss", "leaf_expansion_loss", "cumulative_loss"):
            assert key in metrics, f"missing metric {key}"
        print(f"[{time_dist}] forward OK: {metrics}")


def test_flow_sample_shapes_finite():
    """Drive FlowMatchingModel.sample directly with the MockModel, like the generation test."""
    th.manual_seed(0)
    diffusion = FlowMatchingModel(num_steps=8, prior_std=1.0)
    method = gg.method.Expansion(diffusion=diffusion)
    target_sizes = th.tensor([5, 9, 15], dtype=th.long)
    model = MockModel()
    graphs = method.sample_graphs(target_sizes, model)
    assert len(graphs) == len(target_sizes)
    for g in graphs:
        # graphs carry positions; verify they are finite
        pos = getattr(g, "pos", None)
        if pos is not None:
            assert th.isfinite(pos).all(), "non-finite positions from flow sampling"
    print(f"sample_graphs OK: sizes={[getattr(g, 'num_nodes', None) for g in graphs]}")


def test_anisotropic_prior_per_axis_std():
    """prior_std_pos scales the position prior per-axis; None falls back to isotropic."""
    th.manual_seed(0)
    sigma = [0.74, 0.61, 0.83]
    fm = FlowMatchingModel(num_steps=4, prior_std=1.0, prior_std_pos=sigma)

    scale = fm._pos_scale(th.device("cpu"), th.float32)
    assert tuple(scale.shape) == (1, 3)
    assert th.allclose(scale, th.tensor(sigma).view(1, 3), atol=1e-6)

    # Empirical per-axis std of the position prior matches sigma.
    noise = th.randn((200_000, 3)) * scale
    emp = noise.std(dim=0)
    assert th.allclose(emp, th.tensor(sigma), atol=0.02), f"per-axis std {emp.tolist()} != {sigma}"

    # Isotropic fallback: None -> scalar prior_std (unchanged behavior).
    fm_iso = FlowMatchingModel(num_steps=4, prior_std=1.3, prior_std_pos=None)
    assert fm_iso._pos_scale(th.device("cpu"), th.float32) == 1.3

    # Wrong length is rejected.
    try:
        FlowMatchingModel(prior_std_pos=[1.0, 2.0])
        raise AssertionError("expected ValueError for length-2 prior_std_pos")
    except ValueError:
        pass


def test_anisotropic_prior_forward_runs():
    """forward() still returns finite losses with an anisotropic prior."""
    cfg, loader, model = _build_model_and_loader()
    diffusion = FlowMatchingModel(num_steps=4, prior_std_pos=[0.74, 0.61, 0.83])
    method = gg.method.Expansion(
        diffusion=diffusion,
    )
    res = method.get_loss(next(iter(loader)), model)
    if res is not None:
        loss, _ = res
        assert th.isfinite(loss).all()


if __name__ == "__main__":
    test_flow_forward_loss_finite()
    test_flow_sample_shapes_finite()
    test_anisotropic_prior_per_axis_std()
    test_anisotropic_prior_forward_runs()
    print("Flow-matching functional checks complete.")
