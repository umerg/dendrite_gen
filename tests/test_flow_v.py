"""Functional checks for VFlowMatchingModel (v-prediction / velocity flow matching).

Verifies:
  - forward() returns finite (exp_loss, pos_loss) + diag on a real training batch (uniform & beta),
  - sample() integrates to finite local-frame predictions of the right shape via sample_graphs,
  - the v-prediction integrator is correct: a CONSTANT velocity field integrates EXACTLY and is
    num_steps-INVARIANT (lands at C_init + v for any num_steps) -- the structural property that
    removes the data-prediction 1/(1-t) terminal amplification,
  - prior_std_pos behaves identically to the flow copy (the copy didn't break the shared helper).
"""
import random

import numpy as np
import torch as th
from torch import nn

import graph_generation as gg
from graph_generation.diffusion.flow_v import VFlowMatchingModel

# Reuse the existing smoke-test fixtures (graph generation, dataloader, cfg).
from tests.test_training_smoke import (
    _build_dataloader,
    _generate_graphs,
    _make_cfg,
)
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


class _ConstVelModel(nn.Module):
    """Stand-in model whose output (interpreted as the VELOCITY by VFlowMatchingModel) is a
    fixed constant for every node/step. Integrating a constant velocity is exact."""
    def __init__(self, v_const, ev_const=0.0):
        super().__init__()
        self.register_buffer("v", th.as_tensor(v_const, dtype=th.float32).view(1, 3))
        self.ev = float(ev_const)
        self.register_buffer("uhat", th.tensor([0., 1., 0.]))

    def forward(self, x, edge_index, batch, edge_attr=None,
                parent_idx=None, tmd=None, pre_geom=None, **kw):
        N = x.size(0)
        return {
            "rel_pred": self.v.expand(N, 3).contiguous(),
            "expansion_pred": th.full((N, 1), self.ev, device=x.device),
        }


def _tiny_star_graph(L, device="cpu"):
    """1 root (node 0) + L leaves, all children of the root. Args for diffusion.sample()."""
    N = L + 1
    leaf_idx = th.arange(1, N, dtype=th.long, device=device)
    parent_idx = th.full((N,), 0, dtype=th.long, device=device)
    parent_idx[0] = -1  # root
    return dict(
        node_feats=None,
        edge_index=th.stack([th.zeros(L, dtype=th.long), leaf_idx]).to(device),
        batch=th.zeros(N, dtype=th.long, device=device),
        edge_attr=th.zeros((L, 1), device=device),
        P_0=th.zeros((N, 3), device=device),
        parent_idx=parent_idx,
        leaf_idx=leaf_idx,
        leaf_parent_idx=th.zeros(L, dtype=th.long, device=device),
        local_forward=None, local_sideways=None, uhat=None, pre_geom_p0=None,
    )


def test_v_forward_loss_finite():
    cfg, loader, model = _build_model_and_loader()
    for time_dist in ("uniform", "beta"):
        diffusion = VFlowMatchingModel(num_steps=4, time_dist=time_dist)
        method = gg.method.Expansion(
            diffusion=diffusion,
        )
        res = method.get_loss(next(iter(loader)), model)
        if res is None:
            continue
        loss, metrics = res
        assert th.isfinite(loss).all(), f"non-finite loss for time_dist={time_dist}"
        for key in ("leaf_pos_loss", "leaf_expansion_loss", "cumulative_loss"):
            assert key in metrics, f"missing metric {key}"
        print(f"[{time_dist}] v-forward OK: {metrics}")


def test_v_sample_shapes_finite():
    th.manual_seed(0)
    diffusion = VFlowMatchingModel(num_steps=8, prior_std=1.0)
    method = gg.method.Expansion(diffusion=diffusion)
    target_sizes = th.tensor([5, 9, 15], dtype=th.long)
    graphs = method.sample_graphs(target_sizes, MockModel())
    assert len(graphs) == len(target_sizes)
    for g in graphs:
        pos = getattr(g, "pos", None)
        if pos is not None:
            assert th.isfinite(pos).all(), "non-finite positions from v-flow sampling"


def test_v_sample_constant_velocity_exact_and_num_steps_invariant():
    """Core integrator property: a constant velocity field integrates EXACTLY, independent of
    num_steps (C_final = C_init + v). This is the structural fix vs data-prediction's 1/(1-t)."""
    L = 4000
    v_const = [0.5, -0.3, 0.2]
    model = _ConstVelModel(v_const)
    args = _tiny_star_graph(L)

    results = {}
    for ns in (1, 10, 100):
        diffusion = VFlowMatchingModel(num_steps=ns, prior_std=1.0)
        th.manual_seed(123)  # same C_init draw across runs
        C, e = diffusion.sample(model=model, **args)
        results[ns] = C
        assert C.shape == (L, 3) and th.isfinite(C).all()

    # num_steps-INVARIANT: identical to float tolerance (exact integration of a constant field).
    assert th.allclose(results[1], results[10], atol=1e-5), "num_steps 1 vs 10 differ"
    assert th.allclose(results[1], results[100], atol=1e-5), "num_steps 1 vs 100 differ"

    # C_final = C_init + v_const, and E[C_init] = 0 -> mean over leaves ≈ v_const.
    mean = results[100].mean(dim=0)
    assert th.allclose(mean, th.tensor(v_const), atol=0.05), f"mean {mean.tolist()} != {v_const}"


def test_v_anisotropic_prior_per_axis_std():
    """prior_std_pos helper carried over unchanged from the flow copy."""
    sigma = [0.74, 0.61, 0.83]
    fm = VFlowMatchingModel(num_steps=4, prior_std=1.0, prior_std_pos=sigma)
    scale = fm._pos_scale(th.device("cpu"), th.float32)
    assert tuple(scale.shape) == (1, 3)
    assert th.allclose(scale, th.tensor(sigma).view(1, 3), atol=1e-6)
    assert VFlowMatchingModel(num_steps=4, prior_std=1.3)._pos_scale(th.device("cpu"), th.float32) == 1.3
    try:
        VFlowMatchingModel(prior_std_pos=[1.0, 2.0])
        raise AssertionError("expected ValueError for length-2 prior_std_pos")
    except ValueError:
        pass


if __name__ == "__main__":
    test_v_forward_loss_finite()
    test_v_sample_shapes_finite()
    test_v_sample_constant_velocity_exact_and_num_steps_invariant()
    test_v_anisotropic_prior_per_axis_std()
    print("V-prediction flow-matching checks complete.")
