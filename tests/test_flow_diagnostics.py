"""Unit tests for the flow-matching stratified training diagnostics.

Primary: the pure helper `compute_flow_diagnostics` on hand-built tensors.
Plus one thin end-to-end check that `Expansion.get_loss` propagates the diag dict.
"""
import math
import random

import numpy as np
import torch as th

from graph_generation.diffusion.diagnostics import compute_flow_diagnostics

import graph_generation as gg
from graph_generation.diffusion.flow import FlowMatchingModel
from tests.test_flow_matching import _build_model_and_loader


# --- helpers --------------------------------------------------------------

def _mk(C_pred, C_0, e_pred, e_0, t, root, prior_var):
    """Build float tensors and call the helper."""
    to = lambda x: th.tensor(x, dtype=th.float32)
    return compute_flow_diagnostics(
        C_pred=to(C_pred), C_0=to(C_0),
        e_pred=to(e_pred).view(-1, 1), e_0=to(e_0).view(-1, 1),
        t_leaf=to(t).view(-1, 1),
        is_root_child=th.tensor(root, dtype=th.bool),
        prior_var=prior_var,
    )


# --- R2 skill score -------------------------------------------------------

def test_r2_is_one_when_prediction_is_perfect():
    C = [[0.3, -0.4, 0.5], [1.0, 0.2, -0.7], [-0.6, 0.9, 0.1]]
    out = _mk(C, C, [0.5, -0.5, 0.5], [1.0, -1.0, 1.0],
              t=[0.1, 0.5, 0.9], root=[True, False, False],
              prior_var=(0.7, 0.6, 0.8))
    for name in ("fwd", "side", "axial"):
        assert math.isclose(out[f"R2_{name}"], 1.0, abs_tol=1e-5)
        assert math.isclose(out[f"pos_mse_{name}"], 0.0, abs_tol=1e-6)


def test_r2_is_zero_when_prediction_is_column_mean():
    # Predicting the per-axis mean gives MSE == population variance per axis;
    # with prior_var set to that variance, R2 == 0 (no skill over the mean baseline).
    th.manual_seed(0)
    C_0 = th.randn(500, 3) * th.tensor([0.7, 0.6, 0.8])
    C_pred = C_0.mean(dim=0, keepdim=True).expand_as(C_0).contiguous()
    prior_var = tuple(C_0.var(dim=0, unbiased=False).tolist())
    out = compute_flow_diagnostics(
        C_pred=C_pred, C_0=C_0,
        e_pred=th.zeros(500, 1), e_0=th.ones(500, 1),
        t_leaf=th.rand(500, 1),
        is_root_child=th.zeros(500, dtype=th.bool),
        prior_var=prior_var,
    )
    for name in ("fwd", "side", "axial"):
        assert abs(out[f"R2_{name}"]) < 1e-4, f"R2_{name}={out[f'R2_{name}']}"


def test_r2_side_lowt_present_only_with_low_t_leaves():
    C = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    # both t < 0.25 -> R2_side_lowt present (perfect pred -> 1.0)
    out = _mk(C, C, [0.0, 0.0], [1.0, 1.0], t=[0.1, 0.2],
              root=[False, False], prior_var=(0.7, 0.6, 0.8))
    assert math.isclose(out["R2_side_lowt"], 1.0, abs_tol=1e-5)
    # all t >= 0.25 -> omitted
    out2 = _mk(C, C, [0.0, 0.0], [1.0, 1.0], t=[0.5, 0.9],
               root=[False, False], prior_var=(0.7, 0.6, 0.8))
    assert "R2_side_lowt" not in out2


# --- t-buckets ------------------------------------------------------------

def test_t_buckets_only_nonempty_emitted():
    C = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    out = _mk(C, C, [0.0, 0.0], [1.0, 1.0], t=[0.95, 0.97],
              root=[False, False], prior_var=(0.7, 0.6, 0.8))
    assert "pos_mse_t3" in out  # [0.75, 1.0]
    for b in (0, 1, 2):
        assert f"pos_mse_t{b}" not in out


def test_t_bucket_last_is_inclusive_of_one():
    C = [[0.0, 0.0, 0.0]]
    out = _mk(C, C, [0.0], [1.0], t=[1.0], root=[False], prior_var=(1.0, 1.0, 1.0))
    assert "pos_mse_t3" in out


# --- node-type split ------------------------------------------------------

def test_node_type_split_counts_and_keys():
    C_pred = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    C_0 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    out = _mk(C_pred, C_0, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0],
              t=[0.1, 0.5, 0.9], root=[True, False, True],
              prior_var=(0.7, 0.6, 0.8))
    assert out["num_root_leaves"] == 2.0
    assert out["num_interior_leaves"] == 1.0
    assert "pos_mse_root" in out and "pos_mse_interior" in out
    # root leaves are perfect -> 0 total error; interior leaf has side error 1.0
    assert math.isclose(out["pos_mse_root"], 0.0, abs_tol=1e-6)
    assert math.isclose(out["pos_mse_interior"], 1.0, abs_tol=1e-5)
    assert math.isclose(out["R2_side_root"], 1.0, abs_tol=1e-5)


def test_all_interior_omits_root_keys():
    C = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    out = _mk(C, C, [0.0, 0.0], [1.0, 1.0], t=[0.1, 0.5],
              root=[False, False], prior_var=(0.7, 0.6, 0.8))
    assert out["num_root_leaves"] == 0.0
    assert "pos_mse_root" not in out and "R2_side_root" not in out
    assert "pos_mse_interior" in out


# --- expansion classifier -------------------------------------------------

def test_expansion_classifier_metrics():
    # true labels (e_0>0): [T, T, F, F] -> base_rate 0.5
    # pred  labels (e_pred>0): [T, F, F, T]
    # acc = 2/4 = 0.5 ; tp=1, fp=1, fn=1 -> prec=rec=0.5 -> f1=0.5
    out = _mk([[0, 0, 0]] * 4, [[0, 0, 0]] * 4,
              e_pred=[0.5, -0.5, -0.5, 0.5], e_0=[1.0, 1.0, -1.0, -1.0],
              t=[0.5] * 4, root=[False] * 4, prior_var=(1.0, 1.0, 1.0))
    assert math.isclose(out["exp_acc"], 0.5, abs_tol=1e-6)
    assert math.isclose(out["exp_base_rate"], 0.5, abs_tol=1e-6)
    assert math.isclose(out["exp_f1"], 0.5, abs_tol=1e-6)


def test_single_class_expansion_omits_f1():
    # all true negative, all predicted negative -> acc 1.0, base_rate 0, no f1
    out = _mk([[0, 0, 0]] * 3, [[0, 0, 0]] * 3,
              e_pred=[-0.5, -0.2, -0.9], e_0=[-1.0, -1.0, -1.0],
              t=[0.5] * 3, root=[False] * 3, prior_var=(1.0, 1.0, 1.0))
    assert math.isclose(out["exp_acc"], 1.0, abs_tol=1e-6)
    assert math.isclose(out["exp_base_rate"], 0.0, abs_tol=1e-6)
    assert "exp_f1" not in out


# --- guards ---------------------------------------------------------------

def test_zero_prior_var_omits_r2_but_keeps_mse():
    C_pred = [[0.0, 1.0, 0.0]]
    C_0 = [[0.0, 0.0, 0.0]]
    out = _mk(C_pred, C_0, [0.0], [1.0], t=[0.5], root=[False],
              prior_var=(0.0, 0.6, 0.8))  # fwd prior_var == 0
    assert "pos_mse_fwd" in out          # MSE still reported
    assert "R2_fwd" not in out           # R2 would divide by zero -> omitted
    assert "R2_side" in out


def test_empty_input_returns_empty_dict():
    out = compute_flow_diagnostics(
        C_pred=th.zeros(0, 3), C_0=th.zeros(0, 3),
        e_pred=th.zeros(0, 1), e_0=th.zeros(0, 1),
        t_leaf=th.zeros(0, 1), is_root_child=th.zeros(0, dtype=th.bool),
        prior_var=(1.0, 1.0, 1.0),
    )
    assert out == {}


def test_all_values_are_finite_floats():
    th.manual_seed(1)
    C_0 = th.randn(40, 3)
    C_pred = C_0 + 0.1 * th.randn(40, 3)
    out = compute_flow_diagnostics(
        C_pred=C_pred, C_0=C_0,
        e_pred=th.randn(40, 1), e_0=th.sign(th.randn(40, 1)),
        t_leaf=th.rand(40, 1),
        is_root_child=(th.rand(40) < 0.3),
        prior_var=(0.7, 0.6, 0.8),
    )
    assert out, "expected a populated diagnostics dict"
    for k, v in out.items():
        assert isinstance(v, float) and math.isfinite(v), f"{k}={v}"


# --- thin end-to-end: get_loss propagates diag ----------------------------

def test_get_loss_propagates_diag():
    random.seed(0)
    np.random.seed(0)
    cfg, loader, model = _build_model_and_loader()
    diffusion = FlowMatchingModel(num_steps=4, prior_std_pos=[0.74, 0.61, 0.83])
    method = gg.method.Expansion(
        diffusion=diffusion, red_threshold=cfg.reduction.red_threshold,
    )
    saw_diag = False
    for batch in loader:
        res = method.get_loss(batch, model)
        if res is None:
            continue
        loss, metrics = res
        # original metrics still present
        for key in ("leaf_pos_loss", "leaf_expansion_loss", "cumulative_loss"):
            assert key in metrics
        if metrics.get("num_leaves", 0) > 0:
            assert "diag" in metrics, "diag dict missing for a batch with leaves"
            diag = metrics["diag"]
            assert isinstance(diag, dict) and diag
            assert "exp_acc" in diag
            assert all(isinstance(v, float) for v in diag.values())
            saw_diag = True
            break
    assert saw_diag, "no batch produced leaves; cannot verify diag propagation"


if __name__ == "__main__":
    test_r2_is_one_when_prediction_is_perfect()
    test_r2_is_zero_when_prediction_is_column_mean()
    test_r2_side_lowt_present_only_with_low_t_leaves()
    test_t_buckets_only_nonempty_emitted()
    test_t_bucket_last_is_inclusive_of_one()
    test_node_type_split_counts_and_keys()
    test_all_interior_omits_root_keys()
    test_expansion_classifier_metrics()
    test_single_class_expansion_omits_f1()
    test_zero_prior_var_omits_r2_but_keeps_mse()
    test_empty_input_returns_empty_dict()
    test_all_values_are_finite_floats()
    test_get_loss_propagates_diag()
    print("Flow diagnostics checks complete.")
