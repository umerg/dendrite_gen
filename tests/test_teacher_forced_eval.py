"""Unit tests for validation.teacher_forced_eval metric functions (no model needed)."""

import math
import numpy as np
import pytest

from validation.teacher_forced_eval import (
    compute_tf_distribution_metrics,
    compute_tf_expansion_metrics,
    compute_tf_pos_mse,
    _metrics_from_pools,
    _sibling_angles,
    _auc,
)


def _synthetic(L=400, seed=0):
    """Synthetic captured pools: GT offsets + a sampler that over-extends along forward."""
    rng = np.random.default_rng(seed)
    c0 = np.stack([
        np.abs(rng.normal(0.7, 0.2, L)),   # forward (radial outgrowth, positive)
        rng.normal(0.0, 0.3, L),           # sideways
        rng.normal(0.0, 0.2, L),           # axial
    ], axis=1)
    cs = c0.copy()
    cs[:, 0] *= 1.3                          # over-extend forward (sampler inflation)
    fwd = np.tile(np.array([1.0, 0.0, 0.0]), (L, 1))
    side = np.tile(np.array([0.0, 0.0, 1.0]), (L, 1))
    uhat = np.array([0.0, 1.0, 0.0])
    lp = rng.integers(0, L // 2, L)          # parents (some shared -> siblings)
    es = rng.normal(0.0, 1.0, (L, 1))
    # get_loss passes leaf_expansion in {0,1} (expand=1); make it correlate with es -> AUC>0.5
    lexp = np.where(es.reshape(-1) + rng.normal(0, 0.3, L) > 0, 1, 0)
    level = rng.integers(0, 4, L)
    return dict(cs=cs, c0=c0, fwd=fwd, side=side, uhat=uhat, lp=lp, es=es, lexp=lexp, level=level)


def test_distribution_metrics_keys_and_finite():
    s = _synthetic()
    d = compute_tf_distribution_metrics(s["cs"], s["c0"], s["fwd"], s["side"], s["uhat"], s["lp"])
    for k in ("branch_length_w1", "branch_length_ks", "fwd_signed_w1", "fwd_mag_w1",
              "side_signed_w1", "axial_signed_w1", "turning_angle_w1", "axial_frac_w1",
              "bifurcation_angle_w1", "bifurcation_angle_ks"):
        assert k in d, f"missing {k}"
        assert math.isfinite(d[k]), f"{k} not finite"
    # forward was inflated -> forward W1 should be clearly > sideways W1 (which is unchanged)
    assert d["fwd_signed_w1"] > d["side_signed_w1"]
    # directional means present, finite, and capture the over/under sign (symmetric W1 cannot)
    for k in ("branch_length_mean_samp", "branch_length_mean_gt",
              "fwd_mag_mean_samp", "fwd_mag_mean_gt", "fwd_signed_mean_samp",
              "side_mag_mean_samp", "axial_mag_mean_gt"):
        assert k in d and math.isfinite(d[k]), f"bad {k}"
    # forward magnitude was inflated in the synthetic -> sampled mean > GT mean (over-production)
    assert d["fwd_mag_mean_samp"] > d["fwd_mag_mean_gt"]


def test_expansion_metrics():
    s = _synthetic()
    e = compute_tf_expansion_metrics(s["es"], s["lexp"])
    for k in ("acc", "precision", "recall", "f1", "auc", "base_rate", "n"):
        assert k in e and math.isfinite(e[k]), f"bad {k}"
    assert 0.0 <= e["acc"] <= 1.0
    assert e["auc"] > 0.5  # scores correlate with labels by construction


def test_per_level_stratification():
    s = _synthetic()
    res = _metrics_from_pools(s, level_min=10)
    assert res["n_leaves"] == s["cs"].shape[0]
    assert res["dist"] and res["exp"]
    assert len(res["by_level"]) >= 2  # multiple reduction levels present
    for lv, blk in res["by_level"].items():
        assert "dist" in blk and "exp" in blk and blk["n"] >= 10


def test_auc_perfect_and_sibling_angles():
    # perfect separation -> AUC 1.0
    scores = np.array([0.1, 0.2, 0.9, 1.0])
    labels = np.array([False, False, True, True])
    assert _auc(scores, labels) == pytest.approx(1.0)
    # two opposite offsets sharing a parent -> 180 degrees
    off = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    ang = _sibling_angles(off, np.array([0, 0]))
    assert ang.size == 1 and ang[0] == pytest.approx(180.0, abs=1e-3)


def test_empty_inputs():
    z3 = np.zeros((0, 3)); z1 = np.zeros((0, 1))
    assert compute_tf_distribution_metrics(z3, z3, z3, z3, np.array([0., 1., 0.]), np.zeros(0)) == {}
    assert compute_tf_expansion_metrics(z1, np.zeros(0)) == {}
    assert compute_tf_pos_mse(z3, z3) == {}


def test_pos_mse_node_wise():
    s = _synthetic()  # forward offset inflated x1.3, sideways/axial unchanged
    m = compute_tf_pos_mse(s["cs"], s["c0"])
    for k in ("fwd", "side", "axial", "total", "dist_mean", "dist_median"):
        assert k in m and math.isfinite(m[k]), f"bad {k}"
    # per-axis MSEs sum to the total (mean squared 3D error)
    assert m["total"] == pytest.approx(m["fwd"] + m["side"] + m["axial"], rel=1e-6)
    # only forward was perturbed -> forward MSE dominates; side/axial ~ 0 (GT == sampled there)
    assert m["fwd"] > m["side"]
    assert m["side"] == pytest.approx(0.0, abs=1e-12)
    assert m["dist_mean"] > 0.0 and m["dist_median"] > 0.0


def test_skip_ks_keeps_w1():
    s = _synthetic()
    d = compute_tf_distribution_metrics(
        s["cs"], s["c0"], s["fwd"], s["side"], s["uhat"], s["lp"], enable_ks=False)
    # every KS twin dropped ...
    assert not any(k.endswith("_ks") for k in d), f"KS keys leaked: {[k for k in d if k.endswith('_ks')]}"
    # ... while the W1 distances (and directional means) remain
    for k in ("branch_length_w1", "fwd_signed_w1", "fwd_mag_w1", "turning_angle_w1",
              "axial_frac_w1", "bifurcation_angle_w1", "fwd_mag_mean_samp"):
        assert k in d and math.isfinite(d[k]), f"missing {k}"


def test_pooled_only_no_breakdowns():
    s = _synthetic()
    res = _metrics_from_pools(s, level_min=10, include_breakdowns=False, enable_ks=False)
    # pooled blocks present, including the new node-wise pos_mse
    assert res["dist"] and res["exp"] and res["pos_mse"]
    assert "total" in res["pos_mse"]
    # no per-level / per-depth breakdowns, no KS in the pooled dist block
    assert res["by_level"] == {} and res["by_depth"] == {}
    assert not any(k.endswith("_ks") for k in res["dist"])
