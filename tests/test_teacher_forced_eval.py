"""Unit tests for validation.teacher_forced_eval metric functions (no model needed)."""

import math
import numpy as np
import pytest

from validation.teacher_forced_eval import (
    compute_tf_distribution_metrics,
    compute_tf_expansion_metrics,
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
