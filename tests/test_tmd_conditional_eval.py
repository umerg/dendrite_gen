"""Unit tests for validation.tmd_conditional_eval.

Covers the matched, TMD-conditioned pairwise metric function: expected keys + finiteness,
identical-pair ~0, the empty-input {} contract, the persim-missing guard (PD -> nan, geometric
and W1 stay finite, no crash), bad-pair skip counting, the max_pairs cap, and the all-float
output contract (only floats reach wandb via Trainer.log).
"""
import copy
import math
import random

import numpy as np
import networkx as nx
import pytest

from validation.tmd_conditional_eval import compute_conditional_pairwise_metrics

_UHAT = (0.0, 0.0, 1.0)
_FILTS = ("path", "radial_root")

# Keys the default call (wasserstein on, bottleneck off, filtrations path+radial_root) must emit.
_EXPECTED_KEYS = {
    "pd_wasserstein_path_mean", "pd_wasserstein_path_median", "pd_nan_frac_path",
    "pd_wasserstein_radial_root_mean", "pd_wasserstein_radial_root_median", "pd_nan_frac_radial_root",
    "height_absdiff_mean", "height_absdiff_median",
    "span_xy_absdiff_mean", "span_xy_absdiff_median",
    "bbox_diag_absdiff_mean", "bbox_diag_absdiff_median",
    "branch_length_w1_pairwise_mean", "branch_length_w1_pairwise_median",
    "bifurcation_angle_w1_pairwise_mean", "bifurcation_angle_w1_pairwise_median",
    "n_pairs", "n_pairs_skipped",
}


def _tree(n, seed):
    """A random binary-ish tree with 3D `pos` node attrs and node 0 as the rooted soma."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_node(0, pos=np.zeros(3, np.float32))
    dirs = np.eye(3, dtype=np.float32)
    queue, nid = [0], 1
    while queue and nid < n:
        p = queue.pop(0)
        for _ in range(rng.choice([1, 2])):
            if nid >= n:
                break
            pos = (G.nodes[p]["pos"] + dirs[rng.randint(0, 2)] * rng.uniform(0.5, 2.0)
                   + nrng.normal(0, 0.2, 3).astype(np.float32))
            G.add_node(nid, pos=pos)
            G.add_edge(p, nid)
            queue.append(nid)
            nid += 1
    G.graph["root"] = 0
    return G


def _jitter(G, seed, scale=0.05):
    """Deep copy of G with a small position perturbation (a 'nearby' generated tree)."""
    H = copy.deepcopy(G)
    nrng = np.random.default_rng(seed)
    for nid in H.nodes():
        H.nodes[nid]["pos"] = (np.asarray(H.nodes[nid]["pos"], dtype=np.float64)
                               + nrng.normal(0, scale, 3))
    return H


def _gt_set(k=6, base=40):
    return [_tree(base + i, seed=100 + i) for i in range(k)]


def test_keys_and_finite():
    gt = _gt_set()
    gen = [_jitter(g, seed=200 + i) for i, g in enumerate(gt)]
    out = compute_conditional_pairwise_metrics(gen, gt, uhat=_UHAT, pd_filtrations=_FILTS)
    assert set(out) == _EXPECTED_KEYS
    assert out["n_pairs"] == float(len(gt))
    # geometric + pairwise-W1 aggregates are always finite for well-formed jittered trees
    for k in ("height_absdiff_mean", "span_xy_absdiff_mean", "bbox_diag_absdiff_mean",
              "branch_length_w1_pairwise_mean", "bifurcation_angle_w1_pairwise_mean"):
        assert math.isfinite(out[k]), f"{k} not finite"


def test_all_values_are_float():
    gt = _gt_set()
    gen = [_jitter(g, seed=300 + i) for i, g in enumerate(gt)]
    out = compute_conditional_pairwise_metrics(gen, gt, uhat=_UHAT, pd_filtrations=_FILTS)
    assert out, "expected a non-empty metric dict"
    assert all(isinstance(v, float) for v in out.values()), \
        [k for k, v in out.items() if not isinstance(v, float)]


def test_identical_pairs_near_zero():
    gt = _gt_set()
    gen = [copy.deepcopy(g) for g in gt]  # exact copies -> every matched distance ~ 0
    out = compute_conditional_pairwise_metrics(gen, gt, uhat=_UHAT, pd_filtrations=_FILTS)
    for k in ("height_absdiff_mean", "span_xy_absdiff_mean", "bbox_diag_absdiff_mean",
              "branch_length_w1_pairwise_mean", "bifurcation_angle_w1_pairwise_mean"):
        assert out[k] == pytest.approx(0.0, abs=1e-9), f"{k}={out[k]}"


def test_identical_pairs_pd_zero():
    pytest.importorskip("persim")
    gt = _gt_set()
    gen = [copy.deepcopy(g) for g in gt]
    out = compute_conditional_pairwise_metrics(gen, gt, uhat=_UHAT, pd_filtrations=_FILTS)
    for f in _FILTS:
        assert out[f"pd_wasserstein_{f}_mean"] == pytest.approx(0.0, abs=1e-6)
        assert out[f"pd_nan_frac_{f}"] == pytest.approx(0.0)


def test_empty_inputs():
    gt = _gt_set()
    assert compute_conditional_pairwise_metrics([], gt) == {}
    assert compute_conditional_pairwise_metrics(gt, []) == {}


def test_persim_missing_guard(monkeypatch):
    # Simulate persim unavailable: PD distances -> nan, but geometric / W1 stay finite and
    # the function must not raise.
    monkeypatch.setattr("validation.tmd_conditional_eval._persim", None)
    gt = _gt_set()
    gen = [_jitter(g, seed=400 + i) for i, g in enumerate(gt)]
    out = compute_conditional_pairwise_metrics(gen, gt, uhat=_UHAT, pd_filtrations=_FILTS)
    for f in _FILTS:
        assert math.isnan(out[f"pd_wasserstein_{f}_mean"])
        assert out[f"pd_nan_frac_{f}"] == pytest.approx(1.0)
    assert math.isfinite(out["height_absdiff_mean"])
    assert math.isfinite(out["branch_length_w1_pairwise_mean"])


def test_bad_pair_is_skipped_not_crashed():
    gt = _gt_set(k=5)
    gen = [copy.deepcopy(g) for g in gt]
    del gen[0].graph["root"]          # break diagram construction for one pred (no root)
    out = compute_conditional_pairwise_metrics(gen, gt, uhat=_UHAT, pd_filtrations=_FILTS)
    assert out["n_pairs"] == 5.0
    assert out["n_pairs_skipped"] >= 1.0
    # aggregates over the remaining good pairs are still finite
    assert math.isfinite(out["height_absdiff_mean"])


def test_max_pairs_cap():
    gt = _gt_set(k=10)
    gen = [_jitter(g, seed=500 + i) for i, g in enumerate(gt)]
    out = compute_conditional_pairwise_metrics(gen, gt, uhat=_UHAT, pd_filtrations=_FILTS, max_pairs=4)
    assert out["n_pairs"] == 4.0


def test_bottleneck_toggle_adds_keys():
    pytest.importorskip("persim")
    gt = _gt_set(k=4)
    gen = [_jitter(g, seed=600 + i) for i, g in enumerate(gt)]
    out = compute_conditional_pairwise_metrics(
        gen, gt, uhat=_UHAT, pd_filtrations=("path",), enable_bottleneck=True)
    assert "pd_bottleneck_path_mean" in out and "pd_wasserstein_path_mean" in out
