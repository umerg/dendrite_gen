"""Probe library on synthetic data with known answers."""
import numpy as np
import pytest

from interp import probe as P


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n_groups, per = 200, 20
    groups = np.repeat(np.arange(n_groups), per)
    n = len(groups)
    X = rng.normal(size=(n, 32))
    obs = rng.normal(size=(n, 3))
    return rng, groups, X, obs, n


def test_folds_are_grouped(data):
    _, groups, *_ = data
    for tr, te in P.folds(groups, 5, seed=3):
        assert not set(groups[tr]) & set(groups[te])


def test_ridge_recovers_linear_signal_and_not_noise(data):
    rng, groups, X, _, n = data
    w = rng.normal(size=32)
    y = X @ w + 0.3 * rng.normal(size=n)
    assert P.ridge_cv(X, y, groups).score > 0.95
    assert P.ridge_cv(X, rng.normal(size=n), groups).score < 0.02


def test_logistic_binary_and_multiclass(data):
    rng, groups, X, _, n = data
    yb = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)
    assert P.logistic_cv(X, yb, groups).score > 0.9
    ym = np.digitize(X[:, 1], [-0.7, 0.0, 0.7])
    r = P.logistic_cv(X, ym, groups)
    assert r.kind == "multiclass" and r.score > 0.6


def test_incremental_detects_only_extra_information(data):
    rng, groups, X, obs, n = data
    # y = nonlinear function of obs (+ noise): activations X carry nothing extra
    y0 = np.sin(2 * obs[:, 0]) + obs[:, 1] ** 2 + 0.1 * rng.normal(size=n)
    r0 = P.incremental(obs, X, y0, groups)
    assert r0["score_obs"] > 0.8 and abs(r0["delta"]) < 0.02
    # y has an extra linear term in X beyond obs: Δ must be clearly positive
    y1 = y0 + X[:, :4].sum(1)
    r1 = P.incremental(obs, X, y1, groups)
    assert r1["delta"] > 0.3
    # binary version
    yb = (y1 > np.median(y1)).astype(int)
    rb = P.incremental(obs, X, yb, groups)
    assert rb["delta"] > 0.05


def test_obs_stack_reuse_matches_one_shot(data):
    rng, groups, X, obs, n = data
    y = obs[:, 0] + X[:, 0] + 0.1 * rng.normal(size=n)
    a = P.incremental(obs, X, y, groups)
    b = P.incremental_with(P.obs_stack(obs, y, groups), X)
    np.testing.assert_allclose(a["oof_full"], b["oof_full"])


def test_controls(data):
    rng, groups, X, _, n = data
    keys = rng.integers(0, 5, size=n)
    y = rng.integers(0, 2, size=n)
    c = P.control_task(y, keys)
    for k in range(5):
        assert len(np.unique(c[keys == k])) == 1
    s = P.shuffle_within(y, groups)
    for g in np.unique(groups)[:10]:
        assert sorted(s[groups == g]) == sorted(y[groups == g])


def test_bootstrap_ci_brackets_point(data):
    rng, groups, X, _, n = data
    y = X[:, 0] + rng.normal(size=n)
    r = P.ridge_cv(X, y, groups)
    pt, lo, hi = P.bootstrap_ci(y, r.oof, groups, "regression", B=200)
    assert lo <= pt <= hi and hi - lo < 0.2
    d, dlo, dhi = P.paired_bootstrap(y, r.oof, r.oof, groups, "regression", B=50)
    assert d == 0 and dlo == 0 and dhi == 0


def test_mass_mean_stratification_removes_confound():
    rng = np.random.default_rng(1)
    n = 20000
    depth = rng.integers(1, 8, size=n)
    label = rng.random(n) < (depth / 8)          # label correlated with depth
    u, v = np.eye(16)[0], np.eye(16)[1]
    X = depth[:, None] * u + label[:, None] * v + 0.1 * rng.normal(size=(n, 16))
    raw = P.mass_mean(X, label)
    strat = P.mass_mean(X, label, strata=depth)
    assert abs(raw["unit"] @ u) > 0.5            # confounded by depth
    assert strat["unit"] @ v > 0.99 and abs(strat["unit"] @ u) < 0.05


def test_leace_erases_linear_information(data):
    rng, groups, X, _, n = data
    z = X[:, :3] @ rng.normal(size=3)
    Xs = X + z[:, None] * rng.normal(size=32)
    assert P.ridge_cv(Xs, z, groups).score > 0.9
    er = P.leace_fit(Xs, z)
    Xe = er.apply(Xs)
    assert P.ridge_cv(Xe, z, groups).score < 0.02
    # erasure is a rank-1 edit here: most of X is untouched
    assert np.linalg.matrix_rank(er.P, tol=1e-6) == 1
    import torch as th
    np.testing.assert_allclose(er.apply(th.tensor(Xs)).numpy(), Xe, atol=1e-8)


def test_incremental_not_inflated_by_neuron_memorisation():
    """Neuron-constant target, row-level observables that identify the neuron but carry no signal
    about the target: an unregularised baseline would overfit, making Δ for pure-noise X look
    large. With CV-selected obs models Δ must stay ≈ 0."""
    rng = np.random.default_rng(5)
    n_groups, per = 150, 30
    groups = np.repeat(np.arange(n_groups), per)
    ident = rng.normal(size=(n_groups, 4))[groups] + 0.05 * rng.normal(size=(len(groups), 4))
    y = rng.normal(size=n_groups)[groups] * 0.1 + 1.0       # neuron-constant, unpredictable
    X = rng.normal(size=(len(groups), 32))                  # pure noise activations
    r = P.incremental(ident, X, y, groups)
    assert r["score_obs"] > -0.1
    assert abs(r["delta"]) < 0.05


def test_binary_incremental_no_penalty_artifact():
    rng = np.random.default_rng(6)
    groups = np.repeat(np.arange(200), 20)
    obs = rng.normal(size=(len(groups), 3))
    yb = (obs[:, 0] + 0.3 * rng.normal(size=len(groups)) > 0).astype(int)   # obs nearly determines y
    X = rng.normal(size=(len(groups), 64))
    r = P.incremental(obs, X, yb, groups)
    assert r["score_obs"] > 0.9 and r["delta"] > -0.01


def test_delta_does_not_credit_recalibration_multiclass():
    """X carrying no information must not gain Δ by letting the stacker re-threshold ô."""
    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(120), 15)
    obs = rng.normal(size=(len(groups), 3))
    y = np.digitize(np.sin(2 * obs[:, 0]) + obs[:, 1] ** 2 + 0.3 * rng.normal(size=len(groups)), [0.3, 1.2])
    r = P.incremental(obs, np.zeros((len(groups), 1)), y, groups)
    assert abs(r["delta"]) < 0.01
    r = P.incremental(obs, rng.normal(size=(len(groups), 16)), y, groups)
    assert r["delta"] < 0.02


def test_weighted_ridge_has_intercept():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(400, 3))
    X[:200] += 5.0                       # shifted half-sample
    y = X @ np.array([1.0, -2.0, 0.5]) + 3.0
    w = np.where(np.arange(400) < 200, 3.0, 1.0)
    pred = P._ridge_path(X, y, w, X, np.array([1e-8]))[0]
    assert np.average((pred - y) ** 2, weights=w) < 1e-6


def test_folds_change_with_seed():
    groups = np.repeat(np.arange(100), np.arange(1, 101))   # distinct sizes
    a = {tuple(sorted(np.unique(groups[te]))) for _, te in P.folds(groups, 5, 0)}
    b = {tuple(sorted(np.unique(groups[te]))) for _, te in P.folds(groups, 5, 1)}
    assert a != b
