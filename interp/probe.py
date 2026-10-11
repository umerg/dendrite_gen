"""Probe library: grouped cross-validated linear probes, incremental-over-observables
statistics, controls, neuron bootstrap, mass-mean directions and LEACE erasure.

Conventions
  * ``groups`` is the neuron id per row. Every split is grouped, so no neuron appears in
    both the fitting and scoring part of a fold (rows of one neuron are highly correlated).
  * ``weights`` default to 1/rows-per-neuron (each neuron counts equally); pass ones to
    weight rows equally.
  * Every CV function returns out-of-fold (OOF) predictions so any metric / bootstrap can
    be computed afterwards on exactly the same predictions.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ALPHAS = np.logspace(-2, 5, 15)
CS = np.logspace(-4, 1, 6)


# ----------------------------------------------------------------------------- utilities

def neuron_weights(groups: np.ndarray) -> np.ndarray:
    _, inv, cnt = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    return w * (len(w) / w.sum())


def folds(groups: np.ndarray, n_splits: int = 5, seed: int = 0):
    """Grouped folds, randomised by ``seed`` (GroupKFold without shuffle assigns groups greedily
    by size, so permuting labels would barely change the folds)."""
    groups = np.asarray(groups)
    n_splits = min(n_splits, len(np.unique(groups)))
    return list(GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                .split(np.zeros(len(groups)), groups=groups))


FREE_SCALE = 30.0   # baseline columns: std = FREE_SCALE × activation scale → ~unpenalised


def _prepare(Xtr, Xte, free: int = 0, scale: bool = False):
    """Centre every column on the training split; activations are otherwise left at the
    model's own scale (no clipping, no per-channel rescaling), as in standard probing practice.

    Returns (Xtr, Xte, s2) with s2 = mean training variance of the non-free columns. The
    penalty grids are expressed relative to s2 (ridge: α·s2, logistic: C/s2). That only places
    the grid where the data are: a probe on X with penalty α·s2 makes exactly the same
    predictions as a probe on X/√s2 with penalty α, so the per-channel scales the model chose
    are untouched, and layers whose activations differ in overall size get comparable grids.

    ``scale=True`` additionally z-scores each non-free column (s2 is then 1). Use it only for
    hand-built feature sets with mixed units (e.g. the k-hop profile baseline), never for
    activations. The first ``free`` columns (stacked baseline predictions) are z-scored and
    multiplied by FREE_SCALE·√s2 so that the penalty on them is negligible.
    """
    Xtr = np.array(Xtr, dtype=np.float64, copy=True)
    Xte = np.array(Xte, dtype=np.float64, copy=True)
    mu = Xtr.mean(0)
    Xtr -= mu
    Xte -= mu
    sd = Xtr.std(0)
    sd[sd < 1e-8] = 1.0
    if scale:
        Xtr[:, free:] /= sd[free:]
        Xte[:, free:] /= sd[free:]
        s2 = 1.0
    else:
        v = Xtr[:, free:].var(0)
        s2 = float(v.mean()) if v.size and v.mean() > 1e-12 else 1.0
    if free:
        f = FREE_SCALE * np.sqrt(s2) / sd[:free]
        Xtr[:, :free] *= f
        Xte[:, :free] *= f
    return Xtr, Xte, s2


def task_kind(y: np.ndarray) -> str:
    y = np.asarray(y)
    u = np.unique(y[~np.isnan(y)] if y.dtype.kind == "f" else y)
    if len(u) == 2:
        return "binary"
    if y.dtype.kind in "biu" and len(u) <= 20:
        return "multiclass"
    return "regression"


def score(y, p, kind: str, w=None) -> float:
    """R² (regression), AUROC (binary; p = P(class 1)), balanced accuracy (multiclass; p = labels),
    Spearman ρ ("rank"; unweighted — for scoring a pooled regression fit inside a subgroup, where
    R² is distorted by the pooled intercept but the ranking is still meaningful)."""
    if kind == "rank":
        if len(np.unique(y)) < 2 or len(np.unique(p)) < 2:
            return float("nan")
        return float(spearmanr(y, p).correlation)
    if kind == "regression":
        return float(r2_score(y, p, sample_weight=w))
    if kind == "binary":
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p, sample_weight=w))
    return float(balanced_accuracy_score(y, p, sample_weight=w))


# ----------------------------------------------------------------------------- linear probes

def _ridge_path(Xtr, ytr, wtr, Xte, alphas):
    """Weighted ridge with an unpenalised intercept, for every alpha via one eigendecomposition.

    X and y are centred by their *weighted* means (the weighted-LS intercept), so the penalised
    coefficients only model within-sample variation. Returns pred_te [A, n_te].
    """
    xm = np.average(Xtr, axis=0, weights=wtr)
    ym = np.average(ytr, weights=wtr)
    sw = np.sqrt(wtr)[:, None]
    Xw = (Xtr - xm) * sw
    yw = (ytr - ym) * sw[:, 0]
    lam, V = np.linalg.eigh(Xw.T @ Xw)
    lam = np.clip(lam, 0, None)
    z = V.T @ (Xw.T @ yw)
    XV_te = (Xte - xm) @ V
    return np.asarray([XV_te @ (z / (lam + a)) + ym for a in alphas])


def _ridge_fit_predict(Xtr, ytr, wtr, gtr, Xte, alphas=ALPHAS, seed=0, n_inner=3, free_cols=0,
                       scale=False):
    """Ridge with alpha chosen by grouped inner CV on the training fold (held-out weighted MSE).

    GCV is unreliable here: with heavy-tailed activations (notably random-init controls) it
    picks small alphas whose weights extrapolate wildly on held-out neurons. Inputs are raw;
    centring is redone inside every split and ``alphas`` are relative to the split's mean
    feature variance (see ``_prepare``). Returns (pred_te, relative alpha).
    """
    mse = np.zeros(len(alphas))
    for itr, iva in folds(gtr, n_inner, seed + 7):
        A, Bv, s2 = _prepare(Xtr[itr], Xtr[iva], free_cols, scale)
        preds = _ridge_path(A, ytr[itr], wtr[itr], Bv, alphas * s2)
        mse += np.average((preds - ytr[iva][None, :]) ** 2, axis=1, weights=wtr[iva])
    k = int(np.argmin(mse))
    A, B, s2 = _prepare(Xtr, Xte, free_cols, scale)
    return _ridge_path(A, ytr, wtr, B, alphas[k:k + 1] * s2)[0], float(alphas[k])


def _logit_fit_predict(Xtr, ytr, wtr, gtr, Xte, kind, Cs=CS, seed=0, n_inner=3, free_cols=0, max_iter=500,
                       scale=False):
    """Logistic probe with C chosen by grouped inner CV; first ``free_cols`` columns ~unpenalised.
    ``Cs`` are relative to the mean feature variance (C/s2, see ``_prepare``). If no inner split
    can be scored (e.g. single-class validation folds) C falls back to 1.0."""
    def pred(m, B):
        return m.predict_proba(B)[:, 1] if kind == "binary" else m.predict(B)

    inner = folds(gtr, n_inner, seed + 7)
    best, bestC = -np.inf, None
    for C in Cs:
        sc = []
        for itr, iva in inner:
            if len(np.unique(ytr[itr])) < 2:
                continue
            A, B, s2 = _prepare(Xtr[itr], Xtr[iva], free_cols, scale)
            m = LogisticRegression(C=C / s2, max_iter=max_iter).fit(A, ytr[itr], sample_weight=wtr[itr])
            sc.append(score(ytr[iva], pred(m, B), kind, wtr[iva]))
        sc = np.asarray(sc, dtype=float)
        if np.isfinite(sc).any() and np.nanmean(sc) > best:
            best, bestC = float(np.nanmean(sc)), C
    if bestC is None:
        warnings.warn("logistic probe: no scorable inner split; using C=1.0")
        bestC = 1.0
    A, B, s2 = _prepare(Xtr, Xte, free_cols, scale)
    m = LogisticRegression(C=bestC / s2, max_iter=max_iter).fit(A, ytr, sample_weight=wtr)
    return pred(m, B), float(bestC)


@dataclass
class ProbeResult:
    kind: str
    oof: np.ndarray                 # OOF prediction (regression value / P(1) / class label)
    score: float                    # pooled OOF score (neuron-weighted)
    fold_scores: list = field(default_factory=list)
    params: list = field(default_factory=list)


def ridge_cv(X, y, groups, weights=None, alphas=ALPHAS, n_splits=5, seed=0, X_eval=None,
             scale=False) -> ProbeResult:
    """Grouped-CV ridge. With ``X_eval`` (same rows, e.g. activations under an ablation), each
    fold's probe is fit on ``X`` and *scored on* ``X_eval`` — the fixed-probe transfer test.
    ``scale``: z-score columns (mixed-unit feature sets only; see ``_prepare``)."""
    X = np.asarray(X, dtype=np.float64)
    Xe = X if X_eval is None else np.asarray(X_eval, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    w = neuron_weights(groups) if weights is None else np.asarray(weights)
    oof = np.zeros(len(y))
    fs, ps = [], []
    for tr, te in folds(groups, n_splits, seed):
        oof[te], a = _ridge_fit_predict(X[tr], y[tr], w[tr], groups[tr], Xe[te], alphas, seed,
                                          scale=scale)
        fs.append(score(y[te], oof[te], "regression", w[te]))
        ps.append(a)
    return ProbeResult("regression", oof, score(y, oof, "regression", w), fs, ps)


def logistic_cv(X, y, groups, weights=None, Cs=CS, n_splits=5, seed=0, max_iter=300,
                X_eval=None, scale=False) -> ProbeResult:
    """L2 logistic probe; C picked by grouped inner CV inside each training fold.
    ``X_eval``, ``scale``: see ``ridge_cv``."""
    X = np.asarray(X, dtype=np.float64)
    Xe = X if X_eval is None else np.asarray(X_eval, dtype=np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)
    kind = "binary" if len(np.unique(y)) == 2 else "multiclass"
    w = neuron_weights(groups) if weights is None else np.asarray(weights)
    oof = np.zeros(len(y), dtype=np.float64 if kind == "binary" else y.dtype)
    fs, ps = [], []
    for tr, te in folds(groups, n_splits, seed):
        oof[te], C = _logit_fit_predict(X[tr], y[tr], w[tr], groups[tr], Xe[te], kind, Cs, seed,
                                        max_iter=max_iter, scale=scale)
        fs.append(score(y[te], oof[te], kind, w[te]))
        ps.append(C)
    return ProbeResult(kind, oof, score(y, oof, kind, w), fs, ps)


def linear_probe(X, y, groups, weights=None, kind=None, **kw) -> ProbeResult:
    kind = kind or task_kind(np.asarray(y))
    return ridge_cv(X, y, groups, weights, **kw) if kind == "regression" else logistic_cv(X, y, groups, weights, **kw)


# ----------------------------------------------------------------------------- beyond-observables

# Candidate models for the observables baseline. The best one is picked per outer fold by
# grouped inner CV, so the baseline is neither under-fit (linear only) nor allowed to memorise
# neurons (neuron-constant targets + row-level features make a free GBM overfit badly, which
# would *inflate* Δ in the activations' favour).
OBS_MODELS = {
    "gbm_small": dict(learning_rate=0.05, max_iter=100, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0),
    "gbm_mid": dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0),
    "gbm_big": dict(learning_rate=0.1, max_iter=200, max_leaf_nodes=31, min_samples_leaf=20),
    "linear": None,
}


def _gbm(kind, seed, name="gbm_big"):
    kw = OBS_MODELS[name]
    if kind == "regression":
        return HistGradientBoostingRegressor(random_state=seed, **kw)
    return HistGradientBoostingClassifier(random_state=seed, **kw)


class _LinearObs:
    """Linear model on standardised observables, same interface as the GBMs."""

    def __init__(self, kind):
        self.kind = kind

    def fit(self, X, y, sample_weight=None):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd < 1e-8] = 1.0
        A = (X - self.mu) / self.sd
        if self.kind == "regression":
            from sklearn.linear_model import Ridge
            self.m = Ridge(alpha=1.0).fit(A, y, sample_weight=sample_weight)
        else:
            self.m = LogisticRegression(C=1.0, max_iter=500).fit(A, y, sample_weight=sample_weight)
        self.classes_ = getattr(self.m, "classes_", None)
        return self

    def _A(self, X):
        return (X - self.mu) / self.sd

    def predict(self, X):
        return self.m.predict(self._A(X))

    def predict_proba(self, X):
        return self.m.predict_proba(self._A(X))


def _obs_model(kind, seed, name):
    return _LinearObs(kind) if OBS_MODELS[name] is None else _gbm(kind, seed, name)


def _aligned_proba(m, A, classes):
    """predict_proba with columns mapped onto the global ``classes`` (a class missing from this
    model's training data gets probability ~0), so stacking features never misalign."""
    P = np.full((len(A), len(classes)), 1e-6)
    pm = m.predict_proba(A)
    pos = np.searchsorted(classes, m.classes_)
    P[:, pos] = pm
    return np.clip(P / P.sum(1, keepdims=True), 1e-6, 1 - 1e-6)


def _obs_feat(m, A, kind, classes):
    """Stacking feature: value (regression), logit (binary), log-probs (multiclass)."""
    if kind == "regression":
        return m.predict(A)[:, None]
    p = _aligned_proba(m, A, classes)
    if kind == "binary":
        return np.log(p[:, 1] / p[:, 0])[:, None]
    return np.log(p)


def _obs_pred(m, A, kind, classes):
    """Prediction in score() format: value / P(class 1) / label."""
    if kind == "regression":
        return m.predict(A)
    p = _aligned_proba(m, A, classes)
    return p[:, 1] if kind == "binary" else classes[p.argmax(1)]


def _stack_fit_predict(A, y, w, g, B, kind, free_cols):
    if kind == "regression":
        return _ridge_fit_predict(A, y.astype(float), w, g, B, free_cols=free_cols)[0]
    return _logit_fit_predict(A, y, w, g, B, kind, free_cols=free_cols)[0]


@dataclass
class ObsStack:
    """Cross-fitted observables-model predictions, reusable across activation sites.

    ``oof_obs`` / ``score_obs`` are the *stacked* baseline: the same second-stage linear model
    used with activations, fit on ô alone. Comparing [ô, X] against [ô] (not against the raw
    GBM output) means Δ cannot credit X for merely recalibrating ô.
    """
    kind: str
    y: np.ndarray
    groups: np.ndarray
    w: np.ndarray
    folds: list
    o_tr: list          # per fold: in-fold OOF obs feature for training rows [n_tr, c]
    o_te: list          # per fold: obs feature for test rows [n_te, c]
    oof_obs: np.ndarray
    score_obs: float
    oof_obs_raw: np.ndarray = None
    score_obs_raw: float = float("nan")
    chosen: list = field(default_factory=list)   # obs model picked per outer fold


def obs_stack(obs, y, groups, weights=None, kind=None, n_splits=5, seed=0, models=None) -> ObsStack:
    obs = np.asarray(obs, dtype=np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)
    kind = kind or task_kind(y)
    w = neuron_weights(groups) if weights is None else np.asarray(weights)
    classes = np.unique(y) if kind != "regression" else None
    models = list(models or OBS_MODELS)
    fl = folds(groups, n_splits, seed)
    raw = np.zeros(len(y), dtype=np.float64 if kind != "multiclass" else y.dtype)
    stacked = raw.copy()
    o_trs, o_tes, chosen = [], [], []
    for tr, te in fl:
        inner = folds(groups[tr], 4, seed + 1)
        best = None
        for name in models:
            o_tr, pred = None, np.zeros(len(tr), dtype=raw.dtype)
            for itr, iva in inner:
                m = _obs_model(kind, seed, name).fit(obs[tr][itr], y[tr][itr], sample_weight=w[tr][itr])
                f = _obs_feat(m, obs[tr][iva], kind, classes)
                if o_tr is None:
                    o_tr = np.zeros((len(tr), f.shape[1]))
                o_tr[iva] = f
                pred[iva] = _obs_pred(m, obs[tr][iva], kind, classes)
            sc = score(y[tr], pred, kind, w[tr])
            if best is None or (np.isfinite(sc) and sc > best[0]):
                best = (sc if np.isfinite(sc) else -np.inf, name, o_tr)
        _, name, o_tr = best
        m = _obs_model(kind, seed, name).fit(obs[tr], y[tr], sample_weight=w[tr])
        o_te = _obs_feat(m, obs[te], kind, classes)
        o_trs.append(o_tr)
        o_tes.append(o_te)
        raw[te] = _obs_pred(m, obs[te], kind, classes)
        stacked[te] = _stack_fit_predict(o_tr, y[tr], w[tr], groups[tr], o_te, kind, o_tr.shape[1])
        chosen.append(name)
    return ObsStack(kind, y, groups, w, fl, o_trs, o_tes, stacked, score(y, stacked, kind, w),
                    raw, score(y, raw, kind, w), chosen)


def incremental_with(stack: ObsStack, X) -> dict:
    """Linear probe on [ô, X] per fold of ``stack``; Δ against the same probe on [ô] alone.

    ô enters (almost) unpenalised, so the stacked probe can always reproduce the
    baseline; Δ < 0 only reflects genuine overfitting to X.
    """
    X = np.asarray(X, dtype=np.float64)
    y, w, kind = stack.y, stack.w, stack.kind
    full = np.zeros_like(stack.oof_obs)
    for (tr, te), o_tr, o_te in zip(stack.folds, stack.o_tr, stack.o_te):
        full[te] = _stack_fit_predict(np.hstack([o_tr, X[tr]]), y[tr], w[tr], stack.groups[tr],
                                      np.hstack([o_te, X[te]]), kind, o_tr.shape[1])
    s_full = score(y, full, kind, w)
    return dict(kind=kind, oof_obs=stack.oof_obs, oof_full=full, score_obs=stack.score_obs,
                score_full=s_full, delta=s_full - stack.score_obs)


def incremental(obs, X, y, groups, weights=None, kind=None, n_splits=5, seed=0) -> dict:
    """Does the activation X add *linear* information about y beyond a nonlinear model of obs?

    Cross-fitted stacking. Per outer fold: a CV-selected observables model gives an in-fold OOF
    stacking feature ô for training rows and a full-fold feature for test rows; a linear probe
    on [ô, X] is compared with the same probe on [ô] alone (both scored out of fold).
    When probing many sites against the same observables, call ``obs_stack`` once and
    ``incremental_with`` per site.
    """
    return incremental_with(obs_stack(obs, y, groups, weights, kind, n_splits, seed), X)


def residualize(y, obs, groups, n_splits=5, seed=0) -> np.ndarray:
    """Out-of-fold residual of y after a GBM on obs (regression targets)."""
    y = np.asarray(y, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    groups = np.asarray(groups)
    r = np.zeros_like(y)
    w = neuron_weights(groups)
    for tr, te in folds(groups, n_splits, seed):
        m = _gbm("regression", seed).fit(obs[tr], y[tr], sample_weight=w[tr])
        r[te] = y[te] - m.predict(obs[te])
    return r


# ----------------------------------------------------------------------------- controls

def control_task(y, keys, seed=0) -> np.ndarray:
    """Hewitt & Liang control task: every distinct ``keys`` combination gets a label drawn
    from the empirical distribution of y. A probe that scores well here is memorising types."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    keys = np.asarray(keys)
    if keys.ndim == 1:
        keys = keys[:, None]
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    lab = rng.choice(y, size=inv.max() + 1, replace=True)
    return lab[inv.ravel()]


def shuffle_within(y, groups, seed=0) -> np.ndarray:
    """Permute y within each neuron: keeps neuron-level information, destroys node-level."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).copy()
    for g in np.unique(groups):
        idx = np.nonzero(groups == g)[0]
        y[idx] = y[rng.permutation(idx)]
    return y


# ----------------------------------------------------------------------------- uncertainty

def _q(vals, alpha):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def bootstrap_ci(y, p, groups, kind, weights=None, B=1000, seed=0, alpha=0.05) -> tuple[float, float, float]:
    """Neuron-resampled CI of score(y, p) over fixed OOF predictions (does not include refit
    variance, so it is somewhat narrow). Returns (point, lo, hi)."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    w = neuron_weights(groups) if weights is None else np.asarray(weights)
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_of = [np.nonzero(inv == k)[0] for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_of[k] for k in pick])
        vals.append(score(y[idx], p[idx], kind, w[idx]))
    return (score(y, p, kind, w), *_q(vals, alpha))


def paired_bootstrap(y, p_a, p_b, groups, kind, weights=None, B=1000, seed=0, alpha=0.05):
    """CI of score(p_b) - score(p_a) on the same rows, resampling neurons."""
    y, p_a, p_b, groups = np.asarray(y), np.asarray(p_a), np.asarray(p_b), np.asarray(groups)
    w = neuron_weights(groups) if weights is None else np.asarray(weights)
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_of = [np.nonzero(inv == k)[0] for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_of[k] for k in pick])
        d.append(score(y[idx], p_b[idx], kind, w[idx]) - score(y[idx], p_a[idx], kind, w[idx]))
    point = score(y, p_b, kind, w) - score(y, p_a, kind, w)
    return (point, *_q(d, alpha))


# ----------------------------------------------------------------------------- directions

def mass_mean(X, y, strata=None, weights=None) -> dict:
    """Stratified difference of class means, Σ_s w_s (μ⁺_s − μ⁻_s) / Σ w_s, w_s = min(n⁺, n⁻).

    Stratifying (e.g. by depth × t) keeps the direction from absorbing confounds that differ
    between the classes. Returns the raw vector ``v``, its norm and the unit vector.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    strata = np.zeros(len(y), dtype=int) if strata is None else np.asarray(strata)
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    acc, tot = np.zeros(X.shape[1]), 0.0
    for s in np.unique(strata):
        m = strata == s
        pos, neg = m & y, m & ~y
        n = min(pos.sum(), neg.sum())
        if n == 0:
            continue
        mu_p = np.average(X[pos], axis=0, weights=w[pos])
        mu_n = np.average(X[neg], axis=0, weights=w[neg])
        acc += n * (mu_p - mu_n)
        tot += n
    if tot == 0:
        raise ValueError("No stratum contains both classes.")
    v = acc / tot
    nrm = float(np.linalg.norm(v))
    return dict(v=v, norm=nrm, unit=v / max(nrm, 1e-12), n_pairs=tot)


def orthogonalize(v, basis) -> np.ndarray:
    """Remove the span of ``basis`` (list of vectors) from v."""
    B = np.stack([np.asarray(b, dtype=np.float64) for b in basis], axis=1)
    Q, _ = np.linalg.qr(B)
    return v - Q @ (Q.T @ v)


@dataclass
class Leace:
    """LEACE eraser (Belrose et al. 2023): least-squares-optimal removal of all linear
    information about Z from X. ``apply`` works on numpy or torch inputs of shape [n, d]."""
    mean: np.ndarray
    P: np.ndarray  # d×d affine projection: x' = x - P (x - mean)

    def apply(self, X):
        try:
            import torch as th
            if isinstance(X, th.Tensor):
                P = th.as_tensor(self.P, dtype=X.dtype, device=X.device)
                mu = th.as_tensor(self.mean, dtype=X.dtype, device=X.device)
                return X - (X - mu) @ P.T
        except ImportError:  # pragma: no cover
            pass
        return X - (X - self.mean) @ self.P.T


def leace_fit(X, Z, eps=1e-6) -> Leace:
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    mu = X.mean(0)
    Xc, Zc = X - mu, Z - Z.mean(0)
    n = len(X)
    Sxx = Xc.T @ Xc / n
    Sxz = Xc.T @ Zc / n
    lam, U = np.linalg.eigh(Sxx)
    keep = lam > eps * lam.max()
    W = U[:, keep] @ np.diag(lam[keep] ** -0.5) @ U[:, keep].T       # whitening
    Winv = U[:, keep] @ np.diag(lam[keep] ** 0.5) @ U[:, keep].T
    M = W @ Sxz
    Q, s, _ = np.linalg.svd(M, full_matrices=False)
    Q = Q[:, s > eps * max(s.max(), eps)]
    P = Winv @ (Q @ Q.T) @ W
    return Leace(mean=mu, P=P)


def spearman(y, p) -> float:
    return float(spearmanr(y, p).correlation)
