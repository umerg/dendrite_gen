"""Stratified training diagnostics for the flow-matching path.

These metrics bridge the gap between the single-scalar training loss and the
full free-running sampling validation metrics. They are computed teacher-forced,
single-step, inside ``FlowMatchingModel.forward`` (so they share the exact targets
and frames the loss is computed against) and are cheap, no-grad reductions.

Headline idea — a per-axis **R2 skill score**:

    R2_axis = 1 - MSE_axis / prior_var_axis

Because the data-prediction model regresses the clean offset ``C_0`` and the prior
std is set to the per-axis std of ``C_0`` (so ``prior_var ~ Var(C_0)``), R2 reads as
"fraction of that axis's variance the model explains, vs predicting the mean".
R2 ~ 0 means no skill (model ignores the input/identity); R2 may be negative
(worse than the mean) — values are NOT clipped.

Caveat: averaged over all flow times ``t``, R2 conflates the easy high-``t`` regime
(input already ~= data) with the hard low-``t`` regime (input ~= pure prior, which
is where the sampling ODE launches). Hence the low-``t`` and per-bucket cuts.

The helper is pure (no model state, device/dtype-agnostic) and returns a flat
``dict[str, float]``. Every value is finite-checked; non-finite or empty-subset
metrics are simply omitted (cleaner than logging NaN to wandb).
"""

import math

import torch as th

# Flow-time bucket edges for the per-t MSE breakdown. 4 buckets over [0, 1].
_T_BUCKET_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)
# Local-frame axis order of C = (forward, sideways, axial).
_AXES = ("fwd", "side", "axial")
_SIDE = 1  # index of the sideways axis (angular placement; one-hot identity lives here)


@th.no_grad()
def compute_flow_diagnostics(
    C_pred: th.Tensor,      # [L, 3] predicted clean local-frame offset
    C_0: th.Tensor,         # [L, 3] ground-truth clean local-frame offset
    e_pred: th.Tensor,      # [L, 1] predicted expansion scalar
    e_0: th.Tensor,         # [L, 1] ground-truth expansion in {-1, +1}
    t_leaf: th.Tensor,      # [L, 1] flow time per leaf in [0, 1]
    is_root_child: th.Tensor,  # [L] bool: leaf's parent is the root (parent_idx == -1)
    prior_var,              # tuple[float, float, float]: per-axis prior variance (~ Var(C_0))
) -> dict[str, float]:
    """Return a flat dict of stratified position + expansion diagnostics (floats only).

    All metrics are finite-guarded: a key is present only if its value is finite and
    its subset/bucket is non-empty. Counts are emitted as floats so ``Trainer.log``
    (floats-only) surfaces them.
    """
    out: dict[str, float] = {}

    def put(key: str, value) -> None:
        """Add ``key`` only if it converts to a finite float."""
        v = float(value)
        if math.isfinite(v):
            out[key] = v

    L = C_pred.shape[0]
    if L == 0:
        return out

    t = t_leaf.reshape(-1)                       # [L]
    sq = (C_pred - C_0) ** 2                     # [L, 3] per-axis squared error
    tot = sq.sum(dim=1)                          # [L] total (sum-over-axes) error per leaf

    # --- Per-axis MSE + R2 skill (all t, all leaves) ---
    mse_axis = sq.mean(dim=0)                    # [3]
    for a, name in enumerate(_AXES):
        put(f"pos_mse_{name}", mse_axis[a])
        pv = prior_var[a]
        if pv > 0:
            put(f"R2_{name}", 1.0 - float(mse_axis[a]) / pv)

    # --- Headline: sideways skill near the noise end (t < 0.25), where the ODE launches ---
    low_t = t < _T_BUCKET_EDGES[1]
    if low_t.any() and prior_var[_SIDE] > 0:
        mse_side_lowt = sq[low_t, _SIDE].mean()
        put("R2_side_lowt", 1.0 - float(mse_side_lowt) / prior_var[_SIDE])

    # --- Where in flow time is the model weak: total MSE per t-bucket ---
    for b in range(len(_T_BUCKET_EDGES) - 1):
        lo, hi = _T_BUCKET_EDGES[b], _T_BUCKET_EDGES[b + 1]
        # Last bucket is inclusive of t == 1.0; others are [lo, hi).
        mask = (t >= lo) & (t < hi) if b < len(_T_BUCKET_EDGES) - 2 else (t >= lo) & (t <= hi)
        if mask.any():
            put(f"pos_mse_t{b}", tot[mask].mean())

    # --- Node-type split: root-children (k-way one-hot) vs binary-interior leaves ---
    root = is_root_child.reshape(-1).bool()
    interior = ~root
    put("num_root_leaves", float(root.sum()))
    put("num_interior_leaves", float(interior.sum()))
    if root.any():
        put("pos_mse_root", tot[root].mean())
        # Sideways skill on root-children only = the shared-frame/one-hot paradigm check.
        # Noisy per-batch (small subset); read smoothed over training.
        if prior_var[_SIDE] > 0:
            put("R2_side_root", 1.0 - float(sq[root, _SIDE].mean()) / prior_var[_SIDE])
    if interior.any():
        put("pos_mse_interior", tot[interior].mean())

    # --- L2: expansion as a classifier at the actual sampling boundary (e_pred > 0) ---
    pred = e_pred.reshape(-1) > 0.0
    true = e_0.reshape(-1) > 0.0
    put("exp_acc", (pred == true).float().mean())
    put("exp_base_rate", true.float().mean())   # fraction that should expand; anchors acc
    tp = (pred & true).sum()
    fp = (pred & ~true).sum()
    fn = (~pred & true).sum()
    prec_denom = tp + fp
    rec_denom = tp + fn
    if prec_denom > 0 and rec_denom > 0:
        prec = tp.float() / prec_denom.float()
        rec = tp.float() / rec_denom.float()
        if (prec + rec) > 0:
            put("exp_f1", 2.0 * prec * rec / (prec + rec))

    return out
