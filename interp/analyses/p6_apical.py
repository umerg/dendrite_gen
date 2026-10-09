"""P6 — apical identity and cross-dendrite awareness, and whether global attention carries it.

(a) Own lineage. Does a node know it is in the apical subtree? The apical flag enters the
    input only at the slot-0 soma child (ordinal one-hot). Under pure message passing it can
    reach a node at depth d only after ≥ d-1 blocks. Rows: every stored node with depth ≥ 2
    (soma children excluded — they carry the flag as input). Probe fit on all rows per site,
    AUROC reported per (site × depth). Decoding above the message-passing diagonal ⇒ global
    routing; confirm with the ablated stores.

(b) Cross-dendrite. Does a frontier node in a *basal* dendrite know about the apical (or other
    arms)? Targets: apical_vis_reach (apical's visible reach / tallest), apical_is_tallest_vis,
    other_arm_tall_vis, own_rank, apical_final_reach (the apical's *final* relative reach —
    a future quantity). Reported per bin of dAB = depth(node) + depth of the apical's
    farthest-reaching visible node (hop distance through the soma); beyond ~12 hops only the
    global tokens can carry it. Δ is over LOCAL observables + k-hop profile only.

(c) Ablation (``--ablated name=path``, stores extracted with ``--ablate``; may cover a subset of
    neurons). For each: fixed-probe transfer (fit on intact activations, score on ablated —
    did the information disappear from where the readout expects it?), refit on ablated (is
    it gone entirely?), and the teacher-forced position error ratio per dAB bin (does the
    model *need* it?). Rows are matched on (neuron, level, node).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import probe as P
from ..baselines import KHOP_COLS, OBS_LOCAL, matrix, row_table
from ..store import Store
from .common import Checkpoint, Lazy, open_store, ordered_sites, row_ref, run_units, site_depth, tmd_obs, write

DAB_BINS = [(2, 4), (5, 8), (9, 12), (13, 16), (17, 99)]
CROSS_TARGETS = ["apical_vis_reach", "apical_is_tallest_vis", "other_arm_tall_vis", "own_rank",
                 "apical_final_reach"]
KEY = ["neuron", "level", "node"]


def _bin_of(x, bins):
    out = np.full(len(x), -1)
    for k, (lo, hi) in enumerate(bins):
        out[(x >= lo) & (x <= hi)] = k
    return out


def _bin_ok(y, groups, m, kind, min_rows=30, min_neurons=10, min_class=10, min_class_neurons=5) -> bool:
    """Enough data in a bin to score it: rows, neurons, and for binary targets enough positives
    and negatives spread over enough neurons (else AUROC CIs are meaningless)."""
    if m.sum() < min_rows or len(np.unique(groups[m])) < min_neurons:
        return False
    if kind == "regression":
        return True
    ym, gm = y[m], groups[m]
    for c in np.unique(y):
        sel = ym == c
        if sel.sum() < min_class or len(np.unique(gm[sel])) < min_class_neurons:
            return False
    return len(np.unique(ym)) >= 2


def _bin_kind(kind):
    # A probe fit on all bins at once: within a bin R² is distorted by the pooled intercept, so
    # continuous targets are scored by rank correlation (Spearman) inside bins.
    return "rank" if kind == "regression" else kind


def _binned(y, p, groups, kind, bins_idx, bins, B, extra):
    recs = []
    bk = _bin_kind(kind)
    for k, (lo, hi) in enumerate(bins):
        m = bins_idx == k
        if not _bin_ok(y, groups, m, kind):
            continue
        pt, plo, phi = P.bootstrap_ci(y[m], p[m], groups[m], bk, B=B)
        recs.append(dict(bin_lo=lo, bin_hi=hi, n=int(m.sum()), metric=bk, score=pt, lo=plo, hi=phi, **extra))
    return recs


DEPTH_BINS = [(d, d) for d in range(2, 13)] + [(13, 99)]
CONT_TARGETS = ("apical_vis_reach", "own_rank", "apical_final_reach")


def _delta_bins(y, groups, kind, bidx, bins, inc, B, rec, part):
    recs = []
    for k, (lo, hi) in enumerate(bins):
        m = bidx == k
        if not _bin_ok(y, groups, m, kind):
            continue
        d, dlo, dhi = P.paired_bootstrap(y[m], inc["oof_obs"][m], inc["oof_full"][m], groups[m],
                                         _bin_kind(kind), B=B)
        recs.append(dict(bin_lo=lo, bin_hi=hi, n=int(m.sum()), metric=_bin_kind(kind), delta=d,
                         delta_lo=dlo, delta_hi=dhi, **{**rec, "part": part}))
    return recs


def baseline_unit(stack, y, groups, bidx, bins, B, rec, overall: bool) -> list[dict]:
    """Observables-only model, scored overall (optional) and per bin."""
    recs = [dict(bin_lo=bins[0][0], bin_hi=99, n=len(y), score=stack.score_obs, **rec)] if overall else []
    return recs + _binned(y, stack.oof_obs, groups, stack.kind, bidx, bins, B, rec)


def probe_unit(path, site, rref, y, groups, kind, bidx, bins, B, stack, rec, delta_part, overall: bool):
    """One site: plain probe (overall + per bin) and Δ over the stack (per bin)."""
    X = open_store(path).acts(site, rref)
    r = P.linear_probe(X, y, groups, kind=kind)
    inc = P.incremental_with(stack, X)
    recs = ([dict(bin_lo=bins[0][0], bin_hi=99, n=len(y), score=r.score, delta=inc["delta"], **rec)]
            if overall else [])
    recs += _binned(y, r.oof, groups, kind, bidx, bins, B, rec)
    return recs + _delta_bins(y, groups, kind, bidx, bins, inc, B, rec, delta_part)


def ablation_unit(path_i, path_a, site, rref_i, rref_a, y, groups, kind, bidx, B, rec):
    """Intact fit, fixed-probe transfer to ablated activations, and refit on ablated."""
    Xi = open_store(path_i).acts(site, rref_i)
    Xa = open_store(path_a).acts(site, rref_a)
    fit = P.ridge_cv if kind == "regression" else P.logistic_cv
    recs = []
    for mode, res in (("intact", fit(Xi, y, groups)),
                      ("fixed_probe", fit(Xi, y, groups, X_eval=Xa)),
                      ("refit", fit(Xa, y, groups))):
        recs += _binned(y, res.oof, groups, kind, bidx, DAB_BINS, B, {**rec, "mode": mode})
    return recs


def own_lineage_units(st, rows, t, split, B):
    df = rows[(rows.depth >= 2) & (rows.owner >= 0)].reset_index(drop=True)
    y = df.is_apical_subtree.to_numpy().astype(int)
    g = df.neuron.to_numpy()
    bidx = _bin_of(df.depth.to_numpy(), DEPTH_BINS)
    # Non-learned baseline: depth + 12-hop neighbourhood profile (apical-subtree nodes sit far
    # along -uhat, which the profile's axial columns partly reveal). Δ = what the activations
    # add beyond it.
    obs = np.hstack([df[["depth"]].to_numpy(float), matrix(df, KHOP_COLS)])
    stack = Lazy(P.obs_stack, obs, y, g, kind="binary")
    base = dict(part="a_own_lineage", target="is_apical_subtree", t=t, split=split)
    units = [("a|baseline:khop", baseline_unit,
              dict(stack=stack, y=y, groups=g, bidx=bidx, bins=DEPTH_BINS, B=B,
                   rec={**base, "site": "baseline:khop", "x": np.nan}, overall=True))]
    rref = row_ref(df)
    for s in ordered_sites(st, dense_ok=False):
        units.append((f"a|{s}", probe_unit,
                      dict(path=str(st.path), site=s, rref=rref, y=y, groups=g, kind="binary", bidx=bidx,
                           bins=DEPTH_BINS, B=B, stack=stack, rec={**base, "site": s, "x": site_depth(s)},
                           delta_part="a_own_lineage_delta", overall=True)))
    return units


def cross_rows(rows):
    df = rows[rows.is_frontier & (rows.owner >= 0) & (~rows.is_apical_subtree.astype(bool))
              & rows.dAB.notna()].reset_index(drop=True)
    return df


def _target(df, name):
    y = df[name].to_numpy()
    ok = pd.notna(y)
    y = y[ok].astype(float if name in CONT_TARGETS else int)
    return y, ok


def cross_units(st, rows, t, split, B):
    df = cross_rows(rows)
    g = df.neuron.to_numpy()
    bidx = _bin_of(df.dAB.to_numpy(), DAB_BINS)
    obs = np.hstack([matrix(df, OBS_LOCAL), matrix(df, KHOP_COLS)])
    t_o = tmd_obs(st, df)
    if t_o is not None:  # TMD arm: the conditioning vector is global info the node is given
        obs = np.hstack([obs, t_o])
    units = []
    for name in CROSS_TARGETS:
        y, ok = _target(df, name)
        if ok.sum() < 50:
            continue
        kind = P.task_kind(y)
        if kind == "binary" and len(np.unique(y)) < 2:
            continue
        gk, bk = g[ok], bidx[ok]
        stack = Lazy(P.obs_stack, obs[ok], y, gk, kind=kind)
        base = dict(part="b_cross", target=name, t=t, split=split)
        units.append((f"b|{name}|baseline:local_obs", baseline_unit,
                      dict(stack=stack, y=y, groups=gk, bidx=bk, bins=DAB_BINS, B=B,
                           rec={**base, "site": "baseline:local_obs", "x": np.nan}, overall=False)))
        rref = row_ref(df[ok])
        for s in ordered_sites(st):
            units.append((f"b|{name}|{s}", probe_unit,
                          dict(path=str(st.path), site=s, rref=rref, y=y, groups=gk, kind=kind, bidx=bk,
                               bins=DAB_BINS, B=B, stack=stack, rec={**base, "site": s, "x": site_depth(s)},
                               delta_part="b_cross_delta", overall=False)))
    return units


def ablation_units(st, rows, abl_name, abl_root, t, split, B):
    """Units for P6c plus the (cheap, computed directly) position-error records."""
    sa = Store.open(abl_root, t, split)
    df = cross_rows(rows)
    ai = sa.index[sa.index.is_frontier][KEY + ["row_dense", "row_sparse", "pos_err"]]
    m = df.merge(ai, on=KEY, how="inner", suffixes=("", "_abl"))
    if len(m) < 100:
        print(f"[P6c] {abl_name}: only {len(m)} matched rows, skipping")
        return [], []
    abl_rows = m.assign(row_dense=m.row_dense_abl, row_sparse=m.row_sparse_abl)
    g = m.neuron.to_numpy()
    bidx = _bin_of(m.dAB.to_numpy(), DAB_BINS)
    direct = []
    # does the model need the global read? teacher-forced position error, ablated / intact
    for k, (lo, hi) in enumerate(DAB_BINS):
        mk = bidx == k
        if mk.sum() < 30:
            continue
        direct.append(dict(part="c_pos_err", ablation=abl_name, bin_lo=lo, bin_hi=hi, n=int(mk.sum()),
                           intact=float(m.pos_err[mk].mean()), ablated=float(m.pos_err_abl[mk].mean()),
                           ratio=float(m.pos_err_abl[mk].mean() / max(m.pos_err[mk].mean(), 1e-12)),
                           t=t, split=split))
    units = []
    for name in CROSS_TARGETS:
        y, ok = _target(m, name)
        kind = P.task_kind(y)
        if ok.sum() < 50 or (kind == "binary" and len(np.unique(y)) < 2):
            continue
        rref_i, rref_a = row_ref(m[ok]), row_ref(abl_rows[ok])
        for s in ordered_sites(st):
            units.append((f"c|{abl_name}|{name}|{s}", ablation_unit,
                          dict(path_i=str(st.path), path_a=str(sa.path), site=s, rref_i=rref_i, rref_a=rref_a,
                               y=y, groups=g[ok], kind=kind, bidx=bidx[ok], B=B,
                               rec=dict(part="c_ablation", ablation=abl_name, target=name, site=s,
                                        x=site_depth(s), t=t, split=split))))
    return units, direct


def run(store_root, t: float, split: str, out, ablated: dict[str, str] | None = None, B: int = 300,
        parts=("a", "b", "c")):
    st = Store.open(store_root, t, split)
    if "c" in parts:  # fail fast, before the expensive parts, if an ablated store is missing
        for name, path in (ablated or {}).items():
            Store.open(path, t, split)
    rows = row_table(st)
    meta = dict(analysis="P6", store=str(store_root), t=t, split=split, store_meta=st.meta,
                ablated=ablated, obs_local=OBS_LOCAL + KHOP_COLS)
    units, direct = [], []
    if "a" in parts:
        units += own_lineage_units(st, rows, t, split, B)
    if "b" in parts:
        units += cross_units(st, rows, t, split, B)
    if "c" in parts:
        for name, path in (ablated or {}).items():
            u, d = ablation_units(st, rows, name, path, t, split, B)
            units += u
            direct += d
    out_name = f"p6_{split}_t{t:g}"
    ckpt = Checkpoint(out, out_name, dict(analysis="P6", store=str(store_root), t=t, split=split, B=B,
                                          ablated=ablated, parts=list(parts)))
    recs = run_units(units, ckpt) + direct
    write(out, out_name, recs, meta)
    return pd.DataFrame(recs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--t", type=float, default=0.0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ablated", action="append", default=[], help="name=/path/to/ablated/store (repeatable)")
    ap.add_argument("--parts", default="abc")
    ap.add_argument("--B", type=int, default=300)
    a = ap.parse_args(argv)
    abl = dict(x.split("=", 1) for x in a.ablated)
    run(a.store, a.t, a.split, a.out, ablated=abl, B=a.B, parts=tuple(a.parts))


if __name__ == "__main__":
    main()
