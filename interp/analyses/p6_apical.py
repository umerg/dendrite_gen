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
from .common import ordered_sites, site_depth, tmd_obs, write

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


def own_lineage(st, rows, t, split, B):
    df = rows[(rows.depth >= 2) & (rows.owner >= 0)].reset_index(drop=True)
    y = df.is_apical_subtree.to_numpy().astype(int)
    g = df.neuron.to_numpy()
    depth = df.depth.to_numpy()
    dbins = [(d, d) for d in range(2, 13)] + [(13, 99)]
    bidx = _bin_of(depth, dbins)
    # Non-learned baseline: depth + 12-hop neighbourhood profile (apical-subtree nodes sit far
    # along -uhat, which the profile's axial columns partly reveal). Δ = what the activations
    # add beyond it.
    obs = np.hstack([df[["depth"]].to_numpy(float), matrix(df, KHOP_COLS)])
    stack = P.obs_stack(obs, y, g, kind="binary")
    base = dict(part="a_own_lineage", target="is_apical_subtree", site="baseline:khop", x=np.nan, t=t, split=split)
    recs = [dict(bin_lo=2, bin_hi=99, n=len(y), score=stack.score_obs, **base)]
    recs += _binned(y, stack.oof_obs, g, "binary", bidx, dbins, B, base)
    print(f"[P6a] baseline:khop AUROC={stack.score_obs:.3f}", flush=True)
    for s in ordered_sites(st, dense_ok=False):
        X = st.acts(s, df)
        r = P.logistic_cv(X, y, g)
        inc = P.incremental_with(stack, X)
        extra = dict(part="a_own_lineage", target="is_apical_subtree", site=s, x=site_depth(s), t=t, split=split)
        recs.append(dict(bin_lo=2, bin_hi=99, n=len(y), score=r.score, delta=inc["delta"], **extra))
        recs += _binned(y, r.oof, g, "binary", bidx, dbins, B, extra)
        for k, (lo, hi) in enumerate(dbins):
            m = bidx == k
            if not _bin_ok(y, g, m, "binary"):
                continue
            d, dlo, dhi = P.paired_bootstrap(y[m], inc["oof_obs"][m], inc["oof_full"][m], g[m], "binary", B=B)
            recs.append(dict(bin_lo=lo, bin_hi=hi, n=int(m.sum()), delta=d, delta_lo=dlo, delta_hi=dhi,
                             **{**extra, "part": "a_own_lineage_delta"}))
        print(f"[P6a] {s:>4} AUROC={r.score:.3f}  Δ over khop={inc['delta']:+.3f}", flush=True)
    return recs


def cross_rows(rows):
    df = rows[rows.is_frontier & (rows.owner >= 0) & (~rows.is_apical_subtree.astype(bool))
              & rows.dAB.notna()].reset_index(drop=True)
    return df


def cross_dendrite(st, rows, t, split, B):
    df = cross_rows(rows)
    g = df.neuron.to_numpy()
    bidx = _bin_of(df.dAB.to_numpy(), DAB_BINS)
    obs = np.hstack([matrix(df, OBS_LOCAL), matrix(df, KHOP_COLS)])
    t_o = tmd_obs(st, df)
    if t_o is not None:  # TMD arm: the conditioning vector is global info the node is given
        obs = np.hstack([obs, t_o])
    sites = ordered_sites(st)
    recs = []
    for name in CROSS_TARGETS:
        y = df[name].to_numpy()
        ok = pd.notna(y)
        if ok.sum() < 50:
            continue
        y = y[ok].astype(float if name in ("apical_vis_reach", "own_rank", "apical_final_reach") else int)
        kind = P.task_kind(y)
        if kind == "binary" and len(np.unique(y)) < 2:
            continue
        gk, bk = g[ok], bidx[ok]
        stack = P.obs_stack(obs[ok], y, gk, kind=kind)
        recs += _binned(y, stack.oof_obs, gk, kind, bk, DAB_BINS, B,
                        dict(part="b_cross", target=name, site="baseline:local_obs", x=np.nan, t=t, split=split))
        for s in sites:
            X = st.acts(s, df[ok])
            inc = P.incremental_with(stack, X)
            r = P.linear_probe(X, y, gk, kind=kind)
            extra = dict(part="b_cross", target=name, site=s, x=site_depth(s), t=t, split=split)
            recs += _binned(y, r.oof, gk, kind, bk, DAB_BINS, B, extra)
            for k, (lo, hi) in enumerate(DAB_BINS):  # Δ over local observables per bin
                m = bk == k
                if not _bin_ok(y, gk, m, kind):
                    continue
                d, dlo, dhi = P.paired_bootstrap(y[m], inc["oof_obs"][m], inc["oof_full"][m], gk[m],
                                                 _bin_kind(kind), B=B)
                recs.append(dict(bin_lo=lo, bin_hi=hi, n=int(m.sum()), metric=_bin_kind(kind), delta=d,
                                 delta_lo=dlo, delta_hi=dhi, **{**extra, "part": "b_cross_delta"}))
        print(f"[P6b] {name}: done ({ok.sum()} rows)", flush=True)
    return recs


def ablation(st, rows, abl_name, abl_root, t, split, B):
    sa = Store.open(abl_root, t, split)
    df = cross_rows(rows)
    ai = sa.index[sa.index.is_frontier][KEY + ["row_dense", "row_sparse", "pos_err"]]
    m = df.merge(ai, on=KEY, how="inner", suffixes=("", "_abl"))
    if len(m) < 100:
        print(f"[P6c] {abl_name}: only {len(m)} matched rows, skipping")
        return []
    intact_rows = m
    abl_rows = m.assign(row_dense=m.row_dense_abl, row_sparse=m.row_sparse_abl)
    g = m.neuron.to_numpy()
    bidx = _bin_of(m.dAB.to_numpy(), DAB_BINS)
    recs = []
    # does the model need the global read? teacher-forced position error, ablated / intact
    for k, (lo, hi) in enumerate(DAB_BINS):
        mk = bidx == k
        if mk.sum() < 30:
            continue
        recs.append(dict(part="c_pos_err", ablation=abl_name, bin_lo=lo, bin_hi=hi, n=int(mk.sum()),
                         intact=float(m.pos_err[mk].mean()), ablated=float(m.pos_err_abl[mk].mean()),
                         ratio=float(m.pos_err_abl[mk].mean() / max(m.pos_err[mk].mean(), 1e-12)), t=t, split=split))
    for name in CROSS_TARGETS:
        y = m[name].to_numpy()
        ok = pd.notna(y)
        y = y[ok].astype(float if name in ("apical_vis_reach", "own_rank", "apical_final_reach") else int)
        kind = P.task_kind(y)
        if ok.sum() < 50 or (kind == "binary" and len(np.unique(y)) < 2):
            continue
        for s in ordered_sites(st):
            Xi = st.acts(s, intact_rows[ok])
            Xa = sa.acts(s, abl_rows[ok])
            fit = P.ridge_cv if kind == "regression" else P.logistic_cv
            for mode, res in (("intact", fit(Xi, y, g[ok])),
                              ("fixed_probe", fit(Xi, y, g[ok], X_eval=Xa)),
                              ("refit", fit(Xa, y, g[ok]))):
                extra = dict(part="c_ablation", ablation=abl_name, mode=mode, target=name, site=s,
                             x=site_depth(s), t=t, split=split)
                recs += _binned(y, res.oof, g[ok], kind, bidx[ok], DAB_BINS, B, extra)
        print(f"[P6c] {abl_name}/{name}: done", flush=True)
    return recs


def run(store_root, t: float, split: str, out, ablated: dict[str, str] | None = None, B: int = 300,
        parts=("a", "b", "c")):
    st = Store.open(store_root, t, split)
    if "c" in parts:  # fail fast, before the expensive parts, if an ablated store is missing
        for name, path in (ablated or {}).items():
            Store.open(path, t, split)
    rows = row_table(st)
    meta = dict(analysis="P6", store=str(store_root), t=t, split=split, store_meta=st.meta,
                ablated=ablated, obs_local=OBS_LOCAL + KHOP_COLS)
    recs = []

    def save():  # written after every part, so a later failure never loses finished parts
        write(out, f"p6_{split}_t{t:g}", recs, meta)

    if "a" in parts:
        recs += own_lineage(st, rows, t, split, B)
        save()
    if "b" in parts:
        recs += cross_dendrite(st, rows, t, split, B)
        save()
    if "c" in parts:
        for name, path in (ablated or {}).items():
            recs += ablation(st, rows, name, path, t, split, B)
            save()
    save()
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
