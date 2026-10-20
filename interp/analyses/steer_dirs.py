"""Steering directions for E1 S-class and E2 S-reach (methodology §11b.4–5), from an existing
teacher-forced store (no new extraction).

Output ``directions_<kind>_<split>_t<t>.npz``, keys per site s:
  ``s/<name>/v``           raw mass-mean vector; its norm is ‖μ⁺ − μ⁻‖, so α = 1 adds one
                           natural unit
  ``s/<name>/random``      Gaussian direction, same norm (seeded)
  ``s/<name>/onmanifold``  mass-mean between two random halves of the same neurons, rescaled
                           to the same norm
  ``s/<name>/n_pairs``     stratum-matched pair count
  ``s/sigma``              RMS of centred activations at s over frontier rows: the scale
                           against which ‖αv‖ is reported

kind = class (E1): per cell class c, one-vs-rest on the pooled node mean ``pool_L{k}``
  (what a uniform add on every node shifts), stratified by visible-depth bin, weights
  1/items-per-neuron. Names ``class<c>``.
kind = reach (E2): on the global write δ_k = A_k − L_{k−1} of basal frontier rows, top vs
  bottom tercile of the apical's absolute visible reach within strata of depth bin × own-arm
  reach tercile, orthogonalised to the least-squares depth read-out of δ_k. Name ``reach``.

Example::
    python -m interp.analyses.steer_dirs --store /.../ema --kind class --sites L3,L5,L7,L9 --out results_v3/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .. import probe as P
from .common import load

V_BINS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 8), (9, 99)]
DEPTH_BINS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 12), (13, 99)]


def _bin(x, bins) -> np.ndarray:
    out = np.full(len(x), -1)
    for k, (lo, hi) in enumerate(bins):
        out[(x >= lo) & (x <= hi)] = k
    return out


def _controls(X, y, strata, w, groups, norm, rng) -> tuple[np.ndarray, np.ndarray]:
    d = X.shape[1]
    r = rng.standard_normal(d)
    r *= norm / np.linalg.norm(r)
    uniq = np.unique(groups)
    half = set(rng.choice(uniq, len(uniq) // 2, replace=False).tolist())
    yr = np.array([g in half for g in groups])
    om = P.mass_mean(X, yr, strata=strata, weights=w)["v"]
    om *= norm / max(np.linalg.norm(om), 1e-12)
    return r, om


def _sigma(X) -> float:
    return float(np.sqrt(((X - X.mean(0)) ** 2).mean()))


def class_dirs(st, sites, min_neurons: int = 20, seed: int = 0) -> dict:
    gi = st.graph_index.sort_values("item").reset_index(drop=True)
    cls = pd.to_numeric(gi.cell_class, errors="coerce").to_numpy(float)
    vis = (gi.n_levels - gi.level).to_numpy()
    keep = np.isfinite(cls) & (vis >= 1)
    gi, cls, strata = gi[keep], cls[keep].astype(int), _bin(vis[keep], V_BINS)
    w = 1.0 / gi.groupby("neuron").item.transform("size").to_numpy(float)
    groups = gi.neuron.to_numpy()
    front = st.index[st.index.is_frontier & (st.index.row_dense >= 0)]
    rng = np.random.default_rng(seed)
    out = {}
    for s in sites:
        X = st.graph_acts(f"pool_{s}")[gi.item.to_numpy()]
        out[f"{s}/sigma"] = np.array(_sigma(st.acts(s, front.sample(min(len(front), 20000), random_state=seed))))
        for c in np.unique(cls):
            if len(np.unique(groups[cls == c])) < min_neurons:
                continue
            mm = P.mass_mean(X, cls == c, strata=strata, weights=w)
            r, om = _controls(X, cls == c, strata, w, groups, mm["norm"], rng)
            k = f"{s}/class{c}"
            out.update({f"{k}/v": mm["v"], f"{k}/random": r, f"{k}/onmanifold": om,
                        f"{k}/n_pairs": np.array(mm["n_pairs"])})
        print(f"[steer_dirs] class @ {s}: " + ", ".join(f"{k.split('/')[1]} ‖v‖={np.linalg.norm(v):.3g}"
                                                        for k, v in out.items() if k.startswith(f"{s}/") and k.endswith("/v")),
              flush=True)
    return out


def reach_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Basal frontier rows with a visible apical, labelled by the apical's absolute visible reach."""
    df = rows[rows.is_frontier & (rows.owner >= 1) & ~rows.is_apical_subtree.astype(bool)
              & rows.apical_vis_reach.notna() & (rows.tallest_vis_reach > 0)].copy()
    df["apical_vis_reach_abs"] = df.apical_vis_reach * df.tallest_vis_reach
    db = _bin(df.depth.to_numpy(), DEPTH_BINS)
    df["dbin"] = db
    df = df[db >= 0]
    df["own_t"] = df.groupby("dbin").own_arm_vis_reach.transform(lambda x: pd.qcut(x.rank(method="first"), 3, labels=False))
    df["stratum"] = df.dbin * 3 + df.own_t
    q = df.groupby("stratum").apical_vis_reach_abs.transform(lambda x: pd.qcut(x.rank(method="first"), 3, labels=False))
    df["label"] = np.where(q == 2, 1, np.where(q == 0, 0, -1))
    return df[df.label >= 0].reset_index(drop=True)


def reach_dirs(st, rows, sites, seed: int = 0) -> dict:
    df = reach_rows(rows)
    y = df.label.to_numpy().astype(bool)
    w = 1.0 / df.groupby("neuron").node.transform("size").to_numpy(float)
    groups, strata = df.neuron.to_numpy(), df.stratum.to_numpy()
    rng = np.random.default_rng(seed)
    print(f"[steer_dirs] reach: {len(df)} basal frontier rows, {df.neuron.nunique()} neurons", flush=True)
    out = {}
    for s in sites:
        k = int(s[1:])
        if not s.startswith("A"):
            raise ValueError("reach directions live on global-block sites A{k}")
        D = st.acts(s, df) - st.acts(f"L{k - 1}", df)
        mm = P.mass_mean(D, y, strata=strata, weights=w)
        Dc = D - D.mean(0)
        v_depth = np.linalg.lstsq(Dc, df.depth.to_numpy(float) - df.depth.mean(), rcond=None)[0]
        v = P.orthogonalize(mm["v"], [v_depth])
        r, om = _controls(D, y, strata, w, groups, float(np.linalg.norm(v)), rng)
        out.update({f"{s}/reach/v": v, f"{s}/reach/random": r, f"{s}/reach/onmanifold": om,
                    f"{s}/reach/n_pairs": np.array(mm["n_pairs"]), f"{s}/sigma": np.array(_sigma(st.acts(s, df))),
                    f"{s}/reach/norm_before_orth": np.array(mm["norm"])})
        print(f"[steer_dirs] reach @ {s}: ‖v‖={np.linalg.norm(v):.3g} (before depth-orth {mm['norm']:.3g}), "
              f"σ={float(out[f'{s}/sigma']):.3g}", flush=True)
    return out


def run(store_root, t: float, split: str, out, kind: str, sites: list[str], seed: int = 0,
        min_neurons: int = 20) -> Path:
    st, rows = load(store_root, t, split)
    d = (class_dirs(st, sites, min_neurons=min_neurons, seed=seed) if kind == "class"
         else reach_dirs(st, rows, sites, seed=seed))
    path = Path(out) / f"directions_{kind}_{split}_t{t:g}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **d)
    print(f"[steer_dirs] wrote {path}", flush=True)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--t", type=float, default=0.0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--kind", required=True, choices=["class", "reach"])
    ap.add_argument("--sites", default=None, help="default: L3,L5,L7,L9 (class) / A3,A5,A7,A9 (reach)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sites = (a.sites or ("L3,L5,L7,L9" if a.kind == "class" else "A3,A5,A7,A9")).split(",")
    run(a.store, a.t, a.split, a.out, a.kind, sites, seed=a.seed)


if __name__ == "__main__":
    main()
