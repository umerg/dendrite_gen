"""Steering directions for E1 S-class and E3 local growth (methodology §11b.4–5), from an existing
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
kind = growth (E3): basal frontier rows (not apical subtree) at depth ≥ 5, sites L7–L9.
  ``grow_raw`` = mass-mean of y_h2 (≥ 2 more generations) among rows that branch next, one
  stratum per dendrite (= dendrite-centred, so it cannot encode which tree or dendrite);
  ``label`` = same for branch-next (label2) over all rows; ``grow`` = grow_raw with label and the
  depth read-out projected out, rescaled to ‖grow_raw‖ (α = 1 = one natural unit). Controls for
  ``grow``: random (same norm) and onmanifold (y_h2 shuffled within each dendrite).

Example::
    python -m interp.analyses.steer_dirs --store /.../ema --kind class --sites L3,L5,L7,L9 --out results_v3/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .. import probe as P
from .common import dend_ids, load

V_BINS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 8), (9, 99)]


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


def growth_rows(rows: pd.DataFrame, dmin: int = 5) -> pd.DataFrame:
    """Basal frontier rows (not in the apical subtree) at depth ≥ dmin, with their dendrite id."""
    df = rows[rows.is_frontier & (rows.owner >= 1) & ~rows.is_apical_subtree.astype(bool)
              & (rows.depth >= dmin)].copy()
    df["dend"] = dend_ids(df)
    return df.reset_index(drop=True)


def _shuffle_within(y: np.ndarray, groups: np.ndarray, rng) -> np.ndarray:
    out = y.copy()
    for g in np.unique(groups):
        m = np.flatnonzero(groups == g)
        out[m] = y[rng.permutation(m)]
    return out


def growth_dirs(st, rows, sites, dmin: int = 5, seed: int = 0) -> dict:
    """E3 look-ahead direction, dendrite-centred (one mass-mean stratum per dendrite), with the
    branch-next and depth directions removed. α = 1 is one ‖v_grow‖."""
    df = growth_rows(rows, dmin)
    lab = df.label2.to_numpy().astype(bool)
    g2 = df[lab].reset_index(drop=True)
    y = g2.y_h2.to_numpy().astype(bool)
    w_all = 1.0 / df.groupby("neuron").node.transform("size").to_numpy(float)
    w = 1.0 / g2.groupby("neuron").node.transform("size").to_numpy(float)
    rng = np.random.default_rng(seed)
    print(f"[steer_dirs] growth: {len(df)} basal frontier rows at depth ≥ {dmin} ({df.neuron.nunique()} neurons); "
          f"{len(g2)} branching, y_h2 rate {y.mean():.2f}", flush=True)
    out = {}
    for s in sites:
        Xa = st.acts(s, df)
        Xg = Xa[lab]
        mm = P.mass_mean(Xg, y, strata=g2.dend.to_numpy(), weights=w)
        lb = P.mass_mean(Xa, lab, strata=df.dend.to_numpy(), weights=w_all)
        Xc = Xa - Xa.mean(0)
        v_depth = np.linalg.lstsq(Xc, df.depth.to_numpy(float) - df.depth.mean(), rcond=None)[0]
        nrm = mm["norm"]
        v = P.orthogonalize(mm["v"], [lb["v"], v_depth])
        v *= nrm / max(np.linalg.norm(v), 1e-12)
        r = rng.standard_normal(len(v))
        r *= nrm / np.linalg.norm(r)
        om = P.mass_mean(Xg, _shuffle_within(y, g2.dend.to_numpy(), rng), strata=g2.dend.to_numpy(), weights=w)["v"]
        om *= nrm / max(np.linalg.norm(om), 1e-12)
        out.update({f"{s}/grow/v": v, f"{s}/grow/random": r, f"{s}/grow/onmanifold": om,
                    f"{s}/grow_raw/v": mm["v"],
                    f"{s}/label/v": lb["v"] * nrm / max(lb["norm"], 1e-12),
                    f"{s}/grow/n_pairs": np.array(mm["n_pairs"]), f"{s}/label/n_pairs": np.array(lb["n_pairs"]),
                    f"{s}/grow/cos_raw_label": np.array(float(mm["v"] @ lb["v"] / max(nrm * lb["norm"], 1e-12))),
                    f"{s}/sigma": np.array(_sigma(Xa))})
        print(f"[steer_dirs] growth @ {s}: ‖v_grow‖={nrm:.3g}, ‖v_label‖={lb['norm']:.3g}, "
              f"cos(grow, label)={float(out[f'{s}/grow/cos_raw_label']):.2f}, σ={float(out[f'{s}/sigma']):.3g}", flush=True)
    return out


def run(store_root, t: float, split: str, out, kind: str, sites: list[str], seed: int = 0,
        min_neurons: int = 20) -> Path:
    st, rows = load(store_root, t, split)
    d = (class_dirs(st, sites, min_neurons=min_neurons, seed=seed) if kind == "class"
         else growth_dirs(st, rows, sites, seed=seed))
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
    ap.add_argument("--kind", required=True, choices=["class", "growth"])
    ap.add_argument("--sites", default=None, help="default: L3,L5,L7,L9 (class) / L7,L8,L9 (growth)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sites = (a.sites or ("L3,L5,L7,L9" if a.kind == "class" else "L7,L8,L9")).split(",")
    run(a.store, a.t, a.split, a.out, a.kind, sites, seed=a.seed)


if __name__ == "__main__":
    main()
