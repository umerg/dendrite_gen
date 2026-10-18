"""Within-tree check for soma distance: does the network know each node's own distance, or only
which kind of tree it is in?

The P1 within-depth-bin scores pool nodes from many partial trees. In deep bins most of the
spread in soma distance is between trees (compact vs stretched neurons), and a single
tree-wide summary can rank that part. Here every depth bin is scored twice from the same
out-of-fold probe predictions:

  rho_pooled   Spearman over all rows of the bin (the existing P1 ``bins`` number)
  rho_within   Spearman after subtracting each partial tree's mean from both target and
               prediction (trees = (neuron, level) snapshots with >= 2 rows in the bin).
               Anything that only knows the tree scores 0 here.

Reference rows (site ``oracle:tree_mean``): the prediction is the true per-tree mean of the
target, i.e. perfect knowledge of the tree and none of the node. Its rho_pooled is how high
the pooled score can get from tree identity alone; its rho_within is 0 by construction.

Targets: frontier rows, the parent's straight-line distance to the soma (``parent_radial``);
placed rows (depth >= 1), the node's own distance (``radial``).

Example::
    python -m interp.analyses.within_tree --store /.../ema --max-neurons 300 --out results_v2/
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import probe as P
from .common import load, ordered_sites, site_depth, write

DEPTH_BINS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 12), (13, 99)]


def bin_scores(y, pred, depth, tree, rec: dict) -> list[dict]:
    out = []
    for lo, hi in DEPTH_BINS:
        m = (depth >= lo) & (depth <= hi) & np.isfinite(y) & np.isfinite(pred)
        if m.sum() < 30:
            continue
        d = pd.DataFrame({"y": y[m], "p": pred[m], "tree": tree[m]})
        size = d.groupby("tree").y.transform("size")
        w = d[size >= 2]
        yc = w.y - w.groupby("tree").y.transform("mean")
        pc = w.p - w.groupby("tree").p.transform("mean")
        if len(w) < 30 or yc.std() == 0:
            within = np.nan
        else:   # a prediction constant within every tree carries no within-tree information
            within = spearmanr(yc, pc).correlation if pc.std() > 1e-12 else 0.0
        out.append(dict(**rec, depth_lo=lo, depth_hi=hi, n=int(m.sum()), n_trees=int(d.tree.nunique()),
                        n_within=int(len(w)), share_between=float(1 - yc.var() / d.y.var()) if d.y.var() > 0 else np.nan,
                        rho_pooled=float(spearmanr(d.y, d.p).correlation), rho_within=float(within)))
    return out


def run(store_root, t: float, split: str, out, max_neurons: int | None = 300, seed: int = 0):
    st, rows = load(store_root, t, split)
    if max_neurons and rows.neuron.nunique() > max_neurons:
        keep = np.random.default_rng(seed).choice(rows.neuron.unique(), size=max_neurons, replace=False)
        rows = rows[rows.neuron.isin(keep)].reset_index(drop=True)
    jobs = [("parent_radial/frontier", rows[rows.is_frontier], "parent_radial", ordered_sites(st)),
            ("radial/placed", rows[~rows.is_frontier & (rows.depth >= 1)], "radial", ordered_sites(st, dense_ok=False))]
    recs = []
    for tag, df, col, sites in jobs:
        df = df[df[col].notna()].reset_index(drop=True)
        y = df[col].to_numpy(float)
        depth = df.depth.to_numpy()
        tree = (df.neuron.astype(np.int64) * 1000 + df.level.astype(np.int64)).to_numpy()
        groups = df.neuron.to_numpy()
        print(f"[within-tree] {tag}: {len(df)} rows, {df.neuron.nunique()} neurons", flush=True)
        oracle = pd.Series(y).groupby(tree).transform("mean").to_numpy()
        recs += bin_scores(y, oracle, depth, tree, dict(tag=tag, site="oracle:tree_mean", x=np.nan))
        for s in sites:
            res = P.linear_probe(st.acts(s, df), y, groups, kind="regression")
            recs += bin_scores(y, res.oof, depth, tree, dict(tag=tag, site=s, x=site_depth(s)))
            r = [x for x in recs if x["site"] == s and x["tag"] == tag]
            print(f"  {s:>4}  " + "  ".join(f"{x['depth_lo']}-{x['depth_hi']}: {x['rho_pooled']:.2f}/{x['rho_within']:.2f}"
                                          for x in r), flush=True)
    name = f"within_tree_{split}_t{t:g}"
    write(out, name, recs, dict(analysis="within_tree", store=str(store_root), t=t, split=split,
                                 max_neurons=max_neurons, seed=seed))
    return pd.DataFrame(recs)


def table(df: pd.DataFrame, value: str) -> str:
    df = df.assign(bin=df.depth_lo.astype(str) + "-" + df.depth_hi.astype(str).str.replace("99", "+"))
    order = list(dict.fromkeys(df.sort_values("depth_lo").bin))
    lines = []
    for tag, g in df.groupby("tag", sort=False):
        piv = g.pivot_table(index="site", columns="bin", values=value, aggfunc="first")
        sites = sorted([s for s in piv.index if not s.startswith("oracle")], key=site_depth) + \
            [s for s in piv.index if s.startswith("oracle")]
        lines += [f"\n[{tag}] {value}", piv.loc[sites, [b for b in order if b in piv.columns]].round(2).to_string()]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--t", type=float, default=0.0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-neurons", type=int, default=300, help="random neuron subset (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    df = run(a.store, a.t, a.split, a.out, max_neurons=a.max_neurons or None, seed=a.seed)
    print(table(df, "rho_pooled"))
    print(table(df, "rho_within"))
    shares = df[df.site == "oracle:tree_mean"][["tag", "depth_lo", "depth_hi", "n", "n_trees", "share_between"]]
    print("\nshare of variance between trees\n" + shares.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
