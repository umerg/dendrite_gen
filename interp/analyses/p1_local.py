"""P1 — local structure sanity probes and the message-passing depth diagonal.

Questions
  * Is basic local structure linearly decodable (depth, #visible children, edge length,
    branch angle, frontier flag), and how does that build up across layers?
  * Depth diagonal: block i of a pure message-passing net sees ≤ i+1 hops, so exact depth of
    a node deeper than that should not be decodable at block i. Decoding above the diagonal
    is the first sign of global routing (the ISAB global tokens).

Outputs (``--out``): p1_<split>_t<t>.csv (layer sweep, baselines),
p1_depth_diag_<split>_t<t>.csv: per true depth d × site, the AUROC with which the depth probe's
out-of-fold score separates depth d from depth d+1 (``auc_next``), plus mean |error|. AUROC is
invariant to the ridge's shrinkage toward the mean depth, which makes exact-integer accuracy
useless for deep nodes.

Example::
    python -m interp.analyses.p1_local --store /.../uncond/step_218500/ema --t 0 --split val --out results/
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import probe as P
from ..baselines import KHOP_COLS, matrix
from .common import Checkpoint, load, ordered_sites, run_units, site_probe_unit, split_tables, sweep_units, write


def run(store_root, t: float, split: str, out, B: int = 500, max_rows: int | None = None, seed: int = 0):
    st, rows = load(store_root, t, split)
    if max_rows and len(rows) > max_rows:
        nn = rows.neuron.nunique()
        keep = np.random.default_rng(seed).choice(rows.neuron.unique(), size=min(nn, max(5, int(nn * max_rows / len(rows)))), replace=False)
        rows = rows[rows.neuron.isin(keep)].reset_index(drop=True)
    sites_all = ordered_sites(st)                    # frontier/root-child rows: every site
    sites_ctx = ordered_sites(st, dense_ok=False)    # context rows include B2: sparse sites only

    front = rows[rows.is_frontier].reset_index(drop=True)
    ctx = rows[~rows.is_frontier].reset_index(drop=True)
    internal = ctx[ctx.depth < ctx.visible_depth].reset_index(drop=True)   # all children visible
    nonroot = ctx[(ctx.depth >= 2)].reset_index(drop=True)                 # has an incoming edge

    jobs = [
        ("depth/context", ctx, ctx.depth.to_numpy(float), sites_ctx),
        ("depth/frontier", front, front.depth.to_numpy(float), sites_all),
        ("n_vis_children/internal", internal, internal.n_vis_children.to_numpy(float), sites_ctx),
        ("parent_dist/context", ctx[ctx.depth >= 1], ctx[ctx.depth >= 1].parent_dist.to_numpy(float), sites_ctx),
        ("branch_angle/context", nonroot, nonroot.branch_angle.to_numpy(float), sites_ctx),
        ("own_edge/frontier", front, front.parent_dist.to_numpy(float), sites_all),
        ("is_frontier/all", rows, rows.is_frontier.to_numpy().astype(int), sites_ctx),
    ]
    path = str(st.path)
    extra = dict(t=t, split=split)
    units = []
    for tag, df, y, sites in jobs:
        df = df.reset_index(drop=True)
        ok = np.isfinite(y.astype(float))
        df, y = df[ok].reset_index(drop=True), y[ok]
        if len(df) < 50 or df.neuron.nunique() < 5:
            print(f"[P1] skip {tag}: too few rows")
            continue
        print(f"[P1] {tag}: {len(df)} rows, {df.neuron.nunique()} neurons", flush=True)
        if tag == "is_frontier/all":
            # pure pipeline check (the flag is an input channel): input + last layer suffice
            sites = [s for s in sites if s in ("in", sites[-1])]
        kind = P.task_kind(y)
        units += sweep_units(path, df, y, sites, tag=tag, B=B, extra=extra, kind=kind,
                             diag=tag.startswith("depth/"))
        # the raw-input baseline is exactly the probe at site `in`; only the k-hop profile is extra
        units.append((f"{tag}|baseline:khop", site_probe_unit,
                      dict(path=path, site="baseline:khop", rref=None, y=y, groups=df.neuron.to_numpy(),
                           kind=kind, rec=dict(tag=tag, **extra), B=B, X=matrix(df, KHOP_COLS),
                           scale=True)))

    name = f"p1_{split}_t{t:g}"
    ckpt = Checkpoint(out, name, dict(analysis="P1", store=str(store_root), t=t, split=split, B=B,
                                      max_rows=max_rows, seed=seed))
    recs, side = split_tables(run_units(units, ckpt))
    diag = side.get("diag", [])
    meta = dict(analysis="P1", store=str(store_root), t=t, split=split, store_meta=st.meta)
    write(out, name, recs, meta)
    write(out, f"p1_depth_diag_{split}_t{t:g}", diag, meta)
    return pd.DataFrame(recs), pd.DataFrame(diag)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--t", type=float, default=0.0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--max-rows", type=int, default=None)
    a = ap.parse_args(argv)
    run(a.store, a.t, a.split, a.out, B=a.B, max_rows=a.max_rows)


if __name__ == "__main__":
    main()
