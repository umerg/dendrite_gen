"""P1 — what the network encodes about the tree it can see, layer by layer.

Groups (``--groups``, default all):
  sanity  Basic local structure (depth, #visible children, edge length, branch angle, frontier
          flag) and the depth diagonal: block i of a pure message-passing net sees ≤ i+1 hops,
          so exact depth of a node deeper than that should not be decodable at block i.
  ctx     P1b, frontier rows: does the context a generated node needs (its parent's edge, angle,
          radial alignment; the grandparent's edge; #frontier siblings) arrive at it, and at
          which layer? 1–2 hops away.
  path    Path geometry: path distance to the soma, tortuosity log(path/radial), radial and axial
          position — for a frontier node's parent and for placed nodes themselves. Path distance
          integrates edge lengths along the whole lineage: it is not a function of the node's
          coordinates and needs either depth-many hops or global routing.
  morpho  Morphometric aggregates the net might build: own dendrite's visible size / reach /
          height, whole-tree size, tallest reach (frontier rows); visible Strahler order,
          visible descendants and height (placed rows).

ctx/path/morpho targets correlate with depth, so each also gets (i) Δ over a small geometry
observable set (GEOM_* below) and (ii) within-depth-bin rank scores (``bins`` table).

Outputs (``--out``): p1_<split>_t<t>.csv (layer sweep, baselines),
p1_bins_<split>_t<t>.csv: per target × site × depth bin, Spearman ρ of the probe's out-of-fold
prediction within the bin (and of the observables-only / full stack: rho_obs, rho_full),
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
from .common import (Checkpoint, Lazy, load, ordered_sites, run_units, site_probe_unit, split_tables, sweep_units,
                     write)

GROUPS = ("sanity", "ctx", "path", "morpho")
DEPTH_ONLY = ["depth"]
GEOM_FRONT = ["depth", "parent_radial", "abs_parent_axial"]   # cheap geometry of the parent
GEOM_PLACED = ["depth", "radial", "abs_axial"]
OBS_MODELS_SMALL = ("linear", "gbm_small", "gbm_mid")         # a few observables: no need for gbm_big


def new_jobs(front: pd.DataFrame, placed: pd.DataFrame, sites_all, sites_ctx, groups) -> list[tuple]:
    """(tag, rows, y, sites, obs_cols) for the ctx / path / morpho groups."""
    lg = np.log1p
    f, c = front, placed[placed.depth >= 1]
    J = []
    if "ctx" in groups:
        J += [("ctx/parent_edge/frontier", f, f.parent_edge, sites_all, DEPTH_ONLY),
              ("ctx/parent_angle/frontier", f, f.parent_angle, sites_all, DEPTH_ONLY),
              ("ctx/parent_radial_cos/frontier", f, f.parent_radial_cos, sites_all, DEPTH_ONLY),
              ("ctx/gp_edge/frontier", f, f.gp_edge, sites_all, DEPTH_ONLY),
              ("ctx/n_front_siblings/frontier", f, f.n_front_siblings, sites_all, DEPTH_ONLY)]
    if "path" in groups:
        J += [("path/parent_pathdist/frontier", f, f.parent_pathdist, sites_all, GEOM_FRONT),
              ("path/parent_log_tort/frontier", f, f.parent_log_tort, sites_all, GEOM_FRONT),
              ("path/parent_radial/frontier", f, f.parent_radial, sites_all, DEPTH_ONLY),
              ("path/parent_axial/frontier", f, f.parent_axial, sites_all, DEPTH_ONLY),
              ("path/pathdist/placed", c, c.pathdist_soma, sites_ctx, GEOM_PLACED),
              ("path/log_tort/placed", c, c.log_tort, sites_ctx, GEOM_PLACED),
              ("path/radial/placed", c, c.radial, sites_ctx, DEPTH_ONLY)]
    if "morpho" in groups:
        J += [("morpho/log_own_arm_size/frontier", f, lg(f.own_arm_vis_size), sites_all, DEPTH_ONLY),
              ("morpho/own_arm_reach/frontier", f, f.own_arm_vis_reach, sites_all, DEPTH_ONLY),
              ("morpho/own_arm_height/frontier", f, f.own_arm_vis_height, sites_all, DEPTH_ONLY),
              ("morpho/log_n_visible/frontier", f, lg(f.n_visible), sites_all, DEPTH_ONLY),
              ("morpho/tallest_reach/frontier", f, f.tallest_vis_reach, sites_all, DEPTH_ONLY),
              ("morpho/vis_strahler/placed", c, c.vis_strahler, sites_ctx, DEPTH_ONLY),
              ("morpho/log_vis_desc/placed", c, lg(c.vis_desc), sites_ctx, DEPTH_ONLY),
              ("morpho/vis_height/placed", c, c.vis_height, sites_ctx, DEPTH_ONLY)]
    return J


def run(store_root, t: float, split: str, out, B: int = 500, max_rows: int | None = None, seed: int = 0,
        groups=GROUPS):
    groups = tuple(groups)
    bad = set(groups) - set(GROUPS)
    if bad:
        raise ValueError(f"unknown P1 groups {sorted(bad)}; choose from {GROUPS}")
    st, rows = load(store_root, t, split)
    rows = rows.assign(abs_parent_axial=rows.parent_axial.abs(), abs_axial=rows.axial_coord.abs())
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
    ] if "sanity" in groups else []
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

    for tag, df, y, sites, obs_cols in new_jobs(front, ctx, sites_all, sites_ctx, groups):
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(y)
        df, y = df[ok].reset_index(drop=True), y[ok]
        if len(df) < 50 or df.neuron.nunique() < 5 or np.std(y) == 0:
            print(f"[P1] skip {tag}: too few rows")
            continue
        print(f"[P1] {tag}: {len(df)} rows, {df.neuron.nunique()} neurons, obs={obs_cols}", flush=True)
        stack = Lazy(P.obs_stack, matrix(df, obs_cols), y, df.neuron.to_numpy(), models=OBS_MODELS_SMALL)
        units += sweep_units(path, df, y, sites, tag=tag, B=B, extra=extra, kind="regression",
                             stack=stack, bins=True)

    name = f"p1_{split}_t{t:g}"
    ckpt = Checkpoint(out, name, dict(analysis="P1", store=str(store_root), t=t, split=split, B=B,
                                      max_rows=max_rows, seed=seed))
    recs, side = split_tables(run_units(units, ckpt))
    diag, bins = side.get("diag", []), side.get("bins", [])
    meta = dict(analysis="P1", store=str(store_root), t=t, split=split, groups=list(groups), store_meta=st.meta)
    write(out, name, recs, meta)
    if diag:
        write(out, f"p1_depth_diag_{split}_t{t:g}", diag, meta)
    if bins:
        write(out, f"p1_bins_{split}_t{t:g}", bins, meta)
    return pd.DataFrame(recs), pd.DataFrame(diag)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--t", type=float, default=0.0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--groups", default=",".join(GROUPS), help=f"comma-separated subset of {GROUPS}")
    a = ap.parse_args(argv)
    run(a.store, a.t, a.split, a.out, B=a.B, max_rows=a.max_rows, groups=[g for g in a.groups.split(",") if g])


if __name__ == "__main__":
    main()
