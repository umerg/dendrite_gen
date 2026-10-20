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
  dend    Dendrite-level targets (methodology §8b.6): own dendrite's visible reach and size;
          for basal frontier rows, own height rank among the dendrites, own reach minus the
          apical's (both relative to the tallest), and whether the own dendrite is taller than
          the apical. Within a snapshot these differ between dendrites, so the within-tree score
          asks whether a node knows *its own* dendrite, not just the tree.
  morpho  Visible-subtree morphometrics on internal placed nodes only (finished tips are given
          away by the input leaf flag): visible Strahler order ≥ 3, visible descendants, height.

Tree-level quantities (visible tree size, tallest reach, the apical's reach) are probed at graph
level in P5 (``tree/`` targets), not here.

Apical / basal split (on by default; ``--no-split-apical`` to skip): every ctx / path target on
frontier rows is also probed on frontier rows inside the apical dendrite (``.../frontier_apical``)
and outside it (``.../frontier_basal``), so a layer profile is not an average of two dendrite
types. ``--min-depth N`` keeps node rows at depth >= N (methodology §8b.8).

ctx/path/dend/morpho targets correlate with depth, so each also gets (i) Δ over a small
observable set (GEOM_* / depth) and (ii) within-depth-bin rank scores (``bins`` table), pooled,
within partial-tree snapshots and within dendrites (§8b.2). The sanity group's own-edge target
gets the bins table too.

Outputs (``--out``): p1_<split>_t<t>.csv (layer sweep, baselines),
p1_bins_<split>_t<t>.csv: per target × site × depth bin, Spearman ρ of the probe's out-of-fold
prediction within the bin (and of the observables-only / full stack: rho_obs, rho_full), and
the same after centring on each snapshot / dendrite (rho_tree, rho_dend, rho_tree_obs,
rho_tree_full; share_tree = between-snapshot share of the bin's variance),
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
from .common import (Checkpoint, Lazy, load, min_depth_rows, ordered_sites, run_units, site_probe_unit,
                     split_tables, sweep_units, with_min_depth, write)

GROUPS = ("sanity", "ctx", "path", "dend", "morpho")
DEPTH_ONLY = ["depth"]
GEOM_FRONT = ["depth", "parent_radial", "abs_parent_axial"]   # cheap geometry of the parent
GEOM_PLACED = ["depth", "radial", "abs_axial"]
OBS_MODELS_SMALL = ("linear", "gbm_small", "gbm_mid")         # a few observables: no need for gbm_big


def new_jobs(front: pd.DataFrame, placed: pd.DataFrame, sites_all, sites_ctx, groups) -> list[tuple]:
    """(tag, rows, y, sites, obs_cols) for the ctx / path / dend / morpho groups (methodology §8b.6).

    Tree-level targets (visible tree size, tallest reach, the apical's reach for basal rows) are
    not probed here: they are constant within a snapshot and live in P5's graph-level ``tree/``
    targets. ``dend`` holds dendrite-level targets, scored within snapshots by the bins table.
    ``morpho`` uses internal placed nodes only: for finished tips the input leaf flag already gives
    visible Strahler order 1, zero descendants and height 0 (§8b.5)."""
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
    if "dend" in groups:
        fa = f[f.owner >= 0]
        b = fa[~fa.is_apical_subtree.astype(bool) & fa.apical_vis_reach.notna()
               & (fa.tallest_vis_reach > 0)]
        own_rel = b.own_arm_vis_reach / b.tallest_vis_reach        # same normalisation as apical_vis_reach
        J += [("dend/own_arm_reach/frontier", fa, fa.own_arm_vis_reach, sites_all, DEPTH_ONLY),
              ("dend/log_own_arm_size/frontier", fa, lg(fa.own_arm_vis_size), sites_all, DEPTH_ONLY),
              ("dend/own_rank/basal", b, b.own_rank, sites_all, DEPTH_ONLY),
              ("dend/own_minus_apical/basal", b, own_rel - b.apical_vis_reach, sites_all, DEPTH_ONLY),
              ("dend/mine_taller/basal", b, (own_rel > b.apical_vis_reach).astype(int), sites_all, DEPTH_ONLY)]
    if "morpho" in groups:
        i = c[c.n_vis_children > 0]
        J += [("morpho/strahler_ge3/internal", i, (i.vis_strahler >= 3).astype(int), sites_ctx, DEPTH_ONLY),
              ("morpho/log_vis_desc/internal", i, lg(i.vis_desc), sites_ctx, DEPTH_ONLY),
              ("morpho/vis_height/internal", i, i.vis_height, sites_ctx, DEPTH_ONLY)]
    return J


def split_apical(jobs: list[tuple]) -> list[tuple]:
    """Apical / basal copies of the frontier ctx / path jobs (rows inside / outside the apical
    dendrite). The sibling count is skipped: binarised trees make it constant beyond depth 1."""
    out = []
    for tag, df, y, sites, obs in jobs:
        if not tag.endswith("/frontier") or not tag.startswith(("ctx/", "path/")) or "n_front_siblings" in tag:
            continue
        ap = df.is_apical_subtree.astype(bool).to_numpy()
        y = np.asarray(y)
        for suffix, m in (("_apical", ap), ("_basal", ~ap)):
            out.append((tag + suffix, df[m], y[m], sites, obs))
    return out


def run(store_root, t: float, split: str, out, B: int = 500, max_rows: int | None = None, seed: int = 0,
        groups=GROUPS, min_depth: int = 1, apical_split: bool = True):
    groups = tuple(groups)
    bad = set(groups) - set(GROUPS)
    if bad:
        raise ValueError(f"unknown P1 groups {sorted(bad)}; choose from {GROUPS}")
    st, rows = load(store_root, t, split)
    rows = rows.assign(abs_parent_axial=rows.parent_axial.abs(), abs_axial=rows.axial_coord.abs())
    rows = min_depth_rows(rows, min_depth)
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
                             diag=tag.startswith("depth/"), bins=tag == "own_edge/frontier")
        # the raw-input baseline is exactly the probe at site `in`; only the k-hop profile is extra
        units.append((f"{tag}|baseline:khop", site_probe_unit,
                      dict(path=path, site="baseline:khop", rref=None, y=y, groups=df.neuron.to_numpy(),
                           kind=kind, rec=dict(tag=tag, **extra), B=B, X=matrix(df, KHOP_COLS),
                           scale=True)))

    nj = new_jobs(front, ctx, sites_all, sites_ctx, groups)
    if apical_split:
        nj += split_apical(nj)
    for tag, df, y, sites, obs_cols in nj:
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(y)
        df, y = df[ok].reset_index(drop=True), y[ok]
        if len(df) < 50 or df.neuron.nunique() < 5 or np.std(y) == 0 or len(np.unique(y)) < 2:
            print(f"[P1] skip {tag}: too few rows")
            continue
        print(f"[P1] {tag}: {len(df)} rows, {df.neuron.nunique()} neurons, obs={obs_cols}", flush=True)
        kind = P.task_kind(y)
        y = y.astype(int) if kind != "regression" else y
        stack = Lazy(P.obs_stack, matrix(df, obs_cols), y, df.neuron.to_numpy(), kind=kind, models=OBS_MODELS_SMALL)
        units += sweep_units(path, df, y, sites, tag=tag, B=B, extra=extra, kind=kind,
                             stack=stack, bins=True)

    name = f"p1_{split}_t{t:g}"
    ckpt = Checkpoint(out, name, with_min_depth(dict(analysis="P1", store=str(store_root), t=t, split=split, B=B,
                                                     max_rows=max_rows, seed=seed), min_depth))
    recs, side = split_tables(run_units(units, ckpt))
    diag, bins = side.get("diag", []), side.get("bins", [])
    meta = dict(analysis="P1", store=str(store_root), t=t, split=split, groups=list(groups), store_meta=st.meta,
                min_depth=min_depth, apical_split=apical_split)
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
    ap.add_argument("--min-depth", type=int, default=1, help="keep node rows at depth >= N (§8b.8)")
    ap.add_argument("--no-split-apical", action="store_true",
                    help="skip the apical / basal copies of the frontier ctx / path targets")
    a = ap.parse_args(argv)
    run(a.store, a.t, a.split, a.out, B=a.B, max_rows=a.max_rows, groups=[g for g in a.groups.split(",") if g],
        min_depth=a.min_depth, apical_split=not a.no_split_apical)


if __name__ == "__main__":
    main()
