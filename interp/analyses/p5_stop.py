"""P5 — stopping: with no size input, does the model represent how much tree is left?

Generation is depth-by-depth, so at reduction item ℓ the remaining number of expansion steps
is ℓ and the final max depth is D = v + ℓ (v = visible depth). The only stop signal the model
has is its own expansion head, so a global "levels remaining" variable would be a clean sign
of a learned progress/budget representation; a purely local stopping rule would not need one.

Targets (one row per reduction item): levels_remaining (ℓ), final_depth (D), is_last (ℓ = 0).
Observables (GBM): visible depth, visible size, frontier size and fraction, the model's own
mean predicted continuation rate over the frontier (``cont_rate``), per-arm visible stats and
partial-tree morphometrics. For TMD-conditioned stores the conditioning vector's PCs are added
to the observables (the model is *given* them, and the path filtration encodes D).

Reported per graph-level site: plain probe score and Δ over observables (with neuron CI).

Tree-level facts (``tree/<target>``, methodology §8b.6): visible tree size (log), the tallest
dendrite's reach, the apical's reach relative to the tallest, and whether the apical is the
tallest. They are constant within a snapshot, so they are probed here, one row per snapshot,
not on node rows. Their baseline is only what one attention read of the L0 node states could
average: the tree-wide means of edge length / axial step / sideways length / leaf flag and the
frontier and soma-child fractions (``baselines.tree_average_table``), plus visible depth. Δ over
that is the part of a tree fact *not* explained by a single tree-wide average. The same tree
averages are also added to the stopping targets' observables.

Per visible depth (``p5_bins_<split>_t<t>.csv``): the same out-of-fold predictions scored inside
visible-depth bins (Spearman; ``rho_obs`` is the observables-only stack in the same bin), so a
graph-level claim can be read for partial trees of a given size. ``--min-depth N`` keeps
snapshots with visible depth >= N (methodology §8b.8).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .. import probe as P
from ..baselines import GRAPH_OBS, TREE_AVG_COLS, graph_table, matrix, morpho_cols, tree_average_table
from ..load import compose_cfg, uhat_of
from ..store import Store
from .common import (Checkpoint, Lazy, is_tmd_store, run_units, site_probe_unit, split_tables, stack_score_unit,
                     with_min_depth, write)
from .p3_latent import graph_sites


def run(store_root, t: float, split: str, out, config: str | None = None, B: int = 300,
        sites: list[str] | None = None, obs_tmd: bool | None = None, min_depth: int = 1):
    st = Store.open(store_root, t, split)
    cfg = compose_cfg(config or st.meta["config"])
    gt, _ = graph_table(st, uhat_of(cfg))
    ta = tree_average_table(st, uhat_of(cfg))
    gt = gt.sort_values("item").reset_index(drop=True)
    tcols = TREE_AVG_COLS + ["log_n_visible", "tallest_reach", "apical_reach_rel", "apical_is_tallest"]
    gt = gt.merge(ta[["item"] + tcols], on="item", how="left", validate="one_to_one")
    if min_depth > 1:
        gt = gt[gt.visible_depth >= min_depth].reset_index(drop=True)
    sites = sites or graph_sites(st)
    if obs_tmd is None:
        obs_tmd = is_tmd_store(st)

    cols = [*GRAPH_OBS, *morpho_cols(gt), *TREE_AVG_COLS]
    obs = matrix(gt, cols)
    if obs_tmd:
        ntmd = st.neuron_tmd()
        n = min(64, ntmd.shape[1], len(ntmd) - 1)
        obs = np.hstack([obs, PCA(n_components=n).fit_transform(ntmd)[gt.neuron.to_numpy()]])
    groups = gt.neuron.to_numpy()
    items = gt.item.to_numpy()
    vdepth = gt.visible_depth.to_numpy()
    targets = {
        "levels_remaining": gt.level.to_numpy(float),
        "final_depth": gt.n_levels.to_numpy(float),
        "is_last": (gt.level == 0).to_numpy().astype(int),
    }
    path = str(st.path)
    units = []
    for name, y in targets.items():
        kind = P.task_kind(y)
        stack = Lazy(P.obs_stack, obs, y, groups, kind=kind)
        ex = dict(t=t, split=split, target=name, tag=name, obs_tmd=obs_tmd)
        units.append((f"{name}|baseline:observables", stack_score_unit,
                      dict(stack=stack, rec=dict(site="baseline:observables", n=len(y), **ex))))
        units.append((f"{name}|baseline:visible_depth_only", site_probe_unit,
                      dict(path=path, site="baseline:visible_depth_only", rref=None, y=y, groups=groups,
                           kind=kind, rec=ex, B=B, X=gt[["visible_depth"]].to_numpy(float))))
        for s_ in sites:
            units.append((f"{name}|{s_}", site_probe_unit,
                          dict(path=path, site=s_, rref=None, y=y, groups=groups, kind=kind, rec=ex, B=B,
                               stack=stack, items=items, bins_by=vdepth)))
    # tree-level facts vs the single-attention-read baseline (§8b.6)
    obs_ta = matrix(gt, [*TREE_AVG_COLS, "visible_depth"])
    tree_targets = {"tree/log_n_visible": gt.log_n_visible.to_numpy(float),
                    "tree/tallest_reach": gt.tallest_reach.to_numpy(float),
                    "tree/apical_reach_rel": gt.apical_reach_rel.to_numpy(float),
                    "tree/apical_is_tallest": gt.apical_is_tallest.to_numpy(float)}
    for name, yall in tree_targets.items():
        keep = np.isfinite(yall)
        if keep.sum() < 50 or len(np.unique(yall[keep])) < 2:
            continue
        y = yall[keep]
        kind = P.task_kind(y)
        y = y.astype(int) if kind != "regression" else y
        gk = groups[keep]
        stack = Lazy(P.obs_stack, obs_ta[keep], y, gk, kind=kind)
        ex = dict(t=t, split=split, target=name, tag=name, obs_tmd=False)
        units.append((f"{name}|baseline:tree_average", stack_score_unit,
                      dict(stack=stack, rec=dict(site="baseline:tree_average", n=len(y), **ex))))
        for s_ in sites:
            units.append((f"{name}|{s_}", site_probe_unit,
                          dict(path=path, site=s_, rref=None, y=y, groups=gk, kind=kind, rec=ex, B=B,
                               stack=stack, items=items, keep=keep, bins_by=vdepth[keep])))
    out_name = f"p5_{split}_t{t:g}"
    ckpt = Checkpoint(out, out_name, with_min_depth(dict(analysis="P5", store=str(store_root), t=t, split=split, B=B,
                                                         sites=sites, obs_tmd=obs_tmd), min_depth))
    recs, side = split_tables(run_units(units, ckpt))
    meta = dict(analysis="P5", store=str(store_root), t=t, split=split, store_meta=st.meta, obs_cols=cols,
                min_depth=min_depth)
    write(out, out_name, recs, meta)
    if side.get("bins"):
        write(out, f"p5_bins_{split}_t{t:g}", side["bins"], meta)
    return pd.DataFrame(recs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--t", type=float, default=0.0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--B", type=int, default=300)
    ap.add_argument("--min-depth", type=int, default=1, help="keep snapshots with visible depth >= N (§8b.8)")
    a = ap.parse_args(argv)
    run(a.store, a.t, a.split, a.out, config=a.config, B=a.B, min_depth=a.min_depth)


if __name__ == "__main__":
    main()
