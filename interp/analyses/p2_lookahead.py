"""P2 — look-ahead: does a frontier node encode its *future* subtree beyond the 1-step label?

The loss only asks the expansion head whether a frontier node branches at the next step.
For frontier nodes that do branch (label2), we probe properties of the subtree they will
eventually grow — y_h2 (≥2 more generations), log descendants, height, Strahler order,
log path length — and report the gain of activations over a GBM on observables + the k-hop
neighbourhood profile + the model's own implied label (``incremental``). Primary analysis
at t=0 (the flowed expansion input is pure noise there); t=0.9 is the contrast.

Secondary (context rows with a hidden future): hidden_desc / hidden_height given the visible
part, again as Δ over observables.

Controls (at every site for the primary target): shuffle-within-neuron. The Hewitt–Liang
control task (``--control-task``, keyed on depth × slot) is opt-in: both keys are essentially
input features, so memorising one random label per type is trivial (AUROC ≈ 0.9+) and it does
not measure selectivity here. The informative controls are the random-init network, the
within-neuron shuffle and the Δ over observables.

Strict variant (``strict/<target>``, on by default): the same Δ with the number of levels still
to be generated added to the observables. A frontier node's future height is capped by how much
tree is left, so the default Δ mixes node-specific look-ahead with the global progress signal P5
measures; the strict Δ is look-ahead *beyond knowing how much tree remains* (the model never sees
the level, so this baseline is deliberately stronger than anything it observes).

Apical variants (on by default, methodology §8b.5–6). The input apical flag (slot 0) is chosen
on the finished neuron, and "this node is in the apical dendrite" alone predicts much of the
future subtree. ``apical/<target>`` adds apical membership to the observables (Δ only);
``nonapical/<target>`` repeats the primary probe on frontier rows outside the apical dendrite,
where the label carries no "grow deep" signal. Look-ahead that survives both is not the label.
``inapical/<target>`` repeats it on rows inside the apical dendrite (one dendrite per snapshot, so
its within-tree and within-dendrite scores coincide): with ``nonapical`` it separates what the
global read tells the apical (between trees) from local ranking inside it.

``--min-depth N`` (methodology §8b.8) keeps frontier rows at depth >= N, i.e. partial trees that
reached N levels; shallower nodes receive their dendrite-level facts within the first two hops.

Bins table (``p2_bins_<split>_t<t>.csv``): per depth bin, pooled and within-snapshot /
within-dendrite Spearman of the probe and of the observables-only / full stacks (§8b.2), for the
primary, apical and non-apical variants.

Also writes steering directions (``directions_<split>_t<t>.npz``) per site:
  v_grow  — mass-mean y_h2=1 vs 0 among label2 frontier rows, stratified by depth
  v_label — mass-mean label2=1 vs 0 among all frontier rows, stratified by depth
  v_orth  — v_grow with the span of {v_label, depth direction} removed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .. import probe as P
from ..baselines import KHOP_COLS, OBS_CONTEXT, OBS_FRONTIER, matrix
from .common import (Checkpoint, Lazy, load, min_depth_rows, ordered_sites, run_units, split_tables, sweep_units,
                     tmd_obs, with_min_depth, write)

PRIMARY = [
    ("y_h2", lambda d: d.y_h2.to_numpy().astype(int)),
    ("log_desc", lambda d: np.log1p(d.desc_final.to_numpy(float))),
    ("height", lambda d: d.height_final.to_numpy(float)),
    ("strahler", lambda d: d.strahler_final.to_numpy(float)),
    ("log_pathlen", lambda d: np.log1p(d.subtree_pathlen.to_numpy(float))),
]
SECONDARY = [
    ("hidden_desc", lambda d: np.log1p(d.hidden_desc.to_numpy(float))),
    ("hidden_height", lambda d: d.hidden_height.to_numpy(float)),
]


def directions(st, front: pd.DataFrame, sites) -> dict[str, np.ndarray]:
    """Steering directions per site (see module docstring)."""
    out = {}
    lab = front.label2.to_numpy()
    grow_rows = front[front.label2].reset_index(drop=True)
    for s in sites:
        Xa = st.acts(s, front)
        v_label = P.mass_mean(Xa, lab, strata=front.depth.to_numpy())
        Xg = st.acts(s, grow_rows)
        v_grow = P.mass_mean(Xg, grow_rows.y_h2.to_numpy(), strata=grow_rows.depth.to_numpy())
        # depth direction: least-squares readout of depth on frontier rows (centred)
        Xc = Xa - Xa.mean(0)
        dc = front.depth.to_numpy(float) - front.depth.mean()
        v_depth = np.linalg.lstsq(Xc, dc, rcond=None)[0]
        v_orth = P.orthogonalize(v_grow["v"], [v_label["v"], v_depth])
        out[f"{s}/v_grow"] = v_grow["v"]
        out[f"{s}/v_label"] = v_label["v"]
        out[f"{s}/v_orth"] = v_orth
        out[f"{s}/mean"] = Xa.mean(0)
        out[f"{s}/std"] = Xa.std(0)
        out[f"{s}/norms"] = np.array([v_grow["norm"], v_label["norm"], np.linalg.norm(v_orth)])
    return out


def run(store_root, t: float, split: str, out, B: int = 500, controls: bool = True,
        control_task: bool = False, obs_tmd: bool | None = None, strict: bool = True, apical: bool = True,
        min_depth: int = 1):
    st, rows = load(store_root, t, split)
    rows = min_depth_rows(rows, min_depth)
    sites = ordered_sites(st)
    front = rows[rows.is_frontier].reset_index(drop=True)
    grow = front[front.label2].reset_index(drop=True)
    print(f"[P2] t={t:g} {split}: {len(front)} frontier rows, {len(grow)} with label2, "
          f"{grow.neuron.nunique()} neurons", flush=True)
    groups = grow.neuron.to_numpy()
    obs_f = np.hstack([matrix(grow, OBS_FRONTIER), matrix(grow, KHOP_COLS)])
    t_f = tmd_obs(st, grow, obs_tmd)
    if t_f is not None:
        obs_f = np.hstack([obs_f, t_f])
        print("[P2] TMD-conditioned store: conditioning-TMD PCs added to observables", flush=True)
    lvl_f = grow[["level"]].to_numpy(float)
    # §8b.5: the input apical flag (slot 0, chosen on the finished neuron) is routed down the
    # lineage; "in the apical dendrite" alone explains much of the future-subtree targets.
    ap_f = grow[["is_apical_subtree"]].to_numpy(float)
    basal = ~grow.is_apical_subtree.astype(bool).to_numpy()
    grow_b = grow[basal].reset_index(drop=True)
    grow_a = grow[~basal].reset_index(drop=True)
    path = str(st.path)
    units = []
    for name, fn in PRIMARY:
        y = fn(grow)
        if len(np.unique(y)) < 2:
            continue
        ex = dict(t=t, split=split, target=name)
        units += sweep_units(path, grow, y, sites, tag=f"primary/{name}", B=B, extra=ex,
                             stack=Lazy(P.obs_stack, obs_f, y, groups), bins=True)
        if apical:
            units += sweep_units(path, grow, y, sites, tag=f"apical/{name}", B=B, extra=ex, plain=False,
                                 stack=Lazy(P.obs_stack, np.hstack([obs_f, ap_f]), y, groups), bins=True)
            yb = y[basal]
            if len(np.unique(yb)) >= 2 and grow_b.neuron.nunique() >= 5:
                units += sweep_units(path, grow_b, yb, sites, tag=f"nonapical/{name}", B=B, extra=ex,
                                     stack=Lazy(P.obs_stack, obs_f[basal], yb, groups[basal]), bins=True)
            # rows inside the apical dendrite (one dendrite per snapshot, so within-tree =
            # within-dendrite for these rows)
            ya = y[~basal]
            if len(np.unique(ya)) >= 2 and grow_a.neuron.nunique() >= 5:
                units += sweep_units(path, grow_a, ya, sites, tag=f"inapical/{name}", B=B, extra=ex,
                                     stack=Lazy(P.obs_stack, obs_f[~basal], ya, groups[~basal]), bins=True)
        if strict:
            units += sweep_units(path, grow, y, sites, tag=f"strict/{name}", B=B, extra=ex, plain=False,
                                 stack=Lazy(P.obs_stack, np.hstack([obs_f, lvl_f]), y, groups))
        if controls and name == "y_h2":
            ctl = [("shuffle_within", P.shuffle_within(y, groups))]
            if control_task:
                ctl.append(("control_task", P.control_task(y, np.c_[grow.depth, grow.slot])))
            for cname, yc in ctl:
                units += sweep_units(path, grow, yc, sites, tag=f"control/{cname}", B=min(B, 200), extra=ex)

    ctx = rows[(~rows.is_frontier) & (rows.hidden_desc > 0)].reset_index(drop=True)
    if len(ctx) > 100:
        cg = ctx.neuron.to_numpy()
        obs_c = np.hstack([matrix(ctx, OBS_CONTEXT), matrix(ctx, KHOP_COLS)])
        t_c = tmd_obs(st, ctx, obs_tmd)
        if t_c is not None:
            obs_c = np.hstack([obs_c, t_c])
        csites = ordered_sites(st, dense_ok=False)
        for name, fn in SECONDARY:
            y = fn(ctx)
            ex = dict(t=t, split=split, target=name)
            units += sweep_units(path, ctx, y, csites, tag=f"secondary/{name}", B=B, extra=ex,
                                 stack=Lazy(P.obs_stack, obs_c, y, cg))
            if strict:
                units += sweep_units(path, ctx, y, csites, tag=f"strict_secondary/{name}", B=B, extra=ex,
                                     plain=False,
                                     stack=Lazy(P.obs_stack, np.hstack([obs_c, ctx[["level"]].to_numpy(float)]), y, cg))

    out_name = f"p2_{split}_t{t:g}"
    ckpt = Checkpoint(out, out_name, with_min_depth(dict(analysis="P2", store=str(store_root), t=t, split=split, B=B,
                                                         controls=controls, control_task=control_task,
                                                         obs_tmd=t_f is not None, strict=strict, apical=apical),
                                                    min_depth))
    recs, side = split_tables(run_units(units, ckpt))
    meta = dict(analysis="P2", store=str(store_root), t=t, split=split, store_meta=st.meta,
                obs_cols=OBS_FRONTIER + KHOP_COLS, obs_tmd=t_f is not None, min_depth=min_depth)
    write(out, out_name, recs, meta)
    if side.get("bins"):
        write(out, f"p2_bins_{split}_t{t:g}", side["bins"], meta)
    dirs = directions(st, front, sites)
    np.savez(Path(out) / f"directions_p2_{split}_t{t:g}.npz", **dirs)
    return pd.DataFrame(recs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--t", type=float, default=0.0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--control-task", action="store_true")
    ap.add_argument("--no-strict", action="store_true", help="skip the +levels-remaining strict variant")
    ap.add_argument("--no-apical", action="store_true",
                    help="skip the +apical-membership baseline and the non-apical / in-apical row variants (§8b.6)")
    ap.add_argument("--min-depth", type=int, default=1, help="keep frontier rows at depth >= N (§8b.8)")
    ap.add_argument("--obs-tmd", choices=["auto", "on", "off"], default="auto",
                    help="add conditioning-TMD PCs to observables (auto: on for TMD-conditioned stores)")
    a = ap.parse_args(argv)
    run(a.store, a.t, a.split, a.out, B=a.B, controls=not a.no_controls, control_task=a.control_task,
        obs_tmd={"auto": None, "on": True, "off": False}[a.obs_tmd], strict=not a.no_strict,
        apical=not a.no_apical, min_depth=a.min_depth)


if __name__ == "__main__":
    main()
