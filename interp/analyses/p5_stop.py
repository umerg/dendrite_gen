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
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .. import probe as P
from ..baselines import GRAPH_OBS, graph_table, matrix, morpho_cols
from ..load import compose_cfg, uhat_of
from ..store import Store
from .common import is_tmd_store, write
from .p3_latent import graph_X, graph_sites


def run(store_root, t: float, split: str, out, config: str | None = None, B: int = 300,
        sites: list[str] | None = None, obs_tmd: bool | None = None):
    st = Store.open(store_root, t, split)
    cfg = compose_cfg(config or st.meta["config"])
    gt, _ = graph_table(st, uhat_of(cfg))
    gt = gt.sort_values("item").reset_index(drop=True)
    sites = sites or graph_sites(st)
    if obs_tmd is None:
        obs_tmd = is_tmd_store(st)

    cols = [*GRAPH_OBS, *morpho_cols(gt)]
    obs = matrix(gt, cols)
    if obs_tmd:
        ntmd = st.neuron_tmd()
        n = min(64, ntmd.shape[1], len(ntmd) - 1)
        obs = np.hstack([obs, PCA(n_components=n).fit_transform(ntmd)[gt.neuron.to_numpy()]])
    groups = gt.neuron.to_numpy()
    items = gt.item.to_numpy()
    targets = {
        "levels_remaining": gt.level.to_numpy(float),
        "final_depth": gt.n_levels.to_numpy(float),
        "is_last": (gt.level == 0).to_numpy().astype(int),
    }
    recs = []
    for name, y in targets.items():
        kind = P.task_kind(y)
        stack = P.obs_stack(obs, y, groups, kind=kind)
        extra = dict(t=t, split=split, target=name, n=len(y), obs_tmd=obs_tmd)
        recs.append(dict(site="baseline:observables", score=stack.score_obs, **extra))
        r_v = P.linear_probe(gt[["visible_depth"]].to_numpy(float), y, groups, kind=kind)
        recs.append(dict(site="baseline:visible_depth_only", score=r_v.score, **extra))
        print(f"[P5] {name}: obs={stack.score_obs:.3f} v-only={r_v.score:.3f}", flush=True)
        for s in sites:
            X = graph_X(st, s, items)
            r = P.linear_probe(X, y, groups, kind=kind)
            pt, lo, hi = P.bootstrap_ci(y, r.oof, groups, r.kind, B=B)
            inc = P.incremental_with(stack, X)
            d, dlo, dhi = P.paired_bootstrap(y, inc["oof_obs"], inc["oof_full"], groups, r.kind, B=B)
            recs.append(dict(site=s, score=pt, lo=lo, hi=hi, score_obs=stack.score_obs,
                             score_full=inc["score_full"], delta=d, delta_lo=dlo, delta_hi=dhi, **extra))
            print(f"  {s:>9} score={pt:.3f}  Δ={d:+.3f} [{dlo:+.3f},{dhi:+.3f}]", flush=True)
    meta = dict(analysis="P5", store=str(store_root), t=t, split=split, store_meta=st.meta, obs_cols=cols)
    write(out, f"p5_{split}_t{t:g}", recs, meta)
    return pd.DataFrame(recs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--t", type=float, default=0.0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--B", type=int, default=300)
    a = ap.parse_args(argv)
    run(a.store, a.t, a.split, a.out, config=a.config, B=a.B)


if __name__ == "__main__":
    main()
