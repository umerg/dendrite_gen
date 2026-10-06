"""P3 — latent discovery: does a *partial* tree's representation already encode what kind of
neuron is being built (cell type, final TMD signature)?

Rows are reduction items (one per neuron per visible depth v). For each visible-depth bin we
probe neuron-level targets from the graph-level readouts:
  * ``G{i}``     — the 8 global tokens after block i's ISAB (flattened, 8×256)
  * ``pool_L{i}`` — mean of node features over the visible tree

Targets: cell class (7-way, balanced accuracy) and the top principal components of the final
neuron's TMD vector (conditioning filtrations; PCA fit on this split's neurons — unsupervised).

Baseline at the same v: a GBM on visible-tree observables + partial-tree morphometrics +
(optionally) the partial tree's own TMD (PCA-reduced). Δ = probe on [baseline-pred, activation]
minus baseline. Claim when Δ > 0: the final neuron's type/signature is *linearly available*
in the representation earlier than our summary of the partial tree makes it — not that the
model knows more than exists in the partial tree.

TMD arm contrast (``--tmd-input``): also probe cell class from the raw 512-d conditioning
vector itself (what the tmd arm is given).
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

V_BINS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 8), (9, 99)]


def graph_sites(st: Store, every_pool: int = 2) -> list[str]:
    g = list(st.meta["sites_graph"])
    pools = [f"pool_{s}" for s in ["in"] + [f"L{i}" for i in range(0, 12, every_pool)] + ["L11"]]
    return g + sorted(set(pools), key=lambda s: -1 if s == "pool_in" else int(s.split("L")[1]))


def graph_X(st: Store, site: str, items: np.ndarray) -> np.ndarray:
    a = st.graph_acts(site)
    a = a[items]
    return a.reshape(len(items), -1)


def run(store_root, t: float, split: str, out, config: str | None = None, n_pcs: int = 5,
        partial_tmd: bool = True, B: int = 300, sites: list[str] | None = None, tmd_input: bool = False,
        min_neurons: int = 40):
    st = Store.open(store_root, t, split)
    cfg = compose_cfg(config or st.meta["config"])
    gt, ptmd = graph_table(st, uhat_of(cfg), partial_tmd_cfg=cfg if partial_tmd else None)
    gt = gt.sort_values("item").reset_index(drop=True)
    sites = sites or graph_sites(st)

    ntmd = st.neuron_tmd()
    pca = PCA(n_components=min(n_pcs, ntmd.shape[1], len(ntmd) - 1)).fit(ntmd)
    pcs = pca.transform(ntmd)
    is_tmd = is_tmd_store(st)
    targets = {}
    cc = pd.to_numeric(gt.cell_class, errors="coerce").to_numpy(float)   # NaN rows dropped per target
    if np.isfinite(cc).any():
        targets["cell_class"] = cc
    if is_tmd:
        # The TMD arm is *given* this vector: its PCs as targets would only measure input passthrough.
        print("[P3] TMD-conditioned store: final-TMD targets skipped (input passthrough); conditioning-TMD "
              "PCs added to the baseline for cell_class", flush=True)
    else:
        for k in range(pcs.shape[1]):
            targets[f"tmd_pc{k + 1}"] = pcs[gt.neuron.to_numpy(), k]

    base = [*GRAPH_OBS, *morpho_cols(gt)]
    obs = matrix(gt, base)
    if ptmd is not None:
        pt = np.nan_to_num(ptmd)
        obs = np.hstack([obs, PCA(n_components=min(16, pt.shape[1], len(pt) - 1)).fit_transform(pt)])
    if is_tmd:
        obs = np.hstack([obs, PCA(n_components=min(64, ntmd.shape[1], len(ntmd) - 1)).fit_transform(ntmd)[gt.neuron.to_numpy()]])

    recs = []
    for lo, hi in V_BINS:
        m = (gt.visible_depth >= lo) & (gt.visible_depth <= hi)
        sub = gt[m]
        # one row per neuron per bin: keep the shallowest item of each neuron in wide bins
        sub = sub.sort_values("visible_depth").drop_duplicates("neuron")
        if sub.neuron.nunique() < min_neurons:
            continue
        idx = sub.index.to_numpy()
        groups = sub.neuron.to_numpy()
        items = sub.item.to_numpy()
        print(f"[P3] v∈[{lo},{hi}]: {len(sub)} neurons", flush=True)
        for name, yall in targets.items():
            y = yall[idx]
            kind = "multiclass" if name == "cell_class" else "regression"
            keep = np.isfinite(y)
            if name == "cell_class":
                ok_cls = [c for c, n in zip(*np.unique(y[keep], return_counts=True)) if n >= 10]
                keep &= np.isin(y, ok_cls)
                if keep.sum() < min_neurons or len(np.unique(y[keep])) < 2:
                    continue
                y = np.where(keep, y, 0).astype(int)
            yk, gk = y[keep], groups[keep]
            if name == "cell_class":
                kind = "binary" if len(np.unique(yk)) == 2 else "multiclass"
            stack = P.obs_stack(obs[idx][keep], yk, gk, kind=kind)
            extra = dict(t=t, split=split, target=name, v_lo=lo, v_hi=hi, n=int(keep.sum()))
            recs.append(dict(site="baseline:partial_tree", score=stack.score_obs, **extra))
            if tmd_input and name == "cell_class":
                r = P.linear_probe(ntmd[gt.neuron.to_numpy()[idx][keep]], yk, gk, kind=kind)
                recs.append(dict(site="baseline:tmd_input", score=r.score, **extra))
            for s in sites:
                X = graph_X(st, s, items)[keep]
                r = P.linear_probe(X, yk, gk, kind=kind)
                inc = P.incremental_with(stack, X)
                d, dlo, dhi = P.paired_bootstrap(yk, inc["oof_obs"], inc["oof_full"], gk, kind, B=B)
                pt, plo, phi = P.bootstrap_ci(yk, r.oof, gk, kind, B=B)
                recs.append(dict(site=s, score=pt, lo=plo, hi=phi, score_obs=stack.score_obs,
                                 score_full=inc["score_full"], delta=d, delta_lo=dlo, delta_hi=dhi, **extra))
            print(f"  {name}: baseline={stack.score_obs:.3f}  best={max(x['score'] for x in recs[-len(sites):]):.3f}",
                  flush=True)
    meta = dict(analysis="P3", store=str(store_root), t=t, split=split, store_meta=st.meta,
                tmd_pca_explained=pca.explained_variance_ratio_.tolist(), obs_cols=base,
                partial_tmd=ptmd is not None)
    write(out, f"p3_{split}_t{t:g}", recs, meta)
    return pd.DataFrame(recs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--t", type=float, default=0.0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="defaults to the config recorded in the store")
    ap.add_argument("--n-pcs", type=int, default=5)
    ap.add_argument("--no-partial-tmd", action="store_true")
    ap.add_argument("--tmd-input", action="store_true", help="also probe class from the raw TMD vector")
    ap.add_argument("--B", type=int, default=300)
    a = ap.parse_args(argv)
    run(a.store, a.t, a.split, a.out, config=a.config, n_pcs=a.n_pcs, partial_tmd=not a.no_partial_tmd,
        B=a.B, tmd_input=a.tmd_input)


if __name__ == "__main__":
    main()
