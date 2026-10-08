"""Shared plumbing for the probe analyses: site ordering, layer sweeps, result records."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import probe as P
from ..baselines import row_table
from ..store import Store


def ordered_sites(store: Store, dense_ok: bool = True) -> list[str]:
    """Sites in network order: in, L0, A1, L1, L2, A3, L3, ... (G sites excluded)."""
    sites = list(store.meta["sites_sparse"]) + (list(store.meta["sites_dense"]) if dense_ok else [])

    def key(s):
        if s == "in":
            return (-1, 0)
        k, i = s[0], int(s[1:])
        return (i, 0 if k == "A" else 1)
    return sorted(set(sites), key=key)


def site_depth(site: str) -> float:
    """Position of a site along the network for plotting (in=-1, A_i = i-0.5, L_i = i)."""
    if site == "in":
        return -1.0
    k, i = site[0], int(site[1:])
    return i - 0.5 if k == "A" else float(i)


def load(store_root: str | Path, t: float, split: str):
    st = Store.open(store_root, t, split)
    return st, row_table(st)


def sweep(st: Store, rows: pd.DataFrame, y, sites, *, kind=None, stack=None, B=500,
          tag: str = "", extra: dict | None = None, plain: bool = True) -> list[dict]:
    """Probe ``y`` at every site. Plain probe score + CI; if ``stack`` given, also Δ over obs.
    ``plain=False`` skips the plain probe (used when only a second Δ variant is needed)."""
    groups = rows.neuron.to_numpy()
    y = np.asarray(y)
    kind = kind or P.task_kind(y)
    recs = []
    for s in sites:
        X = st.acts(s, rows)
        if plain:
            res = P.linear_probe(X, y, groups, kind=kind)
            kind_s = res.kind
            pt, lo, hi = P.bootstrap_ci(y, res.oof, groups, res.kind, B=B)
        else:
            kind_s, pt, lo, hi = stack.kind, np.nan, np.nan, np.nan
        rec = dict(tag=tag, site=s, x=site_depth(s), kind=kind_s, n=len(y),
                   n_neurons=len(np.unique(groups)), score=pt, lo=lo, hi=hi, **(extra or {}))
        if stack is not None:
            inc = P.incremental_with(stack, X)
            d, dlo, dhi = P.paired_bootstrap(y, inc["oof_obs"], inc["oof_full"], groups, kind_s, B=B)
            rec.update(score_obs=inc["score_obs"], score_full=inc["score_full"],
                       delta=d, delta_lo=dlo, delta_hi=dhi)
        recs.append(rec)
        head = f"score={pt:.3f} [{lo:.3f},{hi:.3f}]" if plain else "score=(see primary)"
        print(f"  [{tag}] {s:>4} {head}"
              + (f"  Δ={rec['delta']:+.3f} [{rec['delta_lo']:+.3f},{rec['delta_hi']:+.3f}]" if stack is not None else ""),
              flush=True)
    return recs


def baseline_records(rows, y, groups, feats: dict[str, np.ndarray], kind, tag, B=500, extra=None):
    """Score non-activation baselines (raw input, k-hop profile, observables) with the same probe."""
    recs = []
    for name, X in feats.items():
        res = P.linear_probe(X, y, groups, kind=kind)
        pt, lo, hi = P.bootstrap_ci(np.asarray(y), res.oof, groups, res.kind, B=B)
        recs.append(dict(tag=tag, site=f"baseline:{name}", x=np.nan, kind=res.kind, n=len(y),
                         n_neurons=len(np.unique(groups)), score=pt, lo=lo, hi=hi, **(extra or {})))
    return recs


def write(out: str | Path, name: str, recs: list[dict], meta: dict) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(recs)
    p = out / f"{name}.csv"
    df.to_csv(p, index=False)
    (out / f"{name}.meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return p


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def tmd_obs(st: Store, rows: pd.DataFrame, enabled: bool | None = None, n: int = 64) -> np.ndarray | None:
    """Per-row PCs of the neuron's conditioning TMD vector, for TMD-conditioned stores.

    The TMD arm is *given* a summary of the final tree's topology at every node (tmd_mlp output
    is part of the input features), so "beyond observables" claims for that arm must include
    it. ``enabled=None`` → auto from the store's config name / meta.
    """
    if enabled is None:
        enabled = is_tmd_store(st)
    if not enabled:
        return None
    from sklearn.decomposition import PCA

    ntmd = st.neuron_tmd()
    pcs = PCA(n_components=min(n, ntmd.shape[1], len(ntmd) - 1)).fit_transform(ntmd)
    return pcs[rows.neuron.to_numpy()]


def is_tmd_store(st: Store) -> bool:
    """True if the extracted model was TMD-conditioned (from the store's config)."""
    from ..load import compose_cfg
    try:
        cfg = compose_cfg(st.meta["config"])
        return int(getattr(cfg.model, "tmd_hidden_dim", 0)) > 0
    except Exception:  # noqa: BLE001 — fall back to the config name
        return "tmd" in str(st.meta.get("config", ""))
