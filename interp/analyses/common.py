"""Shared plumbing for the probe analyses: site ordering, layer sweeps, result records."""
from __future__ import annotations

import json
import os
import re
import warnings
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


# ----------------------------------------------------------------------------- execution engine
#
# Every analysis is a list of independent *units* (typically one target × one site). Units run
# in parallel worker processes and each finished unit is appended to a checkpoint file
# (<out>/<name>.partial.jsonl) immediately, so a killed or timed-out job resumes where it stopped.
# A checkpoint is only reused if its signature (analysis, store, t, split, B, options, code
# version) matches; otherwise it is moved aside as *.stale and the analysis starts fresh.

CKPT_VERSION = 3   # 3: within-tree / within-dendrite bins (§8b); 2: probes centre only; bump when unit results would change for the same signature


def n_jobs_default() -> int:
    return max(1, int(os.environ.get("INTERP_JOBS", "1")))


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Checkpoint:
    def __init__(self, out: str | Path, name: str, signature: dict):
        self.path = Path(out) / f"{name}.partial.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sig = json.dumps({**signature, "ckpt_version": CKPT_VERSION}, sort_keys=True, default=_jsonable)
        self.done: dict[str, list[dict]] = {}
        if self.path.exists():
            lines = self.path.read_text().splitlines()
            header = json.loads(lines[0]) if lines else {}
            if header.get("sig") == self.sig:
                for ln in lines[1:]:
                    try:
                        e = json.loads(ln)
                    except json.JSONDecodeError:   # a line cut off by a kill: ignore, recompute
                        continue
                    self.done[e["key"]] = e["recs"]
            else:
                k = 0
                while self.path.with_suffix(f".stale{k}").exists():
                    k += 1
                self.path.rename(self.path.with_suffix(f".stale{k}"))
                print(f"[ckpt] {self.path.name}: settings changed — previous checkpoint moved to .stale{k}",
                      flush=True)
        if not self.path.exists():
            with open(self.path, "w") as f:
                f.write(json.dumps({"sig": self.sig}) + "\n")
        if self.done:
            print(f"[ckpt] {self.path.name}: resuming, {len(self.done)} units already done", flush=True)

    def put(self, key: str, recs: list[dict]):
        with open(self.path, "a") as f:
            f.write(json.dumps({"key": key, "recs": recs}, default=_jsonable) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.done[key] = recs


class Lazy:
    """A value computed in the parent only if some unit that needs it still has to run
    (e.g. the observables model, skipped entirely when every site of a target is checkpointed)."""

    def __init__(self, fn, *args, **kwargs):
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self._v, self._done = None, False

    def get(self):
        if not self._done:
            self._v, self._done = self.fn(*self.args, **self.kwargs), True
        return self._v


def _call(fn, kwargs, threads):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=threads):
        return fn(**kwargs)


def fmt(rec: dict) -> str:
    tag = rec.get("tag") or "/".join(str(rec[k]) for k in ("part", "target") if k in rec and rec[k] == rec[k])
    out = f"  [{tag}] {rec.get('site', ''):>6}"
    if "bin_lo" in rec and rec.get("bin_lo") == rec.get("bin_lo"):
        out += f" bin={rec['bin_lo']}-{rec.get('bin_hi')}"
    sc = rec.get("score", np.nan)
    if sc == sc:
        out += f" score={sc:.3f}"
        if rec.get("lo") is not None and rec["lo"] == rec["lo"]:
            out += f" [{rec['lo']:.3f},{rec['hi']:.3f}]"
    if rec.get("score_obs") is not None and rec["score_obs"] == rec["score_obs"]:
        out += f"  obs={rec['score_obs']:.3f}"
    if "delta" in rec and rec["delta"] == rec["delta"]:
        out += f"  Δ={rec['delta']:+.3f}"
        if rec.get("delta_lo") is not None:
            out += f" [{rec['delta_lo']:+.3f},{rec['delta_hi']:+.3f}]"
    return out


def run_units(units: list[tuple], ckpt: Checkpoint, n_jobs: int | None = None) -> list[dict]:
    """Run ``units`` = [(key, fn, kwargs), ...]; kwargs values may be ``Lazy``. Returns all records in
    unit order. Finished units come from the checkpoint; the rest run in parallel processes."""
    n_jobs = n_jobs or n_jobs_default()
    todo = [(k, fn, kw) for k, fn, kw in units if k not in ckpt.done]
    if todo:
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        threads = max(1, cpus // n_jobs)
        resolved = [(k, fn, {a: (v.get() if isinstance(v, Lazy) else v) for a, v in kw.items()})
                    for k, fn, kw in todo]
        if n_jobs == 1:
            gen = ((k, _call(fn, kw, threads)) for k, fn, kw in resolved)
        else:
            from joblib import Parallel, delayed

            def job(k, fn, kw):
                return k, _call(fn, kw, threads)
            gen = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
                delayed(job)(k, fn, kw) for k, fn, kw in resolved)
        for k, recs in gen:
            ckpt.put(k, recs)
            for r in recs:
                if r.get("_table") is None:
                    print(fmt(r), flush=True)
    out = []
    for k, _, _ in units:
        out += ckpt.done.get(k, [])
    return out


_STORES: dict[str, Store] = {}


def open_store(path: str) -> Store:
    """Per-process cached light store (workers receive only the path)."""
    if path not in _STORES:
        _STORES[path] = Store.light(path)
    return _STORES[path]


def row_ref(rows: pd.DataFrame) -> pd.DataFrame:
    """The only part of a row table a worker needs to fetch activations."""
    return rows[["row_dense", "row_sparse"]].reset_index(drop=True)


DEPTH_BINS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 12), (13, 99)]


def _within(y, p, grp):
    """Spearman of group-centred target vs group-centred prediction (methodology §8b.2), over
    rows whose group has >= 2 rows. A prediction constant within every group scores 0."""
    d = pd.DataFrame({"y": y, "p": p, "g": grp})
    d = d[d.groupby("g").y.transform("size") >= 2]
    if len(d) < 30:
        return np.nan, np.nan
    yc = d.y - d.groupby("g").y.transform("mean")
    pc = d.p - d.groupby("g").p.transform("mean")
    if yc.std() == 0:
        return np.nan, np.nan
    share = float(1 - yc.var() / y.var()) if np.var(y) > 0 else np.nan
    return (P.spearman(yc, pc) if pc.std() > 1e-12 else 0.0), share


def _bin_records(y, oof, groups, bins_by, tag, site, rec, oof_obs=None, oof_full=None,
                 tree=None, dend=None) -> list[dict]:
    """Within-depth-bin Spearman of out-of-fold predictions (P1 ``bins`` table). Ranking inside a
    bin removes what depth alone explains; with a stack the observables-only and full-stack
    predictions are scored the same way (their difference is a within-bin Δ). With ``tree`` /
    ``dend`` ids (partial-tree snapshot, primary dendrite within it) each bin is also scored
    within groups (§8b.2): ``rho_tree`` / ``rho_dend``, ``rho_tree_obs`` / ``rho_tree_full``, and
    ``share_tree`` = the part of the bin's variance that is between snapshots."""
    out = []
    for lo, hi in DEPTH_BINS:
        m = (bins_by >= lo) & (bins_by <= hi) & np.isfinite(y)
        if m.sum() < 30 or len(np.unique(groups[m])) < 5 or np.nanstd(y[m]) == 0:
            continue
        r = dict(_table="bins", tag=tag, site=site, x=site_depth(site), depth_lo=lo, depth_hi=hi,
                 n=int(m.sum()), n_neurons=int(len(np.unique(groups[m]))),
                 rho=P.spearman(y[m], oof[m]) if oof is not None else np.nan,
                 **{k: v for k, v in rec.items() if k in ("t", "split")})
        if oof_obs is not None:
            r["rho_obs"], r["rho_full"] = P.spearman(y[m], oof_obs[m]), P.spearman(y[m], oof_full[m])
        if tree is not None:
            if oof is not None:
                r["rho_tree"], r["share_tree"] = _within(y[m], oof[m], tree[m])
            if oof_obs is not None:
                r["rho_tree_obs"] = _within(y[m], oof_obs[m], tree[m])[0]
                r["rho_tree_full"] = _within(y[m], oof_full[m], tree[m])[0]
        if dend is not None and oof is not None:
            r["rho_dend"] = _within(y[m], oof[m], dend[m])[0]
        out.append(r)
    return out


def site_probe_unit(path: str, site: str, rref, y, groups, kind, rec: dict, B: int, **kw) -> list[dict]:
    """``_site_probe`` plus ``nonconv``: the number of solver fits in this unit that stopped at their
    iteration cap (sklearn ConvergenceWarning) — recorded, not acted on."""
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        out = _site_probe(path, site, rref, y, groups, kind, rec, B, **kw)
    out[0]["nonconv"] = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    return out


def _site_probe(path: str, site: str, rref, y, groups, kind, rec: dict, B: int,
                stack=None, plain: bool = True, X=None, diag_tag: str | None = None,
                items=None, keep=None, scale: bool = False, bins_by=None, bins_tree=None,
                bins_dend=None) -> list[dict]:
    """One site: plain probe score + CI and, with ``stack``, Δ over observables. ``X`` overrides
    loading node activations (graph-level sites, baselines). ``diag_tag``: also emit the P1
    depth-diagonal records from the same out-of-fold predictions. ``bins_by`` (per-row depth):
    also emit within-depth-bin rank scores (``_bin_records``). ``scale``: z-score the plain
    probe's columns (mixed-unit hand-built baselines only; activations are only centred)."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    if X is None and items is not None:      # graph-level site: G{i} tokens or pool_<site>
        a = open_store(path).graph_acts(site)[np.asarray(items)]
        X = a.reshape(len(a), -1)
        if keep is not None:
            X = X[keep]
    elif X is None:
        X = open_store(path).acts(site, rref)
    rec = {**rec, "site": site, "x": site_depth(site) if not site.startswith(("baseline", "G", "pool")) else np.nan,
           "n": len(y), "n_neurons": int(len(np.unique(groups)))}
    out_extra = []
    oof = None
    if plain:
        res = P.linear_probe(X, y, groups, kind=kind, scale=scale)
        kind_s = res.kind
        pt, lo, hi = P.bootstrap_ci(y, res.oof, groups, res.kind, B=B)
        oof = res.oof
        if diag_tag is not None:
            err = np.abs(res.oof - y)
            for d in np.unique(y):
                m = y == d
                pair = (y == d) | (y == d + 1)
                auc = (P.score((y[pair] == d + 1).astype(int), res.oof[pair], "binary")
                       if (y == d + 1).sum() >= 20 and m.sum() >= 20 else np.nan)
                out_extra.append(dict(_table="diag", tag=diag_tag, site=site, x=site_depth(site), depth=int(d),
                                      n=int(m.sum()), mae=float(err[m].mean()), auc_next=auc,
                                      **{k: v for k, v in rec.items() if k in ("t", "split")}))
    else:
        kind_s, pt, lo, hi = stack.kind, np.nan, np.nan, np.nan
    rec.update(kind=kind_s, score=pt, lo=lo, hi=hi)
    if stack is not None:
        inc = P.incremental_with(stack, X)
        d, dlo, dhi = P.paired_bootstrap(y, inc["oof_obs"], inc["oof_full"], groups, kind_s, B=B)
        rec.update(score_obs=inc["score_obs"], score_full=inc["score_full"], delta=d, delta_lo=dlo, delta_hi=dhi)
    if bins_by is not None:
        o_obs, o_full = (inc["oof_obs"], inc["oof_full"]) if stack is not None else (None, None)
        out_extra += _bin_records(y.astype(float), oof, groups, np.asarray(bins_by), rec["tag"], site, rec,
                                  o_obs, o_full,
                                  tree=None if bins_tree is None else np.asarray(bins_tree),
                                  dend=None if bins_dend is None else np.asarray(bins_dend))
    return [rec] + out_extra


def tree_ids(rows: pd.DataFrame) -> np.ndarray:
    """One id per partial-tree snapshot (neuron × reduction level)."""
    return rows.neuron.to_numpy(np.int64) * 1000 + rows.level.to_numpy(np.int64)


def dend_ids(rows: pd.DataFrame) -> np.ndarray:
    """One id per primary dendrite within a snapshot (the soma itself gets its own id)."""
    return tree_ids(rows) * 100000 + (rows.owner.to_numpy(np.int64) + 1)


def sweep_units(path: str, rows: pd.DataFrame, y, sites, *, tag: str, B: int, extra: dict | None = None,
                kind=None, stack=None, plain: bool = True, diag: bool = False, key_prefix: str | None = None,
                bins: bool = False):
    """Units probing ``y`` at every site of ``sites`` (node-level activations). ``bins``: emit
    within-depth-bin rank scores keyed on ``rows.depth``, pooled and within snapshot / dendrite
    (§8b.2; ids from ``tree_ids`` / ``dend_ids``)."""
    y = np.asarray(y)
    kind = kind or P.task_kind(y)
    rref, groups = row_ref(rows), rows.neuron.to_numpy()
    base = dict(tag=tag, **(extra or {}))
    pre = key_prefix or tag
    return [(f"{pre}|{s}", site_probe_unit,
             dict(path=path, site=s, rref=rref, y=y, groups=groups, kind=kind, rec=base, B=B,
                  stack=stack, plain=plain, diag_tag=tag if diag else None,
                  bins_by=rows.depth.to_numpy() if bins else None,
                  bins_tree=tree_ids(rows) if bins else None,
                  bins_dend=dend_ids(rows) if bins else None))
            for s in sites]


def split_tables(recs: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    """Separate main records from side tables marked with ``_table``."""
    main, side = [], {}
    for r in recs:
        t = r.get("_table")
        if t is None:
            main.append(r)
        else:
            side.setdefault(t, []).append({k: v for k, v in r.items() if k != "_table"})
    return main, side


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


def stack_score_unit(stack, rec: dict) -> list[dict]:
    """Record the observables-only (stacked) score as a baseline row."""
    return [{**rec, "score": stack.score_obs}]
