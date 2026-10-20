"""Paired analysis of generation-time interventions (methodology §11b): E0 / E1 / E2.

Reads ``<exp_dir>/<condition>_seed<s>.pkl`` written by ``interp.rollout`` and, per condition:
  * per-tree table (``interp.metrics``: size, tall-arm stats; + class probabilities if a
    classifier is given), written to ``trees.parquet``;
  * paired effect vs ``intact`` (same GT neuron, same seed): mean Δ, Hodges–Lehmann shift and
    Wilcoxon p for continuous outcomes, McNemar p for binary ones, neuron-bootstrap 95 % CI
    of the mean Δ (neurons resampled, seeds kept together);
  * distribution metrics vs GT (``compute_distribution_metrics``), optionally vs the GT
    neurons of one class (``--class-ref``) for S-class.

Example::
    python -m interp.analyses.s_rollouts --exp-dir /.../rollouts/e0 --config parity_neurons_uncond \\
        --split val --clf /.../class_clf --out /.../rollouts/e0/analysis
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .. import metrics as M

CONT = ["n_nodes", "n_tips", "max_depth", "axial_extent", "total_length", "n_tall", "dominance",
        "apical_reach", "basal_reach_max", "apical_nodes", "basal_nodes"]
BIN = ["ge2_tall", "one_tall", "tallest_is_slot0", "tallest_neg", "cap_hit", "t_expanded", "s_expanded"]
GROWTH = ["t_n_desc", "t_height", "t_path_length", "t_strahler", "t_axial_reach", "s_n_desc", "s_height",
          "dend_rest_nodes", "other_basal_nodes"]
DIST_KEYS = ["mmd_morpho", "strahler_w1", "sholl_peak_w1", "axial_extent_w1", "node_count_w1",
             "branch_length_w1", "bifurcation_angle_w1"]


def load_runs(exp_dir: Path) -> dict[tuple[str, int], dict]:
    runs = {}
    for p in sorted(exp_dir.glob("*_seed*.pkl")):
        m = re.match(r"(.+)_seed(\d+)\.pkl$", p.name)
        with open(p, "rb") as f:
            runs[(m.group(1), int(m.group(2)))] = pickle.load(f)
    return runs


def tree_frame(runs, uhat, clfs: dict | None = None) -> pd.DataFrame:
    parts = []
    for (cond, seed), r in runs.items():
        gs = r["pred_graphs"]
        df = M.tree_table(gs, uhat)
        df["condition"], df["seed"] = cond, seed
        df["gt_index"] = [g.graph["gt_index"] for g in gs]
        df["gt_class"] = [g.graph.get("cell_class") for g in gs]
        df["cap_hit"] = [bool(g.graph.get("cap_hit", False)) for g in gs]
        if any("target_key" in g.graph for g in gs):      # E3: the steered node's subtree
            gr = pd.DataFrame([M.growth_row(g, uhat) for g in gs])
            gr["target_key"] = [None if g.graph.get("target_key") is None else str(g.graph["target_key"])
                                for g in gs]   # uint64 → str (parquet-safe)
            df = pd.concat([df, gr.drop(columns=[c for c in gr if c in df])], axis=1)
        for name, clf in (clfs or {}).items():
            pr = clf.proba_features(clf.features(gs))
            for j, c in enumerate(clf.classes_):
                df[f"p_{name}_{int(c)}"] = pr[:, j]
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def _hl(d: np.ndarray) -> float:
    """Hodges–Lehmann shift: median of Walsh averages (subsampled when large)."""
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan")
    if len(d) > 2000:
        d = np.random.default_rng(0).choice(d, 2000, replace=False)
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2))


def _boot_mean(d: np.ndarray, groups: np.ndarray, B: int = 1000, seed: int = 0) -> tuple[float, float]:
    ok = np.isfinite(d)
    d, groups = d[ok], groups[ok]
    if len(d) < 2:
        return float("nan"), float("nan")
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=d, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq))
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uniq), size=(B, len(uniq)))
    bm = sums[pick].sum(1) / np.maximum(cnts[pick].sum(1), 1)
    return float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))


def paired_effects(trees: pd.DataFrame, ref: str = "intact", outcomes: list[str] | None = None,
                   B: int = 1000) -> pd.DataFrame:
    outcomes = outcomes or ([c for c in CONT + BIN + GROWTH if c in trees] + [c for c in trees if c.startswith("p_")])
    base = trees[trees.condition == ref].set_index(["gt_index", "seed"])
    recs = []
    for cond, g in trees[trees.condition != ref].groupby("condition"):
        g = g.set_index(["gt_index", "seed"])
        j = g.join(base, how="inner", lsuffix="", rsuffix="_ref")
        n_mismatch = 0
        if "target_key" in j:      # E3 pairs must refer to the same lineage node
            same = j.target_key.astype(object).eq(j.target_key_ref.astype(object))
            n_mismatch = int((~same & j.target_key.notna()).sum())
            j = j[same & j.target_key.notna()]
        groups = j.index.get_level_values(0).to_numpy()
        for o in outcomes:
            a, b = j[o].to_numpy(float), j[f"{o}_ref"].to_numpy(float)
            d = a - b
            ok = np.isfinite(d)
            rec = dict(condition=cond, outcome=o, n=int(ok.sum()), n_key_mismatch=n_mismatch, mean=float(np.nanmean(a)),
                       mean_ref=float(np.nanmean(b)), delta=float(np.nanmean(d)))
            rec["ci_lo"], rec["ci_hi"] = _boot_mean(d, groups, B)
            if o in BIN:
                up, down = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
                rec.update(n_up=up, n_down=down,
                           p=float(stats.binomtest(up, up + down).pvalue) if up + down else 1.0)
            else:
                rec["hl"] = _hl(d)
                nz = d[ok][d[ok] != 0]
                rec["p"] = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 10 else float("nan")
            recs.append(rec)
    return pd.DataFrame(recs)


def dist_table(runs, gt_graphs, uhat, class_ref: int | None = None) -> pd.DataFrame:
    from validation.dist_metrics import compute_distribution_metrics

    refs = {"all": gt_graphs}
    if class_ref is not None:
        refs[f"class{class_ref}"] = [g for g in gt_graphs if g.graph.get("cell_class") == class_ref]
    recs = []
    for (cond, seed), r in runs.items():
        for rname, gt in refs.items():
            if len(gt) < 10:
                continue
            m = compute_distribution_metrics(r["pred_graphs"], gt, uhat=uhat, ged_enabled=False, enable_ks=False)
            recs.append(dict(condition=cond, seed=seed, ref=rname, **{k: m.get(k, np.nan) for k in DIST_KEYS}))
    return pd.DataFrame(recs)


def spec_table(exp_dir: Path) -> pd.DataFrame:
    p = exp_dir / "conditions.json"
    if not p.exists():
        return pd.DataFrame(columns=["condition"])
    rows = [dict(condition=c["name"], **c["spec"]) for c in json.loads(p.read_text())]
    return pd.DataFrame(rows).drop(columns=["blocks"], errors="ignore")


def report(eff: pd.DataFrame, dist: pd.DataFrame | None, spec: pd.DataFrame, primary: list[str]) -> str:
    lines = []
    e = eff[eff.outcome.isin(primary)].merge(spec, on="condition", how="left")
    for o, g in e.groupby("outcome", sort=False):
        cols = [c for c in ["condition", "n", "mean_ref", "mean", "delta", "ci_lo", "ci_hi", "hl", "p"] if c in g]
        lines += [f"\n[{o}] paired vs intact", g[cols].round(4).to_string(index=False)]
    if dist is not None and len(dist):
        piv = dist.groupby(["ref", "condition"])[DIST_KEYS].mean().round(4)
        lines += ["\n[distribution metrics vs GT, mean over seeds]", piv.to_string()]
    return "\n".join(lines)


def main(argv=None):
    from ..class_clf import MorphoClassifier
    from ..load import compose_cfg, load_split, uhat_of

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--clf", default=None, help="directory with class_clf_full.pkl / class_clf_size_free.pkl")
    ap.add_argument("--class-ref", type=int, default=None)
    ap.add_argument("--no-dist", action="store_true")
    ap.add_argument("--primary", default="ge2_tall,basal_reach_max,apical_reach,n_nodes,dominance,cap_hit,"
                                        "t_expanded,t_n_desc,t_height,s_n_desc,dend_rest_nodes,other_basal_nodes")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    exp = Path(a.exp_dir)
    out = Path(a.out or exp / "analysis")
    out.mkdir(parents=True, exist_ok=True)
    cfg = compose_cfg(a.config)
    uhat = uhat_of(cfg)
    runs = load_runs(exp)
    clfs = {}
    if a.clf:
        for name in ("full", "size_free"):
            p = Path(a.clf) / f"class_clf_{name}.pkl"
            if p.exists():
                clfs[name] = MorphoClassifier.load(p)
    trees = tree_frame(runs, uhat, clfs)
    trees.to_parquet(out / "trees.parquet")
    eff = paired_effects(trees, B=a.B)
    eff.to_csv(out / "effects.csv", index=False)
    dist = None
    if not a.no_dist:
        any_run = next(iter(runs.values()))
        ids = set(any_run["meta"]["neuron_ids"])
        gt = [g for g in load_split(cfg, a.split, data_dir=a.data_dir) if g.graph["neuron_id"] in ids]
        dist = dist_table(runs, gt, uhat, a.class_ref)
        dist.to_csv(out / "dist.csv", index=False)
    primary = a.primary.split(",") + [c for c in eff.outcome.unique() if c.startswith("p_")]
    txt = report(eff, dist, spec_table(exp), primary)
    (out / "report.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
