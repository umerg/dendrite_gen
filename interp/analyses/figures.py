"""Figures for the results report (``docs/INTERP_RESULTS.md``), one function per claim.

    python -m interp.analyses.figures --runs ~/Documents/generations/interp/runs --arm uncond \
        --out docs/interp_figs

``--runs`` holds ``<arm>/trained_v3``, ``<arm>/random_v3`` (analysis CSVs from ``results_v3``) and
``<arm>/trained_v2`` (P6 from ``results_v2``). Plain probe scores only: the input layer and the
random-init network are the references; the Δ-over-observables columns are not used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .common import site_depth  # noqa: E402

TRAINED, RANDOM = "#1f5fa8", "#9a9a9a"
BIN_COLORS = {"3-4": "#1b9e77", "5-6": "#d95f02", "7-8": "#7570b3", "9-12": "#e7298a", "13-+": "#66a61e"}


def graph_x(site: str) -> float:
    """Position of a graph-level site (pool_in = -1, pool_L<i> / G<i> = i)."""
    if site == "pool_in":
        return -1.0
    return float(site.split("_L")[-1]) if site.startswith("pool_L") else float(site[1:])


def _read(runs: Path, arm: str, run: str, name: str) -> pd.DataFrame | None:
    p = runs / arm / run / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def _bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.site.str.startswith("baseline")].copy()
    df["bin"] = df.depth_lo.astype(str) + "-" + df.depth_hi.astype(str).str.replace("99", "+")
    df["x"] = df.site.map(site_depth)
    return df.sort_values("x")


def _layer_axis(ax):
    ax.set_xticks([-1, 0, 1, 3, 5, 7, 9, 11], ["in", "L0", "L1", "L3", "L5", "L7", "L9", "L11"])
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(alpha=0.3)


def fig_broadcast(tr, rd, out: Path, tag="primary/log_desc", bins=("3-4", "5-6", "7-8", "9-12")):
    """C1: pooled vs within-tree vs within-dendrite ρ by layer, trained vs random-init."""
    tr, rd = _bins(tr), _bins(rd)
    fig, axs = plt.subplots(1, len(bins), figsize=(4 * len(bins), 3.4), sharey=True)
    for ax, b in zip(axs, bins):
        for df, col, ls in ((tr, TRAINED, "-"), (rd, RANDOM, "--")):
            g = df[(df.tag == tag) & (df.bin == b)]
            for val, mk, lab in (("rho", "o", "pooled"), ("rho_tree", "s", "within tree"),
                                 ("rho_dend", "^", "within dendrite")):
                if val in g and g[val].notna().any():
                    ax.plot(g.x, g[val], ls, color=col, marker=mk, ms=3, lw=1,
                            label=f"{lab} ({'trained' if col == TRAINED else 'random'})")
        share = tr[(tr.tag == tag) & (tr.bin == b)].share_tree.dropna()
        ax.set_title(f"depth {b}   between-tree share {share.iloc[0]:.2f}" if len(share) else f"depth {b}")
        _layer_axis(ax)
    axs[0].set_ylabel(f"Spearman ρ ({tag})")
    axs[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    return _save(fig, out / "F1_broadcast_vs_within.png")


def fig_lookahead(tr, rd, out: Path, tags=("primary/log_desc", "primary/log_pathlen", "nonapical/log_desc"),
                  bins=("3-4", "5-6", "7-8", "9-12", "13-+")):
    """C2: within-dendrite ρ of future-subtree targets, by layer and depth bin."""
    tr, rd = _bins(tr), _bins(rd)
    fig, axs = plt.subplots(1, len(tags), figsize=(4.4 * len(tags), 3.4), sharey=True)
    for ax, tag in zip(axs, tags):
        for b in bins:
            for df, ls, a in ((tr, "-", 1.0), (rd, "--", 0.5)):
                g = df[(df.tag == tag) & (df.bin == b)]
                if "rho_dend" in g and g.rho_dend.notna().any():
                    ax.plot(g.x, g.rho_dend, ls, color=BIN_COLORS[b], alpha=a, lw=1.2,
                            label=f"depth {b}" if df is tr else None)
        ax.set_title(tag)
        _layer_axis(ax)
    axs[0].set_ylabel("within-dendrite Spearman ρ\n(solid trained, dashed random-init)")
    axs[-1].legend(fontsize=7)
    return _save(fig, out / "F2_lookahead_within_dendrite.png")


def _graph_panel(ax, tr, rd, target, metric_label):
    for df, col in ((tr, TRAINED), (rd, RANDOM)):
        g = df[df.target == target]
        for kind, ls in (("G", "-"), ("pool_L", ":")):
            s = g[g.site.str.startswith(kind)].copy()
            s["x"] = s.site.map(graph_x)
            s = s.sort_values("x")
            ax.plot(s.x, s.score, ls, color=col, marker="o", ms=3, lw=1.2,
                    label=f"{'tokens G' if kind == 'G' else 'node mean'} ({'trained' if col == TRAINED else 'random'})")
            if "lo" in s and s.lo.notna().any():
                ax.fill_between(s.x, s.lo, s.hi, color=col, alpha=0.12, lw=0)
    pin = tr[(tr.target == target) & (tr.site == "pool_in")].score
    if len(pin):
        ax.axhline(pin.iloc[0], color="k", ls="--", lw=0.8, label="input (pool_in)")
    ax.set_title(target)
    ax.set_ylabel(metric_label)
    ax.set_xticks([-1, 1, 3, 5, 7, 9, 11], ["in", "1", "3", "5", "7", "9", "11"])
    ax.set_xlabel("block")
    ax.grid(alpha=0.3)


def fig_tree_summaries(tr, rd, out: Path,
                       targets=("tree/tallest_reach", "tree/apical_reach_rel", "tree/apical_is_tallest",
                                "tree/log_n_visible")):
    """C3: tree-level geometry from graph-level sites."""
    fig, axs = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 3.4))
    for ax, t in zip(axs, targets):
        _graph_panel(ax, tr, rd, t, "AUROC" if "is_tallest" in t else "R²")
    axs[-1].legend(fontsize=7, loc="lower right")
    return _save(fig, out / "F3_tree_summaries.png")


def fig_cell_type(tr, rd, out: Path, targets=("cell_class", "tmd_pc1", "tmd_pc3"), site="pool_L11"):
    """C4: cell type / final-TMD components against visible depth."""
    targets = [t for t in targets if (tr.target == t).any()]
    fig, axs = plt.subplots(1, len(targets), figsize=(4.4 * len(targets), 3.4))
    axs = np.atleast_1d(axs)
    for ax, t in zip(axs, targets):
        for df, col, s, lab in ((tr, TRAINED, site, f"trained {site}"), (tr, TRAINED, "G11", "trained G11"),
                                (rd, RANDOM, site, f"random-init {site}"), (tr, "k", "pool_in", "input (pool_in)"),
                                (tr, "#c47f00", "baseline:tmd_input", "finished-tree TMD (ceiling)")):
            g = df[(df.target == t) & (df.site == s)].sort_values("v_lo")
            if not len(g):
                continue
            ls = "--" if s in ("pool_in", "baseline:tmd_input") else ("-." if s == "G11" else "-")
            ax.plot(g.v_lo, g.score, ls, color=col, marker="o", ms=3, lw=1.2, label=lab)
            if g.lo.notna().any():
                ax.fill_between(g.v_lo, g.lo, g.hi, color=col, alpha=0.1, lw=0)
        ax.set_title(t)
        ax.set_xlabel("visible depth of the partial tree")
        ax.set_ylabel("balanced accuracy" if t == "cell_class" else "R²")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7)
    return _save(fig, out / "F4_cell_type.png")


def fig_progress(tr, rd, out: Path, targets=("levels_remaining", "final_depth", "is_last")):
    """C5: how much tree is left, from graph-level sites."""
    fig, axs = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 3.4))
    for ax, t in zip(axs, targets):
        _graph_panel(ax, tr, rd, t, "AUROC" if t == "is_last" else "R²")
    axs[-1].legend(fontsize=7, loc="lower right")
    return _save(fig, out / "F5_progress.png")


def fig_apical(p6, out: Path, sites=("in", "L1", "L3", "L7", "L11")):
    """C6: in-apical-dendrite AUROC by depth and layer (own-lineage part of P6)."""
    g = p6[(p6.part == "a_own_lineage") & p6.bin_lo.notna()]
    g = g[~((g.bin_lo == 2) & (g.bin_hi == 99))]          # drop the all-depths row
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    cmap = plt.get_cmap("viridis", len(sites))
    for i, s in enumerate(sites):
        r = g[g.site == s].sort_values("bin_lo")
        ax.plot(r.bin_lo, r.score, "-o", ms=3, color=cmap(i), label=s)
        ax.fill_between(r.bin_lo, r.lo, r.hi, color=cmap(i), alpha=0.1, lw=0)
    ax.axhline(0.5, color="k", lw=0.5)
    ax.set_xlabel("node depth (13 = 13+)")
    ax.set_ylabel("AUROC: node is in the apical dendrite")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    return _save(fig, out / "F6_apical_identity.png")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_all(runs: str | Path, arm: str, out: str | Path) -> list[Path]:
    runs, out = Path(runs).expanduser(), Path(out)
    made = []
    tr, rd = _read(runs, arm, "trained_v3", "p2_bins_val_t0"), _read(runs, arm, "random_v3", "p2_bins_val_t0")
    if tr is not None and rd is not None:
        made += [fig_broadcast(tr, rd, out), fig_lookahead(tr, rd, out)]
    tr, rd = _read(runs, arm, "trained_v3", "p5_val_t0"), _read(runs, arm, "random_v3", "p5_val_t0")
    if tr is not None and rd is not None:
        made += [fig_tree_summaries(tr, rd, out), fig_progress(tr, rd, out)]
    tr, rd = _read(runs, arm, "trained_v3", "p3_val_t0"), _read(runs, arm, "random_v3", "p3_val_t0")
    if tr is not None and rd is not None:
        made.append(fig_cell_type(tr, rd, out))
    p6 = _read(runs, arm, "trained_v2", "p6_val_t0")
    if p6 is not None:
        made.append(fig_apical(p6, out))
    return made


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--arm", default="uncond")
    ap.add_argument("--out", default="docs/interp_figs")
    a = ap.parse_args(argv)
    for p in make_all(a.runs, a.arm, Path(a.out) / a.arm if a.arm != "uncond" else a.out):
        print(p)


if __name__ == "__main__":
    main()
