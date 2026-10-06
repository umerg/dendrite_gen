"""Compact text summary of a results directory (optionally against a control, e.g. random-init).

    python -m interp.analyses.report results/tmd_ema [--control results/tmd_random]

For each analysis CSV found: per target, the score (and Δ over observables where present) at
every site, side by side with the control's value at the same site.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(v, lo=None, hi=None, signed=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "     —      "
    s = f"{v:+.3f}" if signed else f"{v:.3f}"
    if lo is not None and not (isinstance(lo, float) and np.isnan(lo)):
        s += f" [{lo:+.3f},{hi:+.3f}]" if signed else f" [{lo:.3f},{hi:.3f}]"
    return s


def _table(df: pd.DataFrame, ctl: pd.DataFrame | None, keys: list[str], label: str):
    lines = []
    for k, g in df.groupby(keys, sort=False):
        lines.append(f"\n  {label} {k}")
        cg = None
        if ctl is not None:
            cg = ctl
            for col, val in zip(keys, k if isinstance(k, tuple) else (k,)):
                cg = cg[cg[col] == val]
        for r in g.itertuples(index=False):
            row = f"    {r.site:>22}  score {_fmt(getattr(r, 'score', np.nan), getattr(r, 'lo', None), getattr(r, 'hi', None))}"
            if "delta" in g.columns and pd.notna(getattr(r, "delta", np.nan)):
                row += f"  Δ {_fmt(r.delta, r.delta_lo, r.delta_hi, signed=True)}"
            if cg is not None and len(cg):
                c = cg[cg.site == r.site]
                if len(c):
                    c = c.iloc[0]
                    row += f"  | ctl score {_fmt(c.get('score', np.nan))}"
                    if "delta" in c and pd.notna(c["delta"]):
                        row += f" Δ {_fmt(c['delta'], signed=True)}"
            lines.append(row)
    return lines


def report(res_dir, control=None) -> str:
    res_dir = Path(res_dir)
    out = [f"# {res_dir}" + (f"  (control: {control})" if control else "")]
    for csv in sorted(res_dir.glob("p*.csv")):
        df = pd.read_csv(csv)
        if df.empty:
            continue
        ctl = None
        if control and (Path(control) / csv.name).exists():
            ctl = pd.read_csv(Path(control) / csv.name)
        out.append(f"\n## {csv.stem}")
        if csv.stem.startswith("p6_"):
            for part, g in df.groupby("part"):
                keys = [c for c in ("ablation", "mode", "target", "site") if c in g.columns and g[c].notna().any()]
                cols = [c for c in ("bin_lo", "bin_hi", "n", "metric", "score", "lo", "hi", "delta", "delta_lo",
                                    "delta_hi", "intact", "ablated", "ratio") if c in g.columns]
                out.append(f"\n  [{part}]")
                piv = g[keys + cols].dropna(axis=1, how="all")
                out.append(piv.to_string(index=False, max_rows=400))
            continue
        if "depth_diag" in csv.stem:
            from .common import site_depth
            for tag, g in df.groupby("tag"):   # context vs frontier rows: separate tables
                piv = g.pivot_table(index="site", columns="depth", values="auc_next", aggfunc="first")
                piv = piv.loc[sorted(piv.index, key=site_depth)]
                out.append(f"\n  [{tag}] depth d vs d+1 AUROC")
                out.append(piv.round(2).to_string())
            continue
        keys = ["tag"] if "tag" in df.columns else [c for c in ("target", "v_lo") if c in df.columns]
        out += _table(df, ctl, keys, "")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--control", default=None)
    a = ap.parse_args(argv)
    print(report(a.results, a.control))


if __name__ == "__main__":
    main()
