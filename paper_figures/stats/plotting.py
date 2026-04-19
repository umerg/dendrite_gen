"""Plotting helpers for statistics-driven paper figures."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from ..utils.styles import DEFAULT_DPI, GT_COLOR, PRED_COLOR


def _nice_metric_name(metric: str) -> str:
    return {
        "num_nodes": "Nodes",
        "num_edges": "Edges",
        "num_tips": "Tips",
        "num_branchpoints": "Branch Points",
        "height": "Height",
        "span_xy": "XY Span",
        "bbox_diag": "BBox Diag",
        "max_path_dist": "Max Path Dist",
        "max_radial_dist": "Max Radial Dist",
        "mean_branch_length": "Mean Branch Length",
        "mean_bifurcation_angle_deg": "Mean Branch Angle",
        "max_branch_order": "Max Branch Order",
        "branch_length": "Branch Length",
        "bifurcation_angle_deg": "Bifurcation Angle",
        "path_dist": "Path Distance",
        "radial_dist": "Radial Distance",
        "branch_order": "Branch Order",
    }.get(metric, metric.replace("_", " ").title())


def _bin_edges_from_values(
    arrays: Sequence[np.ndarray],
    *,
    bins: int,
) -> np.ndarray:
    combined = np.concatenate([arr for arr in arrays if arr.size > 0], axis=0) if any(
        arr.size > 0 for arr in arrays
    ) else np.zeros((0,), dtype=float)
    if combined.size > 0:
        lo = float(np.min(combined))
        hi = float(np.max(combined))
        if hi <= lo:
            hi = lo + 1e-6
        return np.linspace(lo, hi, bins + 1)
    return np.linspace(0.0, 1.0, bins + 1)


def _interleaved_hist_bars(
    ax,
    *,
    gt_hist: np.ndarray,
    pred_hist: np.ndarray,
    bin_edges: np.ndarray,
) -> None:
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + 0.5 * bin_widths
    bar_width = 0.42 * bin_widths

    ax.bar(
        bin_centers - 0.5 * bar_width,
        gt_hist,
        width=bar_width,
        color=GT_COLOR,
        edgecolor=GT_COLOR,
        alpha=0.35,
        linewidth=1.0,
        align="center",
    )
    ax.bar(
        bin_centers + 0.5 * bar_width,
        pred_hist,
        width=bar_width,
        color=PRED_COLOR,
        edgecolor=PRED_COLOR,
        alpha=0.35,
        linewidth=1.0,
        align="center",
    )


def _tree_average_hist(
    metric_df: pd.DataFrame,
    *,
    source: str,
    bin_edges: np.ndarray,
) -> np.ndarray:
    source_df = metric_df.loc[metric_df["source"] == source]
    if source_df.empty:
        return np.zeros((len(bin_edges) - 1,), dtype=float)

    group_col = "pair_index" if "pair_index" in source_df.columns else "tree_name"
    histograms = []
    for _, group in source_df.groupby(group_col):
        vals = group["value"].to_numpy(dtype=float)
        hist, _ = np.histogram(vals, bins=bin_edges, density=True)
        histograms.append(hist)
    if not histograms:
        return np.zeros((len(bin_edges) - 1,), dtype=float)
    return np.mean(np.stack(histograms, axis=0), axis=0)


def plot_tree_level_hist_grid(
    df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    out_path: Path,
    ncols: int = 3,
    bins: int = 24,
) -> Path:
    """Plot interleaved histogram densities for tree-level metrics in a grid."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_metrics = len(metrics)
    if n_metrics == 0:
        raise ValueError("At least one metric is required.")
    ncols = max(1, ncols)
    nrows = ceil(n_metrics / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
    colors = {"gt": GT_COLOR, "pred": PRED_COLOR}
    labels = {"gt": "Reference", "pred": "Ours"}

    for idx, metric in enumerate(metrics):
        ax = axes.flat[idx]
        metric_df = df[["source", metric]].dropna()
        gt_vals = metric_df.loc[metric_df["source"] == "gt", metric].to_numpy(dtype=float)
        pred_vals = metric_df.loc[metric_df["source"] == "pred", metric].to_numpy(dtype=float)
        combined = np.concatenate([vals for vals in (gt_vals, pred_vals) if vals.size > 0], axis=0) if (gt_vals.size or pred_vals.size) else np.zeros((0,), dtype=float)

        if combined.size > 0:
            lo = float(np.min(combined))
            hi = float(np.max(combined))
            if hi <= lo:
                hi = lo + 1e-6
            bin_edges = np.linspace(lo, hi, bins + 1)
        else:
            bin_edges = np.linspace(0.0, 1.0, bins + 1)

        gt_hist, _ = np.histogram(gt_vals, bins=bin_edges, density=True)
        pred_hist, _ = np.histogram(pred_vals, bins=bin_edges, density=True)
        bin_widths = np.diff(bin_edges)
        bin_centers = bin_edges[:-1] + 0.5 * bin_widths
        bar_width = 0.42 * bin_widths

        ax.bar(
            bin_centers - 0.5 * bar_width,
            gt_hist,
            width=bar_width,
            color=colors["gt"],
            edgecolor=colors["gt"],
            alpha=0.35,
            linewidth=1.0,
            align="center",
        )
        ax.bar(
            bin_centers + 0.5 * bar_width,
            pred_hist,
            width=bar_width,
            color=colors["pred"],
            edgecolor=colors["pred"],
            alpha=0.35,
            linewidth=1.0,
            align="center",
        )

        ax.set_title(_nice_metric_name(metric))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("")
        ax.set_yticks([])

    for idx in range(n_metrics, nrows * ncols):
        axes.flat[idx].axis("off")

    fig.legend(
        handles=[
            Patch(facecolor=GT_COLOR, edgecolor=GT_COLOR, alpha=0.25, label=labels["gt"]),
            Patch(facecolor=PRED_COLOR, edgecolor=PRED_COLOR, alpha=0.25, label=labels["pred"]),
        ],
        loc="upper center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)
    return out_path


def plot_distribution_hist_grid(
    df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    out_path: Path,
    ncols: int = 3,
    bins: int = 24,
    aggregation: str = "pooled",
) -> Path:
    """Plot interleaved histogram grids for within-tree distribution metrics."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_metrics = len(metrics)
    if n_metrics == 0:
        raise ValueError("At least one metric is required.")
    if aggregation not in {"pooled", "tree_average"}:
        raise ValueError(f"Unsupported aggregation mode: {aggregation}")

    ncols = max(1, ncols)
    nrows = ceil(n_metrics / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)

    for idx, metric in enumerate(metrics):
        ax = axes.flat[idx]
        metric_df = df.loc[df["metric"] == metric, ["source", "value", "tree_name", "pair_index"]].dropna(
            subset=["value"]
        )
        gt_vals = metric_df.loc[metric_df["source"] == "gt", "value"].to_numpy(dtype=float)
        pred_vals = metric_df.loc[metric_df["source"] == "pred", "value"].to_numpy(dtype=float)
        bin_edges = _bin_edges_from_values([gt_vals, pred_vals], bins=bins)

        if aggregation == "pooled":
            gt_hist, _ = np.histogram(gt_vals, bins=bin_edges, density=True)
            pred_hist, _ = np.histogram(pred_vals, bins=bin_edges, density=True)
        else:
            gt_hist = _tree_average_hist(metric_df, source="gt", bin_edges=bin_edges)
            pred_hist = _tree_average_hist(metric_df, source="pred", bin_edges=bin_edges)

        _interleaved_hist_bars(ax, gt_hist=gt_hist, pred_hist=pred_hist, bin_edges=bin_edges)
        ax.set_title(_nice_metric_name(metric))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("")
        ax.set_yticks([])

    for idx in range(n_metrics, nrows * ncols):
        axes.flat[idx].axis("off")

    fig.legend(
        handles=[
            Patch(facecolor=GT_COLOR, edgecolor=GT_COLOR, alpha=0.25, label="Reference"),
            Patch(facecolor=PRED_COLOR, edgecolor=PRED_COLOR, alpha=0.25, label="Ours"),
        ],
        loc="upper center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)
    return out_path
