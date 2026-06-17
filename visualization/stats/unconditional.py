"""Feature vectors and PCA plots for unconditioned GT/pred comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import networkx as nx
import numpy as np
import pandas as pd

from ..utils.styles import DEFAULT_DPI, GT_COLOR, PRED_COLOR
from .distribution_stats import GRAPH_DISTRIBUTION_KEYS, graph_distribution_values
from .plotting import _nice_metric_name
from .tree_stats import graph_tree_scalar_stats


UNCONDITIONAL_DISTRIBUTION_METRICS = GRAPH_DISTRIBUTION_KEYS
UNCONDITIONAL_SCALAR_METRICS = ("height", "span_xy", "bbox_diag")
SUMMARY_SUFFIXES = ("mean", "std")
PCA_COMPONENT_COLUMNS = ("pc1", "pc2")


def unconditional_feature_columns(
    *,
    distribution_metrics: Sequence[str] = UNCONDITIONAL_DISTRIBUTION_METRICS,
    scalar_metrics: Sequence[str] = UNCONDITIONAL_SCALAR_METRICS,
) -> list[str]:
    """Return the feature-column order used for unconditioned diagnostics."""
    columns: list[str] = []
    for metric in distribution_metrics:
        columns.extend([f"{metric}_{suffix}" for suffix in SUMMARY_SUFFIXES])
    columns.extend(scalar_metrics)
    return columns


def _finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def graph_unconditional_feature_row(
    graph: nx.Graph,
    *,
    tree_name: str,
    source: str,
    pair_index: int | None = None,
    distribution_metrics: Sequence[str] = UNCONDITIONAL_DISTRIBUTION_METRICS,
    scalar_metrics: Sequence[str] = UNCONDITIONAL_SCALAR_METRICS,
) -> dict[str, object]:
    """Return one tree-level feature row for unconditioned distribution plots."""
    row: dict[str, object] = {
        "tree_name": tree_name,
        "source": source,
        "pair_index": pair_index,
    }

    for metric in distribution_metrics:
        values = _finite_values(graph_distribution_values(graph, metric))
        row[f"{metric}_mean"] = float(np.mean(values)) if values.size else float("nan")
        row[f"{metric}_std"] = float(np.std(values)) if values.size else float("nan")

    scalar_stats = graph_tree_scalar_stats(graph)
    for metric in scalar_metrics:
        row[metric] = float(scalar_stats.get(metric, float("nan")))

    return row


def _standardize_feature_matrix(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    eps: float = 1e-12,
) -> tuple[np.ndarray, list[str], pd.Series, pd.Series]:
    raw = df.loc[:, list(feature_columns)].replace([np.inf, -np.inf], np.nan)
    available_columns = [
        column for column in raw.columns if raw[column].notna().any()
    ]
    if not available_columns:
        raise ValueError("No finite unconditional feature columns are available.")

    matrix = raw.loc[:, available_columns].to_numpy(dtype=float)
    means = pd.Series(np.nanmean(matrix, axis=0), index=available_columns)
    filled = np.where(np.isnan(matrix), means.to_numpy(dtype=float), matrix)
    stds = pd.Series(np.std(filled, axis=0), index=available_columns)
    variable_columns = [column for column in available_columns if stds[column] > eps]
    if not variable_columns:
        raise ValueError("All unconditional feature columns are constant.")

    matrix = raw.loc[:, variable_columns].to_numpy(dtype=float)
    means = pd.Series(np.nanmean(matrix, axis=0), index=variable_columns)
    filled = np.where(np.isnan(matrix), means.to_numpy(dtype=float), matrix)
    stds = pd.Series(np.std(filled, axis=0), index=variable_columns)
    standardized = (filled - means.to_numpy(dtype=float)) / stds.to_numpy(dtype=float)
    return standardized, variable_columns, means, stds


def compute_unconditional_pca(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute a two-component PCA for combined GT/pred feature rows."""
    if df.empty:
        raise ValueError("Cannot compute unconditional PCA from an empty dataframe.")
    if len(df) < 2:
        raise ValueError("At least two trees are required for unconditional PCA.")

    matrix, used_features, means, stds = _standardize_feature_matrix(
        df, feature_columns=feature_columns
    )
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

    n_components = min(2, vt.shape[0])
    coords = np.zeros((centered.shape[0], 2), dtype=float)
    if n_components:
        coords[:, :n_components] = centered @ vt[:n_components].T

    variances = np.zeros((2,), dtype=float)
    if centered.shape[0] > 1 and singular_values.size:
        all_variances = (singular_values**2) / (centered.shape[0] - 1)
        total_variance = float(np.sum(all_variances))
        if total_variance > 0:
            variances[:n_components] = all_variances[:n_components] / total_variance

    coord_df = df[["tree_name", "source", "pair_index"]].copy()
    coord_df["pc1"] = coords[:, 0]
    coord_df["pc2"] = coords[:, 1]
    coord_df["pc1_explained_variance"] = variances[0]
    coord_df["pc2_explained_variance"] = variances[1]

    loadings = np.zeros((len(used_features), 2), dtype=float)
    loadings[:, :n_components] = vt[:n_components].T
    loadings_df = pd.DataFrame(
        {
            "feature": used_features,
            "pc1": loadings[:, 0],
            "pc2": loadings[:, 1],
            "feature_mean": means.reindex(used_features).to_numpy(dtype=float),
            "feature_std": stds.reindex(used_features).to_numpy(dtype=float),
        }
    )

    metadata_df = pd.DataFrame(
        {
            "component": ["pc1", "pc2"],
            "explained_variance_ratio": variances,
        }
    )
    return coord_df, loadings_df, metadata_df


def _nice_feature_name(feature: str) -> str:
    for suffix in SUMMARY_SUFFIXES:
        marker = f"_{suffix}"
        if feature.endswith(marker):
            metric = feature[: -len(marker)]
            return f"{_nice_metric_name(metric)} {suffix.title()}"
    return _nice_metric_name(feature)


def _axis_limits(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        pad = max(abs(lo) * 0.05, 1e-6)
    else:
        pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def _safe_filename(text: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in text)


def _top_loading_text(loadings_df: pd.DataFrame, component: str, *, count: int = 5) -> str:
    rows = loadings_df.assign(abs_loading=lambda frame: frame[component].abs())
    rows = rows.sort_values("abs_loading", ascending=False).head(count)
    parts = [
        f"{_nice_feature_name(str(row.feature))}: {float(getattr(row, component)):+.2f}"
        for row in rows.itertuples(index=False)
    ]
    return "\n".join(parts)


def plot_unconditional_pca(
    coord_df: pd.DataFrame,
    loadings_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    *,
    out_path: Path,
    point_alpha: float = 0.55,
) -> Path:
    """Plot combined GT/pred PCA coordinates for unconditional feature vectors."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax, text_ax) = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.2),
        gridspec_kw={"width_ratios": [3.2, 1.5]},
    )
    labels = {"gt": "Reference", "pred": "Ours"}
    colors = {"gt": GT_COLOR, "pred": PRED_COLOR}

    for source in ("gt", "pred"):
        source_df = coord_df.loc[coord_df["source"] == source]
        if source_df.empty:
            continue
        ax.scatter(
            source_df["pc1"].to_numpy(dtype=float),
            source_df["pc2"].to_numpy(dtype=float),
            s=4,
            c=colors[source],
            alpha=point_alpha,
            edgecolors="white",
            linewidths=0.35,
            label=labels[source],
        )

    pc1_ratio = float(
        metadata_df.loc[
            metadata_df["component"] == "pc1", "explained_variance_ratio"
        ].iloc[0]
    )
    pc2_ratio = float(
        metadata_df.loc[
            metadata_df["component"] == "pc2", "explained_variance_ratio"
        ].iloc[0]
    )
    ax.set_xlabel(f"PC1 ({pc1_ratio:.1%})")
    ax.set_ylabel(f"PC2 ({pc2_ratio:.1%})")
    ax.set_title("Unconditional Tree-Feature PCA")
    xlim = _axis_limits(coord_df["pc1"].to_numpy(dtype=float))
    ylim = _axis_limits(coord_df["pc2"].to_numpy(dtype=float))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axhline(0.0, color="#9ca3af", linewidth=0.75, alpha=0.55)
    ax.axvline(0.0, color="#9ca3af", linewidth=0.75, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor=GT_COLOR, edgecolor=GT_COLOR, alpha=0.45, label=labels["gt"]),
            Patch(facecolor=PRED_COLOR, edgecolor=PRED_COLOR, alpha=0.45, label=labels["pred"]),
        ],
        loc="best",
        frameon=True,
    )

    text_ax.axis("off")
    text_ax.text(
        0.0,
        1.0,
        "Top loadings\n\n"
        f"PC1\n{_top_loading_text(loadings_df, 'pc1')}\n\n"
        f"PC2\n{_top_loading_text(loadings_df, 'pc2')}",
        ha="left",
        va="top",
        fontsize=9.5,
        linespacing=1.35,
        transform=text_ax.transAxes,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)
    return out_path


def plot_unconditional_pca_by_feature(
    coord_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    out_dir: Path,
    point_alpha: float = 0.72,
    cmap: str = "viridis",
) -> list[Path]:
    """Write one PCA scatter colored by each unconditional feature value."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pc1_ratio = float(
        metadata_df.loc[
            metadata_df["component"] == "pc1", "explained_variance_ratio"
        ].iloc[0]
    )
    pc2_ratio = float(
        metadata_df.loc[
            metadata_df["component"] == "pc2", "explained_variance_ratio"
        ].iloc[0]
    )
    xlim = _axis_limits(coord_df["pc1"].to_numpy(dtype=float))
    ylim = _axis_limits(coord_df["pc2"].to_numpy(dtype=float))

    plot_df = coord_df[["tree_name", "source", "pair_index", "pc1", "pc2"]].copy()
    for feature in feature_columns:
        if feature in feature_df.columns:
            plot_df[feature] = feature_df[feature].to_numpy(dtype=float)

    written: list[Path] = []
    for feature in feature_columns:
        if feature not in plot_df.columns:
            continue

        values = plot_df[feature].to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            continue

        lo = float(np.min(finite_values))
        hi = float(np.max(finite_values))
        if hi <= lo:
            hi = lo + 1e-6
        norm = Normalize(vmin=lo, vmax=hi)

        fig, ax = plt.subplots(figsize=(6.6, 5.3))
        for source, marker, label in (
            ("gt", "o", "Reference"),
            ("pred", "^", "Ours"),
        ):
            source_df = plot_df.loc[plot_df["source"] == source]
            if source_df.empty:
                continue
            scatter = ax.scatter(
                source_df["pc1"].to_numpy(dtype=float),
                source_df["pc2"].to_numpy(dtype=float),
                c=source_df[feature].to_numpy(dtype=float),
                cmap=cmap,
                norm=norm,
                marker=marker,
                s=4,
                alpha=point_alpha,
                edgecolors="none",
                linewidths=0.0,
                label=label,
            )

        ax.set_xlabel(f"PC1 ({pc1_ratio:.1%})")
        ax.set_ylabel(f"PC2 ({pc2_ratio:.1%})")
        ax.set_title(f"PCA Colored by {_nice_feature_name(feature)}")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.axhline(0.0, color="#9ca3af", linewidth=0.75, alpha=0.55)
        ax.axvline(0.0, color="#9ca3af", linewidth=0.75, alpha=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", frameon=True)
        colorbar = fig.colorbar(scatter, ax=ax, shrink=0.86, pad=0.02)
        colorbar.set_label(_nice_feature_name(feature))
        fig.tight_layout()

        out_path = out_dir / f"pca_color_{_safe_filename(feature)}.png"
        fig.savefig(out_path, dpi=DEFAULT_DPI)
        plt.close(fig)
        written.append(out_path)

    return written
