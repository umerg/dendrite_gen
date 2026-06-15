"""Persistence-diagram plotting helpers for TMD paper figures."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..utils.styles import DEFAULT_DPI, GT_COLOR, PRED_COLOR


def _diagram_pairs(diagram) -> np.ndarray:
    if diagram is None:
        return np.zeros((0, 2), dtype=np.float64)
    pairs = diagram.as_pairs()
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(pairs, dtype=np.float64)


def _filtration_title(name: str) -> str:
    return {
        "path": "Path Length",
        "height": "Height",
        "rho": "Radial Distance",
    }.get(name, name.replace("_", " ").title())


def _axis_limits(
    gt_pairs: np.ndarray,
    pred_pairs: np.ndarray,
    *,
    normalize_mode: str,
    pad_frac: float = 0.05,
) -> tuple[float, float]:
    if normalize_mode != "none":
        return -0.02, 1.02

    if gt_pairs.size and pred_pairs.size:
        all_pairs = np.vstack([gt_pairs, pred_pairs])
    elif gt_pairs.size:
        all_pairs = gt_pairs
    elif pred_pairs.size:
        all_pairs = pred_pairs
    else:
        all_pairs = np.zeros((0, 2), dtype=np.float64)

    if all_pairs.size == 0:
        return 0.0, 1.0

    min_val = min(0.0, float(all_pairs.min()))
    max_val = max(1.0, float(all_pairs.max()))
    span = max(max_val - min_val, 1e-6)
    pad = span * pad_frac
    return min_val - pad, max_val + pad


def plot_tmd_persistence_grid(
    gt_diagrams: Mapping[str, object],
    pred_diagrams: Mapping[str, object],
    *,
    filtrations: Sequence[str],
    out_path: Path,
    normalize_mode: str = "minmax",
    ncols: int = 3,
    show_x_axis: bool = True,
    show_titles: bool = True,
    point_alpha: float = 0.75,
) -> Path:
    """Plot GT/pred persistence-diagram overlays for several filtrations in a grid."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_filtrations = len(filtrations)
    if n_filtrations == 0:
        raise ValueError("At least one filtration is required.")

    ncols = max(1, min(ncols, n_filtrations))
    nrows = ceil(n_filtrations / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.8 * nrows), squeeze=False)

    for idx, filtration in enumerate(filtrations):
        ax = axes.flat[idx]
        gt_pairs = _diagram_pairs(gt_diagrams.get(filtration))
        pred_pairs = _diagram_pairs(pred_diagrams.get(filtration))
        lo, hi = _axis_limits(gt_pairs, pred_pairs, normalize_mode=normalize_mode)

        ax.plot([lo, hi], [lo, hi], color="gray", linewidth=1.0, linestyle="--", alpha=0.7)
        if gt_pairs.size:
            ax.scatter(
                gt_pairs[:, 0],
                gt_pairs[:, 1],
                s=26,
                c=GT_COLOR,
                alpha=point_alpha,
                edgecolors="k",
                linewidths=0.25,
                label="GT",
            )
        if pred_pairs.size:
            ax.scatter(
                pred_pairs[:, 0],
                pred_pairs[:, 1],
                s=26,
                c=PRED_COLOR,
                alpha=point_alpha,
                edgecolors="k",
                linewidths=0.25,
                label="Pred",
            )

        if show_titles:
            ax.set_title(_filtration_title(filtration))
        if show_x_axis:
            ax.set_xlabel("Birth")
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax.set_ylabel("Death")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(n_filtrations, nrows * ncols):
        axes.flat[idx].axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)
    return out_path
