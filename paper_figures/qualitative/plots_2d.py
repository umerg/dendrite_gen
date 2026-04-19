"""2D qualitative plotting helpers for paper figures."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..utils.styles import DEFAULT_DPI, EDGE_COLOR, GT_COLOR, PRED_COLOR

if TYPE_CHECKING:
    import networkx as nx


def _pos_to_xyz(pos: np.ndarray | list | tuple) -> np.ndarray:
    arr = np.asarray(pos, dtype=float).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size), mode="constant", constant_values=0.0)
    return arr[:3]


def _projection_indices(projection: str) -> tuple[int, int]:
    # Projection-name mapping is adapted from MorphoPy's NeuronTree.draw_2D()
    # implementation (MorphoPy project, BSD-3-Clause). We use the same axis
    # vocabulary here, but the plotting code itself is reimplemented for this
    # repository's NetworkX tree representation.
    projection_map = {
        "xy": (0, 1),
        "xz": (0, 2),
        "yz": (1, 2),
        "yx": (1, 0),
        "zx": (2, 0),
        "zy": (2, 1),
    }
    if projection not in projection_map:
        raise ValueError(f"Unsupported projection '{projection}'.")
    return projection_map[projection]


def plot_tree_2d(
    ax: plt.Axes,
    graph: "nx.Graph",
    *,
    projection: str = "xy",
    edge_color: str = EDGE_COLOR,
    node_color: str | None = None,
    title: str | None = None,
    linewidth: float = 1.2,
    node_size: float = 10.0,
) -> None:
    """Plot a simple 2D projection of a tree graph."""
    x_idx, y_idx = _projection_indices(projection)

    coords: dict[int, np.ndarray] = {}
    for node in graph.nodes:
        coords[node] = _pos_to_xyz(graph.nodes[node].get("pos", np.zeros(3)))

    for u, v in graph.edges():
        p0 = coords[u]
        p1 = coords[v]
        ax.plot(
            [p0[x_idx], p1[x_idx]],
            [p0[y_idx], p1[y_idx]],
            color=edge_color,
            linewidth=linewidth,
        )

    if node_color is not None and coords:
        pts = np.stack(list(coords.values()), axis=0)
        ax.scatter(
            pts[:, x_idx],
            pts[:, y_idx],
            s=node_size,
            c=node_color,
            edgecolors="none",
            zorder=3,
        )

    if title:
        ax.set_title(title)
    ax.set_xlabel(projection[0].upper())
    ax.set_ylabel(projection[1].upper())
    ax.set_aspect("equal", adjustable="box")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def plot_tree_pair_2d(
    gt: "nx.Graph",
    pred: "nx.Graph",
    *,
    projection: str = "xy",
    out_path: Path,
    title_gt: str = "Ground Truth",
    title_pred: str = "Prediction",
) -> Path:
    """Render a side-by-side 2D pair plot for one GT/pred example."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8))
    plot_tree_2d(
        axes[0],
        gt,
        projection=projection,
        edge_color=GT_COLOR,
        title=title_gt,
    )
    plot_tree_2d(
        axes[1],
        pred,
        projection=projection,
        edge_color=PRED_COLOR,
        title=title_pred,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)
    return out_path


def plot_tree_overlay_2d(
    gt: "nx.Graph",
    pred: "nx.Graph",
    *,
    projection: str = "xy",
    out_path: Path,
    title: str = "GT vs Prediction Overlay",
    gt_color: str = GT_COLOR,
    pred_color: str = PRED_COLOR,
    linewidth: float = 1.4,
    alpha_gt: float = 0.8,
    alpha_pred: float = 0.8,
) -> Path:
    """Render a single-axis 2D overlay of GT and predicted trees."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(5.6, 4.8))
    plot_tree_2d(
        ax,
        gt,
        projection=projection,
        edge_color=gt_color,
        title=title,
        linewidth=linewidth,
    )
    for line in ax.lines:
        line.set_alpha(alpha_gt)

    gt_line_count = len(ax.lines)
    plot_tree_2d(
        ax,
        pred,
        projection=projection,
        edge_color=pred_color,
        linewidth=linewidth,
    )
    for line in ax.lines[gt_line_count:]:
        line.set_alpha(alpha_pred)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)
    return out_path


def plot_tree_gallery_2d(
    gt_graphs: list["nx.Graph"],
    pred_graphs: list["nx.Graph"],
    labels: list[str],
    *,
    projection: str = "xy",
    out_path: Path,
    max_examples: int = 6,
    overlay: bool = False,
) -> Path:
    """Render a small qualitative gallery for several GT/pred pairs."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_examples = min(max_examples, len(gt_graphs), len(pred_graphs), len(labels))
    if n_examples <= 0:
        raise ValueError("At least one example is required to build a gallery.")

    ncols = min(3, n_examples)
    nrows = int(np.ceil(n_examples / ncols))
    width = 4.8 * ncols
    height = 4.2 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)

    for idx in range(nrows * ncols):
        ax = axes.flat[idx]
        if idx >= n_examples:
            ax.axis("off")
            continue

        title = labels[idx]
        if overlay:
            plot_tree_2d(
                ax,
                gt_graphs[idx],
                projection=projection,
                edge_color=GT_COLOR,
                title=title,
                linewidth=1.3,
            )
            for line in ax.lines:
                line.set_alpha(0.8)
            gt_line_count = len(ax.lines)
            plot_tree_2d(
                ax,
                pred_graphs[idx],
                projection=projection,
                edge_color=PRED_COLOR,
                linewidth=1.3,
            )
            for line in ax.lines[gt_line_count:]:
                line.set_alpha(0.8)
        else:
            plot_tree_2d(
                ax,
                gt_graphs[idx],
                projection=projection,
                edge_color=GT_COLOR,
                title=f"{title}\nGT",
                linewidth=1.3,
            )
            plot_tree_2d(
                ax,
                pred_graphs[idx],
                projection=projection,
                edge_color=PRED_COLOR,
                linewidth=1.3,
            )
            pred_edge_count = len(list(pred_graphs[idx].edges()))
            if pred_edge_count > 0:
                for line in ax.lines[:-pred_edge_count]:
                    line.set_alpha(0.25)
                for line in ax.lines[-pred_edge_count:]:
                    line.set_alpha(0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)
    return out_path

