"""2D qualitative plotting helpers."""

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
    title: str | None = None,
    linewidth: float = 1.2,
    show_nodes: bool = False,
    nonroot_node_color: str | None = None,
    root_node_color: str | None = None,
    nonroot_node_size: float = 10.0,
    root_node_size: float = 18.0,
    coord_offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Plot a simple 2D projection of a tree graph."""
    x_idx, y_idx = _projection_indices(projection)
    dx, dy = coord_offset

    coords: dict[int, np.ndarray] = {}
    for node in graph.nodes:
        coords[node] = _pos_to_xyz(graph.nodes[node].get("pos", np.zeros(3)))

    for u, v in graph.edges():
        p0 = coords[u]
        p1 = coords[v]
        ax.plot(
            [p0[x_idx] + dx, p1[x_idx] + dx],
            [p0[y_idx] + dy, p1[y_idx] + dy],
            color=edge_color,
            linewidth=linewidth,
        )

    if show_nodes and coords:
        root = graph.graph.get("root")
        nonroot_ids = [node for node in graph.nodes if node != root]
        if nonroot_ids:
            pts = np.stack([coords[node] for node in nonroot_ids], axis=0)
            ax.scatter(
                pts[:, x_idx] + dx,
                pts[:, y_idx] + dy,
                s=nonroot_node_size,
                c=nonroot_node_color or edge_color,
                edgecolors="none",
                zorder=3,
            )
        if root in coords:
            root_pt = coords[root]
            ax.scatter(
                [root_pt[x_idx] + dx],
                [root_pt[y_idx] + dy],
                s=root_node_size,
                c=root_node_color or edge_color,
                edgecolors="none",
                zorder=4,
            )

    if title:
        ax.set_title(title)
    ax.set_xlabel(projection[0].upper())
    ax.set_ylabel(projection[1].upper())
    ax.set_aspect("equal", adjustable="box")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def _projected_bounds(graph: "nx.Graph", projection: str) -> tuple[float, float, float, float]:
    x_idx, y_idx = _projection_indices(projection)
    if graph.number_of_nodes() == 0:
        return 0.0, 0.0, 0.0, 0.0
    pts = np.stack(
        [_pos_to_xyz(graph.nodes[node].get("pos", np.zeros(3))) for node in graph.nodes],
        axis=0,
    )
    xs = pts[:, x_idx]
    ys = pts[:, y_idx]
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())


def _offset_pair_values(
    gt: "nx.Graph",
    pred: "nx.Graph",
    *,
    projection: str,
    x_gap_scale: float = 0.05,
    y_offset_scale: float = 0.2,
) -> tuple[float, float]:
    """Return the 2D offset used to place the prediction beside the GT tree."""
    gt_xmin, gt_xmax, gt_ymin, gt_ymax = _projected_bounds(gt, projection)
    pred_xmin, pred_xmax, pred_ymin, pred_ymax = _projected_bounds(pred, projection)
    width = max(gt_xmax - gt_xmin, pred_xmax - pred_xmin, 1.0)
    height = max(gt_ymax - gt_ymin, pred_ymax - pred_ymin, 1.0)
    x_gap = x_gap_scale * width
    x_offset = (gt_xmax - pred_xmin) + x_gap
    y_offset = y_offset_scale * height
    return x_offset, y_offset


def plot_tree_pair_2d(
    gt: "nx.Graph",
    pred: "nx.Graph",
    *,
    projection: str = "xy",
    out_path: Path,
    title_gt: str = "Ground Truth",
    title_pred: str = "Prediction",
    show_nodes: bool = False,
    nonroot_node_color: str | None = None,
    root_node_color: str | None = None,
    nonroot_node_size: float = 10.0,
    root_node_size: float = 18.0,
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
        show_nodes=show_nodes,
        nonroot_node_color=nonroot_node_color,
        root_node_color=root_node_color,
        nonroot_node_size=nonroot_node_size,
        root_node_size=root_node_size,
    )
    plot_tree_2d(
        axes[1],
        pred,
        projection=projection,
        edge_color=PRED_COLOR,
        title=title_pred,
        show_nodes=show_nodes,
        nonroot_node_color=nonroot_node_color,
        root_node_color=root_node_color,
        nonroot_node_size=nonroot_node_size,
        root_node_size=root_node_size,
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
    show_nodes: bool = False,
    nonroot_node_color: str | None = None,
    root_node_color: str | None = None,
    nonroot_node_size: float = 10.0,
    root_node_size: float = 18.0,
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
        show_nodes=show_nodes,
        nonroot_node_color=nonroot_node_color,
        root_node_color=root_node_color,
        nonroot_node_size=nonroot_node_size,
        root_node_size=root_node_size,
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
        show_nodes=show_nodes,
        nonroot_node_color=nonroot_node_color,
        root_node_color=root_node_color,
        nonroot_node_size=nonroot_node_size,
        root_node_size=root_node_size,
    )
    for line in ax.lines[gt_line_count:]:
        line.set_alpha(alpha_pred)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)
    return out_path


def plot_tree_offset_pair_2d(
    gt: "nx.Graph",
    pred: "nx.Graph",
    *,
    projection: str = "xy",
    out_path: Path,
    title: str = "GT and Prediction",
    linewidth: float = 1.4,
    x_gap_scale: float = 0.05,
    y_offset_scale: float = 0.2,
    show_nodes: bool = False,
    nonroot_node_color: str | None = None,
    root_node_color: str | None = None,
    nonroot_node_size: float = 10.0,
    root_node_size: float = 18.0,
) -> Path:
    """Render GT and prediction in one axis with a spatial offset between them."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_offset, y_offset = _offset_pair_values(
        gt,
        pred,
        projection=projection,
        x_gap_scale=x_gap_scale,
        y_offset_scale=y_offset_scale,
    )

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.2))
    plot_tree_2d(
        ax,
        gt,
        projection=projection,
        edge_color=GT_COLOR,
        linewidth=linewidth,
        show_nodes=show_nodes,
        nonroot_node_color=nonroot_node_color,
        root_node_color=root_node_color,
        nonroot_node_size=nonroot_node_size,
        root_node_size=root_node_size,
    )
    plot_tree_2d(
        ax,
        pred,
        projection=projection,
        edge_color=PRED_COLOR,
        linewidth=linewidth,
        coord_offset=(x_offset, y_offset),
        show_nodes=show_nodes,
        nonroot_node_color=nonroot_node_color,
        root_node_color=root_node_color,
        nonroot_node_size=nonroot_node_size,
        root_node_size=root_node_size,
    )
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)
    return out_path


def plot_tree_offset_gallery_2d(
    gt_graphs: list["nx.Graph"],
    pred_graphs: list["nx.Graph"],
    labels: list[str],
    *,
    projection: str = "xy",
    out_path: Path,
    max_examples: int = 6,
    x_gap_scale: float = 0.05,
    y_offset_scale: float = 0.2,
    subplot_wspace: float = 0.06,
    subplot_hspace: float = 0.10,
    linewidth: float = 1.3,
    show_nodes: bool = False,
    nonroot_node_color: str | None = None,
    root_node_color: str | None = None,
    nonroot_node_size: float = 10.0,
    root_node_size: float = 18.0,
) -> Path:
    """Render a gallery of GT/pred offset comparisons."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_examples = min(max_examples, len(gt_graphs), len(pred_graphs), len(labels))
    if n_examples <= 0:
        raise ValueError("At least one example is required to build an offset gallery.")

    ncols = min(3, n_examples)
    nrows = int(np.ceil(n_examples / ncols))
    width = 5.2 * ncols
    height = 4.6 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)

    for idx in range(nrows * ncols):
        ax = axes.flat[idx]
        if idx >= n_examples:
            ax.axis("off")
            continue

        gt = gt_graphs[idx]
        pred = pred_graphs[idx]
        x_offset, y_offset = _offset_pair_values(
            gt,
            pred,
            projection=projection,
            x_gap_scale=x_gap_scale,
            y_offset_scale=y_offset_scale,
        )
        plot_tree_2d(
            ax,
            gt,
            projection=projection,
            edge_color=GT_COLOR,
            linewidth=linewidth,
            show_nodes=show_nodes,
            nonroot_node_color=nonroot_node_color,
            root_node_color=root_node_color,
            nonroot_node_size=nonroot_node_size,
            root_node_size=root_node_size,
        )
        plot_tree_2d(
            ax,
            pred,
            projection=projection,
            edge_color=PRED_COLOR,
            linewidth=linewidth,
            coord_offset=(x_offset, y_offset),
            show_nodes=show_nodes,
            nonroot_node_color=nonroot_node_color,
            root_node_color=root_node_color,
            nonroot_node_size=nonroot_node_size,
            root_node_size=root_node_size,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.02,
        top=0.98,
        wspace=subplot_wspace,
        hspace=subplot_hspace,
    )
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
    show_nodes: bool = False,
    nonroot_node_color: str | None = None,
    root_node_color: str | None = None,
    nonroot_node_size: float = 10.0,
    root_node_size: float = 18.0,
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
                show_nodes=show_nodes,
                nonroot_node_color=nonroot_node_color,
                root_node_color=root_node_color,
                nonroot_node_size=nonroot_node_size,
                root_node_size=root_node_size,
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
                show_nodes=show_nodes,
                nonroot_node_color=nonroot_node_color,
                root_node_color=root_node_color,
                nonroot_node_size=nonroot_node_size,
                root_node_size=root_node_size,
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
                show_nodes=show_nodes,
                nonroot_node_color=nonroot_node_color,
                root_node_color=root_node_color,
                nonroot_node_size=nonroot_node_size,
                root_node_size=root_node_size,
            )
            plot_tree_2d(
                ax,
                pred_graphs[idx],
                projection=projection,
                edge_color=PRED_COLOR,
                linewidth=1.3,
                show_nodes=show_nodes,
                nonroot_node_color=nonroot_node_color,
                root_node_color=root_node_color,
                nonroot_node_size=nonroot_node_size,
                root_node_size=root_node_size,
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
