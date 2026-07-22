"""
Plot helpers for GT/pred tree visualization.

Produces side-by-side 3D plots at multiple angles for:
  1) tree graphs (edges + nodes)
  2) point clouds
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"axes.labelsize": 24, "axes.titlesize": 24})
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

try:
    from utils.tmd_conditioning_utils import PersistenceDiagram0D  # type: ignore
except ModuleNotFoundError:
    from tmd_conditioning_utils import PersistenceDiagram0D  # type: ignore


DEFAULT_ANGLES = [(20, 30), (20, 120), (20, 210)]
GT_COLOR = "#1f77b4"
PRED_COLOR = "#8b1e3f"
NODE_SIZE = 18
POINT_SIZE = 14
EDGE_WIDTH = 1.4
SKELETON_WIDTH = 1.8


def _pos_to_xyz(pos: np.ndarray | list | tuple) -> np.ndarray:
    arr = np.asarray(pos, dtype=float).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size), mode="constant", constant_values=0.0)
    return arr[:3]


def _graph_positions(G: nx.Graph) -> dict[int, np.ndarray]:
    return {n: _pos_to_xyz(G.nodes[n].get("pos", np.zeros(3))) for n in G.nodes()}


def _set_axes_tight(ax, pts: np.ndarray, pad_frac: float = 0.04) -> None:
    if pts.size == 0:
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ranges = maxs - mins
    pad = np.maximum(ranges * pad_frac, 1e-3)
    mins = mins - pad
    maxs = maxs + pad
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(maxs - mins)


def _plot_graph(
    ax,
    G: nx.Graph,
    title: str,
    *,
    node_color: str,
    edge_color: str,
    show_nodes: bool = True,
    show_edges: bool = True,
) -> None:
    pos = _graph_positions(G)
    if not pos:
        ax.set_title(title)
        return
    pts = np.stack(list(pos.values()), axis=0)
    if show_edges:
        lw = SKELETON_WIDTH if not show_nodes else EDGE_WIDTH
        for u, v in G.edges():
            p0 = pos[u]
            p1 = pos[v]
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color=edge_color,
                linewidth=lw,
            )
    if show_nodes:
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=NODE_SIZE,
            c=node_color,
            edgecolors="k",
            linewidths=0.3,
        )
    ax.set_title(title)
    _set_axes_tight(ax, pts)


def _plot_graph_layer(
    ax,
    pos: dict[int, np.ndarray],
    edges: Iterable[tuple[int, int]],
    *,
    node_color: str,
    edge_color: str,
    show_nodes: bool = True,
    show_edges: bool = True,
    node_alpha: float = 0.9,
    edge_alpha: float = 0.8,
) -> None:
    if show_edges:
        for u, v in edges:
            p0 = pos.get(u)
            p1 = pos.get(v)
            if p0 is None or p1 is None:
                continue
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color=edge_color,
                linewidth=EDGE_WIDTH,
                alpha=edge_alpha,
            )
    if show_nodes and pos:
        pts = np.stack(list(pos.values()), axis=0)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=NODE_SIZE,
            c=node_color,
            edgecolors="k",
            linewidths=0.3,
            alpha=node_alpha,
        )


def _plot_points(ax, pts: np.ndarray, title: str, *, color: str) -> None:
    if pts.size == 0:
        ax.set_title(title)
        return
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=POINT_SIZE, c=color, edgecolors="k", linewidths=0.2)
    ax.set_title(title)
    _set_axes_tight(ax, pts)


def _diagram_pairs(diagram: PersistenceDiagram0D | None) -> np.ndarray:
    if diagram is None:
        return np.zeros((0, 2), dtype=np.float64)
    pairs = diagram.as_pairs()
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(pairs, dtype=np.float64)


def _nice_title(label: str, n_nodes: int, suffix: str = "") -> str:
    suffix_str = f" - {suffix}" if suffix else ""
    return f"{label}{suffix_str} (n={n_nodes})"


def _rotation_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix R (3x3) with R @ a == b for unit vectors a, b (Rodrigues)."""
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-8:  # already aligned
        return np.eye(3)
    if c < -1.0 + 1e-8:  # antipodal: rotate 180 deg about any axis perpendicular to a
        p = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(a, np.array([0.0, 1.0, 0.0]))
        p = p / (np.linalg.norm(p) + 1e-12)
        return 2.0 * np.outer(p, p) - np.eye(3)
    v = np.cross(a, b)
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def align_uhat_to_z(G: nx.Graph, uhat) -> nx.Graph:
    """
    Return a copy of ``G`` whose node positions are rotated so the model's symmetry
    axis ``uhat`` points along +z. Plotting this copy and sweeping the matplotlib
    azimuth then orbits the *true* uhat axis for any ``so2_axis`` (not hardcoded z).

    The original graph's positions are left untouched (plotting-only transform).
    """
    R = _rotation_align(np.asarray(uhat, dtype=np.float64).reshape(3), np.array([0.0, 0.0, 1.0]))
    H = G.copy()
    for n in H.nodes():
        p = _pos_to_xyz(H.nodes[n].get("pos", np.zeros(3)))
        H.nodes[n]["pos"] = R @ p
    if "root" in G.graph:
        H.graph["root"] = G.graph["root"]
    return H


def plot_graph_grid_angles(
    graphs: list[nx.Graph],
    *,
    out_dir: Path,
    stem: str,
    file_tag: str,
    angles: Iterable[tuple[float, float]] = DEFAULT_ANGLES,
    uhat=None,
    title_prefix: str = "",
    per_graph_titles: list[str] | None = None,
    node_color: str = PRED_COLOR,
    edge_color: str = "lightgray",
    max_graphs: int = 8,
):
    """
    Build a single figure: one row per graph, one column per (elev, azim) angle,
    rendered in 3D. Azimuths orbit ``uhat`` when provided. Returns (fig, out_path).

    ``per_graph_titles``, when given, supplies a per-row label (e.g. a cell-class
    name) that replaces the shared ``title_prefix`` for that row; otherwise every
    row uses ``title_prefix``.

    The figure is NOT closed here so callers can log it (e.g. to wandb); release it
    with ``plt.close`` afterwards.
    """
    angles = list(angles)
    graphs = list(graphs)[:max_graphs]
    n_rows = max(len(graphs), 1)
    n_cols = max(len(angles), 1)
    fig = plt.figure(figsize=(n_cols * 3.2, n_rows * 3.0))
    for r, G in enumerate(graphs):
        Gp = align_uhat_to_z(G, uhat) if uhat is not None else G
        n_nodes = G.number_of_nodes()
        row_label = (
            per_graph_titles[r]
            if per_graph_titles is not None and r < len(per_graph_titles)
            else title_prefix
        )
        for c, (elev, azim) in enumerate(angles):
            ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1, projection="3d")
            _plot_graph(
                ax,
                Gp,
                _nice_title(row_label, n_nodes, f"az{int(azim)}"),
                node_color=node_color,
                edge_color=edge_color,
            )
            ax.title.set_fontsize(9)  # grid is dense; module default (24) overlaps
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_{file_tag}.png"
    fig.savefig(out_path, dpi=150)
    return fig, out_path


def plot_graph_pair_separate(
    gt: nx.Graph,
    pred: nx.Graph,
    *,
    out_dir: Path,
    stem: str,
    file_tag: str = "graph",
    angles: Iterable[tuple[float, float]] = DEFAULT_ANGLES,
    title_gt: str = "Ground Truth Tree",
    title_pred: str = "Generated Tree",
    node_color_gt: str = GT_COLOR,
    node_color_pred: str = PRED_COLOR,
    edge_color_gt: str = "lightgray",
    edge_color_pred: str = "lightgray",
    show_nodes: bool = True,
    show_edges: bool = True,
    title_suffix: str = "",
) -> list[Path]:
    angles = list(angles)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_gt = gt.number_of_nodes()
    n_pred = pred.number_of_nodes()
    out_paths: list[Path] = []
    for elev, azim in angles:
        fig = plt.figure(figsize=(9, 4.5))
        ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
        ax_pred = fig.add_subplot(1, 2, 2, projection="3d")
        _plot_graph(
            ax_gt,
            gt,
            _nice_title(title_gt, n_gt, title_suffix),
            node_color=node_color_gt,
            edge_color=edge_color_gt,
            show_nodes=show_nodes,
            show_edges=show_edges,
        )
        _plot_graph(
            ax_pred,
            pred,
            _nice_title(title_pred, n_pred, title_suffix),
            node_color=node_color_pred,
            edge_color=edge_color_pred,
            show_nodes=show_nodes,
            show_edges=show_edges,
        )
        ax_gt.view_init(elev=elev, azim=azim)
        ax_pred.view_init(elev=elev, azim=azim)
        ax_gt.set_axis_off()
        ax_pred.set_axis_off()
        fig.tight_layout()
        out_path = out_dir / f"{stem}_{file_tag}_e{int(elev)}_a{int(azim)}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def plot_pointcloud_pair_separate(
    gt_pts: np.ndarray,
    pred_pts: np.ndarray,
    *,
    out_dir: Path,
    stem: str,
    angles: Iterable[tuple[float, float]] = DEFAULT_ANGLES,
    title_gt: str = "Ground Truth Tree",
    title_pred: str = "Generated Tree",
    color_gt: str = GT_COLOR,
    color_pred: str = PRED_COLOR,
    n_nodes_gt: int = 0,
    n_nodes_pred: int = 0,
) -> list[Path]:
    angles = list(angles)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_paths: list[Path] = []
    for elev, azim in angles:
        fig = plt.figure(figsize=(9, 4.5))
        ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
        ax_pred = fig.add_subplot(1, 2, 2, projection="3d")
        _plot_points(
            ax_gt,
            gt_pts,
            _nice_title(title_gt, n_nodes_gt, "Point Cloud"),
            color=color_gt,
        )
        _plot_points(
            ax_pred,
            pred_pts,
            _nice_title(title_pred, n_nodes_pred, "Point Cloud"),
            color=color_pred,
        )
        ax_gt.view_init(elev=elev, azim=azim)
        ax_pred.view_init(elev=elev, azim=azim)
        ax_gt.set_axis_off()
        ax_pred.set_axis_off()
        fig.tight_layout()
        out_path = out_dir / f"{stem}_points_e{int(elev)}_a{int(azim)}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def plot_graph_single_angles(
    G: nx.Graph,
    *,
    out_dir: Path,
    stem: str,
    file_tag: str,
    angles: Iterable[tuple[float, float]] = DEFAULT_ANGLES,
    title: str,
    node_color: str,
    edge_color: str,
    show_nodes: bool = True,
    show_edges: bool = True,
    title_suffix: str = "",
) -> list[Path]:
    angles = list(angles)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_nodes = G.number_of_nodes()
    out_paths: list[Path] = []
    for elev, azim in angles:
        fig = plt.figure(figsize=(5, 4.5))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        _plot_graph(
            ax,
            G,
            _nice_title(title, n_nodes, title_suffix),
            node_color=node_color,
            edge_color=edge_color,
            show_nodes=show_nodes,
            show_edges=show_edges,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        fig.tight_layout()
        out_path = out_dir / f"{stem}_{file_tag}_e{int(elev)}_a{int(azim)}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def plot_pointcloud_single_angles(
    pts: np.ndarray,
    *,
    out_dir: Path,
    stem: str,
    file_tag: str,
    angles: Iterable[tuple[float, float]] = DEFAULT_ANGLES,
    title: str,
    color: str,
    n_nodes: int,
) -> list[Path]:
    angles = list(angles)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for elev, azim in angles:
        fig = plt.figure(figsize=(5, 4.5))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        _plot_points(
            ax,
            pts,
            _nice_title(title, n_nodes, "Point Cloud"),
            color=color,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        fig.tight_layout()
        out_path = out_dir / f"{stem}_{file_tag}_e{int(elev)}_a{int(azim)}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def plot_graph_overlay_azimuths(
    gt: nx.Graph,
    pred: nx.Graph,
    *,
    out_dir: Path,
    stem: str,
    file_tag: str = "overlay",
    angles: Iterable[tuple[float, float]] = DEFAULT_ANGLES,
    title: str = "GT + Pred Overlay",
    node_color_gt: str = GT_COLOR,
    node_color_pred: str = PRED_COLOR,
    edge_color_gt: str = GT_COLOR,
    edge_color_pred: str = PRED_COLOR,
    show_nodes: bool = True,
    show_edges: bool = True,
    node_alpha: float = 0.9,
    edge_alpha: float = 0.75,
    title_suffix: str = "",
) -> list[Path]:
    angles = list(angles)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_gt = _graph_positions(gt)
    pos_pred = _graph_positions(pred)
    pts_list: list[np.ndarray] = []
    if pos_gt:
        pts_list.append(np.stack(list(pos_gt.values()), axis=0))
    if pos_pred:
        pts_list.append(np.stack(list(pos_pred.values()), axis=0))
    pts = np.vstack(pts_list) if pts_list else np.zeros((0, 3), dtype=float)

    title_str = title
    if title_suffix:
        title_str = f"{title_str} - {title_suffix}"
    if gt.number_of_nodes() or pred.number_of_nodes():
        title_str = f"{title_str} (n_gt={gt.number_of_nodes()}, n_pred={pred.number_of_nodes()})"

    out_paths: list[Path] = []
    for elev, azim in angles:
        fig = plt.figure(figsize=(5.5, 4.5))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        _plot_graph_layer(
            ax,
            pos_gt,
            gt.edges(),
            node_color=node_color_gt,
            edge_color=edge_color_gt,
            show_nodes=show_nodes,
            show_edges=show_edges,
            node_alpha=node_alpha,
            edge_alpha=edge_alpha,
        )
        _plot_graph_layer(
            ax,
            pos_pred,
            pred.edges(),
            node_color=node_color_pred,
            edge_color=edge_color_pred,
            show_nodes=show_nodes,
            show_edges=show_edges,
            node_alpha=node_alpha,
            edge_alpha=edge_alpha,
        )
        ax.set_title(title_str)
        _set_axes_tight(ax, pts)
        ax.view_init(elev=elev, azim=float(azim))
        ax.set_axis_off()
        fig.tight_layout()
        out_path = out_dir / f"{stem}_{file_tag}_e{int(elev)}_a{int(azim)}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def plot_persistence_diagram_overlay(
    gt_diag: PersistenceDiagram0D | None,
    pred_diag: PersistenceDiagram0D | None,
    *,
    out_dir: Path,
    stem: str,
    file_tag: str = "tmd_pd",
    title: str = "Persistence Diagram (Path Length From Root)",
    color_gt: str = GT_COLOR,
    color_pred: str = PRED_COLOR,
    alpha_gt: float = 0.7,
    alpha_pred: float = 0.7,
    size_gt: float = 24,
    size_pred: float = 24,
    draw_diagonal: bool = True,
    pad_frac: float = 0.05,
    show_x_axis: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    legend_fontsize: float | None = 16.0,
    y_tick_fontsize: float | None = 20.0,
    x_tick_fontsize: float | None = 20.0,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_pairs = _diagram_pairs(gt_diag)
    pred_pairs = _diagram_pairs(pred_diag)
    if gt_pairs.size and pred_pairs.size:
        all_pairs = np.vstack([gt_pairs, pred_pairs])
    elif gt_pairs.size:
        all_pairs = gt_pairs
    elif pred_pairs.size:
        all_pairs = pred_pairs
    else:
        all_pairs = np.zeros((0, 2), dtype=np.float64)

    if all_pairs.size == 0:
        min_val = 0.0
        max_val = 1.0
    else:
        min_val = min(0.0, float(all_pairs.min()))
        max_val = max(1.0, float(all_pairs.max()))

    span = max(max_val - min_val, 1e-6)
    pad = span * pad_frac
    lo = min_val - pad
    hi = max_val + pad

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    if draw_diagonal:
        ax.plot([lo, hi], [lo, hi], color="gray", linewidth=1.0, linestyle="--", alpha=0.7)
    if gt_pairs.size:
        ax.scatter(
            gt_pairs[:, 0],
            gt_pairs[:, 1],
            s=size_gt,
            c=color_gt,
            alpha=alpha_gt,
            edgecolors="k",
            linewidths=0.3,
            label="GT",
        )
    if pred_pairs.size:
        ax.scatter(
            pred_pairs[:, 0],
            pred_pairs[:, 1],
            s=size_pred,
            c=color_pred,
            alpha=alpha_pred,
            edgecolors="k",
            linewidths=0.3,
            label="Pred",
        )

    ax.set_title(title)
    if show_x_axis:
        ax.set_xlabel("Birth")
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.set_ylabel("Death")
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
        if ylim is None:
            ax.set_ylim(float(xlim[0]), float(xlim[1]))
    else:
        ax.set_xlim(lo, hi)
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    elif xlim is None:
        ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    if gt_pairs.size or pred_pairs.size:
        if legend_fontsize is None:
            ax.legend(frameon=False, loc="upper left")
        else:
            ax.legend(frameon=False, loc="upper left", fontsize=legend_fontsize)
    if x_tick_fontsize is not None:
        ax.tick_params(axis="x", labelsize=x_tick_fontsize)
    if y_tick_fontsize is not None:
        ax.tick_params(axis="y", labelsize=y_tick_fontsize)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    out_path = out_dir / f"{stem}_{file_tag}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_tornado_histogram(
    gt_vals: np.ndarray,
    pred_vals: np.ndarray,
    *,
    bin_edges: np.ndarray,
    out_dir: Path,
    stem: str,
    file_tag: str,
    title: str,
    color_gt: str = GT_COLOR,
    color_pred: str = PRED_COLOR,
    value_label: str = "value",
    density_label: str = "Density",
    show_x_axis: bool = True,
    xlim: tuple[float, float] | None = None,
    alpha_fill: float = 0.35,
    line_width: float = 2.0,
    legend_fontsize: float = 20.0,
    y_tick_fontsize: float = 20.0,
    x_tick_fontsize: float | None = 20.0,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_edges = np.asarray(bin_edges, dtype=float).reshape(-1)
    if bin_edges.size < 2:
        raise ValueError("bin_edges must have at least 2 values.")

    def _hist_density(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size == 0:
            return np.zeros((bin_edges.size - 1,), dtype=float)
        hist, _ = np.histogram(values, bins=bin_edges, density=True)
        return hist.astype(float, copy=False)

    gt_hist = _hist_density(gt_vals)
    pred_hist = _hist_density(pred_vals)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    max_density = float(max(gt_hist.max(initial=0.0), pred_hist.max(initial=0.0), 1e-8))
    x_lim = max_density * 1.15

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    ax.fill_betweenx(centers, 0.0, -gt_hist, color=color_gt, alpha=alpha_fill)
    ax.fill_betweenx(centers, 0.0, pred_hist, color=color_pred, alpha=alpha_fill)
    ax.plot(-gt_hist, centers, color=color_gt, linewidth=line_width, label="GT")
    ax.plot(pred_hist, centers, color=color_pred, linewidth=line_width, label="Pred")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(value_label)
    if show_x_axis:
        ax.set_xlabel(density_label)
        if x_tick_fontsize is not None:
            ax.tick_params(axis="x", labelsize=x_tick_fontsize)
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    else:
        ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(float(bin_edges[0]), float(bin_edges[-1]))
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(frameon=False, loc="upper right", fontsize=legend_fontsize)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.tick_params(axis="y", labelsize=y_tick_fontsize)
    fig.tight_layout()

    out_path = out_dir / f"{stem}_{file_tag}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
