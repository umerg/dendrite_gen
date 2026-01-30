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
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


DEFAULT_ANGLES = [(20, 30), (20, 120), (20, 210)]
GT_COLOR = "#1f77b4"
PRED_COLOR = "#8b1e3f"
NODE_SIZE = 26
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


def _plot_points(ax, pts: np.ndarray, title: str, *, color: str) -> None:
    if pts.size == 0:
        ax.set_title(title)
        return
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=POINT_SIZE, c=color, edgecolors="k", linewidths=0.2)
    ax.set_title(title)
    _set_axes_tight(ax, pts)


def _nice_title(label: str, n_nodes: int, suffix: str = "") -> str:
    suffix_str = f" - {suffix}" if suffix else ""
    return f"{label}{suffix_str} (n={n_nodes})"


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
