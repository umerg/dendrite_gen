#!/usr/bin/env python3
"""Compute SO(2) geometry stats (axis-parallel + in-plane distances) for SWC graphs."""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np

# Ensure repository root is on sys.path when running the script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.data_loading import load_swc_graphs_from_dir, nx_graph_to_adj_pos


def _parse_axis(arg: str) -> np.ndarray:
    """Parse --axis argument into a unit vector."""
    ax = arg.strip().lower()
    mapping = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    if ax in mapping:
        vec = mapping[ax]
    else:
        parts = [p for p in ax.replace(",", " ").split() if p]
        if len(parts) != 3:
            raise ValueError(
                f"Axis '{arg}' must be 'x', 'y', 'z', or three floats like '0 0 1'."
            )
        vec = np.array([float(p) for p in parts], dtype=np.float64)
    norm = np.linalg.norm(vec)
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError(f"Axis '{arg}' has invalid norm {norm}.")
    return vec / norm


def _parents_from_tree(G, node_order: Iterable[int]) -> np.ndarray:
    """Recover parent indices (0-based, -1 for roots) for nodes in node_order."""
    order = list(node_order)
    idx_map = {nid: i for i, nid in enumerate(order)}
    parent_idx = np.full(len(order), -1, dtype=np.int64)
    if not order:
        return parent_idx
    root = order[0]
    visited = {root}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        node_idx = idx_map[node]
        for nbr in G.neighbors(node):
            if nbr in visited:
                continue
            visited.add(nbr)
            nbr_idx = idx_map[nbr]
            parent_idx[nbr_idx] = node_idx
            queue.append(nbr)
    if len(visited) != len(order):
        missing = set(order) - visited
        raise ValueError(f"Tree traversal missed nodes: {sorted(missing)}")
    return parent_idx


def _directed_edges_from_parent(parent_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build directed parent <-> child edge indices from parent array."""
    src = []
    dst = []
    for child, parent in enumerate(parent_idx.tolist()):
        if parent < 0:
            continue
        src.append(parent)
        dst.append(child)
        src.append(child)
        dst.append(parent)
    if not src:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)


def _summarize(values: np.ndarray) -> dict[str, float]:
    """Return mean, percentile, and range stats for a 1D array."""
    stats = {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p99": float(np.percentile(values, 99)),
        "min": float(values.min()),
        "max": float(values.max()),
    }
    stats["range"] = stats["max"] - stats["min"]
    return stats


def _print_stats(title: str, stats: dict[str, float]) -> None:
    print(f"\n{title}")
    print(f"  count: {stats['count']}")
    print(f"  mean: {stats['mean']:.6f}")
    print(f"  median: {stats['median']:.6f}")
    print(f"  99th percentile: {stats['p99']:.6f}")
    print(f"  min: {stats['min']:.6f}")
    print(f"  max: {stats['max']:.6f}")
    print(f"  range (max-min): {stats['range']:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SO(2) axis / in-plane distance stats for SWC graphs."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing training SWC files (e.g., data/train).",
    )
    parser.add_argument(
        "--axis",
        type=str,
        default="z",
        help="SO(2) axis: 'x', 'y', 'z', or three floats (default: z-axis).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise NotADirectoryError(f"{data_dir} is not a valid directory.")

    uhat = _parse_axis(args.axis)
    graphs = load_swc_graphs_from_dir(data_dir)
    if not graphs:
        raise RuntimeError(f"No SWC graphs found under {data_dir}.")

    du_values: list[np.ndarray] = []
    rperp_values: list[np.ndarray] = []
    ui_values: list[np.ndarray] = []
    abs_ui_values: list[np.ndarray] = []
    total_edges = 0
    skipped = 0

    for G in graphs:
        _, pos, node_order = nx_graph_to_adj_pos(G)
        num_nodes = pos.shape[0]
        if num_nodes <= 1:
            skipped += 1
            continue
        parent_idx = _parents_from_tree(G, node_order)
        src, dst = _directed_edges_from_parent(parent_idx)
        if src.size == 0:
            skipped += 1
            continue
        coords = pos.astype(np.float64, copy=False)
        rel = coords[dst] - coords[src]
        du = rel @ uhat
        r_par = np.outer(du, uhat)
        r_perp_vec = rel - r_par
        r_perp = np.linalg.norm(r_perp_vec, axis=1)
        u_axis = coords @ uhat
        ui = u_axis[dst]

        du_values.append(du)
        rperp_values.append(r_perp)
        ui_values.append(ui)
        abs_ui_values.append(np.abs(ui))
        total_edges += src.size

    if not du_values or not rperp_values:
        raise RuntimeError("No edges with valid geometry found; check dataset contents.")

    du_all = np.concatenate(du_values)
    rperp_all = np.concatenate(rperp_values)
    ui_all = np.concatenate(ui_values)
    abs_ui_all = np.concatenate(abs_ui_values)
    du_stats = _summarize(du_all)
    rperp_stats = _summarize(rperp_all)
    ui_stats = _summarize(ui_all)
    abs_ui_stats = _summarize(abs_ui_all)

    print(f"Processed {len(graphs)} graphs (skipped {skipped} degenerate graphs).")
    print(f"Total directed edges analyzed: {total_edges}")
    print(f"SO(2) axis (uhat): [{uhat[0]:.3f}, {uhat[1]:.3f}, {uhat[2]:.3f}]")
    _print_stats("Axis-parallel distance (du) stats:", du_stats)
    _print_stats("In-plane distance (||r_perp||) stats:", rperp_stats)
    _print_stats("Axis position (u_i) stats:", ui_stats)
    _print_stats("Axis position (|u_i|) stats:", abs_ui_stats)


if __name__ == "__main__":
    main()
