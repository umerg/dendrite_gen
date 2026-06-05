"""
Distribution-level validation metrics for generated vs ground-truth trees.

The generator only conditions on TMD, so we do NOT expect a generated tree to
match a specific GT tree node-for-node. Instead we compare the *distribution* of
summary statistics pooled over the generated set against the same statistics
pooled over the GT validation set, and reduce each statistic to a single
Wasserstein-1 (Earth-Mover) scalar so it can be tracked as a curve over training.

All heavy lifting (branch lengths, bifurcation angles, tree-edit distance, TMD
barcodes) reuses existing helpers in ``validation/structural_metrics.py`` and
``utils/tmd.py`` -- nothing geometric is reimplemented here.

Returned dict is flat ``{str: float}`` so ``Trainer.log`` auto-logs every entry
to wandb under ``validation/ema_<beta>/dist/<key>``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import networkx as nx
from scipy.stats import wasserstein_distance

from validation.structural_metrics import (
    branch_length_values,
    bifurcation_angle_values,
    graph_edit_distance_topology,
    _root_tree,
    _pos_to_xyz,
)

try:
    from utils.tmd import compute_tmd_barcode_diagram
except ModuleNotFoundError:  # pragma: no cover - fallback when utils already on path
    from tmd import compute_tmd_barcode_diagram  # type: ignore


# Skip tree-edit-distance for pairs above this node count (zss TED is superlinear
# and has no real timeout), and cap how many pairs we evaluate per validation.
GED_MAX_NODES = 200
GED_MAX_PAIRS = 64


def _root_of(G: nx.Graph) -> int | None:
    root = G.graph.get("root")
    if root is None or root not in G.nodes:
        return None
    return int(root)


def _w1(gen_vals: np.ndarray, gt_vals: np.ndarray) -> float:
    """Wasserstein-1 between two pooled value arrays; nan if either is empty."""
    gen_vals = np.asarray(gen_vals, dtype=np.float64)
    gt_vals = np.asarray(gt_vals, dtype=np.float64)
    gen_vals = gen_vals[np.isfinite(gen_vals)]
    gt_vals = gt_vals[np.isfinite(gt_vals)]
    if gen_vals.size == 0 or gt_vals.size == 0:
        return float("nan")
    return float(wasserstein_distance(gen_vals, gt_vals))


# --- per-graph statistic extractors --------------------------------------------------


def _branch_lengths(G: nx.Graph) -> np.ndarray:
    return branch_length_values(G)


def _bifurcation_angles(G: nx.Graph) -> np.ndarray:
    root = _root_of(G)
    if root is None:
        return np.zeros((0,), dtype=np.float64)
    try:
        return bifurcation_angle_values(G, root=root)
    except ValueError:
        return np.zeros((0,), dtype=np.float64)


def _tmd_bar_lengths(G: nx.Graph) -> np.ndarray:
    """|death - birth| for each persistence interval (raw scale, no per-graph norm)."""
    root = _root_of(G)
    if root is None:
        return np.zeros((0,), dtype=np.float64)
    try:
        _barcode, diagram = compute_tmd_barcode_diagram(G, normalize_mode="none")
    except Exception:
        return np.zeros((0,), dtype=np.float64)
    pairs = np.asarray(diagram.as_pairs(), dtype=np.float64).reshape(-1, 2)
    if pairs.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.abs(pairs[:, 1] - pairs[:, 0])


def _size_extent(G: nx.Graph) -> dict[str, float]:
    """Per-tree scalar size/extent stats. Returns nan-filled dict on degenerate trees."""
    out = {
        "node_count": float(G.number_of_nodes()),
        "leaf_count": float("nan"),
        "bifurcation_count": float("nan"),
        "height": float("nan"),
        "span_xy": float("nan"),
        "bbox_diag": float("nan"),
    }
    root = _root_of(G)
    if root is not None:
        _parent, children = _root_tree(G, root)
        out["leaf_count"] = float(sum(1 for ch in children.values() if len(ch) == 0))
        out["bifurcation_count"] = float(sum(1 for ch in children.values() if len(ch) >= 2))
    if G.number_of_nodes() > 0:
        pts = np.stack([_pos_to_xyz(G.nodes[n].get("pos", np.zeros(3))) for n in G.nodes()], axis=0)
        ranges = pts.max(axis=0) - pts.min(axis=0)
        out["height"] = float(ranges[2])
        out["span_xy"] = float(max(ranges[0], ranges[1]))
        out["bbox_diag"] = float(np.linalg.norm(ranges))
    return out


# --- main entry point -----------------------------------------------------------------


def compute_distribution_metrics(
    gen_graphs: list[nx.Graph],
    gt_graphs: list[nx.Graph],
    *,
    ged_enabled: bool = True,
    ged_timeout: float | None = 5.0,  # kept for config compatibility; zss has no real timeout
) -> dict[str, float]:
    """
    Compare distributions of summary statistics between generated and GT trees.

    Returns a flat dict of float scalars. Keys with insufficient data are nan.
    """
    metrics: dict[str, float] = {}

    def _pool(graphs: Iterable[nx.Graph], fn) -> np.ndarray:
        arrs = [np.asarray(fn(G), dtype=np.float64).reshape(-1) for G in graphs]
        arrs = [a for a in arrs if a.size > 0]
        return np.concatenate(arrs) if arrs else np.zeros((0,), dtype=np.float64)

    # Pooled-distribution statistics (every value across every tree contributes).
    metrics["branch_length_w1"] = _w1(
        _pool(gen_graphs, _branch_lengths), _pool(gt_graphs, _branch_lengths)
    )
    metrics["bifurcation_angle_w1"] = _w1(
        _pool(gen_graphs, _bifurcation_angles), _pool(gt_graphs, _bifurcation_angles)
    )
    metrics["tmd_barlen_w1"] = _w1(
        _pool(gen_graphs, _tmd_bar_lengths), _pool(gt_graphs, _tmd_bar_lengths)
    )

    # Per-tree size/extent statistics (one value per tree -> distribution over trees).
    gen_ext = [_size_extent(G) for G in gen_graphs]
    gt_ext = [_size_extent(G) for G in gt_graphs]
    for key in ("node_count", "leaf_count", "bifurcation_count", "height", "span_xy", "bbox_diag"):
        gen_vals = np.array([d[key] for d in gen_ext], dtype=np.float64)
        gt_vals = np.array([d[key] for d in gt_ext], dtype=np.float64)
        metrics[f"{key}_w1"] = _w1(gen_vals, gt_vals)

    # Average tree-edit distance over index-paired trees (both target the same node
    # count, so pairing by index is valid). Capped by node count and pair count.
    if ged_enabled:
        ged_vals: list[float] = []
        skipped_size = 0
        skipped_error = 0
        n_pairs = min(len(gen_graphs), len(gt_graphs))
        evaluated = 0
        for i in range(n_pairs):
            if evaluated >= GED_MAX_PAIRS:
                break
            g, h = gen_graphs[i], gt_graphs[i]
            if _root_of(g) is None or _root_of(h) is None:
                skipped_error += 1
                continue
            if g.number_of_nodes() > GED_MAX_NODES or h.number_of_nodes() > GED_MAX_NODES:
                skipped_size += 1
                continue
            try:
                d = graph_edit_distance_topology(g, h, timeout=ged_timeout)
            except Exception:
                skipped_error += 1
                continue
            if d is not None:
                ged_vals.append(float(d))
                evaluated += 1
        metrics["tree_edit_dist_mean"] = float(np.mean(ged_vals)) if ged_vals else float("nan")
        considered = max(n_pairs, 1)
        metrics["tree_edit_skipped_frac"] = float((skipped_size + skipped_error) / considered)
        metrics["tree_edit_n_pairs"] = float(len(ged_vals))

    return metrics
