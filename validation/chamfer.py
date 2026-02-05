"""
Compute Chamfer distance between ground-truth (SWC) trees and generated trees.

Pipeline:
  1) Load GT graphs via utils.data_loading.load_swc_graphs_from_dir (same as training).
  2) Load predicted graphs from a validation pickle (see notebooks/validation_graph_viewer.ipynb).
  3) Match GT/pred graphs by node count.
  4) Sample equidistant points along edges (plus node positions) to form point clouds.
  5) Compute Chamfer distance for each matched pair and aggregate stats.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

# Ensure repo root is on sys.path when running as a script from validation/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.data_loading import load_swc_graphs_from_dir
from utils.tmd import compute_tmd_barcode_diagram

try:  # allow running as module or script
    from .plot import (
        plot_graph_pair_separate,
        plot_pointcloud_pair_separate,
        plot_graph_single_angles,
        plot_pointcloud_single_angles,
        plot_graph_overlay_azimuths,
        plot_persistence_diagram_overlay,
        plot_tornado_histogram,
        GT_COLOR,
        PRED_COLOR,
    )
    from .geometric_metric import (
        bbox_diag_length,
        height_z_range,
        precision_recall_f1_radius,
        span_xy_diameter,
    )
    from .structural_metrics import (
        bifurcation_angle_values,
        bottleneck_distance,
        branch_length_values,
        mean_branch_amplitude,
        mean_branch_length,
    )
except Exception:
    from plot import (
        plot_graph_pair_separate,
        plot_pointcloud_pair_separate,
        plot_graph_single_angles,
        plot_pointcloud_single_angles,
        plot_graph_overlay_azimuths,
        plot_persistence_diagram_overlay,
        plot_tornado_histogram,
        GT_COLOR,
        PRED_COLOR,
    )
    from geometric_metric import (
        bbox_diag_length,
        height_z_range,
        precision_recall_f1_radius,
        span_xy_diameter,
    )
    from structural_metrics import (
        bifurcation_angle_values,
        bottleneck_distance,
        branch_length_values,
        mean_branch_amplitude,
        mean_branch_length,
    )


# Sampling distance in the same units as node positions.
DEFAULT_POINT_SPACING = 1.0


def _list_swc_files(dir_path: Path) -> list[Path]:
    """Mirror utils.data_loading.load_swc_graphs_from_dir file selection logic."""
    dir_path = Path(dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        raise NotADirectoryError(f"Provided path is not a directory: {dir_path}")
    files: list[Path] = []
    for swc_file in sorted(dir_path.iterdir()):
        if not swc_file.is_file():
            continue
        name = swc_file.name
        if name.startswith("._"):
            continue
        if not name.endswith(".csv.swc"):
            continue
        files.append(swc_file)
    return files


def _pos_to_xyz(pos: Any) -> np.ndarray:
    arr = np.asarray(pos, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size), mode="constant", constant_values=0.0)
    return arr[:3]


def _ensure_root_from_origin(G: nx.Graph, *, tol: float = 1e-5) -> int | None:
    """
    Ensure G.graph["root"] is set. If missing, choose node closest to origin.

    Preference: any node with ||pos|| <= tol; if multiple, pick the closest.
    Fallback: pick the overall closest node.
    """
    if "root" in G.graph and G.graph["root"] in G.nodes:
        return G.graph["root"]
    if G.number_of_nodes() == 0:
        return None

    best_node = None
    best_norm = None
    within_tol: list[tuple[int, float]] = []
    for nid in G.nodes:
        pos = _pos_to_xyz(G.nodes[nid].get("pos", np.zeros(3)))
        norm = float(np.linalg.norm(pos))
        if norm <= tol:
            within_tol.append((nid, norm))
        if best_norm is None or norm < best_norm:
            best_norm = norm
            best_node = nid

    if within_tol:
        within_tol.sort(key=lambda x: x[1])
        root = within_tol[0][0]
    else:
        root = best_node

    if root is not None:
        G.graph["root"] = root
    return root


def _sample_points_on_graph(G: nx.Graph, spacing: float) -> np.ndarray:
    """Sample points along edges at fixed spacing and include all node positions."""
    if spacing <= 0:
        raise ValueError(f"spacing must be > 0, got {spacing}")
    if G.number_of_nodes() == 0:
        return np.zeros((0, 3), dtype=np.float64)

    points: list[np.ndarray] = []
    # Include all node positions once.
    for n in G.nodes():
        points.append(_pos_to_xyz(G.nodes[n].get("pos", np.zeros(3))))

    # Sample interior points along each edge.
    for u, v in G.edges():
        p0 = _pos_to_xyz(G.nodes[u].get("pos", np.zeros(3)))
        p1 = _pos_to_xyz(G.nodes[v].get("pos", np.zeros(3)))
        vec = p1 - p0
        length = float(np.linalg.norm(vec))
        if length <= spacing:
            continue
        num = int(math.floor(length / spacing))
        if num <= 0:
            continue
        # Distances along edge, excluding endpoints.
        dists = spacing * np.arange(1, num + 1, dtype=np.float64)
        if dists.size > 0 and dists[-1] >= length:
            dists = dists[:-1]
        if dists.size == 0:
            continue
        points_on_edge = p0[None, :] + (dists[:, None] / length) * vec[None, :]
        points.extend(points_on_edge)

    return np.vstack(points).astype(np.float64, copy=False)


def _chamfer_components(
    a: np.ndarray, b: np.ndarray, *, squared: bool = False
) -> tuple[float, float]:
    """Return one-sided Chamfer means: a->b and b->a."""
    if a.size == 0 and b.size == 0:
        return 0.0, 0.0
    if a.size == 0:
        return float("inf"), 0.0
    if b.size == 0:
        return 0.0, float("inf")

    tree_b = cKDTree(b)
    dist_a, _ = tree_b.query(a, k=1)
    tree_a = cKDTree(a)
    dist_b, _ = tree_a.query(b, k=1)

    if squared:
        return float(np.mean(dist_a ** 2)), float(np.mean(dist_b ** 2))
    return float(np.mean(dist_a)), float(np.mean(dist_b))


def _chamfer_distance(a: np.ndarray, b: np.ndarray, *, squared: bool = False) -> float:
    """Symmetric Chamfer distance using nearest-neighbor distances."""
    c_ab, c_ba = _chamfer_components(a, b, squared=squared)
    return float(c_ab + c_ba)


def _barcode_to_list(barcode: np.ndarray) -> list[list[float]]:
    if barcode.size == 0:
        return []
    return barcode.astype(float).tolist()


def _diagram_to_list(diagram: Any) -> list[list[float]]:
    if diagram is None:
        return []
    pairs = diagram.as_pairs()
    if pairs.size == 0:
        return []
    return pairs.astype(float).tolist()


def _extract_pred_graphs(payload: Any, ema_key: str | None) -> list[nx.Graph]:
    """Handle validation pickle formats (ema-keyed dict or direct dict)."""
    if isinstance(payload, dict):
        if "pred_graphs" in payload:
            return payload["pred_graphs"]
        if ema_key is not None:
            if ema_key not in payload:
                available = ", ".join(sorted(payload.keys()))
                raise KeyError(f"EMA key '{ema_key}' not in pickle. Available: {available}")
            inner = payload[ema_key]
            if isinstance(inner, dict) and "pred_graphs" in inner:
                return inner["pred_graphs"]
            raise KeyError(f"EMA entry '{ema_key}' missing 'pred_graphs'.")
        # Try single-entry dict fallback.
        if len(payload) == 1:
            only_val = next(iter(payload.values()))
            if isinstance(only_val, dict) and "pred_graphs" in only_val:
                return only_val["pred_graphs"]
    raise ValueError("Unrecognized pickle format: could not find 'pred_graphs'.")


def _group_by_size(graphs: Iterable[nx.Graph]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, G in enumerate(graphs):
        groups[G.number_of_nodes()].append(idx)
    return groups


def _match_by_size(
    gt_graphs: list[nx.Graph],
    pred_graphs: list[nx.Graph],
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    """Match indices by node count; fall back to closest-size matching."""
    gt_groups = _group_by_size(gt_graphs)
    pred_groups = _group_by_size(pred_graphs)
    unmatched: list[dict[str, int]] = []
    pairs: list[dict[str, int]] = []

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    unmatched_gt: list[int] = []
    unmatched_pred: list[int] = []

    for size in sorted(set(gt_groups) | set(pred_groups)):
        g_list = gt_groups.get(size, [])
        p_list = pred_groups.get(size, [])
        n = min(len(g_list), len(p_list))
        for i in range(n):
            gt_idx = g_list[i]
            pred_idx = p_list[i]
            pairs.append(
                {
                    "gt_idx": gt_idx,
                    "pred_idx": pred_idx,
                    "match_type": "exact",
                    "size_diff": 0,
                }
            )
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
        if len(g_list) != len(p_list):
            unmatched.append(
                {
                    "size": size,
                    "gt_count": len(g_list),
                    "pred_count": len(p_list),
                    "matched": n,
                }
            )
        if len(g_list) > n:
            unmatched_gt.extend(g_list[n:])
        if len(p_list) > n:
            unmatched_pred.extend(p_list[n:])

    if not pred_graphs:
        return pairs, unmatched

    # Match remaining GT graphs to closest-size preds (prefer unused preds first).
    unused_pred = set(unmatched_pred)
    for gt_idx in unmatched_gt:
        gt_size = gt_graphs[gt_idx].number_of_nodes()
        candidate_pool = unused_pred if unused_pred else set(range(len(pred_graphs)))
        best_pred = None
        best_diff = None
        for pred_idx in candidate_pool:
            pred_size = pred_graphs[pred_idx].number_of_nodes()
            diff = abs(gt_size - pred_size)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_pred = pred_idx
        if best_pred is None:
            continue
        pairs.append(
            {
                "gt_idx": gt_idx,
                "pred_idx": best_pred,
                "match_type": "closest",
                "size_diff": int(best_diff) if best_diff is not None else 0,
            }
        )
        matched_gt.add(gt_idx)
        matched_pred.add(best_pred)
        if best_pred in unused_pred:
            unused_pred.remove(best_pred)

    return pairs, unmatched


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "median": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
    }


def _global_hist_edges(
    values: np.ndarray,
    *,
    bins: int,
    default_range: tuple[float, float],
) -> np.ndarray:
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        lo, hi = default_range
    else:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = default_range
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = default_range
    if hi <= lo:
        hi = lo + 1e-6
    return np.linspace(lo, hi, bins + 1, dtype=np.float64)


def run_metrics(
    gt_dir: Path,
    pred_pkl: Path,
    ema_key: str | None,
    spacing: float,
    squared: bool,
    tmd_normalize: Literal["minmax", "max", "none"] = "minmax",
    f1_radius: float = 0.05,
    plot_dir: Path | None = None,
    plot_max: int = 12,
    plot_pairs: bool = False,
    hist_bins: int = 32,
) -> dict[str, Any]:
    gt_dir = Path(gt_dir)
    pred_pkl = Path(pred_pkl)

    gt_files = _list_swc_files(gt_dir)
    gt_graphs = load_swc_graphs_from_dir(gt_dir)
    if len(gt_files) != len(gt_graphs):
        raise RuntimeError("GT file list and loaded graph count mismatch.")

    with pred_pkl.open("rb") as f:
        payload = pickle.load(f)
    pred_graphs = _extract_pred_graphs(payload, ema_key)
    if len(pred_graphs) == 0:
        raise ValueError("No predicted graphs found in pickle.")

    for g in gt_graphs:
        _ensure_root_from_origin(g)
    for g in pred_graphs:
        _ensure_root_from_origin(g)

    length_bin_edges: np.ndarray | None = None
    angle_bin_edges: np.ndarray | None = None
    if plot_dir is not None:
        all_lengths: list[np.ndarray] = []
        all_angles: list[np.ndarray] = []
        for g in gt_graphs:
            all_lengths.append(branch_length_values(g))
            all_angles.append(bifurcation_angle_values(g) if g.number_of_nodes() else np.zeros((0,), dtype=np.float64))
        for g in pred_graphs:
            all_lengths.append(branch_length_values(g))
            all_angles.append(bifurcation_angle_values(g) if g.number_of_nodes() else np.zeros((0,), dtype=np.float64))

        if any(arr.size for arr in all_lengths):
            length_vals = np.concatenate([arr for arr in all_lengths if arr.size], axis=0)
        else:
            length_vals = np.zeros((0,), dtype=np.float64)
        if any(arr.size for arr in all_angles):
            angle_vals = np.concatenate([arr for arr in all_angles if arr.size], axis=0)
        else:
            angle_vals = np.zeros((0,), dtype=np.float64)

        length_bin_edges = _global_hist_edges(length_vals, bins=hist_bins, default_range=(0.0, 1.0))
        angle_bin_edges = _global_hist_edges(angle_vals, bins=hist_bins, default_range=(0.0, 180.0))

    # Debug: sizes overview
    gt_sizes = [g.number_of_nodes() for g in gt_graphs]
    pred_sizes = [g.number_of_nodes() for g in pred_graphs]
    if gt_sizes:
        print(f"Loaded GT graphs: {len(gt_graphs)} | size min/med/max = "
              f"{min(gt_sizes)}/{int(np.median(gt_sizes))}/{max(gt_sizes)}")
    else:
        print("Loaded GT graphs: 0")
    if pred_sizes:
        print(f"Loaded pred graphs: {len(pred_graphs)} | size min/med/max = "
              f"{min(pred_sizes)}/{int(np.median(pred_sizes))}/{max(pred_sizes)}")
    else:
        print("Loaded pred graphs: 0")

    pairs, unmatched = _match_by_size(gt_graphs, pred_graphs)

    per_tree: list[dict[str, Any]] = []
    chamfers: list[float] = []
    by_size: dict[int, list[float]] = defaultdict(list)

    for idx_pair, pair in enumerate(pairs):
        gt_idx = pair["gt_idx"]
        pred_idx = pair["pred_idx"]
        gt = gt_graphs[gt_idx]
        pred = pred_graphs[pred_idx]
        gt_pts = _sample_points_on_graph(gt, spacing)
        pred_pts = _sample_points_on_graph(pred, spacing)
        chamfer_ab, chamfer_ba = _chamfer_components(gt_pts, pred_pts, squared=squared)
        dist = float(chamfer_ab + chamfer_ba)
        mean_len_gt = mean_branch_length(gt)
        mean_len_pred = mean_branch_length(pred)
        mean_amp_gt = mean_branch_amplitude(gt)
        mean_amp_pred = mean_branch_amplitude(pred)
        f1_sampled = precision_recall_f1_radius(gt_pts, pred_pts, radius=f1_radius)
        gt_nodes = np.stack([_pos_to_xyz(gt.nodes[n].get("pos", np.zeros(3))) for n in gt.nodes()], axis=0) if gt.number_of_nodes() else np.zeros((0, 3), dtype=np.float64)
        pred_nodes = np.stack([_pos_to_xyz(pred.nodes[n].get("pos", np.zeros(3))) for n in pred.nodes()], axis=0) if pred.number_of_nodes() else np.zeros((0, 3), dtype=np.float64)
        f1_nodes = precision_recall_f1_radius(gt_nodes, pred_nodes, radius=f1_radius)
        height_gt = height_z_range(gt_nodes)
        height_pred = height_z_range(pred_nodes)
        span_gt = span_xy_diameter(gt_nodes)
        span_pred = span_xy_diameter(pred_nodes)
        bbox_diag_gt = bbox_diag_length(gt_nodes)
        bbox_diag_pred = bbox_diag_length(pred_nodes)

        size = gt.number_of_nodes()
        pred_size = pred.number_of_nodes()
        chamfers.append(dist)
        by_size[size].append(dist)

        gt_barcode, gt_diag = compute_tmd_barcode_diagram(
            gt,
            filtration="path",
            normalize_mode=tmd_normalize,
            weight_edges_by_euclidean=True, # basically geodisc path length instead of number of hops
            simplify_to_critical_tree=True,
        )
        pred_barcode, pred_diag = compute_tmd_barcode_diagram(
            pred,
            filtration="path",
            normalize_mode=tmd_normalize,
            weight_edges_by_euclidean=True, # basically geodisc path length instead of number of hops
            simplify_to_critical_tree=True,
        )
        tmd_bn = bottleneck_distance(gt_diag, pred_diag, canonicalize=False)

        tree_entry: dict[str, Any] = {
            "gt_index": int(gt_idx),
            "gt_name": gt_files[gt_idx].name if gt_idx < len(gt_files) else None,
            "pred_index": int(pred_idx),
            "num_nodes": int(size),
            "pred_num_nodes": int(pred_size),
            "match_type": pair.get("match_type", "exact"),
            "size_diff": int(pair.get("size_diff", abs(size - pred_size))),
            "num_points_gt": int(gt_pts.shape[0]),
            "num_points_pred": int(pred_pts.shape[0]),
            "chamfer": float(dist),
            "chamfer_gt_to_pred": float(chamfer_ab),
            "chamfer_pred_to_gt": float(chamfer_ba),
            "mean_branch_length_gt": float(mean_len_gt),
            "mean_branch_length_pred": float(mean_len_pred),
            "mean_branch_amplitude_deg_gt": float(mean_amp_gt),
            "mean_branch_amplitude_deg_pred": float(mean_amp_pred),
            "f1_sampled": float(f1_sampled["f1"]),
            "precision_sampled": float(f1_sampled["precision"]),
            "recall_sampled": float(f1_sampled["recall"]),
            "f1_nodes": float(f1_nodes["f1"]),
            "precision_nodes": float(f1_nodes["precision"]),
            "recall_nodes": float(f1_nodes["recall"]),
            "height_gt": float(height_gt),
            "height_pred": float(height_pred),
            "span_xy_gt": float(span_gt),
            "span_xy_pred": float(span_pred),
            "bbox_diag_gt": float(bbox_diag_gt),
            "bbox_diag_pred": float(bbox_diag_pred),
            "tmd_path_num_bars_gt": int(gt_barcode.shape[0]),
            "tmd_path_num_bars_pred": int(pred_barcode.shape[0]),
            "tmd_path_barcode_gt": _barcode_to_list(gt_barcode),
            "tmd_path_barcode_pred": _barcode_to_list(pred_barcode),
            "tmd_path_pd_gt": _diagram_to_list(gt_diag),
            "tmd_path_pd_pred": _diagram_to_list(pred_diag),
            "tmd_path_bottleneck": float(tmd_bn),
        }

        if plot_dir is not None and idx_pair < plot_max:
            plot_dir = Path(plot_dir)
            stem = f"gt{gt_idx:04d}_pred{pred_idx:04d}"
            if plot_pairs:
                graph_paths = plot_graph_pair_separate(
                    gt,
                    pred,
                    out_dir=plot_dir,
                    stem=stem,
                    file_tag="graph",
                    title_gt="Ground Truth Tree",
                    title_pred="Reconstructed Tree",
                )
                points_paths = plot_pointcloud_pair_separate(
                    gt_pts,
                    pred_pts,
                    out_dir=plot_dir,
                    stem=stem,
                    title_gt="Ground Truth Tree",
                    title_pred="Reconstructed Tree",
                    color_gt=GT_COLOR,
                    color_pred=PRED_COLOR,
                    n_nodes_gt=gt.number_of_nodes(),
                    n_nodes_pred=pred.number_of_nodes(),
                )
                skeleton_paths = plot_graph_pair_separate(
                    gt,
                    pred,
                    out_dir=plot_dir,
                    stem=stem,
                    file_tag="skeleton",
                    title_gt="Ground Truth Tree",
                    title_pred="Reconstructed Tree",
                    node_color_gt=GT_COLOR,
                    node_color_pred=PRED_COLOR,
                    edge_color_gt=GT_COLOR,
                    edge_color_pred=PRED_COLOR,
                    show_nodes=False,
                    show_edges=True,
                    title_suffix="skeleton",
                )
            gt_graph_single = plot_graph_single_angles(
                gt,
                out_dir=plot_dir,
                stem=f"gt{gt_idx:04d}",
                file_tag="graph",
                title="Ground Truth Tree",
                node_color=GT_COLOR,
                edge_color="lightgray",
                show_nodes=True,
                show_edges=True,
            )
            pred_graph_single = plot_graph_single_angles(
                pred,
                out_dir=plot_dir,
                stem=f"pred{pred_idx:04d}",
                file_tag="graph",
                title="Reconstructed Tree",
                node_color=PRED_COLOR,
                edge_color="lightgray",
                show_nodes=True,
                show_edges=True,
            )
            gt_points_single = plot_pointcloud_single_angles(
                gt_pts,
                out_dir=plot_dir,
                stem=f"gt{gt_idx:04d}",
                file_tag="points",
                title="Ground Truth Tree",
                color=GT_COLOR,
                n_nodes=gt.number_of_nodes(),
            )
            pred_points_single = plot_pointcloud_single_angles(
                pred_pts,
                out_dir=plot_dir,
                stem=f"pred{pred_idx:04d}",
                file_tag="points",
                title="Reconstructed Tree",
                color=PRED_COLOR,
                n_nodes=pred.number_of_nodes(),
            )
            gt_skeleton_single = plot_graph_single_angles(
                gt,
                out_dir=plot_dir,
                stem=f"gt{gt_idx:04d}",
                file_tag="skeleton",
                title="Ground Truth Tree",
                node_color=GT_COLOR,
                edge_color=GT_COLOR,
                show_nodes=False,
                show_edges=True,
                title_suffix="skeleton",
            )
            pred_skeleton_single = plot_graph_single_angles(
                pred,
                out_dir=plot_dir,
                stem=f"pred{pred_idx:04d}",
                file_tag="skeleton",
                title="Reconstructed Tree",
                node_color=PRED_COLOR,
                edge_color=PRED_COLOR,
                show_nodes=False,
                show_edges=True,
                title_suffix="skeleton",
            )
            overlay_paths = plot_graph_overlay_azimuths(
                gt,
                pred,
                out_dir=plot_dir,
                stem=stem,
                file_tag="overlay",
                title="GT + Reconstructed Tree Overlay",
                node_color_gt=GT_COLOR,
                node_color_pred=PRED_COLOR,
                edge_color_gt=GT_COLOR,
                edge_color_pred=PRED_COLOR,
                show_nodes=False,
            )
            pd_overlay_path = plot_persistence_diagram_overlay(
                gt_diag,
                pred_diag,
                out_dir=plot_dir,
                stem=stem,
                file_tag="tmd_pd_path",
                title="PD (Path Length from Root)",
                color_gt=GT_COLOR,
                color_pred=PRED_COLOR,
            )
            hist_paths: list[Path] = []
            if length_bin_edges is not None:
                gt_len_vals = branch_length_values(gt)
                pred_len_vals = branch_length_values(pred)
                hist_paths.append(
                    plot_tornado_histogram(
                        gt_len_vals,
                        pred_len_vals,
                        bin_edges=length_bin_edges,
                        out_dir=plot_dir,
                        stem=stem,
                        file_tag="branch_length_hist",
                        title="Branch Path Length",
                        color_gt=GT_COLOR,
                        color_pred=PRED_COLOR,
                        value_label="Length (metres)",
                    )
                )
            if angle_bin_edges is not None:
                gt_ang_vals = bifurcation_angle_values(gt) if gt.number_of_nodes() else np.zeros((0,), dtype=np.float64)
                pred_ang_vals = bifurcation_angle_values(pred) if pred.number_of_nodes() else np.zeros((0,), dtype=np.float64)
                hist_paths.append(
                    plot_tornado_histogram(
                        gt_ang_vals,
                        pred_ang_vals,
                        bin_edges=angle_bin_edges,
                        out_dir=plot_dir,
                        stem=stem,
                        file_tag="bifurcation_angle_hist",
                        title="Bifurcation Angles",
                        color_gt=GT_COLOR,
                        color_pred=PRED_COLOR,
                        value_label="Angle (deg)",
                    )
                )
            if plot_pairs:
                tree_entry["plot_graph_paths"] = [str(p) for p in graph_paths]
                tree_entry["plot_points_paths"] = [str(p) for p in points_paths]
                tree_entry["plot_skeleton_paths"] = [str(p) for p in skeleton_paths]
            tree_entry["plot_graph_single_paths"] = [str(p) for p in gt_graph_single + pred_graph_single]
            tree_entry["plot_points_single_paths"] = [str(p) for p in gt_points_single + pred_points_single]
            tree_entry["plot_skeleton_single_paths"] = [str(p) for p in gt_skeleton_single + pred_skeleton_single]
            tree_entry["plot_overlay_paths"] = [str(p) for p in overlay_paths]
            tree_entry["plot_tmd_pd_path"] = str(pd_overlay_path)
            if hist_paths:
                tree_entry["plot_hist_paths"] = [str(p) for p in hist_paths]

        per_tree.append(
            tree_entry
        )

    summary = _summarize(chamfers)
    per_size_summary = {str(k): _summarize(v) for k, v in sorted(by_size.items())}

    return {
        "config": {
            "gt_dir": str(gt_dir),
            "pred_pkl": str(pred_pkl),
            "ema_key": ema_key,
            "spacing": float(spacing),
            "squared": bool(squared),
            "tmd": {
                "filtration": "path",
                "normalize_mode": tmd_normalize,
                "weight_edges_by_euclidean": True,
                "simplify_to_critical_tree": True,
            },
            "f1_radius": float(f1_radius),
            "hist_bins": int(hist_bins),
        },
        "summary": summary,
        "per_size_summary": per_size_summary,
        "per_tree": per_tree,
        "unmatched": unmatched,
    }


def run_chamfer(
    gt_dir: Path,
    pred_pkl: Path,
    ema_key: str | None,
    spacing: float,
    squared: bool,
    tmd_normalize: Literal["minmax", "max", "none"] = "minmax",
    f1_radius: float = 0.05,
    plot_dir: Path | None = None,
    plot_max: int = 12,
    plot_pairs: bool = False,
    hist_bins: int = 32,
) -> dict[str, Any]:
    return run_metrics(
        gt_dir=gt_dir,
        pred_pkl=pred_pkl,
        ema_key=ema_key,
        spacing=spacing,
        squared=squared,
        tmd_normalize=tmd_normalize,
        f1_radius=f1_radius,
        plot_dir=plot_dir,
        plot_max=plot_max,
        plot_pairs=plot_pairs,
        hist_bins=hist_bins,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chamfer distance evaluation for generated trees.")
    parser.add_argument("--gt-dir", type=Path, required=True, help="Directory containing GT SWC files.")
    parser.add_argument("--pred-pkl", type=Path, required=True, help="Pickle file with predicted graphs.")
    parser.add_argument("--ema-key", type=str, default=None, help="EMA key inside pickle (e.g., 'ema_0.999').")
    parser.add_argument("--spacing", type=float, default=DEFAULT_POINT_SPACING, help="Point sampling spacing.")
    parser.add_argument("--squared", action="store_true", help="Use squared distances for Chamfer.")
    parser.add_argument("--tmd-normalize", type=str, default="minmax", choices=["minmax", "max", "none"],
                        help="Normalization for TMD filtrations before PD (minmax, max, none).")
    parser.add_argument("--f1-radius", type=float, default=0.2, help="Neighborhood radius for F1/precision/recall.")
    parser.add_argument("--plot-dir", type=Path, default=None, help="Optional directory to save plots.")
    parser.add_argument("--plot-max", type=int, default=12, help="Max number of graph pairs to plot.")
    parser.add_argument("--plot-pairs", action="store_true", help="Also save side-by-side GT/pred plots.")
    parser.add_argument("--hist-bins", type=int, default=32, help="Histogram bin count for branch metrics.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save JSON output.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    results = run_metrics(
        gt_dir=args.gt_dir,
        pred_pkl=args.pred_pkl,
        ema_key=args.ema_key,
        spacing=args.spacing,
        squared=args.squared,
        tmd_normalize=args.tmd_normalize,
        f1_radius=args.f1_radius,
        plot_dir=args.plot_dir,
        plot_max=args.plot_max,
        plot_pairs=args.plot_pairs,
        hist_bins=args.hist_bins,
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(results, f, indent=2)

    summary = results["summary"]
    print("Chamfer summary:")
    print(f"  count={summary['count']} mean={summary['mean']:.6f} std={summary['std']:.6f} "
          f"min={summary['min']:.6f} max={summary['max']:.6f} median={summary['median']:.6f}")
    if results["per_tree"]:
        print("Per-graph stats:")
        for item in results["per_tree"]:
            print(
                f"  gt={item['gt_index']}({item['gt_name']}) "
                f"pred={item['pred_index']} "
                f"n(gt/pred)={item['num_nodes']}/{item['pred_num_nodes']} "
                f"match={item.get('match_type','exact')} diff={item.get('size_diff',0)} "
                f"pts(gt/pred)={item['num_points_gt']}/{item['num_points_pred']} "
                f"chamfer={item['chamfer']:.6f} "
                f"chamfer(gt->pred/pred->gt)={item.get('chamfer_gt_to_pred', float('nan')):.6f}/"
                f"{item.get('chamfer_pred_to_gt', float('nan')):.6f} "
                f"mean_len(gt/pred)={item.get('mean_branch_length_gt', float('nan')):.4f}/"
                f"{item.get('mean_branch_length_pred', float('nan')):.4f} "
                f"mean_amp_deg(gt/pred)={item.get('mean_branch_amplitude_deg_gt', float('nan')):.2f}/"
                f"{item.get('mean_branch_amplitude_deg_pred', float('nan')):.2f} "
                f"tmd_bn={item.get('tmd_path_bottleneck', float('nan')):.6f} "
                f"f1_samp={item.get('f1_sampled', float('nan')):.3f} "
                f"p/r_samp={item.get('precision_sampled', float('nan')):.3f}/"
                f"{item.get('recall_sampled', float('nan')):.3f} "
                f"f1_nodes={item.get('f1_nodes', float('nan')):.3f} "
                f"p/r_nodes={item.get('precision_nodes', float('nan')):.3f}/"
                f"{item.get('recall_nodes', float('nan')):.3f} "
                f"height(gt/pred)={item.get('height_gt', float('nan')):.3f}/"
                f"{item.get('height_pred', float('nan')):.3f} "
                f"span_xy(gt/pred)={item.get('span_xy_gt', float('nan')):.3f}/"
                f"{item.get('span_xy_pred', float('nan')):.3f} "
                f"bbox_diag(gt/pred)={item.get('bbox_diag_gt', float('nan')):.3f}/"
                f"{item.get('bbox_diag_pred', float('nan')):.3f}"
            )
    if results["unmatched"]:
        print("Unmatched sizes:")
        for item in results["unmatched"]:
            print(f"  size={item['size']} gt={item['gt_count']} pred={item['pred_count']} matched={item['matched']}")


if __name__ == "__main__":
    main()
