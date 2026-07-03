"""
Geometric comparison metrics for point sets derived from trees.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.spatial import cKDTree


def precision_recall_f1_radius(
    gt_pts: np.ndarray,
    pred_pts: np.ndarray,
    *,
    radius: float,
) -> Dict[str, float]:
    """
    Compute precision/recall/F1 for two point sets under a neighborhood radius.

    A pred point is a TP if it is within `radius` of any GT point.
    A GT point is recovered if it is within `radius` of any pred point.
    """
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")

    gt_pts = np.asarray(gt_pts, dtype=np.float64).reshape(-1, 3)
    pred_pts = np.asarray(pred_pts, dtype=np.float64).reshape(-1, 3)

    if gt_pts.size == 0 and pred_pts.size == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if pred_pts.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if gt_pts.size == 0:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    tree_gt = cKDTree(gt_pts)
    dist_pred, _ = tree_gt.query(pred_pts, k=1, distance_upper_bound=radius)
    precision = float(np.mean(dist_pred <= radius))

    tree_pred = cKDTree(pred_pts)
    dist_gt, _ = tree_pred.query(gt_pts, k=1, distance_upper_bound=radius)
    recall = float(np.mean(dist_gt <= radius))

    if precision + recall <= 0:
        f1 = 0.0
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))

    return {"precision": precision, "recall": recall, "f1": f1}


def _unit_axis(uhat) -> np.ndarray:
    """Coerce ``uhat`` to a unit 3-vector."""
    u = np.asarray(uhat, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(u))
    if n < 1e-12:
        raise ValueError("uhat must be a non-zero 3-vector.")
    return u / n


def height_z_range(pts: np.ndarray, uhat=(0.0, 0.0, 1.0)) -> float:
    """Height = extent (max - min) of the projection onto the equivariance axis ``uhat``.

    Defaults to z (``uhat=(0,0,1)`` ⇒ z-range) for back-compat; pass the dataset's
    ``so2_axis`` for neurons so "height" tracks the real axis (matches
    ``dist_metrics`` ``axial_extent``).
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return float("nan")
    s = pts @ _unit_axis(uhat)
    return float(s.max() - s.min())


def span_xy_diameter(pts: np.ndarray, uhat=(0.0, 0.0, 1.0)) -> float:
    """Max pairwise distance in the plane perpendicular to the axis ``uhat``.

    Defaults to the XY plane (``uhat=z``) for back-compat; pass ``so2_axis`` for
    neurons (matches ``dist_metrics`` ``radial_span``).
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return float("nan")
    if pts.shape[0] < 2:
        return 0.0
    u = _unit_axis(uhat)
    perp = pts - np.outer(pts @ u, u)
    diff = perp[:, None, :] - perp[None, :, :]
    dist2 = np.sum(diff ** 2, axis=-1)
    return float(np.sqrt(np.max(dist2)))


def bbox_diag_length(pts: np.ndarray) -> float:
    """Diagonal length of the 3D bounding box."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return float("nan")
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    return float(np.linalg.norm(maxs - mins))
