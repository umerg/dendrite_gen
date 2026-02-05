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


def height_z_range(pts: np.ndarray) -> float:
    """Height as z-range (max z - min z)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return float("nan")
    z = pts[:, 2]
    return float(z.max() - z.min())


def span_xy_diameter(pts: np.ndarray) -> float:
    """Max pairwise distance in the XY plane."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return float("nan")
    if pts.shape[0] < 2:
        return 0.0
    xy = pts[:, :2]
    diff = xy[:, None, :] - xy[None, :, :]
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
