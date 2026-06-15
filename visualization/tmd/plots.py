"""Persistence-diagram plotting helpers for TMD figures."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
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


def plot_tmd_embedding_scatter(
    records: Sequence[object],
    coords: np.ndarray,
    *,
    out_path: Path,
    reducer: str,
    title: str = "TMD Persistence-Image Embedding",
    connect_pairs: bool = False,
    point_alpha: float = 0.25,
    point_size: float = 5.0,
    marginal_densities: bool = True,
    color_attribute: str | None = None,
    color_label: str | None = None,
    color_cmap: str = "viridis",
) -> Path:
    """Plot a joint 2D embedding of GT and predicted tree TMD vectors."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    coords = np.asarray(coords, dtype=np.float64)
    if len(records) != coords.shape[0]:
        raise ValueError("records and coords must have matching lengths.")

    if marginal_densities:
        fig = plt.figure(figsize=(7.8, 6.7))
        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=(1.15, 4.6),
            height_ratios=(4.6, 1.15),
            hspace=0.04,
            wspace=0.04,
        )
        ax_left = fig.add_subplot(grid[0, 0])
        ax = fig.add_subplot(grid[0, 1], sharey=ax_left)
        ax_bottom = fig.add_subplot(grid[1, 1], sharex=ax)
        ax_corner = fig.add_subplot(grid[1, 0])
        ax_corner.axis("off")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.8))
        ax_left = None
        ax_bottom = None

    finite_coords = coords[np.all(np.isfinite(coords), axis=1)]
    if finite_coords.size:
        x_span = max(float(finite_coords[:, 0].max() - finite_coords[:, 0].min()), 1e-6)
        y_span = max(float(finite_coords[:, 1].max() - finite_coords[:, 1].min()), 1e-6)
        xlim = (
            float(finite_coords[:, 0].min() - 0.05 * x_span),
            float(finite_coords[:, 0].max() + 0.05 * x_span),
        )
        ylim = (
            float(finite_coords[:, 1].min() - 0.05 * y_span),
            float(finite_coords[:, 1].max() + 0.05 * y_span),
        )
    else:
        xlim = (-1.0, 1.0)
        ylim = (-1.0, 1.0)

    if connect_pairs:
        by_pair: dict[int, dict[str, int]] = {}
        for idx, record in enumerate(records):
            by_pair.setdefault(int(record.pair_index), {})[str(record.source)] = idx
        for source_idx in by_pair.values():
            if "gt" not in source_idx or "pred" not in source_idx:
                continue
            gt_coord = coords[source_idx["gt"]]
            pred_coord = coords[source_idx["pred"]]
            ax.plot(
                [gt_coord[0], pred_coord[0]],
                [gt_coord[1], pred_coord[1]],
                color="#9ca3af",
                linewidth=0.8,
                alpha=0.45,
                zorder=1,
            )

    sources = {
        "gt": {"label": "GT", "color": GT_COLOR, "marker": "o"},
        "pred": {"label": "Pred", "color": PRED_COLOR, "marker": "^"},
    }
    color_values = _record_attribute_values(records, color_attribute)
    color_norm = _attribute_color_norm(color_values)
    color_map = plt.get_cmap(color_cmap)

    for source, style in sources.items():
        idxs = [idx for idx, record in enumerate(records) if record.source == source]
        if not idxs:
            continue
        pts = coords[idxs]
        if marginal_densities and ax_left is not None and ax_bottom is not None:
            _plot_marginal_density(
                ax_bottom,
                pts[:, 0],
                value_range=xlim,
                color=style["color"],
                label=style["label"],
                orientation="x",
            )
            _plot_marginal_density(
                ax_left,
                pts[:, 1],
                value_range=ylim,
                color=style["color"],
                label=None,
                orientation="y",
            )
        _scatter_embedding_points(
            ax,
            pts,
            values=color_values[idxs] if color_values is not None else None,
            marker=style["marker"],
            point_size=point_size,
            point_alpha=point_alpha,
            color_norm=color_norm,
            color_map=color_map,
        )

    label_prefix = reducer.upper() if reducer != "umap" else "UMAP"
    ax.set_xlabel(f"{label_prefix} 1")
    ax.set_ylabel(f"{label_prefix} 2")
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if marginal_densities and ax_left is not None and ax_bottom is not None:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)

        ax_bottom.set_xlabel(f"{label_prefix} 1")
        ax_bottom.spines["top"].set_visible(False)
        ax_bottom.spines["right"].set_visible(False)
        ax_bottom.spines["left"].set_visible(False)
        ax_bottom.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax_bottom.set_yticks([])
        handles, labels = ax_bottom.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper right", frameon=False, ncol=2)

        ax_left.set_ylabel(f"{label_prefix} 2")
        ax_left.spines["top"].set_visible(False)
        ax_left.spines["right"].set_visible(False)
        ax_left.spines["bottom"].set_visible(False)
        ax_left.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_left.set_xticks([])
        ax_left.invert_xaxis()
    else:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, frameon=True)
    if color_attribute is not None and color_norm is not None:
        mappable = plt.cm.ScalarMappable(norm=color_norm, cmap=color_map)
        mappable.set_array([])
        colorbar_axes = [ax, ax_bottom] if marginal_densities and ax_bottom is not None else ax
        colorbar = fig.colorbar(mappable, ax=colorbar_axes, fraction=0.046, pad=0.02)
        colorbar.set_label(color_label or _nice_attribute_name(color_attribute))
    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)
    return out_path


def plot_tmd_pair_distance_attribute_scatter(
    records: Sequence[object],
    *,
    out_path: Path,
    attribute: str,
    embedding_name: str,
    distance_label: str = "Persistence-diagram Wasserstein distance",
    point_alpha: float = 0.75,
    point_size: float = 5.0,
) -> Path:
    """Plot one point per GT/pred pair: diagram distance against a GT tree attribute."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    distances = np.asarray(
        [float(record.distance) for record in records],
        dtype=np.float64,
    )
    attribute_values = np.asarray(
        [float(record.attribute_value) for record in records],
        dtype=np.float64,
    )
    finite = np.isfinite(distances) & np.isfinite(attribute_values)

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
    if np.any(finite):
        ax.scatter(
            distances[finite],
            attribute_values[finite],
            s=point_size,
            c=PRED_COLOR,
            alpha=point_alpha,
            edgecolors="black",
            linewidths=0.25,
        )
        ax.set_xlim(_value_axis_limits(distances[finite], include_zero=True))
        ax.set_ylim(_value_axis_limits(attribute_values[finite], include_zero=False))
        ax.text(
            0.04,
            0.96,
            f"n={int(np.sum(finite))}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            color="#4b5563",
            fontsize=9,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No finite paired values",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#6b7280",
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

    nice_attribute = _nice_attribute_name(attribute)
    ax.set_xlabel(distance_label)
    ax.set_ylabel(f"GT {nice_attribute}")
    ax.set_title(f"{_filtration_title(embedding_name)}: Diagram Distance vs GT {nice_attribute}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)
    return out_path


def plot_tmd_mean_persistence_images(
    gt_images: Sequence[np.ndarray],
    pred_images: Sequence[np.ndarray],
    *,
    out_path: Path,
    filtration: str,
    n_bins: int,
    birth_range: tuple[float, float] = (0.0, 1.0),
    persistence_range: tuple[float, float] = (0.0, 1.0),
) -> Path:
    """Plot mean GT/pred persistence images and their difference for one filtration."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gt_stack = _persistence_image_stack(gt_images, n_bins=n_bins)
    pred_stack = _persistence_image_stack(pred_images, n_bins=n_bins)
    mean_gt = _mean_persistence_image(gt_stack, n_bins=n_bins)
    mean_pred = _mean_persistence_image(pred_stack, n_bins=n_bins)
    diff = mean_pred - mean_gt
    abs_diff = np.abs(diff)

    mean_vmax = _positive_vmax([mean_gt, mean_pred])
    diff_absmax = _positive_vmax([np.abs(diff)])
    abs_vmax = _positive_vmax([abs_diff])

    panels = [
        ("Mean GT", mean_gt, "viridis", Normalize(vmin=0.0, vmax=mean_vmax)),
        ("Mean Pred", mean_pred, "viridis", Normalize(vmin=0.0, vmax=mean_vmax)),
        (
            "Pred - GT",
            diff,
            "coolwarm",
            TwoSlopeNorm(vmin=-diff_absmax, vcenter=0.0, vmax=diff_absmax),
        ),
        ("|Pred - GT|", abs_diff, "magma", Normalize(vmin=0.0, vmax=abs_vmax)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.8), constrained_layout=True)
    extent = (
        float(birth_range[0]),
        float(birth_range[1]),
        float(persistence_range[0]),
        float(persistence_range[1]),
    )
    for idx, (panel_title, image, cmap, norm) in enumerate(panels):
        ax = axes[idx]
        im = ax.imshow(
            image.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            norm=norm,
        )
        ax.set_title(panel_title)
        ax.set_xlabel("Birth")
        if idx == 0:
            ax.set_ylabel("Persistence")
        else:
            ax.set_ylabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Mean Persistence Images: {_filtration_title(filtration)}")
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)
    return out_path


def _persistence_image_stack(
    images: Sequence[np.ndarray],
    *,
    n_bins: int,
) -> np.ndarray:
    """Return images as an (n_images, n_bins, n_bins) stack."""
    matrices = []
    for image in images:
        arr = np.asarray(image, dtype=np.float64)
        if arr.size == 0:
            continue
        if arr.shape == (n_bins, n_bins):
            matrices.append(arr)
        elif arr.size == n_bins * n_bins:
            matrices.append(arr.reshape(n_bins, n_bins))
        else:
            raise ValueError(
                f"Persistence image has shape {arr.shape}; expected "
                f"({n_bins}, {n_bins}) or a flat vector of length {n_bins * n_bins}."
            )
    if not matrices:
        return np.zeros((0, n_bins, n_bins), dtype=np.float64)
    return np.stack(matrices, axis=0)


def _mean_persistence_image(stack: np.ndarray, *, n_bins: int) -> np.ndarray:
    if stack.shape[0] == 0:
        return np.zeros((n_bins, n_bins), dtype=np.float64)
    return np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0).mean(axis=0)


def _positive_vmax(arrays: Sequence[np.ndarray], *, fallback: float = 1.0) -> float:
    values = []
    for arr in arrays:
        finite = np.asarray(arr, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            values.append(float(np.max(finite)))
    if not values:
        return fallback
    vmax = max(values)
    if vmax <= 0.0:
        return fallback
    return vmax


def _record_attribute_values(
    records: Sequence[object],
    attribute: str | None,
) -> np.ndarray | None:
    """Extract a numeric attribute value per embedding record."""
    if attribute is None:
        return None
    values = []
    for record in records:
        attributes = getattr(record, "attributes", {}) or {}
        try:
            value = float(attributes.get(attribute, np.nan))
        except (TypeError, ValueError):
            value = float("nan")
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def _attribute_color_norm(values: np.ndarray | None) -> Normalize | None:
    if values is None:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None

    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo:
        pad = max(abs(lo) * 0.05, 1e-6)
        lo -= pad
        hi += pad
    return Normalize(vmin=lo, vmax=hi)


def _value_axis_limits(values: np.ndarray, *, include_zero: bool) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0

    lo = float(np.min(values))
    hi = float(np.max(values))
    lower_floor = 0.0 if include_zero else None
    if lower_floor is not None:
        lo = min(lower_floor, lo)
    if hi <= lo:
        pad = max(abs(lo) * 0.05, 1e-6)
    else:
        pad = (hi - lo) * 0.05
    lower = lo - pad
    if lower_floor is not None:
        lower = lower_floor
    return lower, hi + pad


def _scatter_embedding_points(
    ax: plt.Axes,
    pts: np.ndarray,
    *,
    values: np.ndarray | None,
    marker: str,
    point_size: float,
    point_alpha: float,
    color_norm: Normalize | None,
    color_map,
) -> None:
    """Scatter one source group, optionally colored by a scalar attribute."""
    if values is None or color_norm is None:
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=point_size,
            c="#4b5563",
            marker=marker,
            alpha=point_alpha,
            edgecolors="none",
            zorder=3,
        )
        return

    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if np.any(finite):
        ax.scatter(
            pts[finite, 0],
            pts[finite, 1],
            s=point_size,
            c=values[finite],
            cmap=color_map,
            norm=color_norm,
            marker=marker,
            alpha=point_alpha,
            edgecolors="none",
            zorder=3,
        )
    if np.any(~finite):
        ax.scatter(
            pts[~finite, 0],
            pts[~finite, 1],
            s=point_size,
            c="#d1d5db",
            marker=marker,
            alpha=point_alpha,
            edgecolors="none",
            zorder=2,
        )


def _nice_attribute_name(attribute: str) -> str:
    return {
        "num_nodes": "Nodes",
        "num_edges": "Edges",
        "num_tips": "Tips",
        "num_branchpoints": "Branch Points",
        "height": "Height",
        "span_xy": "XY Span",
        "bbox_diag": "BBox Diag",
        "max_path_dist": "Max Path Dist",
        "max_radial_dist": "Max Radial Dist",
        "mean_branch_length": "Mean Branch Length",
        "mean_bifurcation_angle_deg": "Mean Branch Angle",
        "max_branch_order": "Max Branch Order",
    }.get(attribute, attribute.replace("_", " ").title())


def _density_curve(
    values: np.ndarray,
    *,
    value_range: tuple[float, float],
    n_points: int = 160,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a smooth-ish 1D density curve, falling back to a histogram."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None

    lo, hi = value_range
    if hi <= lo:
        return None
    xs = np.linspace(lo, hi, n_points)

    if float(np.max(values) - np.min(values)) <= 1e-12:
        return None

    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(values)
        ys = kde(xs)
    except Exception:
        hist, edges = np.histogram(values, bins=min(60, max(10, values.size // 4)), range=value_range, density=True)
        centers = edges[:-1] + 0.5 * np.diff(edges)
        ys = np.interp(xs, centers, hist, left=0.0, right=0.0)

    ys = np.nan_to_num(ys, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.any(ys > 0):
        return None
    return xs, ys


def _plot_marginal_density(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    color: str,
    label: str | None,
    value_range: tuple[float, float],
    orientation: str,
) -> None:
    """Draw one marginal density line on an axis."""
    curve = _density_curve(values, value_range=value_range)
    if curve is None:
        return
    xs, ys = curve
    if orientation == "x":
        ax.plot(xs, ys, color=color, linewidth=1.4, label=label)
        ax.set_xlim(value_range)
    elif orientation == "y":
        ax.plot(ys, xs, color=color, linewidth=1.4, label=label)
        ax.set_ylim(value_range)
    else:
        raise ValueError(f"Unknown density orientation: {orientation!r}")
