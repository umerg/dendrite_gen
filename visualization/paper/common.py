"""Shared axis-level helpers for the paper figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import (
    AsinhNorm,
    Colormap,
    Normalize,
    TwoSlopeNorm,
)
import networkx as nx
import numpy as np
from scipy.ndimage import gaussian_filter1d

from dendrite_gen.metrics.distributions import tree_distribution
from dendrite_gen.visualization.paper.colors import (
    BACKGROUND_COLOR,
    BASELINE_COLOR,
    DEPTH_GREEN_CMAP,
    DIAGONAL_COLOR,
    LIGHT_GRID_COLOR,
    NEUTRAL_COLOR,
    OURS_COLOR,
    REFERENCE_COLOR,
    TREE_DEPTH_CMAP,
)
from dendrite_gen.visualization.qualitative.plots_2d import plot_tree_2d


@dataclass(frozen=True)
class DensitySamples:
    """Weighted observations and the number of independent sampling units."""

    values: np.ndarray
    weights: np.ndarray
    independent_count: int


def configure_paper_style() -> None:
    """Apply a compact style suitable for a two-column conference paper."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "lines.linewidth": 1.2,
            "savefig.facecolor": BACKGROUND_COLOR,
            "figure.facecolor": BACKGROUND_COLOR,
        }
    )


def save_png_pdf(fig: plt.Figure, output_stem: Path, *, dpi: int = 300) -> tuple[Path, Path]:
    """Save a figure as a quick-look PNG and vector PDF."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=BACKGROUND_COLOR)
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=BACKGROUND_COLOR)
    return png_path, pdf_path


def _projection_indices(projection: str) -> tuple[int, int]:
    mapping = {
        "xy": (0, 1),
        "xz": (0, 2),
        "yz": (1, 2),
        "yx": (1, 0),
        "zx": (2, 0),
        "zy": (2, 1),
    }
    try:
        return mapping[projection]
    except KeyError as exc:
        raise ValueError(f"Unsupported projection: {projection!r}") from exc


def _projection_depth_index(projection: str) -> int:
    """Return the coordinate axis omitted by a 2D projection."""
    x_index, y_index = _projection_indices(projection)
    return next(index for index in range(3) if index not in (x_index, y_index))


def _node_position(graph: nx.Graph, node: object) -> np.ndarray:
    position = np.asarray(
        graph.nodes[node].get("pos", np.zeros(3)), dtype=float
    ).reshape(-1)
    if position.size < 3:
        position = np.pad(position, (0, 3 - position.size))
    return position[:3]


def symmetric_depth_norm(
    graphs: Sequence[nx.Graph],
    *,
    projection: str = "xy",
    percentile: float = 100.0,
    nonlinear_width: float | None = None,
) -> Normalize:
    """Return one root-relative, symmetric depth scale for several trees.

    ``nonlinear_width`` is the approximate width of the central linear region as
    a fraction of the displayed half-range. Smaller values devote more of the
    colormap to depths near the root plane and compress the outlying depths.
    ``None`` retains a linear scale.
    """
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in the interval (0, 100].")
    if nonlinear_width is not None and (
        not np.isfinite(nonlinear_width) or nonlinear_width <= 0.0
    ):
        raise ValueError("nonlinear_width must be a positive finite value.")
    depth_index = _projection_depth_index(projection)
    absolute_depths = []
    for graph in graphs:
        root = graph.graph.get("root")
        root_depth = _node_position(graph, root)[depth_index] if root in graph else 0.0
        for node in graph.nodes:
            depth = _node_position(graph, node)[depth_index] - root_depth
            if np.isfinite(depth):
                absolute_depths.append(abs(float(depth)))
    maximum = (
        float(np.percentile(absolute_depths, percentile)) if absolute_depths else 0.0
    )
    maximum = max(maximum, 1.0)
    if nonlinear_width is not None:
        return AsinhNorm(
            linear_width=nonlinear_width * maximum,
            vmin=-maximum,
            vmax=maximum,
        )
    return TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)


def _plot_depth_colored_edges(
    ax: plt.Axes,
    graph: nx.Graph,
    *,
    projection: str,
    cmap: Colormap,
    norm: Normalize,
    linewidth: float,
    target_segment_length: float,
    layout_center: tuple[float, float] | None = None,
    layout_rotation_deg: float = 0.0,
    layout_scale: float = 1.0,
    zorder: float = 2.0,
) -> LineCollection:
    """Draw projected edges with a continuous, root-relative depth gradient."""
    if not np.isfinite(target_segment_length) or target_segment_length <= 0.0:
        raise ValueError("target_segment_length must be a positive finite value.")
    if not np.isfinite(layout_rotation_deg):
        raise ValueError("layout_rotation_deg must be finite.")
    if not np.isfinite(layout_scale) or layout_scale <= 0.0:
        raise ValueError("layout_scale must be a positive finite value.")
    if layout_center is None and (
        not np.isclose(layout_rotation_deg, 0.0)
        or not np.isclose(layout_scale, 1.0)
    ):
        raise ValueError(
            "layout_center is required when rotating or scaling a tree for layout."
        )

    x_index, y_index = _projection_indices(projection)
    depth_index = _projection_depth_index(projection)
    root = graph.graph.get("root")
    root_position = _node_position(graph, root) if root in graph else np.zeros(3)
    root_depth = root_position[depth_index]

    rotation = None
    layout_offset = None
    if layout_center is not None:
        center = np.asarray(layout_center, dtype=float).reshape(-1)
        if center.size != 2 or not np.all(np.isfinite(center)):
            raise ValueError("layout_center must contain two finite coordinates.")

        angle = np.deg2rad(layout_rotation_deg)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=float,
        )
        root_projected = root_position[[x_index, y_index]]
        if graph.number_of_nodes() > 0:
            projected_nodes = np.stack(
                [
                    _node_position(graph, node)[[x_index, y_index]]
                    for node in graph.nodes
                ],
                axis=0,
            )
            projected_nodes = layout_scale * (
                (projected_nodes - root_projected) @ rotation.T
            )
            bounding_center = 0.5 * (
                projected_nodes.min(axis=0) + projected_nodes.max(axis=0)
            )
        else:
            bounding_center = np.zeros(2)
        layout_offset = center - bounding_center

    segment_blocks: list[np.ndarray] = []
    depth_blocks: list[np.ndarray] = []
    for first, second in graph.edges():
        start = _node_position(graph, first)
        end = _node_position(graph, second)
        edge_length = float(np.linalg.norm(end - start))
        if not np.isfinite(edge_length) or edge_length <= 1e-12:
            continue

        count = max(1, int(np.ceil(edge_length / target_segment_length)))
        interpolation = np.linspace(0.0, 1.0, count + 1)[:, None]
        points = start + interpolation * (end - start)
        projected = points[:, [x_index, y_index]]
        if rotation is not None and layout_offset is not None:
            projected = layout_scale * (
                (projected - root_position[[x_index, y_index]]) @ rotation.T
            ) + layout_offset
        segment_blocks.append(np.stack([projected[:-1], projected[1:]], axis=1))
        depth_blocks.append(
            0.5 * (points[:-1, depth_index] + points[1:, depth_index]) - root_depth
        )

    if segment_blocks:
        segments = np.concatenate(segment_blocks, axis=0)
        depths = np.concatenate(depth_blocks, axis=0)
        # Paint the back of the tree first so front-facing segments win at crossings.
        order = np.argsort(depths, kind="stable")
        segments = segments[order]
        depths = depths[order]
    else:
        segments = np.zeros((0, 2, 2), dtype=float)
        depths = np.zeros(0, dtype=float)

    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidths=linewidth,
        antialiaseds=True,
        capstyle="round",
        zorder=zorder,
    )
    collection.set_array(depths)
    ax.add_collection(collection)
    ax.autoscale_view()
    return collection


def projected_limits(
    graphs: Sequence[nx.Graph],
    *,
    projection: str = "xy",
    pad_fraction: float = 0.06,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return shared projection limits for one or more root-centred trees."""
    x_index, y_index = _projection_indices(projection)
    arrays = []
    for graph in graphs:
        if graph.number_of_nodes() == 0:
            continue
        arrays.append(
            np.stack(
                [np.asarray(graph.nodes[node]["pos"], dtype=float) for node in graph.nodes],
                axis=0,
            )
        )
    if not arrays:
        return (-1.0, 1.0), (-1.0, 1.0)

    points = np.concatenate(arrays, axis=0)
    x_min, x_max = float(points[:, x_index].min()), float(points[:, x_index].max())
    y_min, y_max = float(points[:, y_index].min()), float(points[:, y_index].max())
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    return (
        (x_min - pad_fraction * x_span, x_max + pad_fraction * x_span),
        (y_min - pad_fraction * y_span, y_max + pad_fraction * y_span),
    )


def plot_tree_panel(
    ax: plt.Axes,
    graph: nx.Graph,
    *,
    color: str,
    projection: str = "xy",
    title: str | None = None,
    linewidth: float = 0.85,
    limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
    depth_coloring: bool = False,
    depth_cmap: Colormap = TREE_DEPTH_CMAP,
    depth_norm: Normalize | None = None,
    depth_segment_length: float = 1.0,
    layout_center: tuple[float, float] | None = None,
    layout_rotation_deg: float = 0.0,
    layout_scale: float = 1.0,
    layout_zorder: float = 2.0,
) -> LineCollection | None:
    """Draw an axis-free tree silhouette onto an existing paper panel."""
    collection = None
    if depth_coloring:
        collection = _plot_depth_colored_edges(
            ax,
            graph,
            projection=projection,
            cmap=depth_cmap,
            norm=depth_norm or symmetric_depth_norm([graph], projection=projection),
            linewidth=linewidth,
            target_segment_length=depth_segment_length,
            layout_center=layout_center,
            layout_rotation_deg=layout_rotation_deg,
            layout_scale=layout_scale,
            zorder=layout_zorder,
        )
        if title:
            ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
    else:
        if layout_center is not None:
            raise ValueError("Free-layout transforms require depth_coloring=True.")
        plot_tree_2d(
            ax,
            graph,
            projection=projection,
            edge_color=color,
            title=title,
            linewidth=linewidth,
        )
    if limits is not None:
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
    ax.set_axis_off()
    return collection


def draw_scale_bar(
    ax: plt.Axes,
    *,
    length: float,
    label: str,
    color: str = NEUTRAL_COLOR,
) -> None:
    """Draw a physical scale bar in the lower-left of an axis."""
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x0 = x_min + 0.08 * (x_max - x_min)
    y0 = y_min + 0.08 * (y_max - y_min)
    ax.plot([x0, x0 + length], [y0, y0], color=color, linewidth=1.4, clip_on=False)
    ax.text(
        x0 + 0.5 * length,
        y0 + 0.035 * (y_max - y_min),
        label,
        ha="center",
        va="bottom",
        color=color,
        fontsize=6.5,
    )


def scalar_density_samples(values: Sequence[float]) -> DensitySamples:
    """Build equally weighted samples where each scalar corresponds to one tree."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return DensitySamples(np.zeros(0), np.zeros(0), 0)
    weights = np.full(array.size, 1.0 / array.size, dtype=float)
    return DensitySamples(array, weights, int(array.size))


def tree_balanced_distribution_samples(
    graphs: Sequence[nx.Graph],
    distribution_name: str,
) -> DensitySamples:
    """Combine per-tree distributions while assigning equal mass to each tree."""
    parts: list[tuple[np.ndarray, np.ndarray]] = []
    for graph_index, graph in enumerate(graphs):
        try:
            distribution = tree_distribution(graph, distribution_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to compute {distribution_name!r} for graph {graph_index}."
            ) from exc
        values = np.asarray(distribution.values, dtype=float)
        finite = np.isfinite(values)
        values = values[finite]
        if values.size == 0:
            continue
        if distribution.weights is None:
            weights = np.ones(values.size, dtype=float)
        else:
            weights = np.asarray(distribution.weights, dtype=float)[finite]
        valid_weight = np.isfinite(weights) & (weights > 0.0)
        values = values[valid_weight]
        weights = weights[valid_weight]
        if values.size == 0 or float(weights.sum()) <= 0.0:
            continue
        parts.append((values, weights / weights.sum()))

    if not parts:
        return DensitySamples(np.zeros(0), np.zeros(0), 0)

    tree_count = len(parts)
    values = np.concatenate([part[0] for part in parts])
    weights = np.concatenate([part[1] / tree_count for part in parts])
    return DensitySamples(values, weights, tree_count)


def single_tree_distribution_samples(
    graph: nx.Graph,
    distribution_name: str,
) -> DensitySamples:
    """Return one normalized within-tree distribution for a paired panel."""
    distribution = tree_distribution(graph, distribution_name)
    values = np.asarray(distribution.values, dtype=float)
    finite = np.isfinite(values)
    values = values[finite]
    if distribution.weights is None:
        weights = np.ones(values.size, dtype=float)
    else:
        weights = np.asarray(distribution.weights, dtype=float)[finite]
    valid_weight = np.isfinite(weights) & (weights > 0.0)
    values = values[valid_weight]
    weights = weights[valid_weight]
    if values.size == 0 or float(weights.sum()) <= 0.0:
        return DensitySamples(np.zeros(0), np.zeros(0), 0)
    weights = weights / weights.sum()
    return DensitySamples(values, weights, int(values.size))


def merge_density_samples(samples: Sequence[DensitySamples]) -> DensitySamples:
    """Merge normalized distributions with equal mass per supplied sample set."""
    nonempty = [sample for sample in samples if sample.values.size > 0]
    if not nonempty:
        return DensitySamples(np.zeros(0), np.zeros(0), 0)
    count = len(nonempty)
    values = np.concatenate([sample.values for sample in nonempty])
    weights = np.concatenate([sample.weights / sample.weights.sum() / count for sample in nonempty])
    independent_count = sum(max(sample.independent_count, 1) for sample in nonempty)
    return DensitySamples(values, weights, independent_count)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cumulative /= cumulative[-1]
    return float(np.interp(quantile, cumulative, sorted_values))


def density_domain(
    sample_sets: Sequence[DensitySamples],
    *,
    lower: float | None = None,
    upper: float | None = None,
    quantiles: tuple[float, float] = (0.0025, 0.9975),
) -> tuple[float, float]:
    """Choose one robust, shared plotting domain for several distributions."""
    merged = merge_density_samples(sample_sets)
    if merged.values.size == 0:
        return (0.0 if lower is None else lower, 1.0 if upper is None else upper)
    lo = _weighted_quantile(merged.values, merged.weights, quantiles[0])
    hi = _weighted_quantile(merged.values, merged.weights, quantiles[1])
    if lower is not None:
        lo = float(lower)
    if upper is not None:
        hi = float(upper)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def density_bandwidth(
    reference: DensitySamples,
    domain: tuple[float, float],
    *,
    scale_factor: float = 1.0,
) -> float:
    """Estimate one absolute Gaussian bandwidth from the reference distribution."""
    if reference.values.size < 2:
        return max((domain[1] - domain[0]) / 20.0, 1e-6)
    weights = reference.weights / reference.weights.sum()
    mean = float(np.sum(weights * reference.values))
    variance = float(np.sum(weights * (reference.values - mean) ** 2))
    std = np.sqrt(max(variance, 0.0))
    q25 = _weighted_quantile(reference.values, weights, 0.25)
    q75 = _weighted_quantile(reference.values, weights, 0.75)
    robust_std = (q75 - q25) / 1.349
    positive_scales = [value for value in (std, robust_std) if np.isfinite(value) and value > 0]
    data_scale = min(positive_scales) if positive_scales else (domain[1] - domain[0]) / 6.0
    n = max(reference.independent_count, 2)
    bandwidth = 0.9 * data_scale * n ** (-0.2) * scale_factor
    domain_span = domain[1] - domain[0]
    return float(np.clip(bandwidth, domain_span / 150.0, domain_span / 3.0))


def _smoothed_density(
    samples: DensitySamples,
    *,
    domain: tuple[float, float],
    bandwidth: float,
    bins: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    if samples.values.size == 0:
        grid = np.linspace(domain[0], domain[1], bins)
        return grid, np.zeros_like(grid)
    edges = np.linspace(domain[0], domain[1], bins + 1)
    bin_width = float(edges[1] - edges[0])
    hist, _ = np.histogram(samples.values, bins=edges, weights=samples.weights, density=False)
    density = hist.astype(float) / max(float(samples.weights.sum()) * bin_width, 1e-12)
    sigma_bins = max(bandwidth / bin_width, 0.35)
    density = gaussian_filter1d(density, sigma=sigma_bins, mode="reflect", truncate=4.0)
    centers = edges[:-1] + 0.5 * bin_width
    area = float(np.trapz(density, centers))
    if area > 0.0:
        density /= area
    return centers, density


def plot_density_comparison(
    ax: plt.Axes,
    samples_by_label: Mapping[str, DensitySamples],
    *,
    colors: Mapping[str, str],
    domain: tuple[float, float],
    bandwidth: float,
    title: str | None = None,
    xlabel: str | None = None,
    fill_alpha: float = 0.14,
) -> None:
    """Plot several consistently smoothed empirical densities on one axis."""
    for label, samples in samples_by_label.items():
        x, y = _smoothed_density(samples, domain=domain, bandwidth=bandwidth)
        color = colors[label]
        ax.fill_between(x, 0.0, y, color=color, alpha=fill_alpha, linewidth=0)
        ax.plot(x, y, color=color, linewidth=1.35, label=label)
    if title:
        ax.set_title(title, pad=3.0)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=2.0)
    ax.set_xlim(*domain)
    ax.set_ylim(bottom=0.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="x", length=2.5, pad=1.5)


def _diagram_pairs(diagram: object) -> np.ndarray:
    if diagram is None:
        return np.zeros((0, 2), dtype=float)
    pairs = np.asarray(diagram.as_pairs(), dtype=float)
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=float)
    return pairs.reshape(-1, 2)


def plot_persistence_overlay(
    ax: plt.Axes,
    target_diagram: object,
    generated_diagram: object,
    *,
    target_color: str = REFERENCE_COLOR,
    generated_color: str = OURS_COLOR,
    normalized: bool = True,
) -> None:
    """Draw a compact target/generated persistence-diagram overlay."""
    target_pairs = _diagram_pairs(target_diagram)
    generated_pairs = _diagram_pairs(generated_diagram)
    if normalized:
        lo, hi = -0.02, 1.02
    else:
        arrays = [array for array in (target_pairs, generated_pairs) if array.size]
        values = np.concatenate(arrays, axis=0) if arrays else np.array([[0.0, 1.0]])
        lo = min(0.0, float(values.min()))
        hi = max(1.0, float(values.max()))
        pad = 0.04 * max(hi - lo, 1.0)
        lo, hi = lo - pad, hi + pad
    ax.plot(
        [lo, hi],
        [lo, hi],
        color=DIAGONAL_COLOR,
        linestyle="--",
        linewidth=0.7,
        zorder=1,
    )
    if target_pairs.size:
        ax.scatter(
            target_pairs[:, 0],
            target_pairs[:, 1],
            s=11,
            color=target_color,
            alpha=0.72,
            linewidths=0,
            label="Target",
            zorder=2,
        )
    if generated_pairs.size:
        ax.scatter(
            generated_pairs[:, 0],
            generated_pairs[:, 1],
            s=11,
            color=generated_color,
            alpha=0.72,
            linewidths=0,
            label="Generated",
            zorder=3,
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(length=2.5, pad=1.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
