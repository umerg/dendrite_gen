"""Generate the figures for the compact Elastic SRVFT matrix report."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import seaborn as sns
from sklearn.metrics import silhouette_samples

try:
    from dendrite_gen.visualization.metric_study.plots import classical_mds
except ModuleNotFoundError:
    from visualization.metric_study.plots import classical_mds  # type: ignore


NAVY = "#19344D"
TEAL = "#287B7A"
ORANGE = "#C66A3D"
BLUE = "#4C78A8"
RED = "#B14A4A"
MUTED = "#5C6B76"
LIGHT = "#E8EEF2"
INK = "#22313D"
CLASS_COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
)
MARKERS = ("o", "s", "^", "D", "P", "X", "v")


@dataclass
class ElasticArtifacts:
    root: Path
    run: dict[str, object]
    manifest: pd.DataFrame
    eligibility: pd.DataFrame
    pairs: pd.DataFrame
    pair_results: pd.DataFrame
    full_matrix: np.ndarray
    missing_edges: list[tuple[int, int]]
    removed_indices: list[int]
    analysis_manifest: pd.DataFrame
    analysis_matrix: np.ndarray

    @property
    def labels(self) -> np.ndarray:
        return self.analysis_manifest["cell_class"].to_numpy(dtype=np.int64)

    @property
    def classes(self) -> tuple[int, ...]:
        return tuple(sorted(int(value) for value in np.unique(self.labels)))

    @property
    def class_names(self) -> dict[int, str]:
        rows = self.manifest[["cell_class", "cell_type"]].drop_duplicates()
        return {
            int(row.cell_class): str(row.cell_type)
            for row in rows.itertuples(index=False)
        }


def _read_shards(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((root / "shard_results").glob("shard_*.json")):
        shard = json.loads(path.read_text(encoding="utf-8"))
        for row in shard.get("results", []):
            if row.get("status") != "ok":
                continue
            flat = {
                "pair_index": int(row["pair_index"]),
                "index_a": int(row["index_a"]),
                "index_b": int(row["index_b"]),
                "tree_a_id": str(row["tree_a_id"]),
                "tree_b_id": str(row["tree_b_id"]),
                "value": float(row["value"]),
            }
            flat.update(row["result"])
            rows.append(flat)
    return rows


def _greedy_missing_edge_cover(
    missing_edges: Iterable[tuple[int, int]],
) -> list[int]:
    """Drop high-missingness vertices until the induced matrix is complete."""

    remaining = set(missing_edges)
    removed: list[int] = []
    while remaining:
        degree = Counter(vertex for edge in remaining for vertex in edge)
        vertex = max(degree, key=lambda item: (degree[item], -item))
        removed.append(vertex)
        remaining = {edge for edge in remaining if vertex not in edge}
    return removed


def load_artifacts(root: Path) -> ElasticArtifacts:
    root = Path(root).expanduser().resolve()
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(root / "selected_trees.csv")
    eligibility = pd.read_csv(root / "eligibility.csv")
    pairs = pd.read_csv(root / "pairs.csv")
    manifest["tree_id"] = manifest["tree_id"].astype(str)
    manifest["cell_class"] = manifest["cell_class"].astype(int)
    eligibility["tree_id"] = eligibility["tree_id"].astype(str)
    eligibility["cell_class"] = eligibility["cell_class"].astype(int)

    pair_results = pd.DataFrame(_read_shards(root))
    if pair_results.empty:
        raise ValueError(f"No successful pair results found below {root}.")
    pair_results = pair_results.sort_values("pair_index").reset_index(drop=True)

    count = len(manifest)
    matrix = np.full((count, count), np.nan, dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    for row in pair_results.itertuples(index=False):
        matrix[row.index_a, row.index_b] = row.value
        matrix[row.index_b, row.index_a] = row.value

    missing_edges = [
        (int(row.index_a), int(row.index_b))
        for row in pairs.itertuples(index=False)
        if not np.isfinite(matrix[int(row.index_a), int(row.index_b)])
    ]
    removed_indices = _greedy_missing_edge_cover(missing_edges)
    kept_indices = np.asarray(
        [index for index in range(count) if index not in set(removed_indices)],
        dtype=np.int64,
    )
    analysis_matrix = matrix[np.ix_(kept_indices, kept_indices)]
    if not np.all(np.isfinite(analysis_matrix)):
        raise RuntimeError("Could not recover a complete induced distance matrix.")
    analysis_manifest = manifest.iloc[kept_indices].reset_index(drop=True)
    analysis_manifest["source_matrix_index"] = kept_indices
    analysis_manifest["matrix_index"] = np.arange(len(analysis_manifest))

    class_by_index = manifest["cell_class"].to_dict()
    type_by_index = manifest["cell_type"].to_dict()
    pair_results["class_a"] = pair_results["index_a"].map(class_by_index).astype(int)
    pair_results["class_b"] = pair_results["index_b"].map(class_by_index).astype(int)
    pair_results["type_a"] = pair_results["index_a"].map(type_by_index)
    pair_results["type_b"] = pair_results["index_b"].map(type_by_index)
    pair_results["same_class"] = (
        pair_results["class_a"] == pair_results["class_b"]
    )
    return ElasticArtifacts(
        root=root,
        run=run,
        manifest=manifest,
        eligibility=eligibility,
        pairs=pairs,
        pair_results=pair_results,
        full_matrix=matrix,
        missing_edges=missing_edges,
        removed_indices=removed_indices,
        analysis_manifest=analysis_manifest,
        analysis_matrix=analysis_matrix,
    )


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "axes.titlecolor": NAVY,
            "axes.labelcolor": INK,
            "text.color": INK,
            "axes.edgecolor": "#AAB8C2",
            "grid.color": "#DCE4E9",
            "grid.linewidth": 0.55,
            "figure.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for extension in ("pdf", "png"):
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths[extension] = path
    plt.close(fig)
    return paths


def _class_colors(artifacts: ElasticArtifacts) -> dict[int, str]:
    return {
        class_id: CLASS_COLORS[index % len(CLASS_COLORS)]
        for index, class_id in enumerate(artifacts.classes)
    }


def _class_layout(
    artifacts: ElasticArtifacts,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped = np.concatenate(
        [np.flatnonzero(artifacts.labels == class_id) for class_id in artifacts.classes]
    )
    counts = np.asarray(
        [np.sum(artifacts.labels == class_id) for class_id in artifacts.classes]
    )
    boundaries = np.cumsum(np.concatenate(([0], counts)))
    centers = 0.5 * (boundaries[:-1] + boundaries[1:] - 1)
    return grouped, boundaries, centers


def _annotate_bars(ax: plt.Axes, bars, *, fmt: str = "{:.0f}") -> None:
    for bar in bars:
        value = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
            color=MUTED,
        )


def plot_coverage(
    artifacts: ElasticArtifacts, output_dir: Path
) -> dict[str, Path]:
    eligibility = artifacts.eligibility.copy()
    class_names = artifacts.class_names
    classes = artifacts.classes
    labels = [class_names[class_id] for class_id in classes]
    status_counts = (
        eligibility.groupby(["cell_class", "status"]).size().unstack(fill_value=0)
    )
    for status in ("eligible", "truncated", "error"):
        if status not in status_counts:
            status_counts[status] = 0
    status_counts = status_counts.loc[list(classes)]
    totals = status_counts.sum(axis=1)
    fractions = status_counts.div(totals, axis=0)

    fig = plt.figure(figsize=(13.2, 7.1))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.02), hspace=0.42, wspace=0.28)

    ax = fig.add_subplot(grid[0, 0])
    left = np.zeros(len(classes))
    status_handles = []
    for status, color, label in (
        ("eligible", TEAL, "Fits four layers"),
        ("truncated", ORANGE, "Would be truncated"),
        ("error", RED, "Unsupported/error"),
    ):
        values = fractions[status].to_numpy()
        ax.barh(labels, values, left=left, color=color, label=label)
        status_handles.append(
            Line2D([0], [0], color=color, linewidth=7, label=label)
        )
        left += values
    for row, (eligible, total) in enumerate(zip(status_counts["eligible"], totals)):
        ax.text(
            1.01,
            row,
            f"{eligible}/{total} ({eligible / total:.1%})",
            va="center",
            fontsize=8,
            color=MUTED,
        )
    ax.set_xlim(0, 1.27)
    ax.set_xlabel("Fraction of test trees")
    ax.set_title("Four-layer compatibility is strongly class-dependent")
    ax.grid(axis="y", visible=False)

    ax = fig.add_subplot(grid[0, 1])
    selected_counts = (
        artifacts.manifest.groupby("cell_class").size().reindex(classes, fill_value=0)
    )
    analysis_counts = (
        artifacts.analysis_manifest.groupby("cell_class")
        .size()
        .reindex(classes, fill_value=0)
    )
    positions = np.arange(len(classes))
    width = 0.34
    bars_selected = ax.bar(
        positions - width / 2,
        selected_counts,
        width,
        color=LIGHT,
        edgecolor=BLUE,
        linewidth=1.0,
        label="Selected for cluster run",
    )
    bars_analysis = ax.bar(
        positions + width / 2,
        analysis_counts,
        width,
        color=BLUE,
        label="Complete analysis cohort",
    )
    _annotate_bars(ax, bars_selected)
    _annotate_bars(ax, bars_analysis)
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set_ylabel("Trees")
    ax.set_ylim(0, max(selected_counts) + 5)
    ax.set_title("Capped selection and the recovered complete cohort")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 0])
    truncated = eligibility[eligibility["status"] == "truncated"].copy()
    truncated["cell_type"] = truncated["cell_class"].map(class_names)
    sns.boxplot(
        data=truncated,
        x="cell_type",
        y="omitted_frontier_branches",
        order=labels,
        color=ORANGE,
        fliersize=1.7,
        linewidth=0.8,
        ax=ax,
    )
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("")
    ax.set_ylabel("Branches beyond layer 4 (symlog)")
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("How much is omitted when the representation truncates")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 1])
    expected_pairs = int(artifacts.run["pairs"]["strict_upper_triangle"])
    observed_pairs = len(artifacts.pair_results)
    missing_pairs = expected_pairs - observed_pairs
    bars = ax.bar(
        ["Observed pairs", "Missing pairs"],
        [observed_pairs, missing_pairs],
        color=(TEAL, RED),
        width=0.62,
    )
    _annotate_bars(ax, bars, fmt="{:,.0f}")
    ax.set_ylim(0, expected_pairs * 1.10)
    ax.set_ylabel("Pair results")
    ax.set_title("Copied cluster result is 98.55% complete")
    ax.text(
        0.98,
        0.72,
        "789 / 791 shard files copied\n"
        "25 copied shards are partial\n"
        "92 / 6,328 pairs absent\n\n"
        "Dropping 7 affected trees leaves\n"
        "106 trees and all 5,565 distances.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        color=MUTED,
        bbox={"facecolor": "#F5F8FA", "edgecolor": "#CBD7DE", "pad": 7},
    )
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Elastic SRVFT cohort: depth-limit screening and artifact coverage",
        fontsize=17,
        color=NAVY,
        fontweight="bold",
        y=1.035,
    )
    fig.legend(
        handles=status_handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.055, 0.965),
        ncol=3,
        fontsize=8.5,
    )
    fig.legend(
        handles=(bars_selected, bars_analysis),
        labels=("Selected for cluster run", "Complete analysis cohort"),
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.965),
        ncol=2,
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.88)
    return _save(fig, output_dir, "01_coverage_and_selection")


def _class_medians(artifacts: ElasticArtifacts) -> np.ndarray:
    matrix = artifacts.analysis_matrix
    labels = artifacts.labels
    result = np.zeros((len(artifacts.classes), len(artifacts.classes)))
    for row, class_a in enumerate(artifacts.classes):
        indices_a = np.flatnonzero(labels == class_a)
        for column, class_b in enumerate(artifacts.classes):
            indices_b = np.flatnonzero(labels == class_b)
            block = matrix[np.ix_(indices_a, indices_b)]
            values = (
                block[np.triu_indices_from(block, k=1)]
                if class_a == class_b
                else block.ravel()
            )
            result[row, column] = np.median(values)
    return result


def _triangle_diagnostics(matrix: np.ndarray) -> tuple[int, int, float]:
    count = len(matrix)
    violations = 0
    comparisons = 0
    worst = 0.0
    for intermediate in range(count):
        excess = matrix - (
            matrix[:, intermediate, None] + matrix[intermediate, None, :]
        )
        mask = np.ones_like(excess, dtype=bool)
        np.fill_diagonal(mask, False)
        mask[:, intermediate] = False
        mask[intermediate, :] = False
        selected = excess[mask]
        comparisons += len(selected)
        violations += int(np.sum(selected > 1e-10))
        worst = max(worst, float(np.max(selected)))
    # The distance matrix is symmetric, so (i, j, k) and (j, i, k) are
    # duplicate tests. Report each unordered endpoint pair once.
    return violations // 2, comparisons // 2, worst


def plot_distance_structure(
    artifacts: ElasticArtifacts, output_dir: Path
) -> tuple[dict[str, Path], dict[str, float]]:
    matrix = artifacts.analysis_matrix
    labels = artifacts.labels
    classes = artifacts.classes
    class_names = artifacts.class_names
    grouped, boundaries, centers = _class_layout(artifacts)
    grouped_matrix = matrix[np.ix_(grouped, grouped)]
    class_labels = [class_names[class_id] for class_id in classes]
    upper_values = matrix[np.triu_indices_from(matrix, k=1)]
    vmax = float(np.quantile(upper_values, 0.98))

    coordinates, eigenvalues = classical_mds(matrix)
    positive = np.clip(eigenvalues, 0.0, None)
    negative_fraction = float(
        np.abs(eigenvalues[eigenvalues < 0]).sum() / np.abs(eigenvalues).sum()
    )
    top_two_fraction = float(positive[:2].sum() / positive.sum())
    embedded = squareform(pdist(coordinates))
    normalized_stress = float(
        np.sqrt(np.square(matrix - embedded).sum() / np.square(matrix).sum())
    )
    triangle_violations, triangle_comparisons, worst_violation = (
        _triangle_diagnostics(matrix)
    )

    fig = plt.figure(figsize=(13.2, 7.15))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.08, 1.0),
        height_ratios=(1.0, 1.0),
        wspace=0.27,
        hspace=0.48,
    )

    ax = fig.add_subplot(grid[:, 0])
    image = ax.imshow(
        grouped_matrix,
        cmap="magma",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    for boundary in boundaries[1:-1]:
        ax.axhline(boundary - 0.5, color="white", linewidth=0.7)
        ax.axvline(boundary - 0.5, color="white", linewidth=0.7)
    ax.set_xticks(centers, class_labels, rotation=45, ha="right")
    ax.set_yticks(centers, class_labels)
    ax.set_title("Class-ordered 106 × 106 distance matrix")
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Elastic alignment energy (clipped at p98)")

    medians = _class_medians(artifacts)
    ax = fig.add_subplot(grid[0, 1])
    image = ax.imshow(medians, cmap="magma", interpolation="nearest", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks(range(len(classes)), class_labels)
    threshold = float(np.nanmin(medians) + 0.52 * np.ptp(medians))
    for row in range(len(classes)):
        for column in range(len(classes)):
            ax.text(
                column,
                row,
                f"{medians[row, column]:.0f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if medians[row, column] < threshold else "black",
            )
    ax.set_title("Median distance by class pair (same row/column order)")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Median energy")

    ax = fig.add_subplot(grid[1, 1])
    colors = _class_colors(artifacts)
    for class_index, class_id in enumerate(classes):
        selected = labels == class_id
        ax.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=36,
            marker=MARKERS[class_index],
            color=colors[class_id],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.88,
            label=f"{class_names[class_id]} (n={selected.sum()})",
        )
    ax.axhline(0, color="#CBD7DE", linewidth=0.6)
    ax.axvline(0, color="#CBD7DE", linewidth=0.6)
    ax.set_xlabel(f"MDS 1 ({positive[0] / positive.sum():.1%} of positive spectrum)")
    ax.set_ylabel(f"MDS 2 ({positive[1] / positive.sum():.1%})")
    ax.set_title(
        f"Classical MDS · 2D positive spectrum {top_two_fraction:.1%}"
    )
    ax.legend(
        frameon=False,
        fontsize=7.2,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.02),
    )
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.985,
        0.02,
        f"Non-Euclidean diagnostics: negative eigenvalue mass {negative_fraction:.1%}; "
        f"2D normalized stress {normalized_stress:.3f}; triangle violations "
        f"{triangle_violations:,}/{triangle_comparisons:,} "
        f"({triangle_violations / triangle_comparisons:.2%}), worst excess "
        f"{worst_violation:.1f}.",
        ha="right",
        fontsize=8.3,
        color=MUTED,
    )
    fig.suptitle(
        "Elastic SRVFT distance structure",
        fontsize=17,
        color=NAVY,
        fontweight="bold",
        y=1.015,
    )
    fig.subplots_adjust(bottom=0.14, top=0.89)
    stats = {
        "mds_negative_absolute_mass_fraction": negative_fraction,
        "mds_top_two_positive_fraction": top_two_fraction,
        "mds_normalized_stress": normalized_stress,
        "triangle_violations": triangle_violations,
        "triangle_comparisons": triangle_comparisons,
        "triangle_worst_excess": worst_violation,
    }
    return _save(fig, output_dir, "02_distance_structure"), stats


def _class_balanced_pair_weights(
    artifacts: ElasticArtifacts,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    labels = artifacts.labels
    upper = np.triu_indices(len(labels), k=1)
    same = labels[upper[0]] == labels[upper[1]]
    weights = np.zeros(len(upper[0]), dtype=np.float64)
    for class_id in artifacts.classes:
        selected = np.flatnonzero(same & (labels[upper[0]] == class_id))
        weights[selected] = 1.0 / (len(artifacts.classes) * len(selected))
    pair_count = len(artifacts.classes) * (len(artifacts.classes) - 1) // 2
    for position, class_a in enumerate(artifacts.classes):
        for class_b in artifacts.classes[position + 1 :]:
            selected = np.flatnonzero(
                (~same)
                & (
                    ((labels[upper[0]] == class_a) & (labels[upper[1]] == class_b))
                    | ((labels[upper[0]] == class_b) & (labels[upper[1]] == class_a))
                )
            )
            weights[selected] = 1.0 / (pair_count * len(selected))
    return upper, same, weights


def _weighted_ecdf(
    values: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(values, kind="stable")
    x = values[order]
    y = np.cumsum(weights[order])
    return x, y / y[-1]


def _weighted_auc(
    same_values: np.ndarray,
    same_weights: np.ndarray,
    different_values: np.ndarray,
    different_weights: np.ndarray,
) -> float:
    order = np.argsort(same_values, kind="stable")
    values = same_values[order]
    weights = same_weights[order] / same_weights.sum()
    cumulative = np.concatenate(([0.0], np.cumsum(weights)))
    left = np.searchsorted(values, different_values, side="left")
    right = np.searchsorted(values, different_values, side="right")
    probabilities = cumulative[left] + 0.5 * (
        cumulative[right] - cumulative[left]
    )
    return float(
        np.sum(probabilities * (different_weights / different_weights.sum()))
    )


def _nearest_neighbor_confusion(
    artifacts: ElasticArtifacts,
) -> tuple[np.ndarray, float, float]:
    matrix = artifacts.analysis_matrix.copy()
    labels = artifacts.labels
    np.fill_diagonal(matrix, np.inf)
    nearest = np.argmin(matrix, axis=1)
    predicted = labels[nearest]
    confusion = np.zeros((len(artifacts.classes), len(artifacts.classes)))
    for row, class_a in enumerate(artifacts.classes):
        selected = labels == class_a
        for column, class_b in enumerate(artifacts.classes):
            confusion[row, column] = np.mean(predicted[selected] == class_b)
    macro = float(np.mean(np.diag(confusion)))
    micro = float(np.mean(predicted == labels))
    return confusion, macro, micro


def plot_class_signal(
    artifacts: ElasticArtifacts, output_dir: Path
) -> tuple[dict[str, Path], dict[str, float]]:
    matrix = artifacts.analysis_matrix
    labels = artifacts.labels
    class_names = artifacts.class_names
    classes = artifacts.classes
    class_labels = [class_names[class_id] for class_id in classes]
    upper, same, weights = _class_balanced_pair_weights(artifacts)
    distances = matrix[upper]
    scale = float(np.median(distances))
    normalized = distances / scale
    auc = _weighted_auc(
        normalized[same],
        weights[same],
        normalized[~same],
        weights[~same],
    )
    confusion, macro_nn, micro_nn = _nearest_neighbor_confusion(artifacts)
    silhouette = silhouette_samples(matrix, labels, metric="precomputed")
    macro_silhouette = float(
        np.mean(
            [
                silhouette[labels == class_id].mean()
                for class_id in artifacts.classes
            ]
        )
    )

    fig = plt.figure(figsize=(13.2, 7.1))
    grid = fig.add_gridspec(2, 2, hspace=0.76, wspace=0.30)

    ax = fig.add_subplot(grid[0, 0])
    for selected, pair_label, color in (
        (same, "Same class", TEAL),
        (~same, "Different class", ORANGE),
    ):
        x, y = _weighted_ecdf(normalized[selected], weights[selected])
        ax.plot(x, y, color=color, linewidth=2.1, label=pair_label)
    ax.set_xlim(0, np.quantile(normalized, 0.99))
    ax.set_xlabel("Distance / global median")
    ax.set_ylabel("Class-balanced cumulative fraction")
    ax.set_title(f"Same vs different classes · AUC {auc:.3f}", pad=44)
    ax.text(
        0.5,
        1.015,
        "Curve height is the fraction of pair distances at or below x; "
        "farther left means smaller distances.\n"
        f"Different-class pairs tend to be farther apart (AUC {auc:.3f}); "
        "0.5 would mean no class ordering.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[:, 1])
    image = ax.imshow(confusion, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(classes)), class_labels, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), class_labels)
    ax.set_xlabel("Nearest-neighbor class")
    ax.set_ylabel("Query class")
    for row in range(len(classes)):
        for column in range(len(classes)):
            value = confusion[row, column]
            ax.text(
                column,
                row,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 0.52 else INK,
            )
    ax.set_title(
        f"Leave-one-out 1-NN confusion · macro {macro_nn:.1%}, micro {micro_nn:.1%}",
        pad=44,
    )
    ax.text(
        0.5,
        1.015,
        "Each row shows the nearest-neighbor classes found for one query "
        "class and sums to 100%.\n"
        "Diagonal cells are correct matches; off-diagonal cells reveal "
        "which classes are confused.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Fraction of query class")

    ax = fig.add_subplot(grid[1, 0])
    source_indices = artifacts.analysis_manifest[
        "source_matrix_index"
    ].to_numpy(dtype=int)
    nodes = artifacts.manifest.loc[source_indices, "node_count"].to_numpy(dtype=float)
    node_sum = nodes[upper[0]] + nodes[upper[1]]
    rho = float(spearmanr(distances, node_sum).statistic)
    ax.scatter(
        node_sum[~same],
        distances[~same],
        s=7,
        color=ORANGE,
        alpha=0.10,
        edgecolors="none",
        label="Different class",
    )
    ax.scatter(
        node_sum[same],
        distances[same],
        s=9,
        color=TEAL,
        alpha=0.18,
        edgecolors="none",
        label="Same class",
    )
    bins = pd.qcut(node_sum, q=18, duplicates="drop")
    trend = (
        pd.DataFrame({"nodes": node_sum, "distance": distances, "bin": bins})
        .groupby("bin", observed=True)
        .median(numeric_only=True)
    )
    ax.plot(
        trend["nodes"],
        trend["distance"],
        color=NAVY,
        linewidth=2.1,
        label="Binned median",
    )
    ax.set_xlabel("Combined node count")
    ax.set_ylabel("Elastic alignment energy")
    ax.set_title(f"Distance rises with tree size · Spearman ρ={rho:.3f}", pad=44)
    ax.text(
        0.5,
        1.015,
        "Each point is one tree pair: x is combined node count and y is its "
        "alignment energy.\n"
        f"The dark binned median rises (ρ={rho:.3f}), so distance is partly "
        "associated with tracing size.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color=MUTED,
        wrap=True,
    )
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Class signal and a basic size association",
        fontsize=17,
        color=NAVY,
        fontweight="bold",
        y=1.025,
    )
    fig.text(
        0.012,
        0.01,
        "Secondary descriptive checks, not primary optimization targets. "
        f"Complete n={len(labels)} cohort; macro silhouette = "
        f"{macro_silhouette:.3f}.",
        fontsize=8.5,
        color=MUTED,
    )
    fig.subplots_adjust(bottom=0.13, top=0.84)
    stats = {
        "same_different_auc": auc,
        "macro_1nn": macro_nn,
        "micro_1nn": micro_nn,
        "macro_silhouette": macro_silhouette,
        "distance_node_sum_spearman": rho,
    }
    return _save(fig, output_dir, "03_class_signal"), stats


def _diagnostic_columns(results: pd.DataFrame) -> pd.DataFrame:
    result = results.copy()
    result["alignment_gain_fraction"] = (
        1.0 - result["energy"] / result["energy_at_zero_rotation"]
    )
    result["refinement_gain_fraction"] = (
        1.0 - result["energy"] / result["grid_energy"]
    )
    result["directional_discrepancy"] = (
        np.abs(result["forward_energy"] - result["reverse_energy"])
        / result["energy"]
    )
    result["angle_degrees"] = np.degrees(result["angle_rad"]) % 360
    wrapped = (
        result["angle_rad"] - result["grid_angle_rad"] + np.pi
    ) % (2 * np.pi) - np.pi
    result["refinement_shift_degrees"] = np.degrees(wrapped)
    result["node_sum"] = result["tree_a_nodes"] + result["tree_b_nodes"]
    return result


def plot_alignment_diagnostics(
    artifacts: ElasticArtifacts, output_dir: Path
) -> tuple[dict[str, Path], dict[str, float]]:
    results = _diagnostic_columns(artifacts.pair_results)
    fig = plt.figure(figsize=(13.2, 7.1))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(0.72, 1.0),
        hspace=0.96,
        wspace=0.30,
    )

    ax = fig.add_subplot(grid[0, :])
    gains = 100 * results["alignment_gain_fraction"].to_numpy()
    ax.hist(gains, bins=55, color=TEAL, alpha=0.88)
    for percentile, style in ((50, "-"), (95, "--")):
        value = float(np.percentile(gains, percentile))
        ax.axvline(value, color=NAVY, linestyle=style, linewidth=1.2)
        ax.text(
            value,
            ax.get_ylim()[1] * (0.86 if percentile == 50 else 0.70),
            f"p{percentile} {value:.1f}%",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color=NAVY,
        )
    ax.set_xlabel("Energy reduction from rotation (%)")
    ax.set_ylabel("Completed pairs")
    ax.set_title(
        f"Rotation gain · median {np.median(gains):.1f}%, "
        f"p95 {np.percentile(gains, 95):.1f}%",
        pad=44,
    )
    ax.text(
        0.5,
        1.015,
        "Bars count pairs by the percentage energy reduction obtained from "
        "rotating one tree.\n"
        f"The median gain is {np.median(gains):.1f}%, but the long tail shows "
        "rotation matters greatly for some pairs.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 0])
    maximum = float(
        max(results["forward_energy"].max(), results["reverse_energy"].max())
    )
    image = ax.hexbin(
        results["forward_energy"],
        results["reverse_energy"],
        gridsize=52,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    ax.plot([0, maximum], [0, maximum], color=RED, linewidth=1.1, linestyle="--")
    ax.set_xlim(0, maximum)
    ax.set_ylim(0, maximum)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Forward energy E(A, RθB)")
    ax.set_ylabel("Reverse energy E(RθB, A)")
    ax.set_title("Directional energies at the shared selected angle", pad=67)
    ax.text(
        0.5,
        1.015,
        "Each hexagon groups pairs by forward energy (x) and\n"
        "reverse energy (y) at the same chosen angle.\n"
        "The dashed diagonal means equality; distance from it is\n"
        "the order dependence removed by averaging.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("log10 pair count")

    ax = fig.add_subplot(grid[1, 1])
    for selected, label, color in (
        (results["same_class"], "Same class", TEAL),
        (~results["same_class"], "Different class", ORANGE),
    ):
        values = np.sort(100 * results.loc[selected, "directional_discrepancy"])
        cumulative = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, cumulative, color=color, linewidth=2.0, label=label)
    median = float(np.median(100 * results["directional_discrepancy"]))
    p95 = float(np.percentile(100 * results["directional_discrepancy"], 95))
    ax.axvline(median, color=NAVY, linewidth=1.0)
    ax.set_xlim(0, np.percentile(100 * results["directional_discrepancy"], 99.5))
    ax.set_ylim(0, 1)
    ax.set_xlabel("|forward − reverse| / mean energy (%)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(
        f"Symmetrization discrepancy · median {median:.1f}%, p95 {p95:.1f}%",
        pad=67,
    )
    ax.text(
        0.5,
        1.015,
        "Curve height is the fraction of pairs below each\n"
        "forward/reverse discrepancy threshold on the x-axis.\n"
        f"Half are within {median:.1f}%, but about 5% exceed {p95:.1f}%,\n"
        "which motivates explicit averaging.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "SO(2) alignment and forward/reverse symmetrization",
        fontsize=17,
        color=NAVY,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.012,
        0.01,
        "All 6,236 successful comparisons. The absolute selected-angle "
        "histogram is omitted because absolute orientations are arbitrary; "
        "the report focuses on energy consequences.",
        fontsize=8.5,
        color=MUTED,
    )
    fig.subplots_adjust(bottom=0.13, top=0.84)
    stats = {
        "alignment_gain_median_fraction": float(
            np.median(results["alignment_gain_fraction"])
        ),
        "alignment_gain_p95_fraction": float(
            np.percentile(results["alignment_gain_fraction"], 95)
        ),
        "directional_discrepancy_median_fraction": float(
            np.median(results["directional_discrepancy"])
        ),
        "directional_discrepancy_p95_fraction": float(
            np.percentile(results["directional_discrepancy"], 95)
        ),
    }
    return _save(fig, output_dir, "04_rotation_and_symmetrization"), stats


def plot_runtime_refinement(
    artifacts: ElasticArtifacts, output_dir: Path
) -> tuple[dict[str, Path], dict[str, float]]:
    results = _diagnostic_columns(artifacts.pair_results)
    runtime = results["runtime_seconds"].to_numpy(dtype=float)
    total_hours = float(runtime.sum() / 3600)
    runtime_rho = float(spearmanr(runtime, results["node_sum"]).statistic)
    refinement = 100 * results["refinement_gain_fraction"].to_numpy()

    fig = plt.figure(figsize=(13.2, 7.1))
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30)

    ax = fig.add_subplot(grid[0, 0])
    bins = np.geomspace(max(runtime.min(), 1.0), runtime.max(), 48)
    ax.hist(runtime, bins=bins, color=BLUE, alpha=0.88)
    ax.set_xscale("log")
    ax.set_xlabel("Wall time per completed pair (seconds, log scale)")
    ax.set_ylabel("Pairs")
    ax.set_title(
        f"Runtime · median {np.median(runtime):.1f}s, p95 "
        f"{np.percentile(runtime, 95):.1f}s"
    )
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[0, 1])
    image = ax.hexbin(
        results["node_sum"],
        runtime,
        gridsize=52,
        bins="log",
        mincnt=1,
        cmap="viridis",
        yscale="log",
    )
    ax.set_xlabel("Combined node count")
    ax.set_ylabel("Pair runtime (seconds, log scale)")
    ax.set_title(f"Runtime tracks tree size · Spearman ρ={runtime_rho:.3f}")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("log10 pair count")

    ax = fig.add_subplot(grid[1, 0])
    positive = np.maximum(refinement, 1e-6)
    bins = np.geomspace(1e-5, max(positive.max(), 1e-4), 55)
    ax.hist(positive, bins=bins, color=TEAL, alpha=0.88)
    ax.set_xscale("log")
    ax.axvline(np.median(positive), color=NAVY, linewidth=1.2)
    ax.axvline(np.percentile(positive, 95), color=NAVY, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Grid → refined energy reduction (%)")
    ax.set_ylabel("Pairs")
    ax.set_title(
        f"36-angle refinement gain · median {np.median(refinement):.3f}%, "
        f"p95 {np.percentile(refinement, 95):.3f}%"
    )
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 1])
    ax.hist(
        results["refinement_shift_degrees"],
        bins=np.linspace(-10, 10, 51),
        color=ORANGE,
        alpha=0.88,
    )
    ax.set_xlabel("Wrapped shift from best grid angle (degrees)")
    ax.set_ylabel("Pairs")
    ax.set_title(
        "Local refinement stays within one 10° grid interval\n"
        f"objective evaluations: median "
        f"{np.median(results['objective_evaluations']):.0f}, range "
        f"{results['objective_evaluations'].min():.0f}–"
        f"{results['objective_evaluations'].max():.0f}"
    )
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Runtime and 36-angle grid refinement",
        fontsize=17,
        color=NAVY,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.012,
        0.01,
        f"Completed comparisons sum to {total_hours:.1f} CPU-hours. Runtime is "
        "right-censored: the 92 unfinished comparisons are absent, and partial "
        "two-hour Slurm tasks preferentially omit slow pairs.",
        fontsize=8.5,
        color=MUTED,
    )
    fig.subplots_adjust(bottom=0.14, top=0.91)
    stats = {
        "runtime_total_hours": total_hours,
        "runtime_median_seconds": float(np.median(runtime)),
        "runtime_p95_seconds": float(np.percentile(runtime, 95)),
        "runtime_max_seconds": float(np.max(runtime)),
        "runtime_node_sum_spearman": runtime_rho,
        "refinement_gain_median_fraction": float(
            np.median(results["refinement_gain_fraction"])
        ),
        "refinement_gain_p95_fraction": float(
            np.percentile(results["refinement_gain_fraction"], 95)
        ),
        "refinement_gain_max_fraction": float(
            np.max(results["refinement_gain_fraction"])
        ),
    }
    return _save(fig, output_dir, "05_runtime_and_refinement"), stats


def _read_swc(path: Path) -> tuple[np.ndarray, list[tuple[int, int]]]:
    table = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=("id", "type", "x", "y", "z", "radius", "parent"),
    )
    identifiers = table["id"].astype(int).to_numpy()
    parents = table["parent"].astype(int).to_numpy()
    points = table[["x", "y", "z"]].to_numpy(dtype=float)
    root_candidates = np.flatnonzero(parents < 0)
    root_index = int(root_candidates[0]) if len(root_candidates) else 0
    points -= points[root_index]
    index_by_id = {identifier: index for index, identifier in enumerate(identifiers)}
    edges = [
        (index_by_id[parent], index)
        for index, parent in enumerate(parents)
        if parent in index_by_id
    ]
    return points, edges


def _rotate_about_scientific_y(points: np.ndarray, angle: float) -> np.ndarray:
    result = points.copy()
    cosine = np.cos(angle)
    sine = np.sin(angle)
    result[:, 0] = cosine * points[:, 0] + sine * points[:, 2]
    result[:, 2] = -sine * points[:, 0] + cosine * points[:, 2]
    return result


def _segments(
    points: np.ndarray,
    edges: Sequence[tuple[int, int]],
    axes: tuple[int, int],
) -> np.ndarray:
    return np.asarray(
        [[points[parent, list(axes)], points[child, list(axes)]] for parent, child in edges]
    )


def _plot_tree_overlay(
    ax: plt.Axes,
    points_a: np.ndarray,
    edges_a: Sequence[tuple[int, int]],
    points_b: np.ndarray,
    edges_b: Sequence[tuple[int, int]],
    axes: tuple[int, int],
    *,
    x_label: str,
    y_label: str,
) -> None:
    ax.add_collection(
        LineCollection(
            _segments(points_a, edges_a, axes),
            colors=NAVY,
            linewidths=0.85,
            alpha=0.78,
        )
    )
    ax.add_collection(
        LineCollection(
            _segments(points_b, edges_b, axes),
            colors=ORANGE,
            linewidths=0.85,
            alpha=0.68,
        )
    )
    combined = np.vstack((points_a[:, list(axes)], points_b[:, list(axes)]))
    minimum = combined.min(axis=0)
    maximum = combined.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    padding = 0.05 * span
    ax.set_xlim(minimum[0] - padding[0], maximum[0] + padding[0])
    ax.set_ylim(minimum[1] - padding[1], maximum[1] + padding[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)


def _representative_rows(artifacts: ElasticArtifacts) -> list[tuple[str, pd.Series]]:
    results = _diagnostic_columns(artifacts.pair_results)
    kept = set(artifacts.analysis_manifest["source_matrix_index"].astype(int))
    results = results[
        results["index_a"].isin(kept) & results["index_b"].isin(kept)
    ].copy()
    choices: list[tuple[str, pd.Series]] = []
    used_pairs: set[int] = set()

    candidates = (
        ("Closest same-class pair", results[results["same_class"]], "energy", True),
        ("Farthest same-class pair", results[results["same_class"]], "energy", False),
        (
            "Closest cross-class pair",
            results[~results["same_class"]],
            "energy",
            True,
        ),
        (
            "Largest rotation gain",
            results,
            "alignment_gain_fraction",
            False,
        ),
    )
    for title, pool, column, ascending in candidates:
        ordered = pool.sort_values(column, ascending=ascending)
        row = next(
            row
            for _, row in ordered.iterrows()
            if int(row["pair_index"]) not in used_pairs
        )
        used_pairs.add(int(row["pair_index"]))
        choices.append((title, row))
    return choices


def plot_representative_pairs(
    artifacts: ElasticArtifacts, swc_root: Path, output_dir: Path
) -> tuple[dict[str, Path], pd.DataFrame]:
    swc_root = Path(swc_root).expanduser().resolve()
    choices = _representative_rows(artifacts)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.1))
    output_rows = []
    for pair_index, (kind, record) in enumerate(choices):
        ax = axes.flat[pair_index]
        path_a = swc_root / "test" / f"{record['tree_a_id']}.swc"
        path_b = swc_root / "test" / f"{record['tree_b_id']}.swc"
        points_a, edges_a = _read_swc(path_a)
        points_b, edges_b = _read_swc(path_b)
        points_b = _rotate_about_scientific_y(points_b, float(record["angle_rad"]))
        _plot_tree_overlay(
            ax,
            points_a,
            edges_a,
            points_b,
            edges_b,
            (0, 2),
            x_label="x",
            y_label="z",
        )
        type_pair = (
            str(record["type_a"])
            if record["same_class"]
            else f"{record['type_a']} ↔ {record['type_b']}"
        )
        detail = (
            f"{kind} · {type_pair} · d={record['energy']:.1f}\n"
            f"θ={np.degrees(record['angle_rad']) % 360:.1f}° · "
            f"rotation gain={100 * record['alignment_gain_fraction']:.1f}%"
        )
        ax.set_title(detail, fontsize=10.0, loc="left", pad=5)
        output_rows.append(
            {
                "kind": kind,
                **{
                    key: record[key]
                    for key in (
                        "pair_index",
                        "tree_a_id",
                        "tree_b_id",
                        "type_a",
                        "type_b",
                        "energy",
                        "angle_rad",
                        "alignment_gain_fraction",
                    )
                },
            }
        )
    legend = [
        Line2D([0], [0], color=NAVY, linewidth=2, label="Tree A"),
        Line2D(
            [0],
            [0],
            color=ORANGE,
            linewidth=2,
            label="Tree B after Elastic-selected rotation",
        ),
    ]
    fig.legend(
        legend,
        [item.get_label() for item in legend],
        frameon=False,
        ncol=2,
        loc="upper right",
    )
    fig.suptitle(
        "Representative root-centred tree pairs",
        fontsize=17,
        color=NAVY,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.012,
        0.01,
        "Top-down x–z projections around the scientific-y rotation axis. "
        "Tree B is shown after applying the Elastic-selected angle; branch "
        "correspondences are not drawn. Exact tree IDs are in "
        "representative_pairs.csv.",
        fontsize=8.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.92), h_pad=1.65, w_pad=1.7)
    table = pd.DataFrame(output_rows)
    table.to_csv(output_dir / "representative_pairs.csv", index=False)
    return _save(fig, output_dir, "06_representative_pairs"), table


def _summary(
    artifacts: ElasticArtifacts,
    extra: dict[str, float],
) -> dict[str, object]:
    upper = artifacts.analysis_matrix[
        np.triu_indices_from(artifacts.analysis_matrix, k=1)
    ]
    classes = artifacts.classes
    return {
        "artifact_root": str(artifacts.root),
        "expected_pairs": int(artifacts.run["pairs"]["strict_upper_triangle"]),
        "observed_pairs": len(artifacts.pair_results),
        "missing_pairs": len(artifacts.missing_edges),
        "selected_trees": len(artifacts.manifest),
        "analysis_trees": len(artifacts.analysis_manifest),
        "removed_source_indices": artifacts.removed_indices,
        "analysis_class_counts": {
            artifacts.class_names[class_id]: int(
                np.sum(artifacts.labels == class_id)
            )
            for class_id in classes
        },
        "distance_min": float(np.min(upper)),
        "distance_median": float(np.median(upper)),
        "distance_mean": float(np.mean(upper)),
        "distance_p95": float(np.percentile(upper, 95)),
        "distance_max": float(np.max(upper)),
        **extra,
    }


def generate_report_plots(
    *,
    run_dir: Path,
    swc_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    _style()
    artifacts = load_artifacts(run_dir)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts.analysis_manifest.to_csv(
        output_dir / "analysis_manifest_106.csv", index=False
    )
    artifacts.manifest.iloc[artifacts.removed_indices].to_csv(
        output_dir / "removed_incomplete_trees.csv", index=False
    )
    np.save(output_dir / "elastic_srvft_complete_106.npy", artifacts.analysis_matrix)

    outputs: dict[str, object] = {}
    outputs["coverage"] = plot_coverage(artifacts, output_dir)
    distance_paths, distance_stats = plot_distance_structure(artifacts, output_dir)
    outputs["distance_structure"] = distance_paths
    class_paths, class_stats = plot_class_signal(artifacts, output_dir)
    outputs["class_signal"] = class_paths
    alignment_paths, alignment_stats = plot_alignment_diagnostics(
        artifacts, output_dir
    )
    outputs["alignment"] = alignment_paths
    runtime_paths, runtime_stats = plot_runtime_refinement(artifacts, output_dir)
    outputs["runtime"] = runtime_paths
    example_paths, _ = plot_representative_pairs(
        artifacts, swc_root, output_dir
    )
    outputs["examples"] = example_paths

    summary = _summary(
        artifacts,
        {
            **distance_stats,
            **class_stats,
            **alignment_stats,
            **runtime_stats,
        },
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    outputs["summary"] = output_dir / "summary.json"
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--swc-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = generate_report_plots(
        run_dir=args.run_dir,
        swc_root=args.swc_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(outputs, default=str, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
