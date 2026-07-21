"""Class-aware plots for any precomputed tree dissimilarity matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..utils.styles import DEFAULT_DPI


_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "h")


def _validated_inputs(
    distances: np.ndarray,
    labels: Sequence[int],
    class_names: Mapping[int, str],
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    matrix = np.asarray(distances, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distances must be a square matrix")
    if matrix.shape[0] != label_array.shape[0]:
        raise ValueError("labels must contain one entry per distance-matrix row")
    if matrix.shape[0] < 2:
        raise ValueError("at least two trees are required for class-aware plots")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("distances must contain only finite values")
    if np.any(matrix < -1e-12):
        raise ValueError("distances must be non-negative")
    if not np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-10):
        raise ValueError("distances must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-10):
        raise ValueError("distance-matrix diagonal must be zero")

    classes = tuple(sorted(int(value) for value in np.unique(label_array)))
    missing_names = [class_id for class_id in classes if class_id not in class_names]
    if missing_names:
        raise ValueError(f"missing names for class IDs {missing_names}")
    return matrix, label_array, classes


def classical_mds(distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Embed a finite distance matrix in two dimensions using classical MDS.

    Returns the coordinates and all eigenvalues of the double-centered squared
    distance matrix. Negative eigenvalues are retained in the diagnostics but
    not used as Euclidean coordinate axes.
    """

    matrix = np.asarray(distances, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distances must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("distances must contain only finite values")

    count = matrix.shape[0]
    squared = np.square(matrix)
    # Algebraically equivalent to J D^2 J, but avoids two cubic matrix
    # multiplications and is more stable across BLAS backends.
    gram = -0.5 * (
        squared
        - squared.mean(axis=1, keepdims=True)
        - squared.mean(axis=0, keepdims=True)
        + squared.mean()
    )
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    coordinates = np.zeros((count, 2), dtype=np.float64)
    positive = np.flatnonzero(eigenvalues > 0.0)[:2]
    for coordinate_index, eigen_index in enumerate(positive):
        coordinates[:, coordinate_index] = (
            eigenvectors[:, eigen_index] * np.sqrt(eigenvalues[eigen_index])
        )
    return coordinates, eigenvalues


def _class_colors(classes: Sequence[int]) -> dict[int, object]:
    palette = plt.get_cmap("tab10")
    return {class_id: palette(index % 10) for index, class_id in enumerate(classes)}


def plot_mds_embedding(
    distances: np.ndarray,
    labels: Sequence[int],
    class_names: Mapping[int, str],
    *,
    metric_label: str,
    out_path: Path,
) -> Path:
    """Plot a two-dimensional classical-MDS embedding colored by class."""

    matrix, label_array, classes = _validated_inputs(distances, labels, class_names)
    coordinates, eigenvalues = classical_mds(matrix)
    colors = _class_colors(classes)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for index, class_id in enumerate(classes):
        selected = label_array == class_id
        ax.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=42,
            marker=_MARKERS[index % len(_MARKERS)],
            color=colors[class_id],
            edgecolor="white",
            linewidth=0.55,
            alpha=0.88,
            label=f"{class_names[class_id]} (n={int(selected.sum())})",
        )

    positive_total = float(np.clip(eigenvalues, 0.0, None).sum())
    axis_fractions = np.zeros(2, dtype=np.float64)
    if positive_total > 0.0:
        axis_fractions[: min(2, len(eigenvalues))] = (
            np.clip(eigenvalues[:2], 0.0, None) / positive_total
        )
    ax.set_xlabel(f"MDS 1 ({axis_fractions[0]:.1%} of positive spectrum)")
    ax.set_ylabel(f"MDS 2 ({axis_fractions[1]:.1%} of positive spectrum)")
    ax.set_title(f"{metric_label}: tree-level embedding")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0.0, color="#d1d5db", linewidth=0.65, zorder=0)
    ax.axvline(0.0, color="#d1d5db", linewidth=0.65, zorder=0)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_class_ordered_matrix(
    distances: np.ndarray,
    labels: Sequence[int],
    class_names: Mapping[int, str],
    *,
    metric_label: str,
    out_path: Path,
) -> Path:
    """Plot the tree-level distance matrix with samples grouped by class."""

    matrix, label_array, classes = _validated_inputs(distances, labels, class_names)
    grouped_indices = np.concatenate(
        [np.flatnonzero(label_array == class_id) for class_id in classes]
    )
    grouped = matrix[np.ix_(grouped_indices, grouped_indices)]
    counts = [int(np.sum(label_array == class_id)) for class_id in classes]
    boundaries = np.cumsum([0, *counts])
    centers = 0.5 * (boundaries[:-1] + boundaries[1:] - 1)

    off_diagonal = grouped[np.triu_indices_from(grouped, k=1)]
    positive = off_diagonal[off_diagonal > 0.0]
    vmax = float(np.quantile(positive, 0.98)) if positive.size else 1.0

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(
        grouped,
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    for boundary in boundaries[1:-1]:
        ax.axhline(boundary - 0.5, color="white", linewidth=0.8)
        ax.axvline(boundary - 0.5, color="white", linewidth=0.8)
    tick_labels = [class_names[class_id] for class_id in classes]
    ax.set_xticks(centers, labels=tick_labels, rotation=45, ha="right")
    ax.set_yticks(centers, labels=tick_labels)
    ax.set_xlabel("Neuron class")
    ax.set_ylabel("Neuron class")
    ax.set_title(f"{metric_label}: individual-tree distances")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Dissimilarity (color clipped at 98th percentile)")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def class_median_distances(
    distances: np.ndarray,
    labels: Sequence[int],
    classes: Sequence[int],
) -> np.ndarray:
    """Aggregate a tree-level matrix to median within/between-class distances."""

    matrix = np.asarray(distances, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    aggregated = np.zeros((len(classes), len(classes)), dtype=np.float64)
    for row, class_a in enumerate(classes):
        indices_a = np.flatnonzero(label_array == class_a)
        for column, class_b in enumerate(classes):
            indices_b = np.flatnonzero(label_array == class_b)
            block = matrix[np.ix_(indices_a, indices_b)]
            if class_a == class_b:
                values = block[np.triu_indices_from(block, k=1)]
            else:
                values = block.ravel()
            aggregated[row, column] = float(np.median(values)) if values.size else np.nan
    return aggregated


def plot_class_median_matrix(
    distances: np.ndarray,
    labels: Sequence[int],
    class_names: Mapping[int, str],
    *,
    metric_label: str,
    out_path: Path,
) -> Path:
    """Plot median within-class and between-class tree dissimilarities."""

    matrix, label_array, classes = _validated_inputs(distances, labels, class_names)
    aggregated = class_median_distances(matrix, label_array, classes)
    labels_text = [class_names[class_id] for class_id in classes]

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    image = ax.imshow(aggregated, cmap="magma", interpolation="nearest", aspect="equal")
    ax.set_xticks(range(len(classes)), labels=labels_text, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), labels=labels_text)
    ax.set_xlabel("Neuron class")
    ax.set_ylabel("Neuron class")
    ax.set_title(f"{metric_label}: median class-to-class dissimilarity")

    threshold = float(np.nanmin(aggregated) + 0.55 * np.ptp(aggregated))
    for row in range(len(classes)):
        for column in range(len(classes)):
            value = aggregated[row, column]
            text_color = "white" if value < threshold else "black"
            ax.text(
                column,
                row,
                f"{value:.2g}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Median dissimilarity")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_class_comparison_plots(
    distances: np.ndarray,
    labels: Sequence[int],
    class_names: Mapping[int, str],
    *,
    metric_label: str,
    out_dir: Path,
) -> dict[str, Path]:
    """Write the standard metric-study plot set and return its paths."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "embedding": plot_mds_embedding(
            distances,
            labels,
            class_names,
            metric_label=metric_label,
            out_path=out_dir / "mds_embedding.png",
        ),
        "ordered_matrix": plot_class_ordered_matrix(
            distances,
            labels,
            class_names,
            metric_label=metric_label,
            out_path=out_dir / "class_ordered_distances.png",
        ),
        "class_medians": plot_class_median_matrix(
            distances,
            labels,
            class_names,
            metric_label=metric_label,
            out_path=out_dir / "class_median_distances.png",
        ),
    }


__all__ = [
    "class_median_distances",
    "classical_mds",
    "plot_class_median_matrix",
    "plot_class_ordered_matrix",
    "plot_mds_embedding",
    "save_class_comparison_plots",
]
