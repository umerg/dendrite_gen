"""Exhaustive visual analysis of completed ground-truth distance matrices."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import seaborn as sns
from sklearn.metrics import silhouette_samples

try:
    from dendrite_gen.metrics.chamfer import tree_chamfer_distance
    from dendrite_gen.metrics.so2 import rotate_points_about_axis
    from dendrite_gen.utils.data_loading import load_swc_graph
    from dendrite_gen.visualization.metric_study.frame import (
        transform_scientific_y_to_internal_z,
    )
    from dendrite_gen.visualization.metric_study.plots import classical_mds
    from dendrite_gen.visualization.qualitative.plots_3d import (
        plot_tree_cylinder_3d,
    )
except ModuleNotFoundError as exc:
    if exc.name != "dendrite_gen":
        raise
    from metrics.chamfer import tree_chamfer_distance  # type: ignore
    from metrics.so2 import rotate_points_about_axis  # type: ignore
    from utils.data_loading import load_swc_graph  # type: ignore
    from visualization.metric_study.frame import (  # type: ignore
        transform_scientific_y_to_internal_z,
    )
    from visualization.metric_study.plots import classical_mds  # type: ignore
    from visualization.qualitative.plots_3d import (  # type: ignore
        plot_tree_cylinder_3d,
    )


METRIC_ORDER = (
    "chamfer",
    "tmd_path_wasserstein",
    "tmd_height_wasserstein",
    "tmd_rho_wasserstein",
    "distribution_branch_length_wasserstein",
    "distribution_sibling_angle_wasserstein",
    "distribution_root_path_wasserstein",
    "distribution_radial_wasserstein",
    "distribution_height_wasserstein",
    "distribution_root_euclidean_wasserstein",
    "distribution_branch_order_wasserstein",
    "fused_gromov_wasserstein",
)

METRIC_LABELS = {
    "chamfer": "Chamfer",
    "tmd_path_wasserstein": "Barcode: path",
    "tmd_height_wasserstein": "Barcode: height",
    "tmd_rho_wasserstein": "Barcode: radius",
    "distribution_branch_length_wasserstein": "Branch length",
    "distribution_sibling_angle_wasserstein": "Sibling angle",
    "distribution_root_path_wasserstein": "Root path",
    "distribution_radial_wasserstein": "Radial coordinate",
    "distribution_height_wasserstein": "Height",
    "distribution_root_euclidean_wasserstein": "Root Euclidean",
    "distribution_branch_order_wasserstein": "Branch order",
    "fused_gromov_wasserstein": "FGW",
}

GEOMETRY_TOPOLOGY_METRICS = (
    "chamfer",
    "tmd_path_wasserstein",
    "tmd_height_wasserstein",
    "tmd_rho_wasserstein",
    "fused_gromov_wasserstein",
)
DISTRIBUTION_METRICS = tuple(
    name for name in METRIC_ORDER if name.startswith("distribution_")
)
EXAMPLE_METRICS = (
    "chamfer",
    "tmd_path_wasserstein",
    "distribution_sibling_angle_wasserstein",
    "fused_gromov_wasserstein",
)
ALL_ASSETS = (
    "correlations",
    "mds",
    "individual_heatmaps",
    "class_medians",
    "same_different",
    "separation",
    "confusions",
    "morphology",
    "examples",
)

_MARKERS = ("o", "s", "^", "D", "P", "X", "v")


@dataclass(frozen=True)
class MatrixStudy:
    run_dir: Path
    manifest: pd.DataFrame
    matrices: Mapping[str, np.ndarray]

    @property
    def metric_names(self) -> tuple[str, ...]:
        return tuple(self.matrices)

    @property
    def labels(self) -> np.ndarray:
        return self.manifest["cell_class"].to_numpy(dtype=np.int64)

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


def metric_label(metric_name: str) -> str:
    return METRIC_LABELS.get(metric_name, metric_name.replace("_", " ").title())


def _metric_groups(study: MatrixStudy) -> tuple[tuple[str, ...], tuple[str, ...]]:
    distributions = tuple(
        name for name in study.metric_names if name.startswith("distribution_")
    )
    other = tuple(name for name in study.metric_names if name not in distributions)
    return other, distributions


def _example_metric_names(study: MatrixStudy) -> tuple[str, ...]:
    preferred = tuple(name for name in EXAMPLE_METRICS if name in study.matrices)
    if preferred:
        return preferred
    return study.metric_names[:4]


def load_matrix_study(run_dir: Path) -> MatrixStudy:
    """Load and validate all completed matrix families in one run directory."""

    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Matrix run directory does not exist: {run_dir}")

    reference_manifest: pd.DataFrame | None = None
    matrices: dict[str, np.ndarray] = {}
    for family_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        progress_path = family_dir / "progress.json"
        manifest_path = family_dir / "selected_trees.csv"
        metric_root = family_dir / "metrics"
        if not progress_path.is_file() or not manifest_path.is_file():
            continue
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if not str(progress.get("status", "")).startswith("complete"):
            raise ValueError(
                f"Matrix family {family_dir.name!r} is not complete: "
                f"{progress.get('status')!r}."
            )
        family_manifest = pd.read_csv(manifest_path)
        family_manifest["tree_id"] = family_manifest["tree_id"].astype(str)
        family_manifest["cell_class"] = family_manifest["cell_class"].astype(int)
        if reference_manifest is None:
            reference_manifest = family_manifest
        elif not family_manifest[
            ["matrix_index", "tree_id", "split", "cell_class", "cell_type"]
        ].equals(
            reference_manifest[
                ["matrix_index", "tree_id", "split", "cell_class", "cell_type"]
            ]
        ):
            raise ValueError(
                f"Matrix family {family_dir.name!r} uses a different tree order."
            )

        for metric_dir in sorted(path for path in metric_root.iterdir() if path.is_dir()):
            distances = np.load(metric_dir / "distances.npy")
            status = np.load(metric_dir / "status.npy")
            if distances.shape != (len(family_manifest), len(family_manifest)):
                raise ValueError(f"Unexpected shape for {metric_dir.name!r}.")
            if status.shape != distances.shape or not np.all(status == 1):
                raise ValueError(f"Metric {metric_dir.name!r} is not fully successful.")
            if not np.all(np.isfinite(distances)):
                raise ValueError(f"Metric {metric_dir.name!r} is not fully finite.")
            if np.any(distances < -1e-10):
                raise ValueError(f"Metric {metric_dir.name!r} contains negative values.")
            if not np.allclose(distances, distances.T, rtol=1e-8, atol=1e-10):
                raise ValueError(f"Metric {metric_dir.name!r} is not symmetric.")
            if not np.allclose(np.diag(distances), 0.0, atol=1e-10):
                raise ValueError(f"Metric {metric_dir.name!r} has a nonzero diagonal.")
            if metric_dir.name in matrices:
                raise ValueError(f"Metric {metric_dir.name!r} appears more than once.")
            matrices[metric_dir.name] = np.asarray(distances, dtype=np.float64)

    if reference_manifest is None:
        raise ValueError(f"No completed matrix families found below {run_dir}.")
    preferred = [name for name in METRIC_ORDER if name in matrices]
    extras = sorted(name for name in matrices if name not in METRIC_ORDER)
    ordered = {name: matrices[name] for name in (*preferred, *extras)}
    return MatrixStudy(run_dir, reference_manifest, ordered)


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for extension in ("png", "pdf"):
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths[extension] = path
    plt.close(fig)
    return paths


def _prepare_output_dir(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _class_palette(study: MatrixStudy) -> dict[int, object]:
    colors = sns.color_palette("colorblind", len(study.classes))
    return dict(zip(study.classes, colors))


def _class_layout(study: MatrixStudy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = study.labels
    grouped = np.concatenate(
        [np.flatnonzero(labels == class_id) for class_id in study.classes]
    )
    counts = np.asarray(
        [np.sum(labels == class_id) for class_id in study.classes], dtype=int
    )
    boundaries = np.cumsum(np.concatenate(([0], counts)))
    centers = 0.5 * (boundaries[:-1] + boundaries[1:] - 1)
    return grouped, boundaries, centers


def plot_metric_correlations(study: MatrixStudy, output_dir: Path) -> dict[str, Path]:
    output_dir = _prepare_output_dir(output_dir)
    count = len(study.manifest)
    upper = np.triu_indices(count, k=1)
    pair_distances = pd.DataFrame(
        {
            metric_label(name): study.matrices[name][upper]
            for name in study.metric_names
        }
    )
    correlations = pair_distances.corr(method="spearman")
    correlations.to_csv(output_dir / "metric_spearman_correlations.csv")

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    mask = np.triu(np.ones_like(correlations, dtype=bool), k=1)
    sns.heatmap(
        correlations,
        mask=mask,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Spearman correlation", "shrink": 0.75},
        ax=ax,
    )
    ax.set_title("Agreement between tree dissimilarities")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return _save_figure(fig, output_dir, "metric_spearman_correlations")


def _compute_mds(study: MatrixStudy) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {name: classical_mds(study.matrices[name]) for name in study.metric_names}


def _plot_mds_group(
    study: MatrixStudy,
    embeddings: Mapping[str, tuple[np.ndarray, np.ndarray]],
    metric_names: Sequence[str],
    output_dir: Path,
    stem: str,
    *,
    ncols: int,
) -> dict[str, Path]:
    nrows = math.ceil(len(metric_names) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.25 * ncols + 1.9, 3.65 * nrows),
        squeeze=False,
    )
    colors = _class_palette(study)
    labels = study.labels
    for ax, metric_name in zip(axes.flat, metric_names):
        coordinates, eigenvalues = embeddings[metric_name]
        for class_index, class_id in enumerate(study.classes):
            selected = labels == class_id
            ax.scatter(
                coordinates[selected, 0],
                coordinates[selected, 1],
                s=22,
                marker=_MARKERS[class_index % len(_MARKERS)],
                color=colors[class_id],
                edgecolor="white",
                linewidth=0.3,
                alpha=0.85,
                label=(
                    f"{study.class_names[class_id]} "
                    f"(n={int(selected.sum())})"
                ),
            )
        positive_total = float(np.clip(eigenvalues, 0.0, None).sum())
        fractions = np.zeros(2)
        if positive_total > 0:
            fractions = np.clip(eigenvalues[:2], 0.0, None) / positive_total
        ax.set_title(metric_label(metric_name))
        ax.set_xlabel(f"MDS 1 ({fractions[0]:.1%})")
        ax.set_ylabel(f"MDS 2 ({fractions[1]:.1%})")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(metric_names) :]:
        ax.axis("off")
    handles, labels_text = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_text,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
    )
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    return _save_figure(fig, output_dir, stem)


def plot_mds_atlas(study: MatrixStudy, output_dir: Path) -> dict[str, dict[str, Path]]:
    output_dir = _prepare_output_dir(output_dir)
    embeddings = _compute_mds(study)
    diagnostics = []
    for metric_name, (_, eigenvalues) in embeddings.items():
        positive = np.clip(eigenvalues, 0.0, None)
        absolute = np.abs(eigenvalues)
        diagnostics.append(
            {
                "metric": metric_name,
                "top_two_positive_fraction": float(positive[:2].sum() / positive.sum()),
                "negative_absolute_mass_fraction": float(
                    absolute[eigenvalues < 0.0].sum() / absolute.sum()
                ),
            }
        )
    pd.DataFrame(diagnostics).to_csv(output_dir / "mds_diagnostics.csv", index=False)
    other_metrics, distribution_metrics = _metric_groups(study)
    outputs: dict[str, dict[str, Path]] = {}
    if other_metrics:
        outputs["geometry_topology"] = _plot_mds_group(
            study,
            embeddings,
            other_metrics,
            output_dir,
            "mds_geometry_topology",
            ncols=3,
        )
    if distribution_metrics:
        outputs["distributions"] = _plot_mds_group(
            study,
            embeddings,
            distribution_metrics,
            output_dir,
            "mds_distributions",
            ncols=3,
        )
    individual_dir = output_dir / "individual"
    for metric_name in study.metric_names:
        outputs[metric_name] = _plot_mds_group(
            study,
            embeddings,
            (metric_name,),
            individual_dir,
            f"mds_{metric_name}",
            ncols=1,
        )
    return outputs


def _plot_matrix_atlas(
    study: MatrixStudy,
    metric_names: Sequence[str],
    normalized_matrices: Mapping[str, np.ndarray],
    output_dir: Path,
    stem: str,
    *,
    ncols: int,
    colorbar_label: str,
    vmax: float,
    annotations: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Path]:
    nrows = math.ceil(len(metric_names) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.7 * ncols + 0.6, 3.55 * nrows),
        squeeze=False,
    )
    grouped, boundaries, centers = _class_layout(study)
    class_labels = [study.class_names[class_id] for class_id in study.classes]
    image = None
    for ax, metric_name in zip(axes.flat, metric_names):
        matrix = normalized_matrices[metric_name]
        if matrix.shape[0] == len(study.manifest):
            matrix = matrix[np.ix_(grouped, grouped)]
        image = ax.imshow(
            matrix,
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        if matrix.shape[0] == len(study.manifest):
            for boundary in boundaries[1:-1]:
                ax.axhline(boundary - 0.5, color="white", linewidth=0.55)
                ax.axvline(boundary - 0.5, color="white", linewidth=0.55)
            ax.set_xticks(centers, labels=class_labels, rotation=50, ha="right")
            ax.set_yticks(centers, labels=class_labels)
        else:
            ax.set_xticks(
                range(len(study.classes)),
                labels=class_labels,
                rotation=50,
                ha="right",
            )
            ax.set_yticks(range(len(study.classes)), labels=class_labels)
            if annotations is not None:
                values = annotations[metric_name]
                for row in range(values.shape[0]):
                    for column in range(values.shape[1]):
                        ax.text(
                            column,
                            row,
                            f"{values[row, column]:.1f}",
                            ha="center",
                            va="center",
                            fontsize=5.5,
                            color=("white" if matrix[row, column] < 0.65 * vmax else "black"),
                        )
        ax.set_title(metric_label(metric_name))
        ax.tick_params(labelsize=6.5)
    for ax in axes.flat[len(metric_names) :]:
        ax.axis("off")
    fig.subplots_adjust(
        left=0.08,
        right=0.875,
        bottom=0.08,
        top=0.94,
        wspace=0.34,
        hspace=0.34,
    )
    if image is not None:
        colorbar_axis = fig.add_axes((0.905, 0.19, 0.016, 0.62))
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label(colorbar_label)
    return _save_figure(fig, output_dir, stem)


def plot_individual_heatmaps(
    study: MatrixStudy, output_dir: Path
) -> dict[str, dict[str, Path]]:
    normalized: dict[str, np.ndarray] = {}
    for metric_name, matrix in study.matrices.items():
        values = matrix[np.triu_indices_from(matrix, k=1)]
        scale = float(np.quantile(values, 0.98))
        normalized[metric_name] = np.clip(matrix / scale, 0.0, 1.0)
    other_metrics, distribution_metrics = _metric_groups(study)
    outputs: dict[str, dict[str, Path]] = {}
    if other_metrics:
        outputs["geometry_topology"] = _plot_matrix_atlas(
            study,
            other_metrics,
            normalized,
            output_dir,
            "individual_heatmaps_geometry_topology",
            ncols=3,
            colorbar_label="Distance / metric-specific 98th percentile",
            vmax=1.0,
        )
    if distribution_metrics:
        outputs["distributions"] = _plot_matrix_atlas(
            study,
            distribution_metrics,
            normalized,
            output_dir,
            "individual_heatmaps_distributions",
            ncols=3,
            colorbar_label="Distance / metric-specific 98th percentile",
            vmax=1.0,
        )
    return outputs


def _class_median_matrix(study: MatrixStudy, matrix: np.ndarray) -> np.ndarray:
    labels = study.labels
    result = np.zeros((len(study.classes), len(study.classes)), dtype=np.float64)
    for row, class_a in enumerate(study.classes):
        indices_a = np.flatnonzero(labels == class_a)
        for column, class_b in enumerate(study.classes):
            indices_b = np.flatnonzero(labels == class_b)
            block = matrix[np.ix_(indices_a, indices_b)]
            if class_a == class_b:
                values = block[np.triu_indices_from(block, k=1)]
            else:
                values = block.ravel()
            result[row, column] = np.median(values)
    return result


def plot_class_medians(
    study: MatrixStudy, output_dir: Path
) -> dict[str, dict[str, Path]]:
    output_dir = _prepare_output_dir(output_dir)
    raw = {
        name: _class_median_matrix(study, matrix)
        for name, matrix in study.matrices.items()
    }
    normalized: dict[str, np.ndarray] = {}
    for name, matrix in study.matrices.items():
        off_diagonal = matrix[np.triu_indices_from(matrix, k=1)]
        normalized[name] = raw[name] / float(np.median(off_diagonal))
    maximum = float(
        np.quantile(np.concatenate([value.ravel() for value in normalized.values()]), 0.98)
    )
    rows = []
    for name in study.metric_names:
        for row, class_a in enumerate(study.classes):
            for column, class_b in enumerate(study.classes):
                rows.append(
                    {
                        "metric": name,
                        "class_a": study.class_names[class_a],
                        "class_b": study.class_names[class_b],
                        "median_distance": raw[name][row, column],
                        "relative_to_global_median": normalized[name][row, column],
                    }
                )
    pd.DataFrame(rows).to_csv(output_dir / "class_median_distances.csv", index=False)
    other_metrics, distribution_metrics = _metric_groups(study)
    outputs: dict[str, dict[str, Path]] = {}
    if other_metrics:
        outputs["geometry_topology"] = _plot_matrix_atlas(
            study,
            other_metrics,
            normalized,
            output_dir,
            "class_medians_geometry_topology",
            ncols=3,
            colorbar_label="Class-pair median / global pair median",
            vmax=maximum,
            annotations=normalized,
        )
    if distribution_metrics:
        outputs["distributions"] = _plot_matrix_atlas(
            study,
            distribution_metrics,
            normalized,
            output_dir,
            "class_medians_distributions",
            ncols=3,
            colorbar_label="Class-pair median / global pair median",
            vmax=maximum,
            annotations=normalized,
        )
    return outputs


def _pair_balance_weights(study: MatrixStudy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(study.manifest)
    upper = np.triu_indices(count, k=1)
    labels = study.labels
    same = labels[upper[0]] == labels[upper[1]]
    weights = np.zeros(len(upper[0]), dtype=np.float64)
    class_count = len(study.classes)
    for class_id in study.classes:
        selected = np.flatnonzero(
            same & (labels[upper[0]] == class_id)
        )
        weights[selected] = 1.0 / (class_count * len(selected))
    class_pairs = class_count * (class_count - 1) // 2
    for index, class_a in enumerate(study.classes):
        for class_b in study.classes[index + 1 :]:
            selected = np.flatnonzero(
                (~same)
                & (
                    ((labels[upper[0]] == class_a) & (labels[upper[1]] == class_b))
                    | ((labels[upper[0]] == class_b) & (labels[upper[1]] == class_a))
                )
            )
            weights[selected] = 1.0 / (class_pairs * len(selected))
    return upper, same, weights


def _weighted_ecdf(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    cumulative /= cumulative[-1]
    return ordered_values, cumulative


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    ordered_values, cumulative = _weighted_ecdf(values, weights)
    return float(np.interp(quantile, cumulative, ordered_values))


def _weighted_auc(
    same_values: np.ndarray,
    same_weights: np.ndarray,
    different_values: np.ndarray,
    different_weights: np.ndarray,
) -> float:
    """Return P(same < different) + 0.5 P(same == different)."""

    order = np.argsort(same_values, kind="stable")
    values = same_values[order]
    weights = same_weights[order] / same_weights.sum()
    cumulative = np.concatenate(([0.0], np.cumsum(weights)))
    left = np.searchsorted(values, different_values, side="left")
    right = np.searchsorted(values, different_values, side="right")
    probabilities = cumulative[left] + 0.5 * (
        cumulative[right] - cumulative[left]
    )
    normalized_different = different_weights / different_weights.sum()
    return float(np.sum(normalized_different * probabilities))


def same_different_summary(study: MatrixStudy) -> pd.DataFrame:
    upper, same, weights = _pair_balance_weights(study)
    rows = []
    for metric_name in study.metric_names:
        values = study.matrices[metric_name][upper]
        scale = float(np.median(values))
        normalized = values / scale
        same_values = normalized[same]
        different_values = normalized[~same]
        same_weights = weights[same]
        different_weights = weights[~same]
        rows.append(
            {
                "metric": metric_name,
                "same_median_relative": _weighted_quantile(
                    same_values, same_weights, 0.5
                ),
                "different_median_relative": _weighted_quantile(
                    different_values, different_weights, 0.5
                ),
                "same_vs_different_auc": _weighted_auc(
                    same_values,
                    same_weights,
                    different_values,
                    different_weights,
                ),
            }
        )
    return pd.DataFrame(rows).set_index("metric")


def plot_same_different(study: MatrixStudy, output_dir: Path) -> dict[str, Path]:
    output_dir = _prepare_output_dir(output_dir)
    upper, same, weights = _pair_balance_weights(study)
    columns = min(4, len(study.metric_names))
    rows = math.ceil(len(study.metric_names) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.3 * columns, 3.0 * rows + 0.2),
        squeeze=False,
    )
    summary = same_different_summary(study)
    for ax, metric_name in zip(axes.flat, study.metric_names):
        values = study.matrices[metric_name][upper]
        normalized = values / float(np.median(values))
        for selected, label, color in (
            (same, "Same class", "#287B7A"),
            (~same, "Different class", "#C66A3D"),
        ):
            x, y = _weighted_ecdf(normalized[selected], weights[selected])
            ax.plot(x, y, label=label, color=color, linewidth=1.8)
        upper_x = float(np.quantile(normalized, 0.98))
        ax.set_xlim(0.0, upper_x)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(
            f"{metric_label(metric_name)}\n"
            f"AUC={summary.loc[metric_name, 'same_vs_different_auc']:.2f}"
        )
        ax.set_xlabel("Distance / global median")
        ax.set_ylabel("Class-balanced cumulative fraction")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(study.metric_names) :]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    summary.to_csv(output_dir / "same_different_summary.csv")
    return _save_figure(fig, output_dir, "same_vs_different_ecdfs")


def _nearest_neighbor_confusion(
    study: MatrixStudy, matrix: np.ndarray
) -> tuple[np.ndarray, float, float, int]:
    labels = study.labels
    classes = study.classes
    class_to_index = {class_id: index for index, class_id in enumerate(classes)}
    working = matrix.copy()
    np.fill_diagonal(working, np.inf)
    scale = float(np.median(matrix[np.triu_indices_from(matrix, k=1)]))
    confusion = np.zeros((len(classes), len(classes)), dtype=np.float64)
    per_query_credit = np.zeros(len(labels), dtype=np.float64)
    tied_queries = 0
    for row in range(len(labels)):
        minimum = float(np.min(working[row]))
        tied = np.flatnonzero(
            np.isclose(working[row], minimum, rtol=1e-12, atol=1e-12 * scale)
        )
        if len(tied) > 1:
            tied_queries += 1
        contribution = 1.0 / len(tied)
        true_index = class_to_index[int(labels[row])]
        for neighbor in tied:
            predicted_index = class_to_index[int(labels[neighbor])]
            confusion[true_index, predicted_index] += contribution
            if labels[neighbor] == labels[row]:
                per_query_credit[row] += contribution
    for row, class_id in enumerate(classes):
        confusion[row] /= float(np.sum(labels == class_id))
    macro = float(
        np.mean(
            [per_query_credit[labels == class_id].mean() for class_id in classes]
        )
    )
    micro = float(per_query_credit.mean())
    return confusion, macro, micro, tied_queries


def separation_results(
    study: MatrixStudy,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    same_summary = same_different_summary(study)
    labels = study.labels
    rows = []
    confusions: dict[str, np.ndarray] = {}
    for metric_name in study.metric_names:
        matrix = study.matrices[metric_name]
        confusion, macro_nn, micro_nn, tied_queries = _nearest_neighbor_confusion(
            study, matrix
        )
        confusions[metric_name] = confusion
        samples = silhouette_samples(matrix, labels, metric="precomputed")
        macro_silhouette = float(
            np.mean(
                [samples[labels == class_id].mean() for class_id in study.classes]
            )
        )
        rows.append(
            {
                "metric": metric_name,
                "macro_1nn": macro_nn,
                "micro_1nn": micro_nn,
                "tied_queries": tied_queries,
                "macro_silhouette": macro_silhouette,
                "micro_silhouette": float(samples.mean()),
                "same_vs_different_auc": float(
                    same_summary.loc[metric_name, "same_vs_different_auc"]
                ),
            }
        )
    return pd.DataFrame(rows).set_index("metric"), confusions


def plot_separation_scores(study: MatrixStudy, output_dir: Path) -> dict[str, Path]:
    output_dir = _prepare_output_dir(output_dir)
    results, _ = separation_results(study)
    results.to_csv(output_dir / "class_separation_scores.csv")
    order = results.sort_values("macro_1nn").index
    positions = np.arange(len(order))
    labels = [metric_label(name) for name in order]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 6.6), sharey=True)
    axes[0].barh(
        positions,
        results.loc[order, "macro_1nn"],
        color="#287B7A",
    )
    axes[0].axvline(
        1.0 / len(study.classes),
        color="#5C6B76",
        linestyle="--",
        linewidth=1,
        label=f"chance = 1/{len(study.classes)}",
    )
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_yticks(positions, labels=labels)
    axes[0].set_xlabel("Macro 1-NN accuracy")
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].barh(
        positions,
        results.loc[order, "same_vs_different_auc"],
        color="#C66A3D",
    )
    axes[1].axvline(
        0.5,
        color="#5C6B76",
        linestyle="--",
        linewidth=1,
        label="chance = 0.5",
    )
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel("Same/different AUC")
    axes[1].legend(frameon=False, loc="lower right", fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)

    axes[2].barh(
        positions,
        results.loc[order, "macro_silhouette"],
        color="#4C78A8",
    )
    axes[2].axvline(0.0, color="#5C6B76", linewidth=0.9)
    axes[2].set_xlabel("Macro silhouette")
    axes[2].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Descriptive class separation by metric", y=1.01)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "class_separation_scores")


def plot_confusion_atlas(
    study: MatrixStudy, output_dir: Path
) -> dict[str, dict[str, Path]]:
    _, confusions = separation_results(study)
    other_metrics, distribution_metrics = _metric_groups(study)
    outputs: dict[str, dict[str, Path]] = {}
    if other_metrics:
        outputs["geometry_topology"] = _plot_matrix_atlas(
            study,
            other_metrics,
            confusions,
            output_dir,
            "nearest_neighbor_confusions_geometry_topology",
            ncols=3,
            colorbar_label="Fraction of queries in true class",
            vmax=1.0,
            annotations=confusions,
        )
    if distribution_metrics:
        outputs["distributions"] = _plot_matrix_atlas(
            study,
            distribution_metrics,
            confusions,
            output_dir,
            "nearest_neighbor_confusions_distributions",
            ncols=3,
            colorbar_label="Fraction of queries in true class",
            vmax=1.0,
            annotations=confusions,
        )
    return outputs


def compute_morphology_table(study: MatrixStudy, swc_root: Path) -> pd.DataFrame:
    swc_root = Path(swc_root).expanduser().resolve()
    rows = []
    for record in study.manifest.itertuples(index=False):
        swc_path = swc_root / str(record.split) / f"{record.tree_id}.swc"
        graph = load_swc_graph(swc_path)
        total_length = 0.0
        for node_a, node_b in graph.edges:
            point_a = np.asarray(graph.nodes[node_a]["pos"], dtype=np.float64)
            point_b = np.asarray(graph.nodes[node_b]["pos"], dtype=np.float64)
            total_length += float(np.linalg.norm(point_a - point_b))
        rows.append(
            {
                "tree_id": str(record.tree_id),
                "node_count": graph.number_of_nodes(),
                "total_neurite_length": total_length,
            }
        )
    return pd.DataFrame(rows)


def morphology_associations(
    study: MatrixStudy, morphology: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    count = len(study.manifest)
    upper = np.triu_indices(count, k=1)
    labels = study.labels
    same = labels[upper[0]] == labels[upper[1]]
    nodes = morphology["node_count"].to_numpy(dtype=np.float64)
    lengths = morphology["total_neurite_length"].to_numpy(dtype=np.float64)
    gaps = {
        "Node-count gap": np.abs(np.log(nodes[upper[0]] / nodes[upper[1]])),
        "Length gap": np.abs(np.log(lengths[upper[0]] / lengths[upper[1]])),
    }
    rows = []
    for metric_name in study.metric_names:
        distances = study.matrices[metric_name][upper]
        row: dict[str, object] = {"metric": metric_name}
        for feature_name, values in gaps.items():
            key = feature_name.lower().replace("-", "_").replace(" ", "_")
            row[f"{key}_all"] = float(spearmanr(distances, values).statistic)
            row[f"{key}_within_class"] = float(
                spearmanr(distances[same], values[same]).statistic
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("metric"), gaps


def plot_morphology_associations(
    study: MatrixStudy, swc_root: Path, output_dir: Path
) -> dict[str, dict[str, Path] | Path]:
    output_dir = _prepare_output_dir(output_dir)
    morphology = compute_morphology_table(study, swc_root)
    morphology.to_csv(output_dir / "selected_tree_morphology.csv", index=False)
    associations, gaps = morphology_associations(study, morphology)
    associations.to_csv(output_dir / "morphology_distance_associations.csv")

    display_columns = {
        "node_count_gap_all": "Node gap\nall pairs",
        "node_count_gap_within_class": "Node gap\nwithin class",
        "length_gap_all": "Length gap\nall pairs",
        "length_gap_within_class": "Length gap\nwithin class",
    }
    table = associations[list(display_columns)].rename(columns=display_columns)
    table.index = [metric_label(name) for name in table.index]
    fig, ax = plt.subplots(figsize=(7.8, 7.0))
    sns.heatmap(
        table,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Spearman correlation"},
        ax=ax,
    )
    ax.set_title("Association with basic morphology differences")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    heatmap_paths = _save_figure(fig, output_dir, "morphology_association_heatmap")

    upper = np.triu_indices(len(study.manifest), k=1)
    length_gap = gaps["Length gap"]
    same_class = study.labels[upper[0]] == study.labels[upper[1]]
    example_metrics = _example_metric_names(study)
    columns = min(2, len(example_metrics))
    rows = math.ceil(len(example_metrics) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.25 * columns, 4.0 * rows),
        squeeze=False,
    )
    for ax, metric_name in zip(axes.flat, example_metrics):
        distances = study.matrices[metric_name][upper]
        normalized = distances / float(np.median(distances))
        for selected, label, color in (
            (np.ones_like(same_class, dtype=bool), "All pairs", "#7F8C96"),
            (same_class, "Within class", "#287B7A"),
        ):
            selected_gaps = length_gap[selected]
            selected_distances = normalized[selected]
            bins = pd.qcut(selected_gaps, q=20, duplicates="drop")
            trend = (
                pd.DataFrame(
                    {"x": selected_gaps, "y": selected_distances, "bin": bins}
                )
                .groupby("bin", observed=True)
                .agg(
                    x=("x", "median"),
                    low=("y", lambda value: np.quantile(value, 0.25)),
                    mid=("y", "median"),
                    high=("y", lambda value: np.quantile(value, 0.75)),
                )
            )
            ax.fill_between(
                trend["x"], trend["low"], trend["high"], color=color, alpha=0.12
            )
            ax.plot(trend["x"], trend["mid"], color=color, linewidth=1.8, label=label)
        ax.set_title(metric_label(metric_name))
        ax.set_xlabel(r"Relative length gap $|\log(L_i/L_j)|$")
        ax.set_ylabel("Distance / global median")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(example_metrics) :]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    trend_paths = _save_figure(fig, output_dir, "morphology_length_trends")
    return {
        "heatmap": heatmap_paths,
        "trends": trend_paths,
        "table": output_dir / "morphology_distance_associations.csv",
    }


def select_representative_pairs(study: MatrixStudy) -> pd.DataFrame:
    labels = study.labels
    rows = []
    for metric_index, metric_name in enumerate(_example_metric_names(study)):
        matrix = study.matrices[metric_name]
        used: set[int] = set()
        for kind_index, (kind, target, lower, upper) in enumerate(
            (("close", 0.10, 0.05, 0.15), ("far", 0.90, 0.85, 0.95))
        ):
            preferred = study.classes[(2 * metric_index + kind_index) % len(study.classes)]
            class_order = (preferred,) + tuple(
                class_id for class_id in study.classes if class_id != preferred
            )
            chosen = None
            for class_id in class_order:
                indices = np.flatnonzero(labels == class_id)
                pair_indices = np.triu_indices(len(indices), k=1)
                first = indices[pair_indices[0]]
                second = indices[pair_indices[1]]
                values = matrix[first, second]
                percentiles = rankdata(values, method="average") / (len(values) + 1)
                candidates = np.flatnonzero(
                    (percentiles >= lower)
                    & (percentiles <= upper)
                    & np.asarray(
                        [a not in used and b not in used for a, b in zip(first, second)]
                    )
                )
                if not len(candidates):
                    continue
                candidate = candidates[np.argmin(np.abs(percentiles[candidates] - target))]
                chosen = (
                    int(first[candidate]),
                    int(second[candidate]),
                    float(values[candidate]),
                    float(percentiles[candidate]),
                    int(class_id),
                )
                break
            if chosen is None:
                raise RuntimeError(f"Could not select a {kind} pair for {metric_name}.")
            index_a, index_b, value, percentile, class_id = chosen
            used.update((index_a, index_b))
            rows.append(
                {
                    "metric": metric_name,
                    "kind": kind,
                    "matrix_index_a": index_a,
                    "matrix_index_b": index_b,
                    "tree_a_id": study.manifest.iloc[index_a]["tree_id"],
                    "tree_b_id": study.manifest.iloc[index_b]["tree_id"],
                    "cell_class": class_id,
                    "cell_type": study.class_names[class_id],
                    "distance": value,
                    "within_class_percentile": percentile,
                }
            )
    return pd.DataFrame(rows)


def _center_graph(graph):
    centered = graph.copy()
    root = centered.graph["root"]
    root_position = np.asarray(centered.nodes[root]["pos"], dtype=np.float64)
    for node in centered.nodes:
        centered.nodes[node]["pos"] = (
            np.asarray(centered.nodes[node]["pos"], dtype=np.float64) - root_position
        )
    return centered


def _align_for_display(graph_a, graph_b):
    result = tree_chamfer_distance(
        graph_a,
        graph_b,
        spacing=1.0,
        quotient_so2=True,
        grid_size=72,
        refine=True,
        refinement_tolerance=1e-8,
    )
    aligned = graph_b.copy()
    nodes = list(aligned.nodes)
    points = np.stack([aligned.nodes[node]["pos"] for node in nodes], axis=0)
    rotated = rotate_points_about_axis(points, result.angle_rad)
    for node, point in zip(nodes, rotated):
        aligned.nodes[node]["pos"] = point
    return aligned, float(result.angle_rad)


def _set_pair_limits(axes: Sequence[plt.Axes], graphs: Sequence[object]) -> None:
    points = np.concatenate(
        [
            np.stack([graph.nodes[node]["pos"] for node in graph.nodes], axis=0)
            for graph in graphs
        ],
        axis=0,
    )
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    padding = 0.06 * span
    minimum -= padding
    maximum += padding
    for ax in axes:
        ax.set_xlim(minimum[0], maximum[0])
        ax.set_ylim(minimum[1], maximum[1])
        ax.set_zlim(minimum[2], maximum[2])
        ax.set_box_aspect(maximum - minimum)


def _plot_z_axis(ax: plt.Axes) -> None:
    """Draw the preferred axial direction through the root without an axis box."""

    z_min, z_max = ax.get_zlim()
    x_min, x_max = ax.get_xlim()
    x_offset = 0.018 * (x_max - x_min)
    ax.plot(
        [0.0, 0.0],
        [0.0, 0.0],
        [z_min, z_max],
        color="#7F8C96",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
    )
    ax.scatter(
        [0.0],
        [0.0],
        [0.0],
        s=7,
        color="#7F8C96",
        depthshade=False,
    )
    ax.text(
        x_offset,
        0.0,
        z_max,
        r"$z$",
        color="#5C6B76",
        fontsize=8,
        ha="left",
        va="bottom",
    )


def plot_representative_pairs(
    study: MatrixStudy, swc_root: Path, output_dir: Path
) -> dict[str, dict[str, Path] | Path]:
    output_dir = _prepare_output_dir(output_dir)
    selections = select_representative_pairs(study)
    selections.to_csv(output_dir / "representative_pairs.csv", index=False)
    swc_root = Path(swc_root).expanduser().resolve()
    outputs: dict[str, dict[str, Path] | Path] = {
        "table": output_dir / "representative_pairs.csv"
    }
    for metric_name in _example_metric_names(study):
        selected_rows = selections[selections["metric"] == metric_name]
        fig = plt.figure(figsize=(10.4, 8.6))
        row_details: list[tuple[list[plt.Axes], str]] = []
        for row_index, record in enumerate(selected_rows.itertuples(index=False)):
            split_a = str(study.manifest.iloc[record.matrix_index_a]["split"])
            split_b = str(study.manifest.iloc[record.matrix_index_b]["split"])
            graph_a = transform_scientific_y_to_internal_z(
                load_swc_graph(swc_root / split_a / f"{record.tree_a_id}.swc")
            )
            graph_b = transform_scientific_y_to_internal_z(
                load_swc_graph(swc_root / split_b / f"{record.tree_b_id}.swc")
            )
            graph_a = _center_graph(graph_a)
            graph_b = _center_graph(graph_b)
            graph_b, angle = _align_for_display(graph_a, graph_b)
            axes = [
                fig.add_subplot(2, 2, row_index * 2 + 1, projection="3d"),
                fig.add_subplot(2, 2, row_index * 2 + 2, projection="3d"),
            ]
            plot_tree_cylinder_3d(
                axes[0],
                graph_a,
                title=_tree_panel_title("A", record.tree_a_id, record.cell_type),
                branch_color="#19344D",
                radius_attr="",
                default_radius=0.45,
                segments=6,
                elev=20,
                azim=35,
            )
            plot_tree_cylinder_3d(
                axes[1],
                graph_b,
                title=_tree_panel_title("B", record.tree_b_id, record.cell_type),
                branch_color="#287B7A",
                radius_attr="",
                default_radius=0.45,
                segments=6,
                elev=20,
                azim=35,
            )
            _set_pair_limits(axes, (graph_a, graph_b))
            for ax in axes:
                _plot_z_axis(ax)
                ax.title.set_fontsize(8.5)
                ax.title.set_color("#22313D")
            row_details.append(
                (
                    axes,
                    (
                        f"{record.kind.capitalize()} pair in {record.cell_type}"
                        f"  ·  d = {record.distance:.3g}"
                        f"  ·  {record.within_class_percentile:.0%} within-class percentile"
                        f"  ·  B rotated {np.degrees(angle):.0f}° for display"
                    ),
                )
            )
        fig.suptitle(
            f"{metric_label(metric_name)}: concrete close and far examples",
            y=0.99,
        )
        fig.subplots_adjust(
            left=0.01,
            right=0.99,
            bottom=0.02,
            top=0.88,
            hspace=0.31,
            wspace=0.02,
        )
        for axes, description in row_details:
            fig.text(
                0.5,
                axes[0].get_position().y1 + 0.045,
                description,
                ha="center",
                va="bottom",
                fontsize=9.5,
                color="#22313D",
            )
        outputs[metric_name] = _save_figure(
            fig, output_dir, f"representative_pairs_{metric_name}"
        )
    return outputs


def _tree_panel_title(panel: str, tree_id: object, cell_type: object) -> str:
    identifier = str(tree_id)
    prefix, separator, suffix = identifier.partition("_")
    if separator:
        identifier = f"{prefix}\n{suffix}"
    return f"{panel} · {identifier} · {cell_type}"


def generate_latex_catalog(
    output_root: Path,
    *,
    include: Iterable[str] = ALL_ASSETS,
    path_prefix: str | Path | None = None,
    example_metrics: Sequence[str] = EXAMPLE_METRICS,
) -> Path:
    requested = set(include)
    figures = [
        ("correlations", "agreement/metric_spearman_correlations.pdf", "Agreement between all pairwise dissimilarities.", "metric-correlations"),
        ("mds", "mds/mds_geometry_topology.pdf", "Class-coloured classical-MDS embeddings for geometry and topology-aware dissimilarities.", "mds-geometry-topology"),
        ("mds", "mds/mds_distributions.pdf", "Class-coloured classical-MDS embeddings for morphology-distribution dissimilarities.", "mds-distributions"),
        ("individual_heatmaps", "individual_heatmaps/individual_heatmaps_geometry_topology.pdf", "Individual-neuron distance matrices grouped by class.", "individual-heatmaps-geometry"),
        ("individual_heatmaps", "individual_heatmaps/individual_heatmaps_distributions.pdf", "Individual-neuron distance matrices for distribution dissimilarities.", "individual-heatmaps-distributions"),
        ("class_medians", "class_medians/class_medians_geometry_topology.pdf", "Median class-to-class dissimilarities.", "class-medians-geometry"),
        ("class_medians", "class_medians/class_medians_distributions.pdf", "Median class-to-class distribution dissimilarities.", "class-medians-distributions"),
        ("same_different", "same_different/same_vs_different_ecdfs.pdf", "Class-balanced same-class and different-class distance distributions.", "same-different"),
        ("separation", "separation/class_separation_scores.pdf", "Descriptive class-separation scores.", "class-separation"),
        ("confusions", "confusions/nearest_neighbor_confusions_geometry_topology.pdf", "Row-normalized nearest-neighbour confusion matrices.", "nn-confusions-geometry"),
        ("confusions", "confusions/nearest_neighbor_confusions_distributions.pdf", "Row-normalized nearest-neighbour confusion matrices for distribution dissimilarities.", "nn-confusions-distributions"),
        ("morphology", "morphology/morphology_association_heatmap.pdf", "Association between tree distances and basic morphology differences.", "morphology-associations"),
        ("morphology", "morphology/morphology_length_trends.pdf", "Distance trends against relative total-neurite-length differences.", "morphology-trends"),
    ]
    for metric_name in example_metrics:
        figures.append(
            (
                "examples",
                f"examples/representative_pairs_{metric_name}.pdf",
                f"Concrete within-class close and far examples for {metric_label(metric_name)}.",
                f"examples-{metric_name.replace('_', '-')}",
            )
        )
    blocks = []
    for group, relative_path, caption, label in figures:
        if group not in requested:
            continue
        include_path = Path(relative_path)
        if path_prefix is not None:
            include_path = Path(path_prefix) / include_path
        blocks.extend(
            [
                r"\begin{figure}[t]",
                r"  \centering",
                f"  \\includegraphics[width=\\linewidth]{{{include_path.as_posix()}}}",
                f"  \\caption{{{caption}}}",
                f"  \\label{{fig:{label}}}",
                r"\end{figure}",
                "",
            ]
        )
    path = output_root / "latex_figure_catalog.tex"
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def generate_report_assets(
    study: MatrixStudy,
    *,
    output_root: Path,
    swc_root: Path,
    include: Iterable[str] = ALL_ASSETS,
    latex_path_prefix: str | Path | None = None,
) -> dict[str, object]:
    """Generate selected report asset groups as matching PDF and PNG files."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    requested = tuple(dict.fromkeys(include))
    unknown = sorted(set(requested) - set(ALL_ASSETS))
    if unknown:
        raise ValueError(f"Unknown report asset groups: {', '.join(unknown)}")
    sns.set_theme(style="whitegrid", context="notebook")
    outputs: dict[str, object] = {}
    if "correlations" in requested:
        outputs["correlations"] = plot_metric_correlations(
            study, output_root / "agreement"
        )
    if "mds" in requested:
        outputs["mds"] = plot_mds_atlas(study, output_root / "mds")
    if "individual_heatmaps" in requested:
        outputs["individual_heatmaps"] = plot_individual_heatmaps(
            study, output_root / "individual_heatmaps"
        )
    if "class_medians" in requested:
        outputs["class_medians"] = plot_class_medians(
            study, output_root / "class_medians"
        )
    if "same_different" in requested:
        outputs["same_different"] = plot_same_different(
            study, output_root / "same_different"
        )
    if "separation" in requested:
        outputs["separation"] = plot_separation_scores(
            study, output_root / "separation"
        )
    if "confusions" in requested:
        outputs["confusions"] = plot_confusion_atlas(
            study, output_root / "confusions"
        )
    if "morphology" in requested:
        outputs["morphology"] = plot_morphology_associations(
            study, swc_root, output_root / "morphology"
        )
    if "examples" in requested:
        outputs["examples"] = plot_representative_pairs(
            study, swc_root, output_root / "examples"
        )
    outputs["latex_catalog"] = generate_latex_catalog(
        output_root,
        include=requested,
        path_prefix=latex_path_prefix,
        example_metrics=_example_metric_names(study),
    )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--swc-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--include", nargs="+", choices=ALL_ASSETS, default=ALL_ASSETS)
    parser.add_argument(
        "--latex-path-prefix",
        help="Optional path prepended to figure paths in latex_figure_catalog.tex.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    study = load_matrix_study(args.run_dir)
    outputs = generate_report_assets(
        study,
        output_root=args.output_dir,
        swc_root=args.swc_root,
        include=args.include,
        latex_path_prefix=args.latex_path_prefix,
    )
    print(json.dumps(_json_paths(outputs), indent=2, sort_keys=True))
    return 0


def _json_paths(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_paths(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
