"""Run one registered tree dissimilarity on a balanced labelled subset."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Sequence

import numpy as np

try:
    from dendrite_gen.utils.data_loading import load_swc_graph
except ModuleNotFoundError as exc:
    if exc.name != "dendrite_gen":
        raise
    from utils.data_loading import load_swc_graph  # type: ignore

from .compute import compute_symmetric_distance_matrix
from .dataset import TreeRecord, discover_tree_records, select_balanced_sample
from .metric_registry import (
    TMD_PATH_WASSERSTEIN,
    available_metric_variants,
    get_metric_variant,
)
from .plots import save_class_comparison_plots


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _default_output_dir(
    *,
    metric_name: str,
    splits: Sequence[str],
    per_class: int,
    seed: int,
) -> Path:
    repository_root = Path(__file__).resolve().parents[2]
    split_tag = "-".join(splits)
    run_name = f"{metric_name}_{split_tag}_{per_class}perclass_seed{seed}"
    return repository_root / "outputs" / "metric_study" / run_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute one registered scalar tree dissimilarity on a deterministic, "
            "class-balanced subset of labelled ground-truth SWCs."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Directory containing the requested split subdirectories.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        help="Split directory names below the dataset root (default: test).",
    )
    parser.add_argument(
        "--metric",
        choices=available_metric_variants(),
        default=TMD_PATH_WASSERSTEIN,
        help="Registered scalar dissimilarity variant.",
    )
    parser.add_argument(
        "--per-class",
        type=_positive_int,
        default=10,
        help="Balanced number of trees selected from every class (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed controlling deterministic within-class selection (default: 0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; defaults to an ignored outputs/metric_study run folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow known run artifacts in an existing output directory to be replaced.",
    )
    return parser


def _prepare_output_dir(path: Path, *, overwrite: bool) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and not resolved.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {resolved}")
    if resolved.is_dir() and any(resolved.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {resolved}. Pass --overwrite to reuse it."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_selected_records(path: Path, records: Sequence[TreeRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "matrix_index",
                "tree_id",
                "swc_path",
                "split",
                "cell_class",
                "cell_type",
            ),
        )
        writer.writeheader()
        for matrix_index, record in enumerate(records):
            writer.writerow(
                {
                    "matrix_index": matrix_index,
                    "tree_id": record.tree_id,
                    "swc_path": str(record.swc_path),
                    "split": record.split,
                    "cell_class": record.cell_class,
                    "cell_type": record.cell_type,
                }
            )


def _write_class_counts(
    path: Path,
    discovered: Sequence[TreeRecord],
    selected: Sequence[TreeRecord],
) -> None:
    available_counts = Counter(record.cell_class for record in discovered)
    selected_counts = Counter(record.cell_class for record in selected)
    class_names = {record.cell_class: record.cell_type for record in discovered}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("cell_class", "cell_type", "available", "selected"),
        )
        writer.writeheader()
        for cell_class in sorted(class_names):
            writer.writerow(
                {
                    "cell_class": cell_class,
                    "cell_type": class_names[cell_class],
                    "available": available_counts[cell_class],
                    "selected": selected_counts[cell_class],
                }
            )


def _matrix_diagnostics(distances: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    upper_rows, upper_columns = np.triu_indices_from(distances, k=1)
    values = distances[upper_rows, upper_columns]
    same_class = labels[upper_rows] == labels[upper_columns]
    within = values[same_class]
    between = values[~same_class]

    def median(array: np.ndarray) -> float:
        return float(np.median(array)) if array.size else math.nan

    nearest = np.array(distances, copy=True)
    np.fill_diagonal(nearest, np.inf)
    nearest_indices = np.argmin(nearest, axis=1)
    nearest_accuracy = float(np.mean(labels[nearest_indices] == labels))
    return {
        "within_class_median": median(within),
        "between_class_median": median(between),
        "nearest_neighbor_class_accuracy": nearest_accuracy,
    }


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def run(args: argparse.Namespace) -> dict[str, object]:
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir or _default_output_dir(
        metric_name=args.metric,
        splits=args.splits,
        per_class=args.per_class,
        seed=args.seed,
    )
    output_dir = _prepare_output_dir(output_dir, overwrite=args.overwrite)

    discovery_started = perf_counter()
    discovered = discover_tree_records(dataset_root, split_dirs=args.splits)
    selected = select_balanced_sample(
        discovered,
        per_class=args.per_class,
        seed=args.seed,
    )
    discovery_seconds = perf_counter() - discovery_started

    print(
        f"Selected {len(selected)} trees across "
        f"{len({record.cell_class for record in selected})} classes.",
        file=sys.stderr,
    )
    loading_started = perf_counter()
    graphs = [load_swc_graph(record.swc_path) for record in selected]
    loading_seconds = perf_counter() - loading_started

    metric = get_metric_variant(args.metric)
    comparison_started = perf_counter()
    distances = compute_symmetric_distance_matrix(graphs, metric)
    comparison_seconds = perf_counter() - comparison_started

    labels = np.asarray([record.cell_class for record in selected], dtype=np.int64)
    tree_ids = np.asarray([record.tree_id for record in selected], dtype=str)
    cell_types = np.asarray([record.cell_type for record in selected], dtype=str)
    class_names = {record.cell_class: record.cell_type for record in selected}

    matrix_path = output_dir / "distance_matrix.npz"
    np.savez_compressed(
        matrix_path,
        distances=distances,
        tree_ids=tree_ids,
        cell_classes=labels,
        cell_types=cell_types,
    )
    records_path = output_dir / "selected_trees.csv"
    _write_selected_records(records_path, selected)
    counts_path = output_dir / "class_counts.csv"
    _write_class_counts(counts_path, discovered, selected)

    metric_label = str(
        getattr(metric, "display_name", metric.name.replace("_", " ").title())
    )
    plot_started = perf_counter()
    plot_paths = save_class_comparison_plots(
        distances,
        labels,
        class_names,
        metric_label=metric_label,
        out_dir=output_dir / "plots",
    )
    plot_seconds = perf_counter() - plot_started

    result: dict[str, object] = {
        "metric": {
            "name": metric.name,
            "display_name": metric_label,
            "configuration": dict(getattr(metric, "configuration", {})),
        },
        "dataset": {
            "root": str(dataset_root),
            "splits": list(args.splits),
            "available_trees": len(discovered),
            "selected_trees": len(selected),
            "classes": {
                str(cell_class): class_names[cell_class]
                for cell_class in sorted(class_names)
            },
            "per_class": args.per_class,
            "selection_seed": args.seed,
        },
        "diagnostics": _matrix_diagnostics(distances, labels),
        "timings_seconds": {
            "discovery_and_selection": discovery_seconds,
            "swc_loading": loading_seconds,
            "metric_preparation_and_pairs": comparison_seconds,
            "plotting": plot_seconds,
        },
        "artifacts": {
            "distance_matrix": str(matrix_path),
            "selected_trees": str(records_path),
            "class_counts": str(counts_path),
            "plots": {name: str(path) for name, path in plot_paths.items()},
        },
    }
    result_path = output_dir / "run.json"
    result["artifacts"]["run_metadata"] = str(result_path)  # type: ignore[index]
    result_path.write_text(
        json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (FileNotFoundError, NotADirectoryError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
