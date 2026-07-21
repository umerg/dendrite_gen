"""Benchmark scalar tree dissimilarities on reproducible ground-truth pairs.

SWC loading and the common scientific-y to internal-z frame transformation are
performed before any metric timer starts.  The long-form output keeps failures
and angle-search diagnostics visible instead of reducing the benchmark to a
single opaque wall-clock number.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
from statistics import mean, median, stdev
import sys
from time import perf_counter, perf_counter_ns
from typing import Callable, Sequence

import networkx as nx
import numpy as np

try:
    from dendrite_gen.metrics.adapters.elastic_srvft import elastic_srvft_distance
    from dendrite_gen.metrics.chamfer import tree_chamfer_distance
    from dendrite_gen.metrics.distributions import (
        CRITICAL_BRANCH_CABLE_LENGTH,
        CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG,
        CRITICAL_NODE_BRANCH_ORDER,
        CRITICAL_NODE_ROOT_PATH_LENGTH,
        UNIFORM_CABLE_HEIGHT_Z,
        UNIFORM_CABLE_RADIAL_XY,
        UNIFORM_CABLE_ROOT_EUCLIDEAN,
        distribution_wasserstein_result,
    )
    from dendrite_gen.metrics.fused_gw import fused_gromov_wasserstein_distance
    from dendrite_gen.metrics.persistence import tmd_persistence_distances
    from dendrite_gen.utils.data_loading import load_swc_graph
except ModuleNotFoundError as exc:
    if exc.name != "dendrite_gen":
        raise
    # Support ``python -m visualization.metric_study...`` from the repo root.
    from metrics.adapters.elastic_srvft import (  # type: ignore
        elastic_srvft_distance,
    )
    from metrics.chamfer import tree_chamfer_distance  # type: ignore
    from metrics.distributions import (  # type: ignore
        CRITICAL_BRANCH_CABLE_LENGTH,
        CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG,
        CRITICAL_NODE_BRANCH_ORDER,
        CRITICAL_NODE_ROOT_PATH_LENGTH,
        UNIFORM_CABLE_HEIGHT_Z,
        UNIFORM_CABLE_RADIAL_XY,
        UNIFORM_CABLE_ROOT_EUCLIDEAN,
        distribution_wasserstein_result,
    )
    from metrics.fused_gw import (  # type: ignore
        fused_gromov_wasserstein_distance,
    )
    from metrics.persistence import tmd_persistence_distances  # type: ignore
    from utils.data_loading import load_swc_graph  # type: ignore

from .dataset import TreeRecord, discover_tree_records
from .frame import (
    INTERNAL_AXIS,
    SCIENTIFIC_AXIS,
    SCIENTIFIC_Y_TO_INTERNAL_Z,
    transform_scientific_y_to_internal_z,
)


MetricEvaluator = Callable[[nx.Graph, nx.Graph], "MetricOutcome"]


@dataclass(frozen=True)
class MetricOutcome:
    """Normalized scalar result and implementation-specific diagnostics."""

    value: float
    result_status: str = "ok"
    objective_evaluations: int = 0
    upstream_evaluations: int | None = None
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkMetric:
    """One fixed scalar metric configuration."""

    name: str
    display_name: str
    family: str
    family_display_name: str
    grid_angles: int
    refine: bool
    evaluate: MetricEvaluator


@dataclass(frozen=True)
class Measurement:
    """One timed metric evaluation on one tree pair."""

    pair_index: int
    metric_name: str
    metric_display_name: str
    family: str
    family_display_name: str
    status: str
    result_status: str
    elapsed_seconds: float
    value: float | None
    grid_angles: int
    refine: bool
    objective_evaluations: int | None
    upstream_evaluations: int | None
    error_type: str = ""
    error_message: str = ""
    details: dict[str, object] = field(default_factory=dict)


_PERSISTENCE_FILTRATIONS: tuple[tuple[str, str, str], ...] = (
    ("tmd_path_wasserstein", "Path-filtration persistence W1", "path"),
    ("tmd_height_wasserstein", "Height-filtration persistence W1", "height"),
    ("tmd_rho_wasserstein", "Radial-filtration persistence W1", "rho"),
)

_DISTRIBUTIONS: tuple[tuple[str, str, str], ...] = (
    (
        "distribution_branch_length_wasserstein",
        "Maximal-branch length W1",
        CRITICAL_BRANCH_CABLE_LENGTH,
    ),
    (
        "distribution_sibling_angle_wasserstein",
        "Sibling-branch angle W1",
        CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG,
    ),
    (
        "distribution_root_path_wasserstein",
        "Critical-node root-path W1",
        CRITICAL_NODE_ROOT_PATH_LENGTH,
    ),
    (
        "distribution_radial_wasserstein",
        "Length-weighted radial-coordinate W1",
        UNIFORM_CABLE_RADIAL_XY,
    ),
    (
        "distribution_height_wasserstein",
        "Length-weighted axial-coordinate W1",
        UNIFORM_CABLE_HEIGHT_Z,
    ),
    (
        "distribution_root_euclidean_wasserstein",
        "Length-weighted root-Euclidean W1",
        UNIFORM_CABLE_ROOT_EUCLIDEAN,
    ),
    (
        "distribution_branch_order_wasserstein",
        "Critical-node branch-order W1",
        CRITICAL_NODE_BRANCH_ORDER,
    ),
)

ALL_METRIC_NAMES: tuple[str, ...] = (
    "chamfer",
    *[name for name, _display, _filtration in _PERSISTENCE_FILTRATIONS],
    *[name for name, _display, _distribution in _DISTRIBUTIONS],
    "fused_gromov_wasserstein",
    "elastic_srvft",
)


def select_random_pairs(
    records: Sequence[TreeRecord],
    pair_count: int,
    *,
    seed: int,
) -> tuple[tuple[TreeRecord, TreeRecord], ...]:
    """Select reproducible disjoint pairs after canonical record sorting."""

    if isinstance(pair_count, bool) or not isinstance(pair_count, int):
        raise TypeError("pair_count must be an integer")
    if pair_count <= 0:
        raise ValueError("pair_count must be positive")

    ordered = tuple(
        sorted(
            records,
            key=lambda record: (
                record.split,
                record.tree_id,
                record.swc_path.as_posix(),
            ),
        )
    )
    needed = 2 * pair_count
    if len(ordered) < needed:
        raise ValueError(
            f"Selecting {pair_count} disjoint pairs requires {needed} trees, "
            f"but only {len(ordered)} are available."
        )

    indices = np.random.default_rng(seed).choice(
        len(ordered),
        size=needed,
        replace=False,
    )
    return tuple(
        (ordered[int(indices[offset])], ordered[int(indices[offset + 1])])
        for offset in range(0, needed, 2)
    )


def build_metric_specs(
    *,
    so2_grid_size: int = 72,
    so2_refine: bool = True,
    elastic_grid_size: int = 8,
    elastic_refine: bool = False,
    elastic_depth_policy: str = "raise",
    fgw_max_nodes: int = 1_000,
) -> tuple[BenchmarkMetric, ...]:
    """Build the fixed metric panel used by the runtime study."""

    def chamfer(graph_a: nx.Graph, graph_b: nx.Graph) -> MetricOutcome:
        result = tree_chamfer_distance(
            graph_a,
            graph_b,
            spacing=1.0,
            squared=False,
            reduction="sum",
            quotient_so2=True,
            grid_size=so2_grid_size,
            refine=so2_refine,
        )
        return MetricOutcome(
            value=result.value,
            objective_evaluations=result.objective_evaluations,
            details={
                "aligned_angle_rad": result.angle_rad,
                "point_count_a": result.point_count_a,
                "point_count_b": result.point_count_b,
            },
        )

    specs: list[BenchmarkMetric] = [
        BenchmarkMetric(
            name="chamfer",
            display_name="Arc-length-sampled Chamfer",
            family="chamfer",
            family_display_name="Chamfer",
            grid_angles=so2_grid_size,
            refine=so2_refine,
            evaluate=chamfer,
        )
    ]

    for metric_name, display_name, filtration in _PERSISTENCE_FILTRATIONS:

        def persistence(
            graph_a: nx.Graph,
            graph_b: nx.Graph,
            selected_filtration: str = filtration,
        ) -> MetricOutcome:
            value = tmd_persistence_distances(
                graph_a,
                graph_b,
                normalize_mode="none",
                filtrations=(selected_filtration,),
                order=1.0,
                ground_norm="chebyshev",
            )[selected_filtration]
            return MetricOutcome(value=float(value))

        specs.append(
            BenchmarkMetric(
                name=metric_name,
                display_name=display_name,
                family="persistence",
                family_display_name="Persistence-diagram Wasserstein",
                grid_angles=0,
                refine=False,
                evaluate=persistence,
            )
        )

    for metric_name, display_name, distribution_name in _DISTRIBUTIONS:

        def distribution(
            graph_a: nx.Graph,
            graph_b: nx.Graph,
            selected_distribution: str = distribution_name,
        ) -> MetricOutcome:
            result = distribution_wasserstein_result(
                graph_a,
                graph_b,
                selected_distribution,
                sample_spacing=1.0,
                empty_policy="nan",
            )
            return MetricOutcome(
                value=result.value,
                result_status=result.status,
                details={
                    "sample_count_a": result.sample_count_a,
                    "sample_count_b": result.sample_count_b,
                    "empty_a": result.empty_a,
                    "empty_b": result.empty_b,
                    "distribution_name": selected_distribution,
                },
            )

        specs.append(
            BenchmarkMetric(
                name=metric_name,
                display_name=display_name,
                family="distribution_wasserstein",
                family_display_name="One-dimensional Wasserstein summaries",
                grid_angles=0,
                refine=False,
                evaluate=distribution,
            )
        )

    def fgw(graph_a: nx.Graph, graph_b: nx.Graph) -> MetricOutcome:
        largest = max(graph_a.number_of_nodes(), graph_b.number_of_nodes())
        if fgw_max_nodes > 0 and largest > fgw_max_nodes:
            raise ValueError(
                f"FGW node guard rejected a {largest}-node tree; "
                f"configured limit is {fgw_max_nodes}."
            )
        result = fused_gromov_wasserstein_distance(
            graph_a,
            graph_b,
            feature_mode="xyz",
            alpha=0.5,
            mass_mode="cable_length",
            normalize=True,
            quotient_so2=True,
            grid_size=so2_grid_size,
            refine=so2_refine,
            max_iter=1_000,
            tol=1e-9,
        )
        return MetricOutcome(
            value=result.value,
            objective_evaluations=result.objective_evaluations,
            details={
                "aligned_angle_rad": result.angle_rad,
                "node_count_a": result.n_nodes_1,
                "node_count_b": result.n_nodes_2,
            },
        )

    specs.append(
        BenchmarkMetric(
            name="fused_gromov_wasserstein",
            display_name="Fused Gromov-Wasserstein",
            family="fused_gromov_wasserstein",
            family_display_name="Fused Gromov-Wasserstein",
            grid_angles=so2_grid_size,
            refine=so2_refine,
            evaluate=fgw,
        )
    )

    def elastic(graph_a: nx.Graph, graph_b: nx.Graph) -> MetricOutcome:
        result = elastic_srvft_distance(
            graph_a,
            graph_b,
            lam_m=0.2,
            lam_s=1.0,
            lam_p=0.2,
            quotient_so2=True,
            grid_size=elastic_grid_size,
            refine=elastic_refine,
            symmetrization="none",
            depth_policy=elastic_depth_policy,  # type: ignore[arg-type]
        )
        return MetricOutcome(
            value=result.value,
            objective_evaluations=result.objective_evaluations,
            upstream_evaluations=result.upstream_energy_evaluations,
            result_status=(
                "full_tree"
                if (
                    result.tree_a_omitted_frontier_branches == 0
                    and result.tree_b_omitted_frontier_branches == 0
                )
                else "truncated_four_layer_representation"
            ),
            details={
                "aligned_angle_rad": result.angle_rad,
                "adapter_runtime_seconds": result.runtime_seconds,
                "tree_a_omitted_frontier_branches": (
                    result.tree_a_omitted_frontier_branches
                ),
                "tree_b_omitted_frontier_branches": (
                    result.tree_b_omitted_frontier_branches
                ),
                "tree_a_represented_branches": result.tree_a_represented_branches,
                "tree_b_represented_branches": result.tree_b_represented_branches,
                "external_revision": result.external_revision,
                "external_checkout": result.external_checkout,
            },
        )

    specs.append(
        BenchmarkMetric(
            name="elastic_srvft",
            display_name="Elastic SRVFT alignment energy",
            family="elastic_srvft",
            family_display_name="Elastic SRVFT",
            grid_angles=elastic_grid_size,
            refine=elastic_refine,
            evaluate=elastic,
        )
    )
    return tuple(specs)


def _timed_measurement(
    spec: BenchmarkMetric,
    graph_a: nx.Graph,
    graph_b: nx.Graph,
    *,
    pair_index: int,
) -> Measurement:
    started_ns = perf_counter_ns()
    try:
        outcome = spec.evaluate(graph_a, graph_b)
    except Exception as exc:  # Keep failures in the benchmark rather than dropping pairs.
        elapsed = (perf_counter_ns() - started_ns) / 1_000_000_000.0
        return Measurement(
            pair_index=pair_index,
            metric_name=spec.name,
            metric_display_name=spec.display_name,
            family=spec.family,
            family_display_name=spec.family_display_name,
            status="error",
            result_status="error",
            elapsed_seconds=elapsed,
            value=None,
            grid_angles=spec.grid_angles,
            refine=spec.refine,
            objective_evaluations=None,
            upstream_evaluations=None,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    elapsed = (perf_counter_ns() - started_ns) / 1_000_000_000.0
    return Measurement(
        pair_index=pair_index,
        metric_name=spec.name,
        metric_display_name=spec.display_name,
        family=spec.family,
        family_display_name=spec.family_display_name,
        status="ok",
        result_status=outcome.result_status,
        elapsed_seconds=elapsed,
        value=float(outcome.value),
        grid_angles=spec.grid_angles,
        refine=spec.refine,
        objective_evaluations=int(outcome.objective_evaluations),
        upstream_evaluations=(
            None
            if outcome.upstream_evaluations is None
            else int(outcome.upstream_evaluations)
        ),
        details=outcome.details,
    )


def _number_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "std": float(stdev(values)) if len(values) >= 2 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def summarize_measurements(
    measurements: Sequence[Measurement],
) -> list[dict[str, object]]:
    """Summarize successful timings without silently losing failed attempts."""

    grouped: dict[str, list[Measurement]] = {}
    for measurement in measurements:
        grouped.setdefault(measurement.metric_name, []).append(measurement)

    summaries: list[dict[str, object]] = []
    for metric_name, group in grouped.items():
        successful = [item for item in group if item.status == "ok"]
        seconds = [item.elapsed_seconds for item in successful]
        evaluations = [
            float(item.objective_evaluations)
            for item in successful
            if item.objective_evaluations is not None
        ]
        upstream = [
            float(item.upstream_evaluations)
            for item in successful
            if item.upstream_evaluations is not None
        ]
        timing = _number_summary(seconds)
        evaluation_summary = _number_summary(evaluations)
        upstream_summary = _number_summary(upstream)
        result_status_counts: dict[str, int] = {}
        for item in successful:
            result_status_counts[item.result_status] = (
                result_status_counts.get(item.result_status, 0) + 1
            )
        summaries.append(
            {
                "metric_name": metric_name,
                "metric_display_name": group[0].metric_display_name,
                "family": group[0].family,
                "family_display_name": group[0].family_display_name,
                "requested_count": len(group),
                "successful_count": len(successful),
                "failed_count": len(group) - len(successful),
                "finite_value_count": sum(
                    item.value is not None and math.isfinite(item.value)
                    for item in successful
                ),
                "mean_seconds": timing["mean"],
                "median_seconds": timing["median"],
                "std_seconds": timing["std"],
                "min_seconds": timing["min"],
                "max_seconds": timing["max"],
                "grid_angles": group[0].grid_angles,
                "refine": group[0].refine,
                "mean_objective_evaluations": evaluation_summary["mean"],
                "min_objective_evaluations": evaluation_summary["min"],
                "max_objective_evaluations": evaluation_summary["max"],
                "mean_upstream_evaluations": upstream_summary["mean"],
                "min_upstream_evaluations": upstream_summary["min"],
                "max_upstream_evaluations": upstream_summary["max"],
                "result_status_counts": result_status_counts,
            }
        )
    return summaries


def summarize_family_totals(
    measurements: Sequence[Measurement],
) -> list[dict[str, object]]:
    """Sum independently timed scalar variants within each family and pair."""

    family_names: dict[str, set[str]] = {}
    family_labels: dict[str, str] = {}
    pair_groups: dict[tuple[str, int], list[Measurement]] = {}
    for measurement in measurements:
        family_names.setdefault(measurement.family, set()).add(
            measurement.metric_name
        )
        family_labels[measurement.family] = measurement.family_display_name
        pair_groups.setdefault((measurement.family, measurement.pair_index), []).append(
            measurement
        )

    summaries: list[dict[str, object]] = []
    for family, metric_names in family_names.items():
        pair_totals: list[float] = []
        requested_pairs = 0
        for (candidate_family, _pair_index), group in pair_groups.items():
            if candidate_family != family:
                continue
            requested_pairs += 1
            names = {item.metric_name for item in group}
            if names == metric_names and all(item.status == "ok" for item in group):
                pair_totals.append(sum(item.elapsed_seconds for item in group))
        timing = _number_summary(pair_totals)
        summaries.append(
            {
                "family": family,
                "family_display_name": family_labels[family],
                "scalar_variant_count": len(metric_names),
                "requested_pair_count": requested_pairs,
                "successful_pair_count": len(pair_totals),
                "failed_pair_count": requested_pairs - len(pair_totals),
                "mean_total_seconds": timing["mean"],
                "median_total_seconds": timing["median"],
                "std_total_seconds": timing["std"],
                "min_total_seconds": timing["min"],
                "max_total_seconds": timing["max"],
                "timing_definition": "sum_of_independently_timed_scalar_variants",
            }
        )
    return summaries


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _csv_value(value: object) -> object:
    safe = _json_safe(value)
    if safe is None:
        return ""
    if isinstance(safe, (dict, list)):
        return json.dumps(safe, sort_keys=True, allow_nan=False)
    return safe


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV table: {path}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _measurement_rows(measurements: Sequence[Measurement]) -> list[dict[str, object]]:
    return [
        {
            "pair_index": item.pair_index,
            "metric_name": item.metric_name,
            "metric_display_name": item.metric_display_name,
            "family": item.family,
            "family_display_name": item.family_display_name,
            "status": item.status,
            "result_status": item.result_status,
            "elapsed_seconds": item.elapsed_seconds,
            "value": item.value,
            "grid_angles": item.grid_angles,
            "refine": item.refine,
            "objective_evaluations": item.objective_evaluations,
            "upstream_evaluations": item.upstream_evaluations,
            "error_type": item.error_type,
            "error_message": item.error_message,
            "details": item.details,
        }
        for item in measurements
    ]


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _grid_size(text: str) -> int:
    value = int(text)
    if value < 3:
        raise argparse.ArgumentTypeError("must be at least 3")
    return value


def _default_dataset_root() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "neurons_conditional"


def _default_output_dir(split: str, pair_count: int, seed: int) -> Path:
    return Path("outputs") / "metric_study" / (
        f"runtime_benchmark_{split}_{pair_count}pairs_seed{seed}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Time scalar tree dissimilarities on reproducible disjoint pairs "
            "of class-labelled ground-truth SWCs."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_default_dataset_root(),
        help="Dataset root containing the selected split directory.",
    )
    parser.add_argument("--split", default="test", help="Dataset split to benchmark.")
    parser.add_argument("--pairs", type=_positive_int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=ALL_METRIC_NAMES,
        default=list(ALL_METRIC_NAMES),
        help="Scalar metric variants to time; the default selects all 13.",
    )
    parser.add_argument("--so2-grid-size", type=_grid_size, default=72)
    parser.add_argument(
        "--no-so2-refine",
        dest="so2_refine",
        action="store_false",
        help="Disable bounded refinement for Chamfer and FGW.",
    )
    parser.add_argument("--elastic-so2-grid-size", type=_grid_size, default=8)
    parser.add_argument(
        "--elastic-so2-refine",
        action="store_true",
        help="Enable bounded refinement for Elastic SRVFT.",
    )
    parser.add_argument(
        "--elastic-depth-policy",
        choices=("raise", "warn", "allow"),
        default="raise",
        help=(
            "The safe default rejects trees deeper than the external four-layer "
            "representation. Use 'allow' explicitly only for a labelled truncation "
            "runtime study."
        ),
    )
    parser.add_argument(
        "--fgw-max-nodes",
        type=int,
        default=1_000,
        help="Dense FGW node limit per input tree; 0 disables the guard.",
    )
    parser.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        help="Do not run one excluded warm-up evaluation per metric variant.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow known benchmark artifacts in an existing directory to be replaced.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    dataset_root = args.dataset_root.expanduser().resolve()
    records = discover_tree_records(dataset_root, split_dirs=(args.split,))
    pairs = select_random_pairs(records, args.pairs, seed=args.seed)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else _default_output_dir(args.split, args.pairs, args.seed)
    ).expanduser().resolve()
    known_artifacts = (
        "pairs.csv",
        "measurements.csv",
        "metric_summary.csv",
        "family_summary.csv",
        "run.json",
    )
    existing = [output_dir / name for name in known_artifacts if (output_dir / name).exists()]
    if existing and not args.overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Benchmark artifacts already exist in {output_dir}: {names}. "
            "Pass --overwrite to replace them."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {2 * len(pairs)} selected SWCs before timing...", file=sys.stderr)
    loaded: dict[Path, nx.Graph] = {}
    for record_a, record_b in pairs:
        for record in (record_a, record_b):
            if record.swc_path not in loaded:
                loaded[record.swc_path] = transform_scientific_y_to_internal_z(
                    load_swc_graph(record.swc_path)
                )

    pair_graphs = tuple(
        (loaded[record_a.swc_path], loaded[record_b.swc_path])
        for record_a, record_b in pairs
    )
    all_specs = build_metric_specs(
        so2_grid_size=args.so2_grid_size,
        so2_refine=args.so2_refine,
        elastic_grid_size=args.elastic_so2_grid_size,
        elastic_refine=args.elastic_so2_refine,
        elastic_depth_policy=args.elastic_depth_policy,
        fgw_max_nodes=args.fgw_max_nodes,
    )
    wanted = set(args.metrics)
    specs = tuple(spec for spec in all_specs if spec.name in wanted)

    warmups: list[dict[str, object]] = []
    if args.warmup:
        warm_graph_a, warm_graph_b = pair_graphs[0]
        for index, spec in enumerate(specs, start=1):
            print(
                f"Warm-up {index}/{len(specs)}: {spec.display_name}",
                file=sys.stderr,
                flush=True,
            )
            warm = _timed_measurement(
                spec,
                warm_graph_a,
                warm_graph_b,
                pair_index=0,
            )
            warmups.append(
                {
                    "metric_name": spec.name,
                    "status": warm.status,
                    "elapsed_seconds": warm.elapsed_seconds,
                    "error_type": warm.error_type,
                    "error_message": warm.error_message,
                }
            )

    measurements: list[Measurement] = []
    total_evaluations = len(specs) * len(pair_graphs)
    progress = 0
    for spec in specs:
        for pair_index, (graph_a, graph_b) in enumerate(pair_graphs, start=1):
            progress += 1
            print(
                f"Timed evaluation {progress}/{total_evaluations}: "
                f"{spec.display_name}, pair {pair_index}/{len(pair_graphs)}",
                file=sys.stderr,
                flush=True,
            )
            measurements.append(
                _timed_measurement(
                    spec,
                    graph_a,
                    graph_b,
                    pair_index=pair_index,
                )
            )

    metric_summary = summarize_measurements(measurements)
    family_summary = summarize_family_totals(measurements)

    pair_rows: list[dict[str, object]] = []
    for pair_index, ((record_a, record_b), (graph_a, graph_b)) in enumerate(
        zip(pairs, pair_graphs, strict=True),
        start=1,
    ):
        pair_rows.append(
            {
                "pair_index": pair_index,
                "tree_a_id": record_a.tree_id,
                "tree_a_path": record_a.swc_path,
                "tree_a_class": record_a.cell_class,
                "tree_a_type": record_a.cell_type,
                "tree_a_nodes": graph_a.number_of_nodes(),
                "tree_a_edges": graph_a.number_of_edges(),
                "tree_b_id": record_b.tree_id,
                "tree_b_path": record_b.swc_path,
                "tree_b_class": record_b.cell_class,
                "tree_b_type": record_b.cell_type,
                "tree_b_nodes": graph_b.number_of_nodes(),
                "tree_b_edges": graph_b.number_of_edges(),
            }
        )

    pairs_path = output_dir / "pairs.csv"
    measurements_path = output_dir / "measurements.csv"
    metric_summary_path = output_dir / "metric_summary.csv"
    family_summary_path = output_dir / "family_summary.csv"
    run_path = output_dir / "run.json"
    _write_rows(pairs_path, pair_rows)
    _write_rows(measurements_path, _measurement_rows(measurements))
    _write_rows(metric_summary_path, metric_summary)
    _write_rows(family_summary_path, family_summary)

    elastic_summary = next(
        (
            summary
            for summary in metric_summary
            if summary["metric_name"] == "elastic_srvft"
        ),
        None,
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "root": dataset_root,
            "split": args.split,
            "available_trees": len(records),
            "selected_pairs": len(pairs),
            "selected_trees": 2 * len(pairs),
            "pair_selection": (
                "sort records, NumPy default_rng(seed).choice without replacement, "
                "pair adjacent selections"
            ),
            "seed": args.seed,
        },
        "frame_contract": {
            "scientific_so2_axis_original_xyz": SCIENTIFIC_AXIS,
            "metric_internal_so2_axis_transformed_xyz": INTERNAL_AXIS,
            "proper_rotation_original_to_internal": SCIENTIFIC_Y_TO_INTERNAL_Z.tolist(),
            "coordinate_map": "(x, y, z) -> (x, -z, y)",
            "determinant": float(np.linalg.det(SCIENTIFIC_Y_TO_INTERNAL_Z)),
        },
        "timing_protocol": {
            "timer": "time.perf_counter_ns",
            "single_process": True,
            "swc_loading_included": False,
            "common_frame_transform_included": False,
            "warmup_enabled": bool(args.warmup),
            "warmup_pair": 1 if args.warmup else None,
            "warmup_included_in_averages": False,
            "repetitions_per_pair_and_metric": 1,
            "family_totals": "sum of independently timed scalar variants",
        },
        "configuration": {
            "selected_metrics": [spec.name for spec in specs],
            "chamfer_fgw_grid_angles": args.so2_grid_size,
            "chamfer_fgw_refine": args.so2_refine,
            "elastic_grid_angles": args.elastic_so2_grid_size,
            "elastic_refine": args.elastic_so2_refine,
            "elastic_depth_policy": args.elastic_depth_policy,
            "fgw_max_nodes": args.fgw_max_nodes,
        },
        "elastic_interpretation": {
            "safe_default_depth_policy": "raise",
            "timed_depth_policy": args.elastic_depth_policy,
            "warning": (
                "When timed_depth_policy is 'allow' or 'warn', result status "
                "'truncated_four_layer_representation' is not a full-tree metric."
            ),
            "result_status_counts": (
                None
                if elastic_summary is None
                else elastic_summary["result_status_counts"]
            ),
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "thread_limits": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "warmups": warmups,
        "metric_summary": metric_summary,
        "family_summary": family_summary,
        "total_run_seconds": perf_counter() - started,
        "artifacts": {
            "pairs": pairs_path,
            "measurements": measurements_path,
            "metric_summary": metric_summary_path,
            "family_summary": family_summary_path,
            "run_metadata": run_path,
        },
    }
    run_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fgw_max_nodes < 0:
        parser.error("--fgw-max-nodes must be non-negative")
    try:
        payload = run(args)
    except (FileExistsError, NotADirectoryError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
