"""Sharded full-depth-compatible Elastic SRVFT distance matrix study.

The scientific definition is intentionally narrow: a relative SO(2) quotient,
forward/reverse mean symmetrization, a 36-angle grid with local refinement, and
rejection of every tree that exceeds the external four-layer representation.

Slurm array elements write separate result shards and a final merge builds the
matrix:

``prepare`` -> ``compute-shard`` -> ``merge``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    from dendrite_gen.metrics.adapters import elastic_srvft as elastic_adapter
    from dendrite_gen.utils.data_loading import load_swc_graph
except ModuleNotFoundError as exc:
    if exc.name != "dendrite_gen":
        raise
    from metrics.adapters import elastic_srvft as elastic_adapter  # type: ignore
    from utils.data_loading import load_swc_graph  # type: ignore

from .dataset import TreeRecord, discover_tree_records, validate_tree_records
from .frame import (
    INTERNAL_AXIS,
    SCIENTIFIC_AXIS,
    SCIENTIFIC_Y_TO_INTERNAL_Z,
    transform_scientific_y_to_internal_z,
)


STATUS_PENDING = np.uint8(0)
STATUS_OK = np.uint8(1)
STATUS_ERROR = np.uint8(3)

DEFAULT_CLASS_CAP = 20
DEFAULT_PAIRS_PER_SHARD = 8
DEFAULT_GRID_SIZE = 36
DEFAULT_REFINEMENT_TOLERANCE = 1e-3
METRIC_NAME = "elastic_srvft"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    rendered = json.dumps(
        _json_safe(payload), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _json_safe(row.get(name, "")) for name in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_npy_atomic(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _default_dataset_root() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "neurons_conditional"


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


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _nonnegative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return value


def _selection_rank(record: TreeRecord, seed: int) -> tuple[bytes, str]:
    identity = f"{seed}\0{record.cell_class}\0{record.tree_id}".encode("utf-8")
    return hashlib.sha256(identity).digest(), record.tree_id


def select_compatible_records(
    records: Sequence[TreeRecord],
    eligible_tree_ids: set[str],
    *,
    class_cap: int,
    seed: int,
) -> tuple[TreeRecord, ...]:
    """Select up to ``class_cap`` compatible trees from every discovered class."""
    if class_cap <= 0:
        raise ValueError("class_cap must be positive.")
    validate_tree_records(records)
    by_class: dict[int, list[TreeRecord]] = defaultdict(list)
    all_classes = sorted({record.cell_class for record in records})
    for record in records:
        if record.tree_id in eligible_tree_ids:
            by_class[record.cell_class].append(record)

    missing = [cell_class for cell_class in all_classes if not by_class[cell_class]]
    if missing:
        raise ValueError(
            "No full four-layer-compatible trees remain for classes "
            f"{missing!r}."
        )

    selected: list[TreeRecord] = []
    for cell_class in all_classes:
        chosen = sorted(
            by_class[cell_class], key=lambda item: _selection_rank(item, seed)
        )[:class_cap]
        selected.extend(sorted(chosen, key=lambda item: item.tree_id))
    return tuple(selected)


def build_pair_plan(
    tree_count: int,
    *,
    pairs_per_shard: int,
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    """Return deterministic strict-upper pair rows and contiguous shard rows."""
    if tree_count < 2:
        raise ValueError("At least two selected trees are required.")
    if pairs_per_shard <= 0:
        raise ValueError("pairs_per_shard must be positive.")

    pairs: list[dict[str, int]] = []
    pair_index = 0
    for index_a in range(tree_count - 1):
        for index_b in range(index_a + 1, tree_count):
            pairs.append(
                {
                    "pair_index": pair_index,
                    "shard_id": pair_index // pairs_per_shard,
                    "index_a": index_a,
                    "index_b": index_b,
                }
            )
            pair_index += 1

    shards: list[dict[str, int]] = []
    for start in range(0, len(pairs), pairs_per_shard):
        stop = min(start + pairs_per_shard, len(pairs))
        shards.append(
            {
                "shard_id": start // pairs_per_shard,
                "pair_start": start,
                "pair_stop": stop,
                "pair_count": stop - start,
            }
        )
    return pairs, shards


def _backend_contract(checkout: str | Path | None) -> dict[str, str]:
    api = elastic_adapter._load_external_api(checkout)
    return {"checkout": str(api.checkout), "revision": api.revision}


def screen_records(
    records: Sequence[TreeRecord],
    *,
    checkout: str | Path | None,
    default_radius: float,
) -> tuple[list[dict[str, object]], set[str], dict[str, str]]:
    """Inspect every tree once and classify fixed-depth compatibility."""
    backend = _backend_contract(checkout)
    rows: list[dict[str, object]] = []
    eligible: set[str] = set()
    for position, record in enumerate(records, start=1):
        if position == 1 or position % 50 == 0 or position == len(records):
            print(
                f"Screening Elastic tree {position}/{len(records)}: "
                f"{record.tree_id}",
                file=sys.stderr,
                flush=True,
            )
        row: dict[str, object] = {
            "tree_id": record.tree_id,
            "swc_path": record.swc_path,
            "split": record.split,
            "cell_class": record.cell_class,
            "cell_type": record.cell_type,
            "status": "error",
            "node_count": "",
            "terminal_leaf_count": "",
            "represented_branch_count": "",
            "omitted_frontier_branches": "",
            "canonical_order_ties": "",
            "error_type": "",
            "error_message": "",
        }
        try:
            graph = transform_scientific_y_to_internal_z(
                load_swc_graph(record.swc_path)
            )
            diagnostics = elastic_adapter.elastic_srvft_tree_diagnostics(
                graph,
                checkout=backend["checkout"],
                depth_policy="allow",
                default_radius=default_radius,
            )
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            row["error_message"] = str(exc)
        else:
            row.update(
                {
                    "node_count": diagnostics.node_count,
                    "terminal_leaf_count": diagnostics.terminal_leaf_count,
                    "represented_branch_count": diagnostics.represented_branch_count,
                    "omitted_frontier_branches": (
                        diagnostics.omitted_frontier_branches
                    ),
                    "canonical_order_ties": diagnostics.canonical_order_ties,
                }
            )
            if diagnostics.omitted_frontier_branches:
                row["status"] = "truncated"
            else:
                row["status"] = "eligible"
                eligible.add(record.tree_id)
        rows.append(row)
    return rows, eligible, backend


def prepare_run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Reuse the existing "
            "plan or choose a new directory."
        )

    dataset_root = args.dataset_root.expanduser().resolve()
    records = discover_tree_records(dataset_root, split_dirs=(args.split,))
    eligibility_rows, eligible_ids, backend = screen_records(
        records,
        checkout=args.elastic_checkout,
        default_radius=args.default_radius,
    )
    selected = select_compatible_records(
        records,
        eligible_ids,
        class_cap=args.max_trees_per_class,
        seed=args.seed,
    )
    pair_rows, shard_rows = build_pair_plan(
        len(selected), pairs_per_shard=args.pairs_per_shard
    )

    diagnostics_by_id = {str(row["tree_id"]): row for row in eligibility_rows}
    selected_rows: list[dict[str, object]] = []
    for matrix_index, record in enumerate(selected):
        diagnostics = diagnostics_by_id[record.tree_id]
        selected_rows.append(
            {
                "matrix_index": matrix_index,
                "tree_id": record.tree_id,
                "swc_path": record.swc_path,
                "split": record.split,
                "cell_class": record.cell_class,
                "cell_type": record.cell_type,
                "node_count": diagnostics["node_count"],
                "terminal_leaf_count": diagnostics["terminal_leaf_count"],
                "represented_branch_count": diagnostics[
                    "represented_branch_count"
                ],
                "canonical_order_ties": diagnostics["canonical_order_ties"],
            }
        )

    pair_manifest_rows: list[dict[str, object]] = []
    for row in pair_rows:
        record_a = selected[row["index_a"]]
        record_b = selected[row["index_b"]]
        pair_manifest_rows.append(
            {
                **row,
                "tree_a_id": record_a.tree_id,
                "tree_b_id": record_b.tree_id,
                "class_a": record_a.cell_class,
                "class_b": record_b.cell_class,
            }
        )

    metric_configuration = {
        "name": METRIC_NAME,
        "value_kind": "upstream_alignment_energy",
        "metric_status": "dissimilarity_not_established_as_a_metric",
        "quotient_so2": True,
        "grid_size": args.so2_grid_size,
        "refine": True,
        "refinement_tolerance": args.refinement_tolerance,
        "symmetrization": "mean",
        "depth_policy": "raise",
        "lam_m": args.lam_m,
        "lam_s": args.lam_s,
        "lam_p": args.lam_p,
        "default_radius": args.default_radius,
    }
    class_available = Counter(
        record.cell_type for record in records if record.tree_id in eligible_ids
    )
    class_selected = Counter(record.cell_type for record in selected)
    screening_counts = Counter(str(row["status"]) for row in eligibility_rows)
    run_payload: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": _now(),
        "dataset": {"root": dataset_root, "split": args.split},
        "selection": {
            "policy": "full_four_layer_compatible_then_per_class_cap",
            "max_trees_per_class": args.max_trees_per_class,
            "seed": args.seed,
            "candidate_trees": len(records),
            "screening_counts": dict(sorted(screening_counts.items())),
            "compatible_by_class": dict(sorted(class_available.items())),
            "selected_by_class": dict(sorted(class_selected.items())),
            "selected_trees": len(selected),
        },
        "pairs": {
            "strict_upper_triangle": len(pair_rows),
            "pairs_per_shard": args.pairs_per_shard,
            "shard_count": len(shard_rows),
        },
        "metric": metric_configuration,
        "backend": backend,
        "frame_contract": {
            "scientific_so2_axis_original_xyz": SCIENTIFIC_AXIS,
            "metric_internal_so2_axis_transformed_xyz": INTERNAL_AXIS,
            "proper_rotation_original_to_internal": (
                SCIENTIFIC_Y_TO_INTERNAL_Z.tolist()
            ),
            "coordinate_map": "(x, y, z) -> (x, -z, y)",
        },
        "artifacts": {
            "eligibility": "eligibility.csv",
            "selected_trees": "selected_trees.csv",
            "pairs": "pairs.csv",
            "task_count": "task_count.txt",
            "shard_results": "shard_results",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shard_results").mkdir()
    _write_csv_atomic(
        output_dir / "eligibility.csv",
        eligibility_rows,
        (
            "tree_id",
            "swc_path",
            "split",
            "cell_class",
            "cell_type",
            "status",
            "node_count",
            "terminal_leaf_count",
            "represented_branch_count",
            "omitted_frontier_branches",
            "canonical_order_ties",
            "error_type",
            "error_message",
        ),
    )
    _write_csv_atomic(
        output_dir / "selected_trees.csv",
        selected_rows,
        (
            "matrix_index",
            "tree_id",
            "swc_path",
            "split",
            "cell_class",
            "cell_type",
            "node_count",
            "terminal_leaf_count",
            "represented_branch_count",
            "canonical_order_ties",
        ),
    )
    _write_csv_atomic(
        output_dir / "pairs.csv",
        pair_manifest_rows,
        (
            "pair_index",
            "shard_id",
            "index_a",
            "index_b",
            "tree_a_id",
            "tree_b_id",
            "class_a",
            "class_b",
        ),
    )
    (output_dir / "task_count.txt").write_text(
        f"{len(shard_rows)}\n", encoding="utf-8"
    )
    _write_json_atomic(output_dir / "run.json", run_payload)
    return run_payload


def _resolve_shard_id(explicit: int | None) -> int:
    if explicit is not None:
        if explicit < 0:
            raise ValueError("shard_id must be non-negative.")
        return explicit
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw is None:
        raise ValueError(
            "Supply --shard-id or run inside a Slurm array with "
            "SLURM_ARRAY_TASK_ID."
        )
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("SLURM_ARRAY_TASK_ID must be an integer.") from exc
    if value < 0:
        raise ValueError("SLURM_ARRAY_TASK_ID must be non-negative.")
    return value


def _load_shard_payload(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _load_json(path)


def _result_by_pair(payload: Mapping[str, object]) -> dict[int, dict[str, Any]]:
    raw_results = payload.get("results", [])
    if not isinstance(raw_results, list):
        raise ValueError("Shard results must be a list.")
    results: dict[int, dict[str, Any]] = {}
    for raw in raw_results:
        if not isinstance(raw, dict):
            raise ValueError("Every shard result must be an object.")
        pair_index = int(raw["pair_index"])
        if pair_index in results:
            raise ValueError(f"Duplicate pair {pair_index} in shard payload.")
        results[pair_index] = raw
    return results


def compute_shard(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir.expanduser().resolve()
    run_payload = _load_json(output_dir / "run.json")
    backend = run_payload.get("backend")
    if not isinstance(backend, Mapping) or not backend.get("checkout"):
        raise ValueError("run.json has no Elastic checkout path.")
    checkout = str(backend["checkout"])
    shard_id = _resolve_shard_id(args.shard_id)

    selected_rows = _read_csv(output_dir / "selected_trees.csv")
    pair_rows = [
        row
        for row in _read_csv(output_dir / "pairs.csv")
        if int(row["shard_id"]) == shard_id
    ]
    if not pair_rows:
        raise ValueError(f"Shard {shard_id} is outside the prepared pair plan.")

    shard_path = output_dir / "shard_results" / f"shard_{shard_id:06d}.json"
    existing = _load_shard_payload(shard_path)
    if existing is None:
        shard_payload: dict[str, Any] = {
            "schema_version": 1,
            "shard_id": shard_id,
            "complete": False,
            "updated_at_utc": _now(),
            "results": [],
        }
    else:
        shard_payload = existing
        if int(shard_payload.get("shard_id", -1)) != shard_id:
            raise ValueError(f"Shard file has the wrong shard_id: {shard_path}.")

    existing_results = _result_by_pair(shard_payload)
    expected_indices = {int(row["pair_index"]) for row in pair_rows}
    unexpected = set(existing_results) - expected_indices
    if unexpected:
        raise ValueError(
            f"Shard {shard_id} contains unexpected pairs: {sorted(unexpected)!r}."
        )

    metric = run_payload.get("metric")
    if not isinstance(metric, Mapping):
        raise ValueError("run.json has no valid Elastic metric configuration.")
    graph_cache: dict[int, object] = {}
    new_pairs = 0
    for position, row in enumerate(pair_rows, start=1):
        pair_index = int(row["pair_index"])
        if pair_index in existing_results:
            continue

        index_a = int(row["index_a"])
        index_b = int(row["index_b"])
        print(
            f"Shard {shard_id}: pair {position}/{len(pair_rows)} "
            f"({row['tree_a_id']}, {row['tree_b_id']})",
            file=sys.stderr,
            flush=True,
        )
        try:
            for index in (index_a, index_b):
                if index not in graph_cache:
                    graph_cache[index] = transform_scientific_y_to_internal_z(
                        load_swc_graph(Path(selected_rows[index]["swc_path"]))
                    )
            result = elastic_adapter.elastic_srvft_distance(
                graph_cache[index_a],  # type: ignore[arg-type]
                graph_cache[index_b],  # type: ignore[arg-type]
                checkout=checkout,
                lam_m=float(metric["lam_m"]),
                lam_s=float(metric["lam_s"]),
                lam_p=float(metric["lam_p"]),
                quotient_so2=True,
                grid_size=int(metric["grid_size"]),
                refine=True,
                refinement_tolerance=float(metric["refinement_tolerance"]),
                symmetrization="mean",
                depth_policy="raise",
                default_radius=float(metric["default_radius"]),
            )
            if (
                result.tree_a_omitted_frontier_branches
                or result.tree_b_omitted_frontier_branches
            ):
                raise RuntimeError(
                    "A selected tree unexpectedly produced a truncated Elastic "
                    "representation."
                )
            pair_result: dict[str, object] = {
                "pair_index": pair_index,
                "index_a": index_a,
                "index_b": index_b,
                "tree_a_id": row["tree_a_id"],
                "tree_b_id": row["tree_b_id"],
                "status": "ok",
                "value": result.value,
                "result": asdict(result),
            }
        except Exception as exc:
            pair_result = {
                "pair_index": pair_index,
                "index_a": index_a,
                "index_b": index_b,
                "tree_a_id": row["tree_a_id"],
                "tree_b_id": row["tree_b_id"],
                "status": "error",
                "value": None,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

        existing_results[pair_index] = pair_result
        shard_payload["results"] = [
            existing_results[index] for index in sorted(existing_results)
        ]
        shard_payload["complete"] = False
        shard_payload["updated_at_utc"] = _now()
        _write_json_atomic(shard_path, shard_payload)
        new_pairs += 1

    missing = expected_indices - set(existing_results)
    shard_payload["results"] = [
        existing_results[index] for index in sorted(existing_results)
    ]
    shard_payload["complete"] = not missing
    shard_payload["updated_at_utc"] = _now()
    _write_json_atomic(shard_path, shard_payload)
    counts = Counter(
        str(result.get("status", "invalid")) for result in existing_results.values()
    )
    return {
        "shard_id": shard_id,
        "complete": not missing,
        "expected_pairs": len(expected_indices),
        "new_pairs": new_pairs,
        "status_counts": dict(sorted(counts.items())),
        "artifact": shard_path,
    }


def merge_run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir.expanduser().resolve()
    run_payload = _load_json(output_dir / "run.json")
    selected_rows = _read_csv(output_dir / "selected_trees.csv")
    pair_rows = _read_csv(output_dir / "pairs.csv")
    expected = {int(row["pair_index"]): row for row in pair_rows}
    if len(expected) != len(pair_rows):
        raise ValueError("pairs.csv contains duplicate pair indices.")

    pairs_by_shard: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in pair_rows:
        pairs_by_shard[int(row["shard_id"])].append(row)

    missing_shards: list[int] = []
    incomplete_shards: list[int] = []
    collected: dict[int, dict[str, Any]] = {}
    for shard_id, planned_rows in sorted(pairs_by_shard.items()):
        shard_path = (
            output_dir / "shard_results" / f"shard_{shard_id:06d}.json"
        )
        shard_payload = _load_shard_payload(shard_path)
        if shard_payload is None:
            missing_shards.append(shard_id)
            continue
        results = _result_by_pair(shard_payload)
        planned_indices = {int(row["pair_index"]) for row in planned_rows}
        if (
            int(shard_payload.get("shard_id", -1)) != shard_id
            or not shard_payload.get("complete")
            or set(results) != planned_indices
        ):
            incomplete_shards.append(shard_id)
            continue
        for pair_index, result in results.items():
            if pair_index in collected:
                raise ValueError(f"Duplicate pair result {pair_index}.")
            plan_row = expected[pair_index]
            identity = (
                int(plan_row["index_a"]),
                int(plan_row["index_b"]),
                plan_row["tree_a_id"],
                plan_row["tree_b_id"],
            )
            observed = (
                int(result["index_a"]),
                int(result["index_b"]),
                str(result["tree_a_id"]),
                str(result["tree_b_id"]),
            )
            if observed != identity:
                raise ValueError(
                    f"Pair identity mismatch for pair {pair_index}: "
                    f"expected {identity!r}, got {observed!r}."
                )
            collected[pair_index] = result
    if missing_shards or incomplete_shards:
        raise ValueError(
            "Cannot merge yet. Missing shards: "
            f"{missing_shards!r}; incomplete shards: {incomplete_shards!r}."
        )
    if set(collected) != set(expected):
        missing = sorted(set(expected) - set(collected))
        raise ValueError(f"Merged shards are missing pairs: {missing[:20]!r}.")

    tree_count = len(selected_rows)
    distances = np.full((tree_count, tree_count), np.nan, dtype=np.float64)
    matrix_status = np.full((tree_count, tree_count), STATUS_PENDING, dtype=np.uint8)
    diagonal = np.arange(tree_count)
    distances[diagonal, diagonal] = 0.0
    matrix_status[diagonal, diagonal] = STATUS_OK
    issue_rows: list[dict[str, object]] = []
    jsonl_rows: list[str] = []
    for pair_index in sorted(collected):
        result = collected[pair_index]
        index_a = int(result["index_a"])
        index_b = int(result["index_b"])
        if result.get("status") == "ok":
            value = float(result["value"])
            if not math.isfinite(value) or value < -1e-10:
                raise ValueError(f"Invalid Elastic value for pair {pair_index}: {value}.")
            value = max(value, 0.0)
            distances[index_a, index_b] = distances[index_b, index_a] = value
            matrix_status[index_a, index_b] = matrix_status[index_b, index_a] = STATUS_OK
        elif result.get("status") == "error":
            matrix_status[index_a, index_b] = matrix_status[index_b, index_a] = STATUS_ERROR
            issue_rows.append(
                {
                    "pair_index": pair_index,
                    "index_a": index_a,
                    "index_b": index_b,
                    "tree_a_id": result["tree_a_id"],
                    "tree_b_id": result["tree_b_id"],
                    "error_type": result.get("error_type", ""),
                    "error_message": result.get("error_message", ""),
                }
            )
        else:
            raise ValueError(
                f"Unknown pair status for pair {pair_index}: {result.get('status')!r}."
            )
        jsonl_rows.append(
            json.dumps(_json_safe(result), sort_keys=True, allow_nan=False)
        )

    metric_dir = output_dir / "metrics" / METRIC_NAME
    metric_dir.mkdir(parents=True, exist_ok=True)
    _write_npy_atomic(metric_dir / "distances.npy", distances)
    _write_npy_atomic(metric_dir / "status.npy", matrix_status)
    _write_json_atomic(
        metric_dir / "metric.json",
        {
            "name": METRIC_NAME,
            "configuration": run_payload["metric"],
            "backend": run_payload["backend"],
        },
    )
    pair_results_path = metric_dir / "pair_results.jsonl"
    pair_results_tmp = pair_results_path.with_name(pair_results_path.name + ".tmp")
    with pair_results_tmp.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(jsonl_rows) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    pair_results_tmp.replace(pair_results_path)
    _write_csv_atomic(
        metric_dir / "issues.csv",
        issue_rows,
        (
            "pair_index",
            "index_a",
            "index_b",
            "tree_a_id",
            "tree_b_id",
            "error_type",
            "error_message",
        ),
    )
    final_status = "complete_with_errors" if issue_rows else "complete"
    progress = {
        "status": final_status,
        "updated_at_utc": _now(),
        "selected_trees": tree_count,
        "strict_upper_triangle_pairs": len(pair_rows),
        "ok_pairs": len(pair_rows) - len(issue_rows),
        "error_pairs": len(issue_rows),
        "artifacts": {
            "distances": metric_dir / "distances.npy",
            "status": metric_dir / "status.npy",
            "metric": metric_dir / "metric.json",
            "pair_results": pair_results_path,
            "issues": metric_dir / "issues.csv",
        },
    }
    _write_json_atomic(output_dir / "progress.json", progress)
    return progress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare, shard, and merge a symmetric Elastic SRVFT matrix."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Screen the full split and freeze the selected pair plan."
    )
    prepare.add_argument("--dataset-root", type=Path, default=_default_dataset_root())
    prepare.add_argument("--split", default="test")
    prepare.add_argument(
        "--max-trees-per-class", type=_positive_int, default=DEFAULT_CLASS_CAP
    )
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument(
        "--pairs-per-shard", type=_positive_int, default=DEFAULT_PAIRS_PER_SHARD
    )
    prepare.add_argument("--so2-grid-size", type=_grid_size, default=DEFAULT_GRID_SIZE)
    prepare.add_argument(
        "--refinement-tolerance",
        type=_positive_float,
        default=DEFAULT_REFINEMENT_TOLERANCE,
    )
    prepare.add_argument("--lam-m", type=_nonnegative_float, default=0.2)
    prepare.add_argument("--lam-s", type=_nonnegative_float, default=1.0)
    prepare.add_argument("--lam-p", type=_nonnegative_float, default=0.2)
    prepare.add_argument("--default-radius", type=_positive_float, default=1.0)
    prepare.add_argument("--elastic-checkout", type=Path)
    prepare.add_argument("--output-dir", type=Path, required=True)

    compute = subparsers.add_parser(
        "compute-shard", help="Compute one independently writable pair shard."
    )
    compute.add_argument("--output-dir", type=Path, required=True)
    compute.add_argument("--shard-id", type=int)

    merge = subparsers.add_parser(
        "merge", help="Validate complete shards and publish the symmetric matrix."
    )
    merge.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            payload = prepare_run(args)
        elif args.command == "compute-shard":
            payload = compute_shard(args)
        elif args.command == "merge":
            payload = merge_run(args)
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except (
        elastic_adapter.ElasticSRVFTError,
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False))
    if args.command == "compute-shard":
        counts = payload.get("status_counts", {})
        return 2 if isinstance(counts, Mapping) and counts.get("error", 0) else 0
    if args.command == "merge":
        return 2 if payload.get("status") == "complete_with_errors" else 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
