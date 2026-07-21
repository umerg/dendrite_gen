"""Resumable distance matrices for selected ground-truth neuron trees."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
from itertools import islice
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import signal
import sys
from time import perf_counter
from typing import Any

import networkx as nx
import numpy as np

try:
    from dendrite_gen.utils.data_loading import load_swc_graph
except ModuleNotFoundError as exc:
    if exc.name != "dendrite_gen":
        raise
    from utils.data_loading import load_swc_graph  # type: ignore

from .dataset import (
    TreeRecord,
    discover_tree_records,
    select_balanced_sample,
    validate_tree_records,
)
from .frame import (
    INTERNAL_AXIS,
    SCIENTIFIC_AXIS,
    SCIENTIFIC_Y_TO_INTERNAL_Z,
    transform_scientific_y_to_internal_z,
)
from .matrix_metrics import (
    METRIC_SELECTORS,
    PreparedMatrixMetric,
    build_matrix_metric,
    expand_metric_selection,
)


STATUS_PENDING = np.uint8(0)
STATUS_OK = np.uint8(1)
STATUS_UNDEFINED = np.uint8(2)
STATUS_ERROR = np.uint8(3)
STATUS_PREPARATION_ERROR = np.uint8(4)
STATUS_LABELS = {
    int(STATUS_PENDING): "pending",
    int(STATUS_OK): "ok",
    int(STATUS_UNDEFINED): "undefined",
    int(STATUS_ERROR): "error",
    int(STATUS_PREPARATION_ERROR): "preparation_error",
}

_LOCK_FILENAME = ".distance_matrix_runner.lock"
_IMPLEMENTATION_SOURCES = (
    "metrics/chamfer.py",
    "metrics/distributions.py",
    "metrics/fused_gw.py",
    "metrics/persistence.py",
    "metrics/so2.py",
    "utils/data_loading.py",
    "utils/tmd.py",
    "visualization/metric_study/frame.py",
    "visualization/metric_study/matrix_metrics.py",
    "visualization/metric_study/run_distance_matrices.py",
    "visualization/tmd/distances.py",
)


@dataclass(frozen=True)
class _PairEvaluation:
    index_a: int
    index_b: int
    value: float | None
    error_type: str | None = None
    error_message: str | None = None


_WORKER_METRIC: PreparedMatrixMetric | None = None
_WORKER_PREPARED: Sequence[object | None] | None = None


def _initialize_pair_worker(
    metric: PreparedMatrixMetric,
    prepared: Sequence[object | None],
) -> None:
    global _WORKER_METRIC, _WORKER_PREPARED
    _WORKER_METRIC = metric
    _WORKER_PREPARED = prepared


def _clear_pair_worker() -> None:
    global _WORKER_METRIC, _WORKER_PREPARED
    _WORKER_METRIC = None
    _WORKER_PREPARED = None


def _evaluate_metric_pair(
    metric: PreparedMatrixMetric,
    prepared: Sequence[object | None],
    pair: tuple[int, int],
) -> _PairEvaluation:
    index_a, index_b = pair
    try:
        value = float(metric.compare(prepared[index_a], prepared[index_b]))
    except Exception as exc:
        return _PairEvaluation(
            index_a=index_a,
            index_b=index_b,
            value=None,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
    return _PairEvaluation(index_a=index_a, index_b=index_b, value=value)


def _evaluate_pair_in_worker(pair: tuple[int, int]) -> _PairEvaluation:
    if _WORKER_METRIC is None or _WORKER_PREPARED is None:
        raise RuntimeError("Pair worker was not initialized.")
    return _evaluate_metric_pair(_WORKER_METRIC, _WORKER_PREPARED, pair)


def _iter_pending_pairs(
    status: np.ndarray,
    *,
    limit: int | None,
) -> Iterator[tuple[int, int]]:
    emitted = 0
    for index_a in range(status.shape[0]):
        for index_b in range(index_a + 1, status.shape[1]):
            if status[index_a, index_b] != STATUS_PENDING:
                continue
            if limit is not None and emitted >= limit:
                return
            emitted += 1
            yield index_a, index_b


def _batched(
    values: Iterable[tuple[int, int]],
    size: int,
) -> Iterator[list[tuple[int, int]]]:
    iterator = iter(values)
    while batch := list(islice(iterator, size)):
        yield batch


def _resolve_worker_count(explicit_workers: int | None) -> tuple[int, str]:
    if explicit_workers is not None:
        if explicit_workers <= 0:
            raise ValueError("workers must be positive.")
        return explicit_workers, "--workers"

    slurm_workers = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_workers is None or not slurm_workers.strip():
        return 1, "default"
    try:
        workers = int(slurm_workers)
    except ValueError as exc:
        raise ValueError(
            "SLURM_CPUS_PER_TASK must be a positive integer or --workers "
            "must be supplied explicitly."
        ) from exc
    if workers <= 0:
        raise ValueError(
            "SLURM_CPUS_PER_TASK must be positive or --workers must be "
            "supplied explicitly."
        )
    return workers, "SLURM_CPUS_PER_TASK"


def _preferred_process_context() -> tuple[mp.context.BaseContext, str]:
    available = mp.get_all_start_methods()
    method = "fork" if "fork" in available else "spawn"
    return mp.get_context(method), method


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"Cannot fingerprint input SWC {path}: {exc}") from exc
    return digest.hexdigest()


def _frame_contract() -> dict[str, object]:
    return {
        "scientific_so2_axis_original_xyz": SCIENTIFIC_AXIS,
        "metric_internal_so2_axis_transformed_xyz": INTERNAL_AXIS,
        "proper_rotation_original_to_internal": SCIENTIFIC_Y_TO_INTERNAL_Z.tolist(),
        "coordinate_map": "(x, y, z) -> (x, -z, y)",
    }


def _implementation_contract() -> dict[str, object]:
    repository_root = Path(__file__).resolve().parents[2]
    source_hashes: dict[str, str] = {}
    combined = hashlib.sha256()
    for relative_path in _IMPLEMENTATION_SOURCES:
        source_path = repository_root / relative_path
        source_digest = _sha256_file(source_path)
        source_hashes[relative_path] = source_digest
        combined.update(relative_path.encode("utf-8"))
        combined.update(b"\0")
        combined.update(source_digest.encode("ascii"))
        combined.update(b"\0")

    dependency_versions: dict[str, str | None] = {}
    for distribution in ("networkx", "numpy", "scipy", "POT"):
        try:
            dependency_versions[distribution] = version(distribution)
        except PackageNotFoundError:
            dependency_versions[distribution] = None

    return {
        "fingerprint": combined.hexdigest(),
        "python": platform.python_version(),
        "dependencies": dependency_versions,
        "source_sha256": source_hashes,
    }


@contextmanager
def _exclusive_run_lock(output_dir: Path):
    """Hold a process lock for one output directory until the run returns."""

    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / _LOCK_FILENAME
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                f"Another distance-matrix process is already using {output_dir}."
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} acquired_at_utc={_now()}\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


class _RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, message: str) -> None:
        line = f"[{_now()}] {message}"
        print(line, file=sys.stderr, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _canonical_records(records: Sequence[TreeRecord]) -> tuple[TreeRecord, ...]:
    return tuple(
        sorted(
            records,
            key=lambda record: (
                record.cell_class,
                record.tree_id,
                record.split,
                record.swc_path.as_posix(),
            ),
        )
    )


def select_records(
    records: Sequence[TreeRecord],
    *,
    mode: str,
    seed: int,
    count: int | None = None,
    per_class: int | None = None,
) -> tuple[TreeRecord, ...]:
    """Select trees using explicit all, random-total, or capped-class semantics."""

    ordered = _canonical_records(records)
    if not ordered:
        raise ValueError("No trees remain available for selection.")
    if mode == "all":
        if count is not None or per_class is not None:
            raise ValueError("'all' selection does not accept count or per_class.")
        return ordered
    if mode == "random":
        if per_class is not None:
            raise ValueError("'random' selection does not accept per_class.")
        if count is None or count <= 0:
            raise ValueError("'random' selection requires a positive count.")
        if count > len(ordered):
            raise ValueError(
                f"Requested {count} random trees, but only {len(ordered)} are available."
            )
        indices = np.random.default_rng(seed).choice(
            len(ordered),
            size=count,
            replace=False,
        )
        return _canonical_records([ordered[int(index)] for index in indices])
    if mode == "balanced":
        if count is not None:
            raise ValueError("'balanced' selection does not accept count.")
        if per_class is None or per_class <= 0:
            raise ValueError(
                "'balanced' selection requires a positive per_class value."
            )
        by_class: dict[int, list[TreeRecord]] = {}
        for record in ordered:
            by_class.setdefault(record.cell_class, []).append(record)
        selected: list[TreeRecord] = []
        for cell_class in sorted(by_class):
            class_records = by_class[cell_class]
            selected.extend(
                select_balanced_sample(
                    class_records,
                    min(per_class, len(class_records)),
                    seed=seed,
                )
            )
        return _canonical_records(selected)
    raise ValueError(
        f"Unknown selection mode {mode!r}; use 'balanced', 'random', or 'all'."
    )


def filter_records_by_classes(
    records: Sequence[TreeRecord],
    selectors: Sequence[str] | None,
) -> tuple[TreeRecord, ...]:
    """Filter by integer class IDs or exact cell-type labels."""

    if not selectors:
        return tuple(records)
    known_ids = {record.cell_class for record in records}
    known_types = {record.cell_type for record in records}
    selected_ids: set[int] = set()
    selected_types: set[str] = set()
    unknown: list[str] = []
    for selector in selectors:
        try:
            class_id = int(selector)
        except ValueError:
            if selector in known_types:
                selected_types.add(selector)
            else:
                unknown.append(selector)
        else:
            if class_id in known_ids:
                selected_ids.add(class_id)
            else:
                unknown.append(selector)
    if unknown:
        raise ValueError(
            f"Unknown class selectors {unknown!r}. Available IDs: "
            f"{sorted(known_ids)!r}; labels: {sorted(known_types)!r}."
        )
    filtered = tuple(
        record
        for record in records
        if record.cell_class in selected_ids or record.cell_type in selected_types
    )
    if not filtered:
        raise ValueError("The class filter selected no trees.")
    return filtered


def _select_from_manifest(
    records: Sequence[TreeRecord],
    manifest_path: Path,
) -> tuple[TreeRecord, ...]:
    by_id = {record.tree_id: record for record in records}
    selected: list[TreeRecord] = []
    seen: set[str] = set()
    with manifest_path.expanduser().open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "tree_id" not in reader.fieldnames:
            raise ValueError("Selection manifest must contain a 'tree_id' column.")
        for row_number, row in enumerate(reader, start=2):
            tree_id = (row.get("tree_id") or "").strip()
            if not tree_id:
                raise ValueError(
                    f"Empty tree_id in selection manifest at row {row_number}."
                )
            if tree_id in seen:
                raise ValueError(f"Duplicate tree_id in manifest: {tree_id!r}.")
            seen.add(tree_id)
            try:
                selected.append(by_id[tree_id])
            except KeyError as exc:
                raise ValueError(
                    f"Manifest tree_id {tree_id!r} was not found in the "
                    "configured dataset splits."
                ) from exc
    if not selected:
        raise ValueError("Selection manifest contains no trees.")
    return tuple(selected)


def _write_manifest(path: Path, records: Sequence[TreeRecord]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
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
        for index, record in enumerate(records):
            writer.writerow(
                {
                    "matrix_index": index,
                    "tree_id": record.tree_id,
                    "swc_path": record.swc_path,
                    "split": record.split,
                    "cell_class": record.cell_class,
                    "cell_type": record.cell_type,
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_manifest(path: Path) -> tuple[TreeRecord, ...]:
    records: list[TreeRecord] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "matrix_index",
            "tree_id",
            "swc_path",
            "split",
            "cell_class",
            "cell_type",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Saved tree manifest is missing required columns: {path}")
        for expected_index, row in enumerate(reader):
            if int(row["matrix_index"]) != expected_index:
                raise ValueError("Saved tree manifest indices are not contiguous.")
            records.append(
                TreeRecord(
                    tree_id=row["tree_id"],
                    swc_path=Path(row["swc_path"]).expanduser().resolve(),
                    split=row["split"],
                    cell_class=int(row["cell_class"]),
                    cell_type=row["cell_type"],
                )
            )
    validate_tree_records(records)
    if not records:
        raise ValueError("Saved tree manifest is empty.")
    return tuple(records)


def _manifest_fingerprint(records: Sequence[TreeRecord]) -> str:
    payload = [
        {
            "tree_id": record.tree_id,
            "swc_path": str(record.swc_path),
            "split": record.split,
            "cell_class": record.cell_class,
            "cell_type": record.cell_type,
            "swc_sha256": _sha256_file(record.swc_path),
        }
        for record in records
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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
    serialized = (
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _status_counts(status: np.ndarray) -> dict[str, int]:
    counts = np.zeros(max(STATUS_LABELS) + 1, dtype=np.int64)
    for row_index in range(status.shape[0] - 1):
        upper_row = np.asarray(status[row_index, row_index + 1 :])
        counts += np.bincount(
            upper_row,
            minlength=len(counts),
        )[: len(counts)]
    return {
        label: int(counts[code])
        for code, label in STATUS_LABELS.items()
    }


def _flush_results(
    distances: np.memmap,
    status: np.memmap,
    buffered: list[tuple[int, int, float, np.uint8]],
) -> None:
    if not buffered:
        return
    for index_a, index_b, value, _state in buffered:
        distances[index_a, index_b] = value
        distances[index_b, index_a] = value
    distances.flush()
    for index_a, index_b, _value, state in buffered:
        # The upper-triangle cell is the commit marker used by traversal on
        # resume, so write its mirror first and commit it last.
        status[index_b, index_a] = state
        status[index_a, index_b] = state
    status.flush()
    buffered.clear()


def _append_issue(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(_json_safe(payload), sort_keys=True, allow_nan=False) + "\n"
        )


def _open_metric_arrays(
    metric_dir: Path,
    *,
    tree_count: int,
    resume: bool,
) -> tuple[np.memmap, np.memmap]:
    distance_path = metric_dir / "distances.npy"
    status_path = metric_dir / "status.npy"
    distance_temporary = metric_dir / ".distances.npy.initializing"
    status_temporary = metric_dir / ".status.npy.initializing"
    shape = (tree_count, tree_count)
    if resume and distance_path.is_file() and status_path.is_file():
        distances = np.load(distance_path, mmap_mode="r+")
        status = np.load(status_path, mmap_mode="r+")
        if distances.shape != shape or status.shape != shape:
            raise ValueError(
                f"Saved matrix shape for {metric_dir.name!r} does not match "
                f"the {tree_count}-tree manifest."
            )
        if distances.dtype != np.float64 or status.dtype != np.uint8:
            raise ValueError(f"Saved matrix dtypes are invalid in {metric_dir}.")
        if int(np.max(status, initial=0)) > max(STATUS_LABELS):
            raise ValueError(f"Saved matrix has invalid status codes in {metric_dir}.")

        # The strict upper triangle is authoritative. Rebuild its mirrors and
        # discard any stale value whose completion state was never committed.
        diagonal = np.arange(tree_count)
        distances[diagonal, diagonal] = 0.0
        status[diagonal, diagonal] = STATUS_OK
        for index_a in range(tree_count - 1):
            upper_status = np.asarray(status[index_a, index_a + 1 :])
            upper_distances = np.asarray(distances[index_a, index_a + 1 :])
            pending = upper_status == STATUS_PENDING
            ok = upper_status == STATUS_OK
            non_success = np.logical_and(~pending, ~ok)
            if np.any(
                np.logical_or(
                    ~np.isfinite(upper_distances[ok]),
                    upper_distances[ok] < -1e-10,
                )
            ):
                raise ValueError(
                    f"Saved completed distances in row {index_a} are invalid "
                    f"in {metric_dir}."
                )
            if np.any(~np.isnan(upper_distances[non_success])):
                raise ValueError(
                    f"Saved non-success distances in row {index_a} must be "
                    f"NaN in {metric_dir}."
                )
            upper_distances[pending] = np.nan
            distances[index_a + 1 :, index_a] = upper_distances
            status[index_a + 1 :, index_a] = upper_status
        distances.flush()
        status.flush()
        return distances, status

    if resume and distance_path.exists() != status_path.exists():
        # Metric metadata is written only after both initialized arrays exist.
        # With no metadata, one final file is a recoverable bootstrap crash.
        if (metric_dir / "metric.json").exists():
            raise ValueError(
                f"One matrix artifact is missing from an established metric "
                f"directory: {metric_dir}."
            )
        distance_path.unlink(missing_ok=True)
        status_path.unlink(missing_ok=True)
    elif distance_path.exists() or status_path.exists():
        raise FileExistsError(
            f"Incomplete metric artifacts already exist in {metric_dir}; use --resume."
        )

    distance_temporary.unlink(missing_ok=True)
    status_temporary.unlink(missing_ok=True)
    distances_initializing = np.lib.format.open_memmap(
        distance_temporary,
        mode="w+",
        dtype=np.float64,
        shape=shape,
    )
    status_initializing = np.lib.format.open_memmap(
        status_temporary,
        mode="w+",
        dtype=np.uint8,
        shape=shape,
    )
    distances_initializing[:] = np.nan
    status_initializing[:] = STATUS_PENDING
    diagonal = np.arange(tree_count)
    distances_initializing[diagonal, diagonal] = 0.0
    distances_initializing.flush()
    status_initializing[diagonal, diagonal] = STATUS_OK
    status_initializing.flush()
    del distances_initializing
    del status_initializing
    distance_temporary.replace(distance_path)
    status_temporary.replace(status_path)
    distances = np.load(distance_path, mmap_mode="r+")
    status = np.load(status_path, mmap_mode="r+")
    return distances, status


def compute_one_metric(
    metric: PreparedMatrixMetric,
    graphs: Sequence[nx.Graph],
    records: Sequence[TreeRecord],
    *,
    metric_dir: Path,
    resume: bool,
    retry_errors: bool,
    checkpoint_every: int,
    fail_fast: bool,
    logger: _RunLogger,
    max_new_pairs: int | None = None,
    workers: int = 1,
) -> dict[str, object]:
    """Prepare every tree once and fill or resume one symmetric matrix."""

    if workers <= 0:
        raise ValueError("workers must be positive.")
    metric_dir.mkdir(parents=True, exist_ok=True)
    distances, status = _open_metric_arrays(
        metric_dir,
        tree_count=len(records),
        resume=resume,
    )
    issues_path = metric_dir / "issues.jsonl"
    metric_metadata_path = metric_dir / "metric.json"
    configuration = dict(metric.configuration)
    if metric_metadata_path.is_file():
        previous = _load_json(metric_metadata_path)
        if previous.get("configuration") != _json_safe(configuration):
            raise ValueError(
                f"Metric configuration changed for {metric.name!r}; use a new "
                "output directory."
            )
    else:
        _write_json_atomic(
            metric_metadata_path,
            {
                "name": metric.name,
                "display_name": metric.display_name,
                "family": metric.family,
                "configuration": configuration,
            },
        )

    if retry_errors:
        for index_a in range(len(records) - 1):
            upper_status = np.asarray(status[index_a, index_a + 1 :])
            retry_mask = np.logical_or(
                upper_status == STATUS_ERROR,
                upper_status == STATUS_PREPARATION_ERROR,
            )
            if not np.any(retry_mask):
                continue
            indices_b = np.flatnonzero(retry_mask) + index_a + 1
            distances[index_a, indices_b] = np.nan
            distances[indices_b, index_a] = np.nan
            status[indices_b, index_a] = STATUS_PENDING
            status[index_a, indices_b] = STATUS_PENDING
        distances.flush()
        status.flush()

    initial_counts = _status_counts(status)
    live_counts = dict(initial_counts)
    if initial_counts["pending"] == 0:
        logger.write(f"{metric.name}: already complete; nothing to resume.")
        return {
            "metric": metric.name,
            "configuration": configuration,
            "status_counts": initial_counts,
            "new_pairs": 0,
            "elapsed_seconds": 0.0,
            "workers_requested": workers,
            "workers_used": 0,
            "worker_start_method": None,
        }

    logger.write(
        f"{metric.name}: preparing {len(graphs)} trees; "
        f"{initial_counts['pending']} pairs pending."
    )
    started = perf_counter()
    prepared: list[object | None] = [None] * len(graphs)
    preparation_errors: set[int] = set()
    for index, graph in enumerate(graphs):
        try:
            prepared[index] = metric.prepare(graph)
        except Exception as exc:
            preparation_errors.add(index)
            _append_issue(
                issues_path,
                {
                    "created_at_utc": _now(),
                    "kind": "preparation_error",
                    "metric": metric.name,
                    "matrix_index": index,
                    "tree_id": records[index].tree_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            if fail_fast:
                raise

    preparation_results: list[tuple[int, int, float, np.uint8]] = []
    for index_a in range(len(records)):
        for index_b in range(index_a + 1, len(records)):
            if status[index_a, index_b] != STATUS_PENDING:
                continue
            if index_a in preparation_errors or index_b in preparation_errors:
                preparation_results.append(
                    (
                        index_a,
                        index_b,
                        float("nan"),
                        STATUS_PREPARATION_ERROR,
                    )
                )
                live_counts["pending"] -= 1
                live_counts["preparation_error"] += 1
                if len(preparation_results) >= checkpoint_every:
                    _flush_results(distances, status, preparation_results)
    _flush_results(distances, status, preparation_results)

    pair_limit = max_new_pairs
    pending_comparisons = live_counts["pending"]
    if pair_limit is not None:
        pending_comparisons = min(pending_comparisons, pair_limit)
    workers_used = min(workers, pending_comparisons)
    worker_start_method: str | None = None
    executor: ProcessPoolExecutor | None = None

    buffered: list[tuple[int, int, float, np.uint8]] = []
    new_pairs = 0
    last_reported = 0
    batch_size = max(1, workers_used * 4)
    pending_pairs = _iter_pending_pairs(status, limit=pair_limit)
    try:
        if workers_used > 1:
            process_context, worker_start_method = _preferred_process_context()
            executor = ProcessPoolExecutor(
                max_workers=workers_used,
                mp_context=process_context,
                initializer=_initialize_pair_worker,
                initargs=(metric, prepared),
            )
            logger.write(
                f"{metric.name}: evaluating pairs with {workers_used} process "
                f"workers ({worker_start_method})."
            )

        for batch in _batched(pending_pairs, batch_size):
            if executor is None:
                evaluations: Iterable[_PairEvaluation] = (
                    _evaluate_metric_pair(metric, prepared, pair)
                    for pair in batch
                )
            else:
                futures = [
                    executor.submit(_evaluate_pair_in_worker, pair)
                    for pair in batch
                ]
                evaluations = (
                    future.result() for future in as_completed(futures)
                )

            for evaluation in evaluations:
                index_a = evaluation.index_a
                index_b = evaluation.index_b
                error_type = evaluation.error_type
                error_message = evaluation.error_message
                value = evaluation.value
                state = STATUS_OK

                if error_type is None:
                    try:
                        if value is None:
                            raise RuntimeError(
                                "Metric comparison returned no value."
                            )
                        if math.isnan(value):
                            if getattr(metric, "allows_undefined", False):
                                state = STATUS_UNDEFINED
                                _append_issue(
                                    issues_path,
                                    {
                                        "created_at_utc": _now(),
                                        "kind": "undefined",
                                        "metric": metric.name,
                                        "index_a": index_a,
                                        "index_b": index_b,
                                        "tree_a_id": records[index_a].tree_id,
                                        "tree_b_id": records[index_b].tree_id,
                                    },
                                )
                            else:
                                raise ValueError(
                                    "Metric returned NaN without declaring "
                                    "that undefined pair values are meaningful."
                                )
                        elif not math.isfinite(value):
                            raise ValueError(
                                "Metric returned an infinite value."
                            )
                        elif value < -1e-10:
                            raise ValueError(
                                f"Metric returned a negative value: {value}."
                            )
                        else:
                            value = max(value, 0.0)
                    except Exception as exc:
                        error_type = type(exc).__name__
                        error_message = str(exc)

                if error_type is not None:
                    value = float("nan")
                    state = STATUS_ERROR
                    _append_issue(
                        issues_path,
                        {
                            "created_at_utc": _now(),
                            "kind": "comparison_error",
                            "metric": metric.name,
                            "index_a": index_a,
                            "index_b": index_b,
                            "tree_a_id": records[index_a].tree_id,
                            "tree_b_id": records[index_b].tree_id,
                            "error_type": error_type,
                            "error_message": error_message,
                        },
                    )

                assert value is not None
                buffered.append((index_a, index_b, value, state))
                new_pairs += 1
                live_counts["pending"] -= 1
                live_counts[STATUS_LABELS[int(state)]] += 1
                if len(buffered) >= checkpoint_every:
                    _flush_results(distances, status, buffered)
                if new_pairs - last_reported >= checkpoint_every:
                    last_reported = new_pairs
                    elapsed = perf_counter() - started
                    rate = new_pairs / elapsed if elapsed > 0.0 else 0.0
                    eta = (
                        live_counts["pending"] / rate
                        if rate > 0.0
                        else math.inf
                    )
                    logger.write(
                        f"{metric.name}: {new_pairs} new pairs; "
                        f"{live_counts['pending']} pending; "
                        f"{rate:.2f} pairs/s; ETA "
                        f"{eta / 60.0:.1f} min."
                    )
                if fail_fast and state == STATUS_ERROR:
                    _flush_results(distances, status, buffered)
                    raise RuntimeError(
                        f"{metric.name} failed for pair ({index_a}, "
                        f"{index_b}): {error_type}: {error_message}"
                    )
    except KeyboardInterrupt:
        _flush_results(distances, status, buffered)
        logger.write(
            f"{metric.name}: interrupted after {new_pairs} new pairs."
        )
        raise
    except Exception:
        _flush_results(distances, status, buffered)
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        _clear_pair_worker()

    _flush_results(distances, status, buffered)
    if live_counts["pending"]:
        logger.write(
            f"{metric.name}: paused with {live_counts['pending']} pairs pending."
        )
    else:
        logger.write(
            f"{metric.name}: finished with {live_counts['ok']} ok, "
            f"{live_counts['undefined']} undefined, "
            f"{live_counts['error']} errors, and "
            f"{live_counts['preparation_error']} preparation errors."
        )
    return {
        "metric": metric.name,
        "configuration": configuration,
        "status_counts": dict(live_counts),
        "new_pairs": new_pairs,
        "elapsed_seconds": perf_counter() - started,
        "workers_requested": workers,
        "workers_used": workers_used,
        "worker_start_method": worker_start_method,
    }


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _grid_size(text: str) -> int:
    value = int(text)
    if value < 3:
        raise argparse.ArgumentTypeError("must be at least 3")
    return value


def _default_dataset_root() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "neurons_conditional"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute resumable symmetric distance matrices on selected "
            "ground-truth neuron trees. Elastic SRVFT is deliberately excluded."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_default_dataset_root(),
    )
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument(
        "--classes",
        nargs="+",
        help="Optional integer class IDs or exact labels such as 23P and 5P-IT.",
    )
    parser.add_argument(
        "--selection",
        choices=("balanced", "random", "all", "manifest"),
        default="balanced",
    )
    parser.add_argument(
        "--per-class",
        type=_positive_int,
        help=(
            "Maximum trees per class for balanced selection; smaller classes "
            "contribute all available trees (default: 10)."
        ),
    )
    parser.add_argument("--count", type=_positive_int)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="CSV with a tree_id column, used with --selection manifest.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRIC_SELECTORS,
        required=True,
        help=(
            "Family aliases or scalar variants. 'all' means all 12 current "
            "non-Elastic outputs."
        ),
    )
    parser.add_argument("--so2-grid-size", type=_grid_size, default=72)
    parser.add_argument(
        "--no-so2-refine",
        dest="so2_refine",
        action="store_false",
        help="Use only the angular grid for Chamfer and xyz-feature FGW.",
    )
    parser.add_argument(
        "--so2-refinement-tolerance",
        type=_positive_float,
        default=1e-8,
    )
    parser.add_argument("--fgw-max-nodes", type=_nonnegative_int, default=1_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-every",
        type=_positive_int,
        default=25,
        help="Flush matrix values and states after this many new pairs.",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        help=(
            "Pair-comparison worker processes. Defaults to "
            "SLURM_CPUS_PER_TASK under Slurm, otherwise 1."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the exact saved manifest and metric configuration.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry comparison and preparation errors; undefined values stay terminal.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first metric preparation or comparison error.",
    )
    parser.add_argument(
        "--max-new-pairs",
        type=_positive_int,
        help="Optional debug/chunk limit across this invocation.",
    )
    return parser


def _new_selection(args: argparse.Namespace) -> tuple[TreeRecord, ...]:
    discovered = discover_tree_records(
        args.dataset_root.expanduser().resolve(),
        split_dirs=args.splits,
    )
    filtered = filter_records_by_classes(discovered, args.classes)
    if args.selection == "manifest":
        if args.selection_manifest is None:
            raise ValueError(
                "--selection manifest requires --selection-manifest."
            )
        if args.count is not None or args.per_class is not None:
            raise ValueError(
                "Manifest selection does not accept --count or --per-class."
            )
        return _select_from_manifest(filtered, args.selection_manifest)
    if args.selection_manifest is not None:
        raise ValueError(
            "--selection-manifest is only valid with --selection manifest."
        )
    return select_records(
        filtered,
        mode=args.selection,
        seed=args.seed,
        count=args.count,
        per_class=(
            10
            if args.selection == "balanced" and args.per_class is None
            else args.per_class
        ),
    )


def _build_metrics(
    names: Sequence[str],
    args: argparse.Namespace,
) -> tuple[PreparedMatrixMetric, ...]:
    return tuple(
        build_matrix_metric(
            name,
            so2_grid_size=args.so2_grid_size,
            so2_refine=args.so2_refine,
            so2_refinement_tolerance=args.so2_refinement_tolerance,
            fgw_max_nodes=args.fgw_max_nodes,
        )
        for name in names
    )


def _static_run_payload(
    args: argparse.Namespace,
    records: Sequence[TreeRecord],
    metrics: Sequence[PreparedMatrixMetric],
) -> dict[str, object]:
    counts = Counter(record.cell_type for record in records)
    return {
        "schema_version": 1,
        "created_at_utc": _now(),
        "dataset": {
            "root": args.dataset_root.expanduser().resolve(),
            "splits": list(args.splits),
        },
        "selection": {
            "mode": args.selection,
            "seed": args.seed,
            "count": args.count,
            "per_class": (
                10
                if args.selection == "balanced" and args.per_class is None
                else args.per_class
            ),
            "per_class_is_cap": args.selection == "balanced",
            "classes": args.classes,
            "selection_manifest": args.selection_manifest,
            "selected_trees": len(records),
            "strict_upper_triangle_pairs": len(records) * (len(records) - 1) // 2,
            "class_counts": dict(sorted(counts.items())),
            "manifest_fingerprint": _manifest_fingerprint(records),
        },
        "frame_contract": _frame_contract(),
        "implementation_contract": _implementation_contract(),
        "metric_order": [metric.name for metric in metrics],
        "metrics": {
            metric.name: {
                "display_name": metric.display_name,
                "family": metric.family,
                "configuration": dict(metric.configuration),
            }
            for metric in metrics
        },
        "status_codes": STATUS_LABELS,
        "operational_note": (
            "Operational flags such as checkpoint cadence are intentionally "
            "excluded from scientific configuration compatibility."
        ),
    }


def _validate_resume(
    saved: Mapping[str, object],
    records: Sequence[TreeRecord],
    metrics: Sequence[PreparedMatrixMetric],
) -> None:
    selection = saved.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Saved run.json has no valid selection block.")
    if selection.get("manifest_fingerprint") != _manifest_fingerprint(records):
        raise ValueError(
            "Saved tree manifest or current SWC contents do not match run.json."
        )
    if saved.get("frame_contract") != _json_safe(_frame_contract()):
        raise ValueError("Saved coordinate-frame contract has changed.")
    if saved.get("implementation_contract") != _json_safe(
        _implementation_contract()
    ):
        raise ValueError(
            "Metric implementation or recorded runtime versions changed; use "
            "a new output directory rather than mixing implementations."
        )
    saved_metrics = saved.get("metrics")
    if not isinstance(saved_metrics, Mapping):
        raise ValueError("Saved run.json has no valid metrics block.")
    current_names = [metric.name for metric in metrics]
    saved_order = saved.get("metric_order")
    if not isinstance(saved_order, list) or not all(
        isinstance(name, str) for name in saved_order
    ):
        raise ValueError("Saved run.json has no valid metric_order list.")
    if saved_order != current_names:
        raise ValueError(
            "Resume requires the same expanded metric list and order. "
            f"Saved: {saved_order!r}; requested: {current_names!r}."
        )
    if set(saved_metrics) != set(saved_order):
        raise ValueError("Saved run.json metrics and metric_order disagree.")
    for metric in metrics:
        entry = saved_metrics[metric.name]
        if not isinstance(entry, Mapping) or entry.get("configuration") != _json_safe(
            dict(metric.configuration)
        ):
            raise ValueError(
                f"Resume configuration changed for metric {metric.name!r}."
            )


def _prepare_new_output_directory(output_dir: Path) -> None:
    entries = {
        path.name: path
        for path in output_dir.iterdir()
        if path.name != _LOCK_FILENAME
    }
    if not entries:
        return

    # These are the only files that can be left before run.json commits a new
    # run. They contain no computed distances and are safe to reconstruct.
    recoverable_bootstrap = {
        "selected_trees.csv",
        "selected_trees.csv.tmp",
        "run.json.tmp",
    }
    if "run.json" not in entries and set(entries).issubset(recoverable_bootstrap):
        for path in entries.values():
            if path.is_file():
                path.unlink()
        return
    raise FileExistsError(
        f"Output directory is not empty: {output_dir}. Use --resume "
        "for an existing run or choose a new directory."
    )


def _run_locked(
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, object]:
    run_path = output_dir / "run.json"
    manifest_path = output_dir / "selected_trees.csv"
    progress_path = output_dir / "progress.json"
    log_path = output_dir / "run.log"
    metric_names = expand_metric_selection(args.metrics)
    metrics = _build_metrics(metric_names, args)
    workers, worker_source = _resolve_worker_count(args.workers)

    if args.resume:
        if not run_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume: {output_dir} has no run.json and selected_trees.csv."
            )
        records = _read_manifest(manifest_path)
        saved = _load_json(run_path)
        _validate_resume(saved, records, metrics)
    else:
        _prepare_new_output_directory(output_dir)
        records = _new_selection(args)
        _write_manifest(manifest_path, records)
        _write_json_atomic(
            run_path,
            _static_run_payload(args, records, metrics),
        )

    logger = _RunLogger(log_path)
    pair_count = len(records) * (len(records) - 1) // 2
    logger.write(
        f"Selected {len(records)} trees ({pair_count} strict upper-triangle "
        f"pairs) for {len(metrics)} scalar metrics."
    )
    logger.write(
        f"Pair evaluation requested {workers} worker(s) from {worker_source}."
    )
    execution_info = {
        "workers_requested": workers,
        "worker_source": worker_source,
    }
    summaries: dict[str, object] = {}
    if args.resume and progress_path.is_file():
        previous_progress = _load_json(progress_path)
        previous_summaries = previous_progress.get("metrics")
        if isinstance(previous_summaries, Mapping):
            summaries.update(previous_summaries)
    remaining_budget = args.max_new_pairs
    overall_status = "running"
    current_metric: PreparedMatrixMetric | None = None
    _write_json_atomic(
        progress_path,
        {
            "status": overall_status,
            "updated_at_utc": _now(),
            "execution": execution_info,
            "metrics": summaries,
        },
    )

    def record_current_checkpoint(*, interrupted: bool) -> None:
        if current_metric is None:
            return
        status_path = (
            output_dir / "metrics" / current_metric.name / "status.npy"
        )
        if not status_path.is_file():
            return
        try:
            status = np.load(status_path, mmap_mode="r")
            counts = _status_counts(status)
        except Exception:
            return
        previous = summaries.get(current_metric.name)
        checkpoint = dict(previous) if isinstance(previous, Mapping) else {}
        checkpoint.update(
            {
                "metric": current_metric.name,
                "configuration": dict(current_metric.configuration),
                "status_counts": counts,
                "checkpointed_at_utc": _now(),
                "interrupted": interrupted,
            }
        )
        summaries[current_metric.name] = checkpoint

    try:
        logger.write("Loading and transforming SWCs before metric preparation.")
        graphs = [
            transform_scientific_y_to_internal_z(load_swc_graph(record.swc_path))
            for record in records
        ]
        for metric in metrics:
            current_metric = metric
            if remaining_budget is not None and remaining_budget <= 0:
                overall_status = "paused"
                break
            summary = compute_one_metric(
                metric,
                graphs,
                records,
                metric_dir=output_dir / "metrics" / metric.name,
                resume=args.resume,
                retry_errors=args.retry_errors,
                checkpoint_every=args.checkpoint_every,
                fail_fast=args.fail_fast,
                logger=logger,
                max_new_pairs=remaining_budget,
                workers=workers,
            )
            invocation_new_pairs = int(summary["new_pairs"])
            invocation_elapsed = float(summary["elapsed_seconds"])
            previous = summaries.get(metric.name)
            previous_total_pairs = 0
            previous_total_elapsed = 0.0
            if isinstance(previous, Mapping):
                previous_total_pairs = int(
                    previous.get(
                        "new_pairs_total",
                        previous.get("new_pairs", 0),
                    )
                )
                previous_total_elapsed = float(
                    previous.get(
                        "elapsed_seconds_total",
                        previous.get("elapsed_seconds", 0.0),
                    )
                )
            summary["new_pairs_this_invocation"] = invocation_new_pairs
            summary["elapsed_seconds_this_invocation"] = invocation_elapsed
            summary["new_pairs_total"] = (
                previous_total_pairs + invocation_new_pairs
            )
            summary["elapsed_seconds_total"] = (
                previous_total_elapsed + invocation_elapsed
            )
            summaries[metric.name] = summary
            if remaining_budget is not None:
                remaining_budget -= invocation_new_pairs
            _write_json_atomic(
                progress_path,
                {
                    "status": overall_status,
                    "updated_at_utc": _now(),
                    "execution": execution_info,
                    "metrics": summaries,
                },
            )
            if summary["status_counts"]["pending"] > 0:  # type: ignore[index]
                overall_status = "paused"
                break
    except KeyboardInterrupt:
        overall_status = "interrupted"
        record_current_checkpoint(interrupted=True)
        _write_json_atomic(
            progress_path,
            {
                "status": overall_status,
                "updated_at_utc": _now(),
                "execution": execution_info,
                "metrics": summaries,
            },
        )
        logger.write("Run interrupted; flushed checkpoints can be resumed.")
        raise
    except Exception as exc:
        overall_status = "failed"
        record_current_checkpoint(interrupted=False)
        _write_json_atomic(
            progress_path,
            {
                "status": overall_status,
                "updated_at_utc": _now(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "execution": execution_info,
                "metrics": summaries,
            },
        )
        logger.write(f"Run failed with {type(exc).__name__}: {exc}")
        raise

    if overall_status == "running":
        has_failures = any(
            summary["status_counts"]["error"]  # type: ignore[index]
            or summary["status_counts"]["preparation_error"]  # type: ignore[index]
            for summary in summaries.values()  # type: ignore[union-attr]
        )
        overall_status = "complete_with_errors" if has_failures else "complete"
    payload = {
        "status": overall_status,
        "updated_at_utc": _now(),
        "selected_trees": len(records),
        "strict_upper_triangle_pairs": pair_count,
        "execution": execution_info,
        "metrics": summaries,
        "artifacts": {
            "configuration": run_path,
            "selected_trees": manifest_path,
            "progress": progress_path,
            "log": log_path,
            "metric_root": output_dir / "metrics",
        },
    }
    _write_json_atomic(progress_path, payload)
    logger.write(f"Run status: {overall_status}.")
    return payload


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir.expanduser().resolve()
    with _exclusive_run_lock(output_dir):
        return _run_locked(args, output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def interrupt_on_sigterm(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt_on_sigterm)
    try:
        try:
            payload = run(args)
        except KeyboardInterrupt:
            return 130
        except (
            FileExistsError,
            FileNotFoundError,
            NotADirectoryError,
            ValueError,
            KeyError,
        ) as exc:
            parser.error(str(exc))
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False))
    return 2 if payload.get("status") == "complete_with_errors" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
