"""Focused tests for the resumable ground-truth distance-matrix runner."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from metrics.chamfer import tree_chamfer_distance
from metrics.so2 import rotate_points_about_axis
from visualization.metric_study.dataset import TreeRecord
from visualization.metric_study.matrix_metrics import (
    ALL_MATRIX_METRICS,
    CHAMFER,
    ChamferMatrixMetric,
    METRIC_SELECTORS,
    PERSISTENCE_VARIANTS,
    TMD_HEIGHT_WASSERSTEIN,
    TMD_PATH_WASSERSTEIN,
    build_matrix_metric,
    expand_metric_selection,
)
from visualization.metric_study.run_distance_matrices import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_PENDING,
    STATUS_UNDEFINED,
    _RunLogger,
    _flush_results,
    _load_json,
    _manifest_fingerprint,
    _open_metric_arrays,
    _resolve_worker_count,
    _static_run_payload,
    _validate_resume,
    _write_json_atomic,
    build_parser,
    compute_one_metric,
    select_records,
)


def _record(
    index: int,
    cell_class: int,
    *,
    fixture_root: Path | None = None,
) -> TreeRecord:
    if fixture_root is None:
        swc_path = Path(f"/fixture/class-{cell_class}-tree-{index}.swc")
    else:
        fixture_root.mkdir(parents=True, exist_ok=True)
        swc_path = fixture_root / f"class-{cell_class}-tree-{index}.swc"
        swc_path.write_text(
            f"# cell_class {cell_class}\n"
            f"# cell_type type-{cell_class}\n"
            "1 1 0 0 0 1 -1\n"
            f"2 3 {index + 1} {cell_class + 1} 0 1 1\n",
            encoding="utf-8",
        )
    return TreeRecord(
        tree_id=f"class-{cell_class}-tree-{index}",
        swc_path=swc_path,
        split="test",
        cell_class=cell_class,
        cell_type=f"type-{cell_class}",
    )


def _records(
    *,
    classes: int = 3,
    per_class: int = 4,
    fixture_root: Path | None = None,
) -> tuple[TreeRecord, ...]:
    return tuple(
        _record(index, cell_class, fixture_root=fixture_root)
        for cell_class in range(classes)
        for index in range(per_class)
    )


def _value_graph(value: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(0)
    graph.graph["value"] = value
    return graph


def _geometric_tree(*, angle: float = 0.0) -> nx.Graph:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.2, 0.7],
            [-0.3, 1.2, 1.1],
            [2.1, 0.9, 1.8],
        ]
    )
    points = rotate_points_about_axis(points, angle)
    graph = nx.Graph()
    for node, point in enumerate(points):
        graph.add_node(node, pos=point)
    graph.add_edges_from(((0, 1), (0, 2), (1, 3)))
    graph.graph["root"] = 0
    return graph


class _CountingMetric:
    name = "counting_metric"
    display_name = "Counting metric"
    family = "fixture"
    symmetric = True

    def __init__(self, *, interrupt_once: bool = False) -> None:
        self.interrupt_once = interrupt_once
        self.did_interrupt = False
        self.prepare_calls: list[int] = []
        self.compare_calls: list[tuple[int, int]] = []

    @property
    def configuration(self) -> dict[str, object]:
        return {"fixture": "absolute_difference"}

    def prepare(self, graph: nx.Graph) -> int:
        value = int(graph.graph["value"])
        self.prepare_calls.append(value)
        return value

    def compare(self, prepared_a: object, prepared_b: object) -> float:
        pair = (int(prepared_a), int(prepared_b))
        self.compare_calls.append(pair)
        if self.interrupt_once and not self.did_interrupt and pair == (0, 2):
            self.did_interrupt = True
            raise KeyboardInterrupt
        return float(abs(pair[0] - pair[1]))


class _StatusMetric(_CountingMetric):
    name = "status_metric"
    allows_undefined = True

    def compare(self, prepared_a: object, prepared_b: object) -> float:
        pair = (int(prepared_a), int(prepared_b))
        self.compare_calls.append(pair)
        if pair == (0, 1):
            return float("nan")
        if pair == (0, 2):
            raise RuntimeError("fixture comparison failure")
        return float(abs(pair[0] - pair[1]))


class _UnexpectedNaNMetric(_CountingMetric):
    name = "unexpected_nan_metric"

    def compare(self, prepared_a: object, prepared_b: object) -> float:
        pair = (int(prepared_a), int(prepared_b))
        self.compare_calls.append(pair)
        return float("nan")


class _ExplicitUndefinedMetric(_UnexpectedNaNMetric):
    name = "explicit_undefined_metric"
    allows_undefined = True


class _WorkerErrorMetric(_CountingMetric):
    name = "worker_error_metric"

    def compare(self, prepared_a: object, prepared_b: object) -> float:
        pair = (int(prepared_a), int(prepared_b))
        if pair == (0, 2):
            raise RuntimeError("fixture worker failure")
        return float(abs(pair[0] - pair[1]))


class _RecordingArray:
    def __init__(self) -> None:
        self.events: list[tuple[str, object, object | None]] = []

    def __setitem__(self, key: object, value: object) -> None:
        self.events.append(("set", key, value))

    def flush(self) -> None:
        self.events.append(("flush", None, None))


def test_selection_modes_are_seeded_and_input_order_independent() -> None:
    records = _records()

    assert select_records(records, mode="all", seed=0) == select_records(
        tuple(reversed(records)), mode="all", seed=99
    )

    random_first = select_records(records, mode="random", count=5, seed=17)
    random_repeated = select_records(
        tuple(reversed(records)), mode="random", count=5, seed=17
    )
    assert random_first == random_repeated
    assert len(random_first) == len({record.tree_id for record in random_first}) == 5

    balanced_first = select_records(
        records,
        mode="balanced",
        per_class=2,
        seed=17,
    )
    balanced_repeated = select_records(
        tuple(reversed(records)),
        mode="balanced",
        per_class=2,
        seed=17,
    )
    assert balanced_first == balanced_repeated
    assert Counter(record.cell_class for record in balanced_first) == {
        0: 2,
        1: 2,
        2: 2,
    }

    with pytest.raises(ValueError, match="does not accept count"):
        select_records(records, mode="all", count=2, seed=0)
    with pytest.raises(ValueError, match="only 12"):
        select_records(records, mode="random", count=13, seed=0)

    mixed_sizes = tuple(
        [*_records(classes=1, per_class=2)]
        + [_record(index, 1) for index in range(4)]
    )
    capped = select_records(
        mixed_sizes,
        mode="balanced",
        per_class=3,
        seed=19,
    )
    capped_repeated = select_records(
        tuple(reversed(mixed_sizes)),
        mode="balanced",
        per_class=3,
        seed=19,
    )
    assert capped == capped_repeated
    assert Counter(record.cell_class for record in capped) == {0: 2, 1: 3}


def test_metric_aliases_expand_canonically_and_exclude_elastic() -> None:
    assert expand_metric_selection(
        ("persistence", CHAMFER, TMD_PATH_WASSERSTEIN)
    ) == (CHAMFER, *PERSISTENCE_VARIANTS)
    assert expand_metric_selection(("all",)) == ALL_MATRIX_METRICS
    assert len(ALL_MATRIX_METRICS) == 12
    assert "elastic" not in METRIC_SELECTORS
    assert "elastic_srvft" not in METRIC_SELECTORS

    with pytest.raises(ValueError, match="intentionally excluded"):
        expand_metric_selection(("elastic_srvft",))
    with pytest.raises(ValueError, match="intentionally excluded"):
        expand_metric_selection(("elastic",))


def test_cached_chamfer_matches_pair_api() -> None:
    tree_a = _geometric_tree()
    tree_b = _geometric_tree(angle=0.713)
    metric = ChamferMatrixMetric(
        grid_size=8,
        refine=True,
        refinement_tolerance=1e-8,
        spacing=0.25,
    )

    cached = metric.compare(metric.prepare(tree_a), metric.prepare(tree_b))
    direct = tree_chamfer_distance(
        tree_a,
        tree_b,
        spacing=0.25,
        grid_size=8,
        refine=True,
        refinement_tolerance=1e-8,
    ).value

    assert cached == pytest.approx(direct, abs=1e-12)


def test_manifest_fingerprint_changes_when_swc_bytes_change(tmp_path: Path) -> None:
    record = _record(0, 0, fixture_root=tmp_path)
    original = _manifest_fingerprint((record,))

    record.swc_path.write_text(
        record.swc_path.read_text(encoding="utf-8").replace(
            "2 3 1 1 0 1 1",
            "2 3 9 1 0 1 1",
        ),
        encoding="utf-8",
    )

    assert _manifest_fingerprint((record,)) != original


def test_resume_repairs_mirrors_and_clears_pending_distance_values(
    tmp_path: Path,
) -> None:
    metric_dir = tmp_path / "metric"
    metric_dir.mkdir()
    distances, status = _open_metric_arrays(
        metric_dir,
        tree_count=3,
        resume=False,
    )

    distances[0, 1], status[0, 1] = 1.25, STATUS_OK
    distances[1, 0], status[1, 0] = 99.0, STATUS_ERROR
    distances[0, 2], status[0, 2] = 42.0, STATUS_PENDING
    distances[2, 0], status[2, 0] = 7.0, STATUS_OK
    distances[1, 2], status[1, 2] = np.nan, STATUS_UNDEFINED
    distances[2, 1], status[2, 1] = 8.0, STATUS_OK
    distances.flush()
    status.flush()
    del distances, status

    repaired_distances, repaired_status = _open_metric_arrays(
        metric_dir,
        tree_count=3,
        resume=True,
    )

    assert repaired_distances[0, 1] == repaired_distances[1, 0] == 1.25
    assert repaired_status[0, 1] == repaired_status[1, 0] == STATUS_OK
    assert np.isnan(repaired_distances[0, 2])
    assert np.isnan(repaired_distances[2, 0])
    assert repaired_status[0, 2] == repaired_status[2, 0] == STATUS_PENDING
    assert np.isnan(repaired_distances[1, 2])
    assert np.isnan(repaired_distances[2, 1])
    assert repaired_status[1, 2] == repaired_status[2, 1] == STATUS_UNDEFINED
    np.testing.assert_allclose(np.diag(repaired_distances), 0.0)
    np.testing.assert_array_equal(
        np.diag(repaired_status),
        np.full(3, STATUS_OK, dtype=np.uint8),
    )


def test_status_flush_commits_lower_mirror_before_authoritative_upper() -> None:
    distances = _RecordingArray()
    status = _RecordingArray()
    buffered = [(0, 1, 2.5, STATUS_OK)]

    _flush_results(distances, status, buffered)  # type: ignore[arg-type]

    assert buffered == []
    assert status.events == [
        ("set", (1, 0), STATUS_OK),
        ("set", (0, 1), STATUS_OK),
        ("flush", None, None),
    ]


def test_interruption_flushes_completed_pairs_and_resume_skips_them(
    tmp_path: Path,
) -> None:
    records = _records(classes=1, per_class=3)
    graphs = [_value_graph(index) for index in range(3)]
    metric = _CountingMetric(interrupt_once=True)
    metric_dir = tmp_path / "metric"
    logger = _RunLogger(tmp_path / "run.log")

    with pytest.raises(KeyboardInterrupt):
        compute_one_metric(
            metric,
            graphs,
            records,
            metric_dir=metric_dir,
            resume=False,
            retry_errors=False,
            checkpoint_every=20,
            fail_fast=False,
            logger=logger,
        )

    interrupted_status = np.load(metric_dir / "status.npy")
    interrupted_distances = np.load(metric_dir / "distances.npy")
    assert interrupted_status[0, 1] == STATUS_OK
    assert interrupted_status[0, 2] == STATUS_PENDING
    assert interrupted_status[1, 2] == STATUS_PENDING
    assert interrupted_distances[0, 1] == pytest.approx(1.0)

    summary = compute_one_metric(
        metric,
        graphs,
        records,
        metric_dir=metric_dir,
        resume=True,
        retry_errors=False,
        checkpoint_every=20,
        fail_fast=False,
        logger=logger,
    )

    assert summary["new_pairs"] == 2
    distances = np.load(metric_dir / "distances.npy")
    status = np.load(metric_dir / "status.npy")
    np.testing.assert_allclose(
        distances,
        np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]),
    )
    np.testing.assert_array_equal(status, np.full((3, 3), STATUS_OK))
    assert metric.compare_calls.count((0, 1)) == 1
    assert metric.compare_calls.count((0, 2)) == 2
    assert metric.compare_calls.count((1, 2)) == 1


def test_undefined_and_error_are_terminal_but_retry_errors_is_selective(
    tmp_path: Path,
) -> None:
    records = _records(classes=1, per_class=3)
    graphs = [_value_graph(index) for index in range(3)]
    metric = _StatusMetric()
    metric_dir = tmp_path / "metric"
    logger = _RunLogger(tmp_path / "run.log")

    first = compute_one_metric(
        metric,
        graphs,
        records,
        metric_dir=metric_dir,
        resume=False,
        retry_errors=False,
        checkpoint_every=20,
        fail_fast=False,
        logger=logger,
    )
    assert first["status_counts"] == {
        "pending": 0,
        "ok": 1,
        "undefined": 1,
        "error": 1,
        "preparation_error": 0,
    }

    ordinary_resume = compute_one_metric(
        metric,
        graphs,
        records,
        metric_dir=metric_dir,
        resume=True,
        retry_errors=False,
        checkpoint_every=20,
        fail_fast=False,
        logger=logger,
    )
    assert ordinary_resume["new_pairs"] == 0
    assert len(metric.compare_calls) == 3

    retry = compute_one_metric(
        metric,
        graphs,
        records,
        metric_dir=metric_dir,
        resume=True,
        retry_errors=True,
        checkpoint_every=20,
        fail_fast=False,
        logger=logger,
    )
    assert retry["new_pairs"] == 1
    assert metric.compare_calls.count((0, 1)) == 1
    assert metric.compare_calls.count((0, 2)) == 2

    distances = np.load(metric_dir / "distances.npy")
    status = np.load(metric_dir / "status.npy")
    assert np.isnan(distances[0, 1]) and np.isnan(distances[1, 0])
    assert np.isnan(distances[0, 2]) and np.isnan(distances[2, 0])
    assert status[0, 1] == status[1, 0] == STATUS_UNDEFINED
    assert status[0, 2] == status[2, 0] == STATUS_ERROR
    assert status[1, 2] == status[2, 1] == STATUS_OK


def test_nan_is_undefined_only_when_metric_explicitly_allows_it(
    tmp_path: Path,
) -> None:
    records = _records(classes=1, per_class=2)
    graphs = [_value_graph(index) for index in range(2)]
    logger = _RunLogger(tmp_path / "run.log")

    unexpected_summary = compute_one_metric(
        _UnexpectedNaNMetric(),
        graphs,
        records,
        metric_dir=tmp_path / "unexpected",
        resume=False,
        retry_errors=False,
        checkpoint_every=20,
        fail_fast=False,
        logger=logger,
    )
    allowed_summary = compute_one_metric(
        _ExplicitUndefinedMetric(),
        graphs,
        records,
        metric_dir=tmp_path / "allowed",
        resume=False,
        retry_errors=False,
        checkpoint_every=20,
        fail_fast=False,
        logger=logger,
    )

    assert unexpected_summary["status_counts"]["error"] == 1
    assert unexpected_summary["status_counts"]["undefined"] == 0
    assert allowed_summary["status_counts"]["error"] == 0
    assert allowed_summary["status_counts"]["undefined"] == 1
    unexpected_status = np.load(tmp_path / "unexpected" / "status.npy")
    allowed_status = np.load(tmp_path / "allowed" / "status.npy")
    assert unexpected_status[0, 1] == unexpected_status[1, 0] == STATUS_ERROR
    assert allowed_status[0, 1] == allowed_status[1, 0] == STATUS_UNDEFINED


def test_worker_count_prefers_explicit_then_slurm_then_safe_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parsed = build_parser().parse_args(
        [
            "--metrics",
            "chamfer",
            "--output-dir",
            str(tmp_path / "run"),
            "--workers",
            "2",
        ]
    )
    assert parsed.workers == 2

    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert _resolve_worker_count(None) == (1, "default")

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
    assert _resolve_worker_count(None) == (6, "SLURM_CPUS_PER_TASK")
    assert _resolve_worker_count(2) == (2, "--workers")

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not-an-integer")
    with pytest.raises(ValueError, match="positive integer"):
        _resolve_worker_count(None)
    assert _resolve_worker_count(3) == (3, "--workers")
    with pytest.raises(ValueError, match="workers must be positive"):
        _resolve_worker_count(0)


def test_process_workers_match_sequential_values_and_statuses(
    tmp_path: Path,
) -> None:
    records = _records(classes=1, per_class=4)
    values = np.asarray([0, 2, 5, 9], dtype=np.float64)
    graphs = [_value_graph(int(value)) for value in values]
    sequential_metric = _CountingMetric()
    process_metric = _CountingMetric()

    sequential = compute_one_metric(
        sequential_metric,
        graphs,
        records,
        metric_dir=tmp_path / "sequential",
        resume=False,
        retry_errors=False,
        checkpoint_every=2,
        fail_fast=False,
        logger=_RunLogger(tmp_path / "sequential.log"),
        workers=1,
    )
    process = compute_one_metric(
        process_metric,
        graphs,
        records,
        metric_dir=tmp_path / "process",
        resume=False,
        retry_errors=False,
        checkpoint_every=2,
        fail_fast=False,
        logger=_RunLogger(tmp_path / "process.log"),
        workers=2,
    )

    expected = np.abs(values[:, np.newaxis] - values[np.newaxis, :])
    sequential_distances = np.load(tmp_path / "sequential" / "distances.npy")
    process_distances = np.load(tmp_path / "process" / "distances.npy")
    sequential_status = np.load(tmp_path / "sequential" / "status.npy")
    process_status = np.load(tmp_path / "process" / "status.npy")
    np.testing.assert_allclose(sequential_distances, expected)
    np.testing.assert_allclose(process_distances, expected)
    np.testing.assert_array_equal(process_distances, sequential_distances)
    np.testing.assert_array_equal(process_status, sequential_status)
    np.testing.assert_array_equal(
        process_status,
        np.full((4, 4), STATUS_OK, dtype=np.uint8),
    )
    assert sequential["workers_used"] == 1
    assert sequential["worker_start_method"] is None
    assert process["workers_used"] == 2
    assert process["worker_start_method"] in {"fork", "spawn"}
    assert sequential_metric.compare_calls == [
        (0, 2),
        (0, 5),
        (0, 9),
        (2, 5),
        (2, 9),
        (5, 9),
    ]


def test_process_pair_cap_is_exact_and_resume_finishes_pending_pairs(
    tmp_path: Path,
) -> None:
    records = _records(classes=1, per_class=4)
    graphs = [_value_graph(index) for index in range(4)]
    metric = _CountingMetric()
    metric_dir = tmp_path / "metric"
    logger = _RunLogger(tmp_path / "run.log")

    partial = compute_one_metric(
        metric,
        graphs,
        records,
        metric_dir=metric_dir,
        resume=False,
        retry_errors=False,
        checkpoint_every=10,
        fail_fast=False,
        logger=logger,
        max_new_pairs=2,
        workers=2,
    )

    assert partial["new_pairs"] == 2
    assert partial["status_counts"]["ok"] == 2
    assert partial["status_counts"]["pending"] == 4
    partial_status = np.load(metric_dir / "status.npy")
    partial_distances = np.load(metric_dir / "distances.npy")
    assert partial_status[0, 1] == partial_status[0, 2] == STATUS_OK
    assert partial_distances[0, 1] == pytest.approx(1.0)
    assert partial_distances[0, 2] == pytest.approx(2.0)
    for index_a, index_b in ((0, 3), (1, 2), (1, 3), (2, 3)):
        assert partial_status[index_a, index_b] == STATUS_PENDING
        assert np.isnan(partial_distances[index_a, index_b])

    resumed = compute_one_metric(
        metric,
        graphs,
        records,
        metric_dir=metric_dir,
        resume=True,
        retry_errors=False,
        checkpoint_every=10,
        fail_fast=False,
        logger=logger,
        workers=1,
    )

    assert resumed["new_pairs"] == 4
    assert resumed["status_counts"]["pending"] == 0
    values = np.arange(4, dtype=np.float64)
    np.testing.assert_allclose(
        np.load(metric_dir / "distances.npy"),
        np.abs(values[:, np.newaxis] - values[np.newaxis, :]),
    )
    np.testing.assert_array_equal(
        np.load(metric_dir / "status.npy"),
        np.full((4, 4), STATUS_OK, dtype=np.uint8),
    )


def test_worker_exception_is_recorded_and_remains_retryable(
    tmp_path: Path,
) -> None:
    records = _records(classes=1, per_class=3)
    graphs = [_value_graph(index) for index in range(3)]
    metric_dir = tmp_path / "metric"
    metric = _WorkerErrorMetric()
    logger = _RunLogger(tmp_path / "run.log")

    first = compute_one_metric(
        metric,
        graphs,
        records,
        metric_dir=metric_dir,
        resume=False,
        retry_errors=False,
        checkpoint_every=10,
        fail_fast=False,
        logger=logger,
        workers=2,
    )

    assert first["new_pairs"] == 3
    assert first["workers_used"] == 2
    assert first["status_counts"]["ok"] == 2
    assert first["status_counts"]["error"] == 1
    distances = np.load(metric_dir / "distances.npy")
    status = np.load(metric_dir / "status.npy")
    assert np.isnan(distances[0, 2]) and np.isnan(distances[2, 0])
    assert status[0, 2] == status[2, 0] == STATUS_ERROR
    assert status[0, 1] == status[1, 0] == STATUS_OK
    assert status[1, 2] == status[2, 1] == STATUS_OK
    issues = [
        json.loads(line)
        for line in (metric_dir / "issues.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(issues) == 1
    assert issues[0]["kind"] == "comparison_error"
    assert issues[0]["error_type"] == "RuntimeError"
    assert issues[0]["error_message"] == "fixture worker failure"

    ordinary_resume = compute_one_metric(
        metric,
        graphs,
        records,
        metric_dir=metric_dir,
        resume=True,
        retry_errors=False,
        checkpoint_every=10,
        fail_fast=False,
        logger=logger,
        workers=2,
    )
    assert ordinary_resume["new_pairs"] == 0

    retry = compute_one_metric(
        metric,
        graphs,
        records,
        metric_dir=metric_dir,
        resume=True,
        retry_errors=True,
        checkpoint_every=10,
        fail_fast=False,
        logger=logger,
        workers=2,
    )
    assert retry["new_pairs"] == 1
    assert retry["status_counts"]["error"] == 1
    assert retry["status_counts"]["pending"] == 0
    retried_status = np.load(metric_dir / "status.npy")
    assert retried_status[0, 2] == retried_status[2, 0] == STATUS_ERROR


def test_persisted_run_json_key_sorting_does_not_break_resume_metric_order(
    tmp_path: Path,
) -> None:
    records = _records(
        classes=1,
        per_class=2,
        fixture_root=tmp_path / "swcs",
    )
    metric_names = (TMD_PATH_WASSERSTEIN, TMD_HEIGHT_WASSERSTEIN)
    metrics = tuple(
        build_matrix_metric(
            name,
            so2_grid_size=4,
            so2_refine=False,
            so2_refinement_tolerance=1e-8,
            fgw_max_nodes=100,
        )
        for name in metric_names
    )
    args = SimpleNamespace(
        dataset_root=tmp_path,
        splits=["test"],
        selection="all",
        seed=0,
        count=None,
        per_class=None,
        classes=None,
        selection_manifest=None,
    )
    run_path = tmp_path / "run.json"
    _write_json_atomic(run_path, _static_run_payload(args, records, metrics))

    # JSON object key order is not semantic. In particular, the atomic writer's
    # stable key sorting must not make a run reject its own saved configuration.
    _validate_resume(_load_json(run_path), records, metrics)
