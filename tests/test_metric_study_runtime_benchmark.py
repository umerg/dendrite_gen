"""Tests for the reproducible ground-truth runtime benchmark."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from visualization.metric_study.dataset import TreeRecord
from visualization.metric_study.run_runtime_benchmark import (
    Measurement,
    main,
    select_random_pairs,
    summarize_family_totals,
    summarize_measurements,
    transform_scientific_y_to_internal_z,
)


def _record(index: int) -> TreeRecord:
    return TreeRecord(
        tree_id=f"tree-{index}",
        swc_path=Path(f"/fixture/tree-{index}.swc"),
        split="test",
        cell_class=index % 2,
        cell_type=f"type-{index % 2}",
    )


def _measurement(
    *,
    pair_index: int,
    metric_name: str,
    family: str,
    elapsed: float,
    status: str = "ok",
    grid_angles: int = 72,
    objective_evaluations: int | None = 80,
) -> Measurement:
    return Measurement(
        pair_index=pair_index,
        metric_name=metric_name,
        metric_display_name=metric_name,
        family=family,
        family_display_name=family,
        status=status,
        result_status="ok" if status == "ok" else "error",
        elapsed_seconds=elapsed,
        value=1.0 if status == "ok" else None,
        grid_angles=grid_angles,
        refine=bool(grid_angles),
        objective_evaluations=objective_evaluations,
        upstream_evaluations=None,
    )


def test_random_pair_selection_is_seeded_order_independent_and_has_no_reuse() -> None:
    records = tuple(_record(index) for index in range(8))

    first = select_random_pairs(records, 3, seed=17)
    repeated = select_random_pairs(records, 3, seed=17)
    reversed_input = select_random_pairs(tuple(reversed(records)), 3, seed=17)
    different_seed = select_random_pairs(records, 3, seed=18)

    assert first == repeated == reversed_input
    selected_ids = [record.tree_id for pair in first for record in pair]
    assert len(selected_ids) == len(set(selected_ids)) == 6
    assert first != different_seed
    with pytest.raises(ValueError, match="requires 10 trees"):
        select_random_pairs(records, 5, seed=0)


def test_frame_rotation_maps_scientific_y_to_internal_z() -> None:
    graph = nx.Graph()
    graph.add_node(0, pos=np.array([1.0, 2.0, 3.0]))
    graph.graph["root"] = 0

    transformed = transform_scientific_y_to_internal_z(graph)

    np.testing.assert_allclose(transformed.nodes[0]["pos"], [1.0, -3.0, 2.0])
    np.testing.assert_allclose(graph.nodes[0]["pos"], [1.0, 2.0, 3.0])


def test_summaries_exclude_failures_and_sum_scalar_family_variants() -> None:
    measurements = [
        _measurement(pair_index=1, metric_name="angular", family="a", elapsed=1.0),
        _measurement(
            pair_index=2,
            metric_name="angular",
            family="a",
            elapsed=3.0,
            objective_evaluations=84,
        ),
        _measurement(
            pair_index=3,
            metric_name="angular",
            family="a",
            elapsed=99.0,
            status="error",
            objective_evaluations=None,
        ),
    ]

    summary = summarize_measurements(measurements)[0]

    assert summary["requested_count"] == 3
    assert summary["successful_count"] == 2
    assert summary["failed_count"] == 1
    assert summary["mean_seconds"] == pytest.approx(2.0)
    assert summary["median_seconds"] == pytest.approx(2.0)
    assert summary["min_seconds"] == pytest.approx(1.0)
    assert summary["max_seconds"] == pytest.approx(3.0)
    assert summary["grid_angles"] == 72
    assert summary["refine"] is True
    assert summary["mean_objective_evaluations"] == pytest.approx(82.0)

    family_measurements = [
        _measurement(pair_index=1, metric_name="one", family="family", elapsed=1.0),
        _measurement(pair_index=1, metric_name="two", family="family", elapsed=2.0),
        _measurement(pair_index=2, metric_name="one", family="family", elapsed=3.0),
        _measurement(pair_index=2, metric_name="two", family="family", elapsed=4.0),
    ]
    family_summary = summarize_family_totals(family_measurements)[0]
    assert family_summary["mean_total_seconds"] == pytest.approx(5.0)
    assert family_summary["scalar_variant_count"] == 2


def _write_tree(
    path: Path,
    *,
    cell_class: int,
    cell_type: str,
    scale: float,
) -> None:
    path.write_text(
        "# fixture\n"
        f"# cell_class {cell_class}\n"
        f"# cell_type {cell_type}\n"
        "1 1 0 0 0 1 -1\n"
        f"2 3 {scale} {scale} 0 1 1\n"
        f"3 3 {-scale} {2 * scale} 0 1 1\n",
        encoding="utf-8",
    )


def test_cli_runs_only_requested_fast_metrics_and_records_angle_counts(
    tmp_path: Path,
    capsys,
) -> None:
    dataset_root = tmp_path / "dataset"
    split = dataset_root / "test"
    split.mkdir(parents=True)
    for index, (cell_class, cell_type) in enumerate(
        ((0, "A"), (0, "A"), (1, "B"), (1, "B")),
        start=1,
    ):
        _write_tree(
            split / f"tree-{index}.swc",
            cell_class=cell_class,
            cell_type=cell_type,
            scale=float(index),
        )
    output_dir = tmp_path / "benchmark"

    exit_code = main(
        [
            "--dataset-root",
            str(dataset_root),
            "--split",
            "test",
            "--pairs",
            "2",
            "--seed",
            "7",
            "--metrics",
            "chamfer",
            "tmd_path_wasserstein",
            "--so2-grid-size",
            "4",
            "--no-so2-refine",
            "--no-warmup",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert stdout_payload == file_payload
    assert file_payload["dataset"]["selected_pairs"] == 2
    assert file_payload["dataset"]["selected_trees"] == 4
    assert file_payload["configuration"]["selected_metrics"] == [
        "chamfer",
        "tmd_path_wasserstein",
    ]

    with (output_dir / "pairs.csv").open(encoding="utf-8", newline="") as handle:
        pair_rows = list(csv.DictReader(handle))
    selected_ids = [
        row[key]
        for row in pair_rows
        for key in ("tree_a_id", "tree_b_id")
    ]
    assert len(selected_ids) == len(set(selected_ids)) == 4

    with (output_dir / "measurements.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["status"] for row in rows} == {"ok"}
    for row in rows:
        assert float(row["elapsed_seconds"]) >= 0.0
        if row["metric_name"] == "chamfer":
            assert row["grid_angles"] == "4"
            assert row["objective_evaluations"] == "4"
        else:
            assert row["grid_angles"] == "0"
            assert row["objective_evaluations"] == "0"

    for filename in (
        "pairs.csv",
        "measurements.csv",
        "metric_summary.csv",
        "family_summary.csv",
        "run.json",
    ):
        assert (output_dir / filename).is_file()
