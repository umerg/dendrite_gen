"""Focused tests for the sharded Elastic SRVFT matrix runner."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from visualization.metric_study.dataset import TreeRecord
from visualization.metric_study import run_elastic_distance_matrix as runner


def _record(cell_class: int, index: int) -> TreeRecord:
    return TreeRecord(
        tree_id=f"class-{cell_class}-tree-{index}",
        swc_path=Path(f"/fixture/class-{cell_class}-tree-{index}.swc"),
        split="test",
        cell_class=cell_class,
        cell_type=f"type-{cell_class}",
    )


def _write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plan(
    output_dir: Path,
    *,
    tree_count: int = 3,
    pairs_per_shard: int = 3,
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    output_dir.mkdir()
    (output_dir / "shard_results").mkdir()
    pair_rows, shard_rows = runner.build_pair_plan(
        tree_count, pairs_per_shard=pairs_per_shard
    )
    run_payload = {
        "schema_version": 1,
        "metric": {
            "name": runner.METRIC_NAME,
            "lam_m": 0.2,
            "lam_s": 1.0,
            "lam_p": 0.2,
            "grid_size": 36,
            "refinement_tolerance": 1e-3,
            "default_radius": 1.0,
        },
        "backend": {
            "checkout": "/fixture/elastic-srvft",
            "revision": "fixture-revision",
        },
    }
    (output_dir / "run.json").write_text(
        json.dumps(run_payload), encoding="utf-8"
    )
    _write_csv(
        output_dir / "selected_trees.csv",
        ("matrix_index", "tree_id", "swc_path"),
        [
            {
                "matrix_index": index,
                "tree_id": f"tree-{index}",
                "swc_path": f"/fixture/tree-{index}.swc",
            }
            for index in range(tree_count)
        ],
    )
    pair_manifest = [
        {
            **row,
            "tree_a_id": f"tree-{row['index_a']}",
            "tree_b_id": f"tree-{row['index_b']}",
        }
        for row in pair_rows
    ]
    _write_csv(
        output_dir / "pairs.csv",
        (
            "pair_index",
            "shard_id",
            "index_a",
            "index_b",
            "tree_a_id",
            "tree_b_id",
        ),
        pair_manifest,
    )
    return pair_rows, shard_rows


def _write_complete_shards(
    output_dir: Path,
    pair_rows: list[dict[str, int]],
    shard_rows: list[dict[str, int]],
) -> None:
    for shard in shard_rows:
        shard_id = shard["shard_id"]
        results = []
        for pair in pair_rows:
            if pair["shard_id"] != shard_id:
                continue
            index_a = pair["index_a"]
            index_b = pair["index_b"]
            results.append(
                {
                    "pair_index": pair["pair_index"],
                    "index_a": index_a,
                    "index_b": index_b,
                    "tree_a_id": f"tree-{index_a}",
                    "tree_b_id": f"tree-{index_b}",
                    "status": "ok",
                    "value": float(10 * index_a + index_b),
                }
            )
        payload = {
            "schema_version": 1,
            "shard_id": shard_id,
            "complete": True,
            "results": results,
        }
        path = output_dir / "shard_results" / f"shard_{shard_id:06d}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")


def test_compatible_selection_is_deterministic_and_capped_per_class() -> None:
    records = tuple(
        _record(cell_class, index)
        for cell_class, count in ((0, 5), (1, 2), (2, 4))
        for index in range(count)
    )
    eligible = {
        record.tree_id
        for record in records
        if record.cell_class == 1
        or (record.cell_class == 0 and record.tree_id != "class-0-tree-4")
        or (record.cell_class == 2 and record.tree_id == "class-2-tree-0")
    }

    selected = runner.select_compatible_records(
        records, eligible, class_cap=3, seed=17
    )
    selected_again = runner.select_compatible_records(
        tuple(reversed(records)), eligible, class_cap=3, seed=17
    )

    assert [record.tree_id for record in selected_again] == [
        record.tree_id for record in selected
    ]
    assert Counter(record.cell_class for record in selected) == {0: 3, 1: 2, 2: 1}
    assert {record.tree_id for record in selected} <= eligible


def test_pair_plan_covers_the_strict_upper_triangle_once() -> None:
    pairs, shards = runner.build_pair_plan(5, pairs_per_shard=3)

    assert [(row["index_a"], row["index_b"]) for row in pairs] == list(
        itertools.combinations(range(5), 2)
    )
    assert [row["pair_index"] for row in pairs] == list(range(10))
    assert [row["shard_id"] for row in pairs] == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
    assert shards == [
        {"shard_id": 0, "pair_start": 0, "pair_stop": 3, "pair_count": 3},
        {"shard_id": 1, "pair_start": 3, "pair_stop": 6, "pair_count": 3},
        {"shard_id": 2, "pair_start": 6, "pair_stop": 9, "pair_count": 3},
        {"shard_id": 3, "pair_start": 9, "pair_stop": 10, "pair_count": 1},
    ]


def test_prepare_parser_exposes_the_fixed_scientific_defaults(
    tmp_path: Path,
) -> None:
    args = runner.build_parser().parse_args(
        ["prepare", "--output-dir", str(tmp_path / "run")]
    )

    assert args.split == "test"
    assert args.max_trees_per_class == 20
    assert args.pairs_per_shard == 8
    assert args.so2_grid_size == 36
    assert args.refinement_tolerance == pytest.approx(1e-3)
    assert (args.lam_m, args.lam_s, args.lam_p) == pytest.approx((0.2, 1.0, 0.2))
    assert args.default_radius == pytest.approx(1.0)


@dataclass(frozen=True)
class _FakeElasticResult:
    value: float
    tree_a_omitted_frontier_branches: int = 0
    tree_b_omitted_frontier_branches: int = 0


def test_compute_shard_uses_mean_symmetrization_and_resumes_after_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    _write_plan(output_dir)
    expected_backend = {
        "checkout": "/fixture/elastic-srvft",
        "revision": "fixture-revision",
    }
    monkeypatch.setattr(runner, "_backend_contract", lambda checkout: expected_backend)
    monkeypatch.setattr(runner, "load_swc_graph", lambda path: path.name)
    monkeypatch.setattr(
        runner,
        "transform_scientific_y_to_internal_z",
        lambda graph: f"internal:{graph}",
    )

    calls: list[tuple[object, object, dict[str, object]]] = []
    interrupt = {"enabled": True}

    def fake_distance(
        graph_a: object, graph_b: object, **kwargs: object
    ) -> _FakeElasticResult:
        calls.append((graph_a, graph_b, kwargs))
        if interrupt["enabled"] and len(calls) == 2:
            raise KeyboardInterrupt
        return _FakeElasticResult(value=float(len(calls)))

    monkeypatch.setattr(
        runner.elastic_adapter, "elastic_srvft_distance", fake_distance
    )
    args = SimpleNamespace(output_dir=output_dir, shard_id=0)

    with pytest.raises(KeyboardInterrupt):
        runner.compute_shard(args)

    shard_path = output_dir / "shard_results" / "shard_000000.json"
    interrupted_payload = json.loads(shard_path.read_text(encoding="utf-8"))
    assert interrupted_payload["complete"] is False
    assert [row["pair_index"] for row in interrupted_payload["results"]] == [0]

    interrupt["enabled"] = False
    resumed = runner.compute_shard(args)
    assert resumed["complete"] is True
    assert resumed["new_pairs"] == 2
    assert resumed["status_counts"] == {"ok": 3}
    assert len(calls) == 4

    already_complete = runner.compute_shard(args)
    assert already_complete["new_pairs"] == 0
    assert len(calls) == 4

    for _, _, kwargs in calls:
        assert kwargs == {
            "checkout": "/fixture/elastic-srvft",
            "lam_m": 0.2,
            "lam_s": 1.0,
            "lam_p": 0.2,
            "quotient_so2": True,
            "grid_size": 36,
            "refine": True,
            "refinement_tolerance": 1e-3,
            "symmetrization": "mean",
            "depth_policy": "raise",
            "default_radius": 1.0,
        }


def test_merge_publishes_a_symmetric_matrix(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    pairs, shards = _write_plan(output_dir, pairs_per_shard=2)
    _write_complete_shards(output_dir, pairs, shards)

    progress = runner.merge_run(SimpleNamespace(output_dir=output_dir))

    assert progress["status"] == "complete"
    distances = np.load(
        output_dir / "metrics" / runner.METRIC_NAME / "distances.npy"
    )
    status = np.load(output_dir / "metrics" / runner.METRIC_NAME / "status.npy")
    np.testing.assert_allclose(
        distances,
        np.asarray([[0.0, 1.0, 2.0], [1.0, 0.0, 12.0], [2.0, 12.0, 0.0]]),
    )
    np.testing.assert_array_equal(status, np.full((3, 3), runner.STATUS_OK))


def test_merge_rejects_missing_pair_coverage(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    pairs, shards = _write_plan(output_dir, pairs_per_shard=2)
    _write_complete_shards(output_dir, pairs, shards)
    final_path = output_dir / "shard_results" / "shard_000001.json"
    payload = json.loads(final_path.read_text(encoding="utf-8"))
    payload["results"] = []
    final_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete shards"):
        runner.merge_run(SimpleNamespace(output_dir=output_dir))
