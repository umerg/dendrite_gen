from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from visualization.metric_study.matrix_report import (
    MatrixStudy,
    _nearest_neighbor_confusion,
    _pair_balance_weights,
    _weighted_auc,
    load_matrix_study,
    metric_label,
)


def _small_study(matrix: np.ndarray) -> MatrixStudy:
    manifest = pd.DataFrame(
        {
            "matrix_index": np.arange(4),
            "tree_id": ["a", "b", "c", "d"],
            "swc_path": ["unused"] * 4,
            "split": ["test"] * 4,
            "cell_class": [0, 0, 1, 1],
            "cell_type": ["A", "A", "B", "B"],
        }
    )
    return MatrixStudy(
        run_dir=Path("."),
        manifest=manifest,
        matrices={"custom_metric": matrix},
    )


def test_weighted_auc_matches_pairwise_definition_with_ties() -> None:
    same = np.asarray([1.0, 2.0, 2.0])
    same_weights = np.asarray([1.0, 2.0, 1.0])
    different = np.asarray([2.0, 3.0])
    different_weights = np.asarray([3.0, 1.0])

    expected = sum(
        (weight_a / same_weights.sum())
        * (weight_b / different_weights.sum())
        * ((value_a < value_b) + 0.5 * (value_a == value_b))
        for value_a, weight_a in zip(same, same_weights)
        for value_b, weight_b in zip(different, different_weights)
    )

    assert np.isclose(
        _weighted_auc(same, same_weights, different, different_weights),
        expected,
    )


def test_fractional_nearest_neighbor_ties_and_pair_balance() -> None:
    matrix = np.asarray(
        [
            [0.0, 1.0, 1.0, 3.0],
            [1.0, 0.0, 3.0, 3.0],
            [1.0, 3.0, 0.0, 0.5],
            [3.0, 3.0, 0.5, 0.0],
        ]
    )
    study = _small_study(matrix)

    confusion, macro, micro, tied = _nearest_neighbor_confusion(study, matrix)
    np.testing.assert_allclose(confusion, [[0.75, 0.25], [0.0, 1.0]])
    assert macro == 0.875
    assert micro == 0.875
    assert tied == 1

    _, same, weights = _pair_balance_weights(study)
    assert np.isclose(weights[same].sum(), 1.0)
    assert np.isclose(weights[~same].sum(), 1.0)


def test_loader_accepts_a_new_metric_name(tmp_path) -> None:
    family = tmp_path / "new_family"
    metric_dir = family / "metrics" / "new_tree_metric"
    metric_dir.mkdir(parents=True)
    (family / "progress.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    manifest = _small_study(np.zeros((4, 4))).manifest
    manifest.to_csv(family / "selected_trees.csv", index=False)
    matrix = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 2.5, 3.5],
            [2.0, 2.5, 0.0, 1.5],
            [3.0, 3.5, 1.5, 0.0],
        ]
    )
    np.save(metric_dir / "distances.npy", matrix)
    np.save(metric_dir / "status.npy", np.ones_like(matrix, dtype=np.uint8))

    study = load_matrix_study(tmp_path)

    assert study.metric_names == ("new_tree_metric",)
    np.testing.assert_array_equal(study.matrices["new_tree_metric"], matrix)
    assert metric_label("new_tree_metric") == "New Tree Metric"
