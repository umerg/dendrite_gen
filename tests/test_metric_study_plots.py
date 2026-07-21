"""Tests for class-aware metric-study plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visualization.metric_study.plots import (
    class_median_distances,
    classical_mds,
    save_class_comparison_plots,
)


def _example_matrix() -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    coordinates = np.asarray([0.0, 0.2, 2.0, 2.3, 5.0, 5.4])
    distances = np.abs(coordinates[:, None] - coordinates[None, :])
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    names = {0: "A", 1: "B", 2: "C"}
    return distances, labels, names


def test_classical_mds_preserves_euclidean_line_distances() -> None:
    distances, _, _ = _example_matrix()

    coordinates, eigenvalues = classical_mds(distances)
    embedded = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)

    assert eigenvalues[0] > 0.0
    assert np.allclose(embedded, distances, atol=1e-10)


def test_class_medians_keep_diagonal_and_cross_class_blocks_separate() -> None:
    distances, labels, _ = _example_matrix()

    medians = class_median_distances(distances, labels, classes=(0, 1, 2))

    assert np.allclose(np.diag(medians), [0.2, 0.3, 0.4])
    assert medians[0, 1] == medians[1, 0]
    assert medians[0, 2] == medians[2, 0]


def test_standard_plot_set_writes_nonempty_pngs(tmp_path: Path) -> None:
    distances, labels, names = _example_matrix()

    outputs = save_class_comparison_plots(
        distances,
        labels,
        names,
        metric_label="Example metric",
        out_dir=tmp_path,
    )

    assert set(outputs) == {"embedding", "ordered_matrix", "class_medians"}
    for path in outputs.values():
        assert path.is_file()
        assert path.stat().st_size > 1000
        image = plt.imread(path)
        assert image.ndim == 3
        assert image.shape[0] > 100
        assert image.shape[1] > 100
