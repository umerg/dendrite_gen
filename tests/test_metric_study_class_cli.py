"""End-to-end test for the class-labelled metric-study runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from visualization.metric_study.run_class_comparison import main


def _write_tree(
    path: Path,
    *,
    cell_class: int,
    cell_type: str,
    lengths: tuple[float, float],
) -> None:
    path.write_text(
        "# fixture\n"
        f"# cell_class {cell_class}\n"
        f"# cell_type {cell_type}\n"
        "1 1 0 0 0 1 -1\n"
        f"2 3 {lengths[0]} 0 0 1 1\n"
        f"3 3 0 {lengths[1]} 0 1 1\n",
        encoding="utf-8",
    )


def test_class_comparison_writes_reusable_matrix_metadata_and_plots(
    tmp_path: Path,
    capsys,
) -> None:
    dataset_root = tmp_path / "dataset"
    split = dataset_root / "test"
    split.mkdir(parents=True)
    _write_tree(
        split / "a0.swc",
        cell_class=0,
        cell_type="A",
        lengths=(1.0, 2.0),
    )
    _write_tree(
        split / "a1.swc",
        cell_class=0,
        cell_type="A",
        lengths=(1.1, 2.1),
    )
    _write_tree(
        split / "b0.swc",
        cell_class=1,
        cell_type="B",
        lengths=(3.0, 5.0),
    )
    _write_tree(
        split / "b1.swc",
        cell_class=1,
        cell_type="B",
        lengths=(3.2, 5.2),
    )
    output_dir = tmp_path / "run"

    exit_code = main(
        [
            "--dataset-root",
            str(dataset_root),
            "--splits",
            "test",
            "--per-class",
            "2",
            "--seed",
            "7",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metric"]["name"] == "tmd_path_wasserstein"
    assert payload["dataset"]["selected_trees"] == 4
    assert payload["dataset"]["classes"] == {"0": "A", "1": "B"}

    matrix_bundle = np.load(output_dir / "distance_matrix.npz")
    distances = matrix_bundle["distances"]
    assert distances.shape == (4, 4)
    np.testing.assert_allclose(distances, distances.T)
    np.testing.assert_allclose(np.diag(distances), 0.0, atol=1e-12)
    assert list(matrix_bundle["cell_classes"]) == [0, 0, 1, 1]

    assert (output_dir / "selected_trees.csv").is_file()
    assert (output_dir / "class_counts.csv").is_file()
    assert (output_dir / "run.json").is_file()
    for filename in (
        "mds_embedding.png",
        "class_ordered_distances.png",
        "class_median_distances.png",
    ):
        path = output_dir / "plots" / filename
        assert path.is_file()
        assert path.stat().st_size > 1000
