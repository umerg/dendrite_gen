"""End-to-end checks for the ground-truth-only single-pair command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from metrics.pair import compare_tree_pair
from utils.data_loading import load_swc_graph
from visualization.metric_study.run_pair import main


def _write_swc(path: Path, *, angle_rad: float = 0.0) -> None:
    points = np.asarray(
        [
            [5.0, 4.0, 3.0],
            [6.2, 4.1, 3.7],
            [4.6, 5.7, 4.4],
            [7.1, 4.8, 5.0],
            [3.9, 6.4, 5.7],
        ],
        dtype=np.float64,
    )
    root = points[0].copy()
    centered = points - root
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    points = centered @ rotation.T + root
    parents = (-1, 1, 1, 2, 3)
    lines = [
        f"{index} 3 {x:.12g} {y:.12g} {z:.12g} 1 {parent}"
        for index, ((x, y, z), parent) in enumerate(
            zip(points, parents), start=1
        )
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_chain_swc(path: Path) -> None:
    path.write_text(
        "1 3 0 0 0 1 -1\n2 3 0 0 2 1 1\n",
        encoding="utf-8",
    )


def test_compare_tree_pair_selects_only_requested_families(tmp_path: Path) -> None:
    swc = tmp_path / "tree.swc"
    _write_swc(swc)
    tree = load_swc_graph(swc)

    result = compare_tree_pair(tree, tree, metric_families=("chamfer",))

    assert list(result) == ["chamfer"]
    assert result["chamfer"]["value"] == pytest.approx(0.0)  # type: ignore[index]

    default_result = compare_tree_pair(tree, tree)
    assert list(default_result) == ["chamfer", "persistence", "distributions"]


def test_compare_tree_pair_rejects_unknown_family(tmp_path: Path) -> None:
    swc = tmp_path / "tree.swc"
    _write_swc(swc)
    tree = load_swc_graph(swc)

    with pytest.raises(ValueError, match="Unknown metric families"):
        compare_tree_pair(tree, tree, metric_families=("not_a_metric",))  # type: ignore[arg-type]


def test_cli_compares_two_swcs_and_writes_standard_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tree_a = tmp_path / "a.swc"
    tree_b = tmp_path / "b.swc"
    output = tmp_path / "results" / "pair.json"
    _write_swc(tree_a)
    _write_swc(tree_b, angle_rad=0.73)

    exit_code = main(
        [
            "--tree-a",
            str(tree_a),
            "--tree-b",
            str(tree_b),
            "--metrics",
            "chamfer",
            "persistence",
            "distributions",
            "--chamfer-spacing",
            "0.5",
            "--output-json",
            str(output),
        ]
    )

    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert stdout_payload == file_payload
    assert list(stdout_payload["results"]) == [
        "chamfer",
        "distributions",
        "persistence",
    ]
    assert stdout_payload["quotient_group"] == {
        "name": "SO(2)",
        "preferred_axis": "z",
        "enabled_for_azimuth_retaining_metrics": True,
        "includes_tilts": False,
        "includes_axis_flips": False,
        "includes_reflections": False,
        "grid_size": 72,
        "local_refinement": True,
        "refinement_angle_tolerance_rad": 1e-8,
    }
    assert stdout_payload["results"]["chamfer"]["value"] < 1e-8
    assert all(
        diagnostic["status"] == "ok"
        for diagnostic in stdout_payload["results"]["distributions"][
            "diagnostics"
        ].values()
    )
    assert stdout_payload["tree_a"]["nodes"] == 5
    assert stdout_payload["tree_b"]["nodes"] == 5


def test_cli_explains_undefined_one_empty_distribution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    chain = tmp_path / "chain.swc"
    branched = tmp_path / "branched.swc"
    _write_chain_swc(chain)
    _write_swc(branched)
    distribution = "critical_branch_chord_sibling_angle_deg"

    assert main(
        [
            "--tree-a",
            str(chain),
            "--tree-b",
            str(branched),
            "--metrics",
            "distributions",
            "--distributions",
            distribution,
        ]
    ) == 0

    result = json.loads(capsys.readouterr().out)["results"]["distributions"]
    assert result["distances"][distribution] is None
    assert result["diagnostics"][distribution] == {
        "empty_a": True,
        "empty_b": False,
        "sample_count_a": 0,
        "sample_count_b": 1,
        "status": "undefined_one_empty",
    }


def test_cli_refuses_to_overwrite_an_input_swc(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tree_a = tmp_path / "a.swc"
    tree_b = tmp_path / "b.swc"
    _write_swc(tree_a)
    _write_swc(tree_b)
    original = tree_a.read_text(encoding="utf-8")

    with pytest.raises(SystemExit):
        main(
            [
                "--tree-a",
                str(tree_a),
                "--tree-b",
                str(tree_b),
                "--metrics",
                "chamfer",
                "--output-json",
                str(tree_a),
            ]
        )

    assert "must not resolve" in capsys.readouterr().err
    assert tree_a.read_text(encoding="utf-8") == original


def test_cli_wires_opt_in_fgw_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("ot", reason="FGW integration requires optional POT")
    tree_a = tmp_path / "a.swc"
    tree_b = tmp_path / "b.swc"
    _write_swc(tree_a)
    _write_swc(tree_b, angle_rad=0.73)

    assert main(
        [
            "--tree-a",
            str(tree_a),
            "--tree-b",
            str(tree_b),
            "--metrics",
            "fgw",
            "--so2-grid-size",
            "12",
        ]
    ) == 0

    result = json.loads(capsys.readouterr().out)["results"]["fgw"]
    assert result["value"] < 1e-7
    assert result["feature_mode"] == "xyz"
    assert result["mass_mode"] == "cable_length"
    assert result["quotient_so2"] is True
    assert result["grid_size"] == 12


def test_cli_dense_fgw_guard_runs_before_solver(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tree_a = tmp_path / "a.swc"
    tree_b = tmp_path / "b.swc"
    _write_swc(tree_a)
    _write_swc(tree_b)

    with pytest.raises(SystemExit):
        main(
            [
                "--tree-a",
                str(tree_a),
                "--tree-b",
                str(tree_b),
                "--metrics",
                "fgw",
                "--fgw-max-nodes",
                "4",
            ]
        )

    assert "dense pairwise matrices" in capsys.readouterr().err
