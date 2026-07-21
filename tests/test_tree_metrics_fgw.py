"""Focused tests for the optional POT-backed tree FGW metric."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pytest


pytest.importorskip("ot", reason="Fused GW tests require the optional POT package")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.fused_gw import fused_gromov_wasserstein_distance
from metrics.so2 import rotate_points_about_axis


def _asymmetric_tree() -> nx.Graph:
    graph = nx.Graph()
    positions = {
        0: (0.0, 0.0, 0.0),
        1: (1.2, 0.1, 0.7),
        2: (-0.4, 1.7, 1.4),
        3: (2.1, 0.8, 2.0),
        4: (-1.1, 2.4, 2.7),
    }
    graph.add_nodes_from((node, {"pos": np.asarray(pos)}) for node, pos in positions.items())
    graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4)])
    graph.graph["root"] = 0
    return graph


def _second_tree() -> nx.Graph:
    graph = _asymmetric_tree()
    graph.nodes[3]["pos"] = np.array([2.6, 0.4, 2.3])
    graph.nodes[4]["pos"] = np.array([-0.8, 2.8, 3.1])
    return graph


def _rotate_tree(graph: nx.Graph, angle_rad: float) -> nx.Graph:
    rotated = copy.deepcopy(graph)
    nodes = list(rotated.nodes)
    points = np.stack([rotated.nodes[node]["pos"] for node in nodes])
    rotated_points = rotate_points_about_axis(points, angle_rad, (0.0, 0.0, 1.0))
    for node, point in zip(nodes, rotated_points):
        rotated.nodes[node]["pos"] = point
    return rotated


def test_identity_is_zero() -> None:
    tree = _asymmetric_tree()
    result = fused_gromov_wasserstein_distance(tree, tree, feature_mode="axis")

    assert result.value == pytest.approx(0.0, abs=1e-10)
    assert result.angle_rad == 0.0
    assert result.n_nodes_1 == result.n_nodes_2 == tree.number_of_nodes()


def test_distance_is_approximately_symmetric() -> None:
    tree_1 = _asymmetric_tree()
    tree_2 = _second_tree()

    forward = fused_gromov_wasserstein_distance(tree_1, tree_2, feature_mode="axis")
    reverse = fused_gromov_wasserstein_distance(tree_2, tree_1, feature_mode="axis")

    assert forward.value == pytest.approx(reverse.value, rel=1e-7, abs=1e-9)


def test_axis_features_are_so2_invariant_without_search() -> None:
    tree = _asymmetric_tree()
    rotated = _rotate_tree(tree, 0.731)

    result = fused_gromov_wasserstein_distance(
        tree,
        rotated,
        feature_mode="axis",
        quotient_so2=False,
    )

    assert result.value == pytest.approx(0.0, abs=1e-10)
    assert result.angle_rad == 0.0


def test_xyz_quotient_recovers_rotated_asymmetric_tree() -> None:
    tree = _asymmetric_tree()
    rotated = _rotate_tree(tree, 0.731)

    unaligned = fused_gromov_wasserstein_distance(
        tree,
        rotated,
        feature_mode="xyz",
        quotient_so2=False,
    )
    aligned = fused_gromov_wasserstein_distance(
        tree,
        rotated,
        grid_size=36,
        refine=True,
    )

    assert unaligned.value > 1e-3
    assert aligned.value < 1e-7
    assert aligned.feature_mode == "xyz"
    assert aligned.mass_mode == "cable_length"
    assert aligned.quotient_so2 is True
    assert 0.0 <= aligned.angle_rad < 2.0 * np.pi


def test_structure_only_alpha_skips_so2_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("metrics.fused_gw")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("SO(2) search should be skipped when alpha=1")

    monkeypatch.setattr(module, "minimize_over_so2", fail_if_called)
    result = module.fused_gromov_wasserstein_distance(
        _asymmetric_tree(),
        _rotate_tree(_asymmetric_tree(), 0.731),
        feature_mode="xyz",
        quotient_so2=True,
        alpha=1.0,
    )

    assert result.angle_rad == 0.0
    assert result.quotient_so2 is False
