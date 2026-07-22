from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from visualization.metric_study.dataset import TreeRecord
from visualization.metric_study.matrix_metrics import (
    CHAMFER,
    DISTRIBUTION_SIBLING_ANGLE_WASSERSTEIN,
    TMD_PATH_WASSERSTEIN,
)
from visualization.metric_study.run_self_rotation_sweep import (
    metric_angle_curve,
    rotate_graph_about_root_z,
    select_rotation_trees,
)


def _toy_tree() -> nx.Graph:
    graph = nx.Graph()
    positions = {
        0: np.asarray([4.0, -2.0, 7.0]),
        1: np.asarray([5.0, -2.0, 8.0]),
        2: np.asarray([4.2, 0.0, 9.0]),
        3: np.asarray([3.0, -1.0, 10.0]),
        4: np.asarray([5.0, 1.0, 11.0]),
    }
    for node, position in positions.items():
        graph.add_node(node, pos=position.copy())
    graph.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 4)])
    graph.graph["root"] = 0
    return graph


def test_rotate_graph_about_root_z_preserves_root_and_original() -> None:
    graph = _toy_tree()
    original_positions = {
        node: graph.nodes[node]["pos"].copy() for node in graph.nodes
    }

    rotated = rotate_graph_about_root_z(graph, np.pi / 2.0)

    np.testing.assert_allclose(rotated.nodes[0]["pos"], original_positions[0])
    for node in graph.nodes:
        np.testing.assert_allclose(graph.nodes[node]["pos"], original_positions[node])
        assert rotated.nodes[node]["pos"][2] == pytest.approx(
            original_positions[node][2]
        )


def test_self_rotation_curves_distinguish_raw_and_invariant_metrics() -> None:
    graph = _toy_tree()
    angles = np.asarray([0.0, 90.0, 360.0])

    chamfer = metric_angle_curve(
        graph,
        CHAMFER,
        angles,
        chamfer_spacing=0.5,
        fgw_max_nodes=1_000,
    )
    path = metric_angle_curve(
        graph,
        TMD_PATH_WASSERSTEIN,
        angles,
        chamfer_spacing=0.5,
        fgw_max_nodes=1_000,
    )
    sibling_angle = metric_angle_curve(
        graph,
        DISTRIBUTION_SIBLING_ANGLE_WASSERSTEIN,
        angles,
        chamfer_spacing=0.5,
        fgw_max_nodes=1_000,
    )

    assert chamfer[0] == pytest.approx(0.0, abs=1e-12)
    assert chamfer[1] > 0.0
    assert chamfer[2] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(path, 0.0, atol=1e-12)
    np.testing.assert_allclose(sibling_angle, 0.0, atol=1e-12)


def test_select_rotation_trees_uses_distinct_classes_deterministically() -> None:
    records = tuple(
        TreeRecord(
            tree_id=f"tree-{cell_class}-{index}",
            swc_path=Path(f"tree-{cell_class}-{index}.swc"),
            split="test",
            cell_class=cell_class,
            cell_type=f"class-{cell_class}",
        )
        for cell_class in range(4)
        for index in range(3)
    )

    first = select_rotation_trees(records, 3, seed=7)
    second = select_rotation_trees(tuple(reversed(records)), 3, seed=7)

    assert first == second
    assert len({record.cell_class for record in first}) == 3
