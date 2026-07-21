from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from visualization.metric_study.compute import (
    compute_registered_distance_matrix,
    compute_symmetric_distance_matrix,
)
from visualization.metric_study.metric_registry import (
    TMD_PATH_WASSERSTEIN,
    available_metric_variants,
    get_metric_variant,
)


def _tree(scale: float = 1.0) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node("root", pos=np.array([0.0, 0.0, 0.0]))
    graph.add_node("a", pos=np.array([scale, 0.0, scale]))
    graph.add_node("b", pos=np.array([0.0, 2.0 * scale, 3.0 * scale]))
    graph.add_edges_from([("root", "a"), ("root", "b")])
    graph.graph["root"] = "root"
    return graph


class _CountingMetric:
    name = "counting_absolute_difference"
    symmetric = True

    def __init__(self) -> None:
        self.prepare_calls: list[int] = []
        self.compare_calls: list[tuple[float, float]] = []

    def prepare(self, graph: nx.Graph) -> float:
        value = int(graph.graph["value"])
        self.prepare_calls.append(value)
        return float(value)

    def compare(self, prepared_a: float, prepared_b: float) -> float:
        self.compare_calls.append((prepared_a, prepared_b))
        return abs(prepared_a - prepared_b)


def _value_graph(value: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(0)
    graph.graph["value"] = value
    return graph


def test_matrix_engine_prepares_once_and_only_evaluates_upper_triangle() -> None:
    metric = _CountingMetric()
    graphs = [_value_graph(1), _value_graph(4), _value_graph(9)]

    matrix = compute_symmetric_distance_matrix(graphs, metric)

    np.testing.assert_allclose(
        matrix,
        np.array(
            [
                [0.0, 3.0, 8.0],
                [3.0, 0.0, 5.0],
                [8.0, 5.0, 0.0],
            ]
        ),
    )
    assert metric.prepare_calls == [1, 4, 9]
    assert metric.compare_calls == [
        (1.0, 1.0),
        (1.0, 4.0),
        (1.0, 9.0),
        (4.0, 4.0),
        (4.0, 9.0),
        (9.0, 9.0),
    ]


def test_matrix_engine_handles_an_empty_collection() -> None:
    metric = _CountingMetric()

    matrix = compute_symmetric_distance_matrix([], metric)

    assert matrix.shape == (0, 0)
    assert metric.prepare_calls == []
    assert metric.compare_calls == []


def test_matrix_engine_rejects_asymmetric_and_invalid_results() -> None:
    asymmetric = _CountingMetric()
    asymmetric.symmetric = False
    with pytest.raises(ValueError, match="not declared symmetric"):
        compute_symmetric_distance_matrix([_value_graph(1)], asymmetric)

    invalid = _CountingMetric()
    invalid.compare = lambda _a, _b: float("nan")  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="non-finite.*pair \\(0, 0\\)"):
        compute_symmetric_distance_matrix([_value_graph(1)], invalid)


def test_path_tmd_variant_is_registered_and_produces_a_symmetric_matrix() -> None:
    assert TMD_PATH_WASSERSTEIN in available_metric_variants()
    metric = get_metric_variant(TMD_PATH_WASSERSTEIN)
    assert metric.name == TMD_PATH_WASSERSTEIN
    assert metric.configuration == {
        "filtration": "path",
        "normalize_mode": "none",
        "wasserstein_order": 1.0,
        "ground_norm": "chebyshev",
        "weight_edges_by_euclidean": True,
        "simplify_to_critical_tree": True,
    }

    matrix = compute_registered_distance_matrix(
        [_tree(scale=1.0), _tree(scale=1.5)],
        TMD_PATH_WASSERSTEIN,
    )

    assert matrix.shape == (2, 2)
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-12)
    assert matrix[0, 1] > 0.0
