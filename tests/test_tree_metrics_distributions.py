"""Tests for standalone tree-distribution Wasserstein dissimilarities."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from metrics.distributions import (
    CRITICAL_BRANCH_CABLE_LENGTH,
    CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG,
    CRITICAL_NODE_BRANCH_ORDER,
    CRITICAL_NODE_ROOT_PATH_LENGTH,
    DEFAULT_DISTRIBUTIONS,
    UNIFORM_CABLE_HEIGHT_Z,
    UNIFORM_CABLE_RADIAL_XY,
    UNIFORM_CABLE_ROOT_EUCLIDEAN,
    all_default_distribution_wasserstein_distances,
    distribution_wasserstein_distance,
    distribution_wasserstein_result,
    tree_distribution,
)


def _branched_tree() -> nx.Graph:
    graph = nx.Graph()
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([0.0, 0.0, 1.0]),
        2: np.array([0.0, 0.0, 2.0]),
        3: np.array([-1.0, 0.0, 3.0]),
        4: np.array([1.0, 0.0, 3.0]),
    }
    graph.add_nodes_from((node, {"pos": position}) for node, position in positions.items())
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (2, 4)])
    graph.graph["root"] = 0
    return graph


def _chain(end: tuple[float, float, float] = (3.0, 4.0, 12.0)) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node("root", pos=np.zeros(3))
    graph.add_node("tip", pos=np.asarray(end, dtype=np.float64))
    graph.add_edge("root", "tip")
    graph.graph["root"] = "root"
    return graph


def _rotate_translate_and_relabel(graph: nx.Graph, theta: float) -> nx.Graph:
    cosine, sine = np.cos(theta), np.sin(theta)
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    translation = np.array([8.0, -3.0, 5.0])
    mapping = {node: f"renamed-{index}" for index, node in enumerate(graph.nodes)}
    transformed = nx.relabel_nodes(graph, mapping, copy=True)
    for old_node, new_node in mapping.items():
        transformed.nodes[new_node]["pos"] = (
            rotation @ np.asarray(graph.nodes[old_node]["pos"], dtype=float)
            + translation
        )
    transformed.graph["root"] = mapping[graph.graph["root"]]
    return transformed


def test_named_distributions_have_explicit_geometric_definitions() -> None:
    chain = _chain()

    branch_length = tree_distribution(
        chain, CRITICAL_BRANCH_CABLE_LENGTH, sample_spacing=13.0
    )
    root_path = tree_distribution(
        chain, CRITICAL_NODE_ROOT_PATH_LENGTH, sample_spacing=13.0
    )
    branch_order = tree_distribution(
        chain, CRITICAL_NODE_BRANCH_ORDER, sample_spacing=13.0
    )
    radial = tree_distribution(chain, UNIFORM_CABLE_RADIAL_XY, sample_spacing=13.0)
    height = tree_distribution(chain, UNIFORM_CABLE_HEIGHT_Z, sample_spacing=13.0)
    euclidean = tree_distribution(
        chain, UNIFORM_CABLE_ROOT_EUCLIDEAN, sample_spacing=13.0
    )

    np.testing.assert_allclose(branch_length.values, [13.0])
    np.testing.assert_allclose(root_path.values, [13.0])
    np.testing.assert_allclose(branch_order.values, [1.0])
    np.testing.assert_allclose(radial.values, [2.5])
    np.testing.assert_allclose(height.values, [6.0])
    np.testing.assert_allclose(euclidean.values, [6.5])
    np.testing.assert_allclose(radial.weights, [13.0])


def test_critical_branches_collapse_degree_two_paths_and_use_chords() -> None:
    tree = _branched_tree()

    lengths = tree_distribution(tree, CRITICAL_BRANCH_CABLE_LENGTH).values
    angles = tree_distribution(
        tree, CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG
    ).values
    path_lengths = tree_distribution(tree, CRITICAL_NODE_ROOT_PATH_LENGTH).values
    orders = tree_distribution(tree, CRITICAL_NODE_BRANCH_ORDER).values

    np.testing.assert_allclose(np.sort(lengths), np.sort([2.0, np.sqrt(2), np.sqrt(2)]))
    np.testing.assert_allclose(angles, [90.0])
    np.testing.assert_allclose(
        np.sort(path_lengths), np.sort([2.0, 2.0 + np.sqrt(2), 2.0 + np.sqrt(2)])
    )
    np.testing.assert_allclose(np.sort(orders), [1.0, 2.0, 2.0])


def test_all_distances_are_identity_so2_translation_and_relabel_invariant() -> None:
    tree = _branched_tree()
    transformed = _rotate_translate_and_relabel(tree, theta=0.731)

    identity = all_default_distribution_wasserstein_distances(
        tree, tree, sample_spacing=0.37
    )
    transformed_distances = all_default_distribution_wasserstein_distances(
        tree, transformed, sample_spacing=0.37
    )

    assert tuple(identity) == DEFAULT_DISTRIBUTIONS
    for name in DEFAULT_DISTRIBUTIONS:
        assert identity[name] == pytest.approx(0.0, abs=1e-12)
        assert transformed_distances[name] == pytest.approx(0.0, abs=1e-10)


def test_known_shape_change_changes_geometry_distributions() -> None:
    first = _branched_tree()
    second = _branched_tree()
    second.nodes[4]["pos"] = np.array([2.0, 0.0, 3.0])

    for name in (
        CRITICAL_BRANCH_CABLE_LENGTH,
        CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG,
        CRITICAL_NODE_ROOT_PATH_LENGTH,
        UNIFORM_CABLE_RADIAL_XY,
        UNIFORM_CABLE_ROOT_EUCLIDEAN,
    ):
        distance = distribution_wasserstein_distance(
            first, second, name, sample_spacing=0.25
        )
        assert distance > 0.0


def test_empty_distribution_policy_is_explicit() -> None:
    chain = _chain()
    branched = _branched_tree()

    assert distribution_wasserstein_distance(
        chain, chain, CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG
    ) == pytest.approx(0.0)
    assert np.isnan(
        distribution_wasserstein_distance(
            chain, branched, CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG
        )
    )
    both_empty = distribution_wasserstein_result(
        chain, chain, CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG
    )
    one_empty = distribution_wasserstein_result(
        chain, branched, CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG
    )
    assert both_empty.status == "both_empty"
    assert both_empty.value == 0.0
    assert both_empty.empty_a and both_empty.empty_b
    assert one_empty.status == "undefined_one_empty"
    assert np.isnan(one_empty.value)
    assert one_empty.empty_a and not one_empty.empty_b
    with pytest.raises(ValueError, match="empty"):
        distribution_wasserstein_distance(
            chain,
            branched,
            CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG,
            empty_policy="raise",
        )


@pytest.mark.parametrize(
    ("graph", "message"),
    [
        (nx.Graph(), "empty graph"),
        (nx.cycle_graph(3), "connected, acyclic"),
    ],
)
def test_invalid_graphs_are_rejected(graph: nx.Graph, message: str) -> None:
    for node in graph:
        graph.nodes[node]["pos"] = np.zeros(3)
    if graph:
        graph.graph["root"] = next(iter(graph))
    with pytest.raises(ValueError, match=message):
        tree_distribution(graph, CRITICAL_BRANCH_CABLE_LENGTH)


def test_missing_root_and_position_are_rejected() -> None:
    no_root = _chain()
    del no_root.graph["root"]
    with pytest.raises(ValueError, match="valid root"):
        tree_distribution(no_root, CRITICAL_BRANCH_CABLE_LENGTH)

    no_position = _chain()
    del no_position.nodes["tip"]["pos"]
    with pytest.raises(ValueError, match="missing"):
        tree_distribution(no_position, CRITICAL_BRANCH_CABLE_LENGTH)


def test_configuration_validation() -> None:
    tree = _chain()
    with pytest.raises(ValueError, match="sample_spacing"):
        tree_distribution(tree, UNIFORM_CABLE_RADIAL_XY, sample_spacing=0.0)
    with pytest.raises(ValueError, match="empty_policy"):
        distribution_wasserstein_distance(
            tree,
            tree,
            CRITICAL_BRANCH_CABLE_LENGTH,
            empty_policy="unsupported",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="Unknown tree distribution"):
        tree_distribution(tree, "raw_node_radius")
