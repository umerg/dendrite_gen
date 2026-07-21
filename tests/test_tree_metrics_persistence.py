from __future__ import annotations

import copy

import networkx as nx
import numpy as np
import pytest

from metrics.persistence import compute_tmd_diagrams, tmd_persistence_distances
from visualization.tmd.distances import persistence_diagram_wasserstein_distance


class _Diagram:
    def __init__(self, pairs: np.ndarray) -> None:
        self._pairs = pairs

    def as_pairs(self) -> np.ndarray:
        return self._pairs


def _tree(*, second_leaf_scale: float = 1.0) -> nx.Graph:
    tree = nx.Graph()
    tree.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    tree.add_node(1, pos=np.array([1.0, 0.0, 1.0]))
    tree.add_node(
        2,
        pos=np.array([0.0, 2.0 * second_leaf_scale, 3.0 * second_leaf_scale]),
    )
    tree.add_edges_from([(0, 1), (0, 2)])
    tree.graph["root"] = 0
    return tree


def _rotate_about_z(tree: nx.Graph, angle: float) -> nx.Graph:
    rotated = copy.deepcopy(tree)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    for node in rotated.nodes:
        rotated.nodes[node]["pos"] = rotation @ rotated.nodes[node]["pos"]
    return rotated


def test_diagram_distance_accepts_arrays_and_as_pairs_objects() -> None:
    pairs_a = np.array([[0.0, 2.0], [1.0, 4.0]])
    pairs_b = np.array([[0.5, 2.5], [1.0, 4.0]])

    array_distance = persistence_diagram_wasserstein_distance(pairs_a, pairs_b)
    object_distance = persistence_diagram_wasserstein_distance(
        _Diagram(pairs_a), _Diagram(pairs_b)
    )

    assert array_distance == pytest.approx(np.sqrt(0.5))
    assert object_distance == pytest.approx(array_distance)


def test_diagram_distance_supports_ground_norm_and_wasserstein_order() -> None:
    pairs = np.array([[0.0, 2.0], [1.0, 3.0]])
    empty = np.zeros((0, 2))

    euclidean = persistence_diagram_wasserstein_distance(
        pairs, empty, order=2, ground_norm="euclidean"
    )
    chebyshev = persistence_diagram_wasserstein_distance(
        pairs, empty, order=2, ground_norm="chebyshev"
    )

    assert euclidean == pytest.approx(2.0)
    assert chebyshev == pytest.approx(np.sqrt(2.0))


def test_diagram_distance_canonicalizes_and_filters_pairs() -> None:
    clean = np.array([[0.0, 2.0]])
    noisy = np.array(
        [[2.0, 0.0], [1.0, 1.0], [np.nan, 3.0], [0.0, np.inf]]
    )

    with pytest.raises(ValueError, match="non-finite"):
        persistence_diagram_wasserstein_distance(clean, noisy)
    assert persistence_diagram_wasserstein_distance(
        clean, noisy, nonfinite_policy="drop"
    ) == pytest.approx(0.0)


def test_diagram_distance_rejects_unknown_nonfinite_policy() -> None:
    with pytest.raises(ValueError, match="nonfinite_policy"):
        persistence_diagram_wasserstein_distance(
            np.zeros((0, 2)),
            np.zeros((0, 2)),
            nonfinite_policy="invent",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("order", [0, -1, np.inf, np.nan])
def test_diagram_distance_rejects_invalid_order(order: float) -> None:
    with pytest.raises(ValueError, match="order"):
        persistence_diagram_wasserstein_distance(
            np.zeros((0, 2)), np.zeros((0, 2)), order=order
        )


def test_diagram_distance_rejects_unknown_ground_norm() -> None:
    with pytest.raises(ValueError, match="ground_norm"):
        persistence_diagram_wasserstein_distance(
            np.zeros((0, 2)),
            np.zeros((0, 2)),
            ground_norm="manhattan",  # type: ignore[arg-type]
        )


def test_tmd_wrapper_returns_a_distance_per_filtration() -> None:
    distances = tmd_persistence_distances(
        _tree(),
        _tree(second_leaf_scale=1.5),
        normalize_mode="none",
    )

    assert list(distances) == ["path", "height", "rho"]
    assert all(np.isfinite(value) and value >= 0.0 for value in distances.values())
    assert any(value > 0.0 for value in distances.values())


def test_tmd_wrapper_is_invariant_to_rotation_about_z() -> None:
    tree = _tree()
    rotated = _rotate_about_z(tree, angle=0.73)

    distances = tmd_persistence_distances(
        tree,
        rotated,
        normalize_mode="none",
        ground_norm="chebyshev",
    )

    assert distances == pytest.approx({"path": 0.0, "height": 0.0, "rho": 0.0})


def test_tmd_wrapper_root_centers_programmatic_inputs() -> None:
    tree = _tree()
    translated = copy.deepcopy(tree)
    translation = np.array([7.0, -4.0, 11.0])
    for node in translated.nodes:
        translated.nodes[node]["pos"] = translated.nodes[node]["pos"] + translation

    distances = tmd_persistence_distances(
        tree,
        translated,
        normalize_mode="none",
    )

    assert distances == pytest.approx({"path": 0.0, "height": 0.0, "rho": 0.0})


def test_tmd_wrapper_rejects_empty_graphs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compute_tmd_diagrams(nx.Graph(), normalize_mode="none")


def test_tmd_wrapper_requires_explicit_normalization() -> None:
    with pytest.raises(TypeError, match="normalize_mode"):
        compute_tmd_diagrams(_tree())  # type: ignore[call-arg]
