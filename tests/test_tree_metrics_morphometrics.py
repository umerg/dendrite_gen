from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pytest

from metrics.morphometrics import (
    MORPHOMETRIC_FEATURES,
    fit_morphometric_reference,
    fit_shared_sholl_radii,
    morphometric_euclidean_distance,
    morphometric_euclidean_distance_prepared,
    prepare_morphometric_tree,
    tree_morphometric_vector,
)
from metrics.so2 import rotate_points_about_axis
from validation.dist_metrics import (
    MORPHO_KEYS,
    _sholl_radii_from_graphs,
    assemble_morpho_vector,
)


def _branching_tree(*, scale: float = 1.0, offset: float = 0.0) -> nx.Graph:
    graph = nx.Graph()
    positions = {
        0: (0.0, 0.0, 0.0),
        1: (1.0 + offset, 0.2, 1.2),
        2: (-0.8, 0.5 + offset, 1.0),
        3: (1.8, -0.5, 2.4 + offset),
        4: (0.7, 1.1, 2.2),
        5: (-1.6, 0.1, 2.0),
        6: (-0.4, 1.4, 2.5),
    }
    for node, position in positions.items():
        graph.add_node(node, pos=scale * np.asarray(position, dtype=np.float64))
    graph.add_edges_from(
        ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6))
    )
    graph.graph["root"] = 0
    return graph


def _rotate_about_root(graph: nx.Graph, angle: float) -> nx.Graph:
    rotated = graph.copy()
    root = graph.graph["root"]
    root_position = np.asarray(graph.nodes[root]["pos"], dtype=np.float64)
    for node in graph.nodes:
        centered = np.asarray(graph.nodes[node]["pos"], dtype=np.float64) - root_position
        rotated.nodes[node]["pos"] = root_position + rotate_points_about_axis(
            centered[None, :],
            angle,
            (0.0, 0.0, 1.0),
        )[0]
    return rotated


def _path_tree() -> nx.Graph:
    graph = nx.Graph()
    for node in range(3):
        graph.add_node(node, pos=np.asarray([float(node), 0.0, float(node)]))
    graph.add_edges_from(((0, 1), (1, 2)))
    graph.graph["root"] = 0
    return graph


def test_descriptor_matches_intentionally_duplicated_validation_version() -> None:
    graphs = (
        _branching_tree(),
        _branching_tree(scale=1.4),
        _branching_tree(offset=0.3),
    )
    radii = fit_shared_sholl_radii(graphs, n_shells=32)
    validation_radii = _sholl_radii_from_graphs(graphs, 32)

    assert MORPHOMETRIC_FEATURES == MORPHO_KEYS
    assert validation_radii is not None
    np.testing.assert_allclose(radii, validation_radii)
    for graph in graphs:
        np.testing.assert_allclose(
            tree_morphometric_vector(graph),
            assemble_morpho_vector(
                graph,
                uhat=(0.0, 0.0, 1.0),
            ),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            tree_morphometric_vector(
                graph,
                axis=(0.0, 0.0, 1.0),
                sholl_radii=radii,
            ),
            assemble_morpho_vector(
                graph,
                uhat=(0.0, 0.0, 1.0),
                radii=validation_radii,
            ),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )


def test_reference_standardized_distance_is_symmetric_and_so2_invariant() -> None:
    base = _branching_tree()
    altered = _branching_tree(offset=0.35)
    scaled = _branching_tree(scale=1.7)
    reference = fit_morphometric_reference((base, altered, scaled))

    distance = morphometric_euclidean_distance(base, altered, reference=reference)
    reverse = morphometric_euclidean_distance(altered, base, reference=reference)
    rotated = _rotate_about_root(base, 1.234)

    assert distance > 0.0
    assert reverse == pytest.approx(distance)
    assert morphometric_euclidean_distance(base, base, reference=reference) == 0.0
    assert morphometric_euclidean_distance(
        base, rotated, reference=reference
    ) == pytest.approx(0.0, abs=1e-11)
    assert morphometric_euclidean_distance(
        base, scaled, reference=reference
    ) > 0.0

    prepared_base = prepare_morphometric_tree(base, reference)
    prepared_altered = prepare_morphometric_tree(altered, reference)
    assert morphometric_euclidean_distance_prepared(
        prepared_base, prepared_altered
    ) == pytest.approx(float(np.linalg.norm(prepared_base - prepared_altered)))


def test_reference_configuration_is_finite_and_json_serializable() -> None:
    graphs = (
        _branching_tree(),
        _branching_tree(scale=1.4),
        _branching_tree(offset=0.3),
    )
    reference = fit_morphometric_reference(graphs)

    assert len(reference.mean) == len(MORPHOMETRIC_FEATURES)
    assert len(reference.scale) == len(MORPHOMETRIC_FEATURES)
    assert len(reference.sholl_radii) == 32
    assert all(scale > 0.0 for scale in reference.scale)
    json.dumps(reference.configuration, allow_nan=False)


def test_nonfinite_policy_is_explicit_for_degenerate_descriptors() -> None:
    path = _path_tree()
    with pytest.raises(ValueError, match="undefined morphometric components"):
        fit_morphometric_reference((path,))

    reference = fit_morphometric_reference(
        (path,),
        nonfinite_policy="reference_mean",
    )
    prepared = prepare_morphometric_tree(path, reference)
    assert np.all(np.isfinite(prepared))
    np.testing.assert_allclose(prepared, 0.0)
