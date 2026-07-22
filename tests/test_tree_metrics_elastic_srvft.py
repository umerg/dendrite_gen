"""Dependency-free tests for the optional Elastic SRVFT adapter."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np
import pytest

from metrics.adapters import elastic_srvft as adapter


def _asymmetric_tree() -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(
        [
            ("root", {"pos": np.array([4.0, -2.0, 1.0]), "radius": 0.8}),
            ("short", {"pos": np.array([5.2, -1.9, 1.7]), "radius": 0.4}),
            ("long", {"pos": np.array([3.6, -0.3, 2.4])}),
            ("tip_a", {"pos": np.array([6.1, -1.2, 3.0])}),
            ("tip_b", {"pos": np.array([2.9, 0.4, 3.6])}),
        ]
    )
    graph.add_edges_from(
        [
            ("root", "short"),
            ("root", "long"),
            ("short", "tip_a"),
            ("long", "tip_b"),
        ]
    )
    graph.graph["root"] = "root"
    return graph


def _rotate_tree_about_root(graph: nx.Graph, angle_rad: float) -> nx.Graph:
    rotated = copy.deepcopy(graph)
    root = rotated.graph["root"]
    origin = np.asarray(rotated.nodes[root]["pos"], dtype=np.float64)
    rotation = adapter.rotation_matrix_about_axis(angle_rad)
    for node in rotated.nodes:
        relative = np.asarray(rotated.nodes[node]["pos"], dtype=np.float64) - origin
        rotated.nodes[node]["pos"] = origin + rotation @ relative
    return rotated


def _fake_api(
    tmp_path: Path,
    *,
    comp_builder: Callable[[np.ndarray], dict[str, Any]] | None = None,
    mutate_energy_inputs: bool = False,
) -> adapter._ExternalAPI:
    def build(raw: np.ndarray, layers: int) -> dict[str, Any]:
        assert layers == 4
        if comp_builder is not None:
            return comp_builder(raw)
        return {"raw": raw.copy(), "beta_children": []}

    def convert(comp_tree: dict[str, Any]) -> dict[str, Any]:
        points = np.asarray(comp_tree["raw"][:, 2:5], dtype=np.float64).T
        return {
            "q0": points.copy(),
            "q": [],
            "q_children": [],
            "b00_startP": points[:, 0].copy(),
        }

    def energy(
        left: dict[str, Any],
        right: dict[str, Any],
        *,
        lam_m: float,
        lam_s: float,
        lam_p: float,
    ) -> tuple[dict[str, float], None, None]:
        assert (lam_m, lam_s, lam_p) == pytest.approx((0.3, 1.2, 0.4))
        value = float(np.sum((left["q0"] - right["q0"]) ** 2))
        if mutate_energy_inputs:
            left["q0"][:] = 12345.0
            right["q0"][:] = -12345.0
        return {"E": value}, None, None

    return adapter._ExternalAPI(
        compute_distance_energy=energy,
        comp_tree_from_swcdata_rad=build,
        comp_tree_to_qcomp_tree_rad_4layers=convert,
        checkout=tmp_path,
        revision="fake-revision",
    )


def test_graph_conversion_is_root_centered_parent_first_and_relabel_invariant() -> None:
    original = _asymmetric_tree()
    relabeled = nx.relabel_nodes(
        original,
        {"root": 91, "short": 3, "long": 44, "tip_a": 8, "tip_b": 2},
        copy=True,
    )
    # Rebuild in another insertion order so neither labels nor NetworkX order
    # can accidentally become the upstream branch order.
    reordered = nx.Graph()
    for node in reversed(list(relabeled.nodes)):
        reordered.add_node(node, **copy.deepcopy(relabeled.nodes[node]))
    reordered.add_edges_from(reversed(list(relabeled.edges)))
    reordered.graph["root"] = 91

    raw_original, leaves_original, ties_original = adapter._graph_to_external_swc(
        original,
        name="original",
        default_radius=1.25,
    )
    raw_reordered, leaves_reordered, ties_reordered = adapter._graph_to_external_swc(
        reordered,
        name="reordered",
        default_radius=1.25,
    )

    np.testing.assert_allclose(raw_original, raw_reordered)
    np.testing.assert_array_equal(raw_original[:, 0], np.arange(1, 6))
    np.testing.assert_allclose(raw_original[0, 2:5], 0.0)
    assert raw_original[0, 6] == -1
    assert np.all(raw_original[1:, 6] < raw_original[1:, 0])
    assert set(raw_original[:, 5]) == {0.4, 0.8, 1.25}
    assert leaves_original == leaves_reordered == 2
    assert ties_original == ties_reordered == 0


def test_graph_conversion_handles_a_chain_deeper_than_python_recursion_limit() -> None:
    node_count = 2500
    graph = nx.Graph()
    graph.add_nodes_from(
        (node, {"pos": np.array([0.0, 0.0, float(node)])})
        for node in range(node_count)
    )
    graph.add_edges_from((node - 1, node) for node in range(1, node_count))
    graph.graph["root"] = 0

    raw, terminal_leaves, canonical_order_ties = adapter._graph_to_external_swc(
        graph,
        name="deep_chain",
        default_radius=1.0,
    )

    assert raw.shape == (node_count, 7)
    np.testing.assert_array_equal(raw[:, 0], np.arange(1, node_count + 1))
    np.testing.assert_array_equal(raw[1:, 6], np.arange(1, node_count))
    assert terminal_leaves == 1
    assert canonical_order_ties == 0


def test_graph_conversion_rejects_zero_length_edges_before_external_code() -> None:
    graph = nx.Graph()
    graph.add_node("root", pos=np.array([1.0, 2.0, 3.0]))
    graph.add_node("duplicate", pos=np.array([1.0, 2.0, 3.0]))
    graph.add_edge("root", "duplicate")
    graph.graph["root"] = "root"

    with pytest.raises(
        adapter.ElasticSRVFTUnsupportedTree,
        match="zero-length edges",
    ):
        adapter._graph_to_external_swc(
            graph,
            name="duplicate_positions",
            default_radius=1.0,
        )


def test_qtree_rotation_does_not_mutate_or_double_rotate_aliased_branches() -> None:
    root_q = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    child_q = np.array([[2.0, 1.0], [1.0, 0.0], [5.0, 6.0]])
    child = {
        "q0": child_q,
        "q": [],
        "q_children": [],
        "b00_startP": np.array([2.0, 1.0, 5.0]),
    }
    qtree = {
        "q0": root_q,
        "q": [child_q],
        "q_children": [child],
        "b00_startP": np.array([1.0, 0.0, 3.0]),
    }
    original = copy.deepcopy(qtree)
    rotation = adapter.rotation_matrix_about_axis(math.pi / 2.0)

    rotated = adapter._rotate_qtree_copy(qtree, math.pi / 2.0)

    np.testing.assert_allclose(rotated["q0"], rotation @ root_q)
    np.testing.assert_allclose(rotated["q"][0], rotation @ child_q)
    np.testing.assert_allclose(rotated["q_children"][0]["q0"], rotation @ child_q)
    np.testing.assert_allclose(
        rotated["q_children"][0]["b00_startP"],
        rotation @ np.array([2.0, 1.0, 5.0]),
    )
    np.testing.assert_allclose(qtree["q0"], original["q0"])
    np.testing.assert_allclose(qtree["q"][0], original["q"][0])
    np.testing.assert_allclose(qtree["q_children"][0]["q0"], original["q_children"][0]["q0"])


def test_so2_quotient_recovers_z_rotation_with_mutating_upstream_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api = _fake_api(tmp_path, mutate_energy_inputs=True)
    monkeypatch.setattr(adapter, "_load_external_api", lambda _checkout=None: api)
    tree = _asymmetric_tree()
    rotated = _rotate_tree_about_root(tree, math.pi / 2.0)

    unaligned = adapter.elastic_srvft_distance(
        tree,
        rotated,
        quotient_so2=False,
        lam_m=0.3,
        lam_s=1.2,
        lam_p=0.4,
        grid_size=4,
    )
    aligned = adapter.elastic_srvft_distance(
        tree,
        rotated,
        quotient_so2=True,
        lam_m=0.3,
        lam_s=1.2,
        lam_p=0.4,
        grid_size=4,
        refine=False,
    )

    assert unaligned.value > 1.0
    assert aligned.value == pytest.approx(0.0, abs=1e-12)
    assert aligned.energy_at_zero_rotation == pytest.approx(unaligned.value)
    assert aligned.angle_rad == pytest.approx(3.0 * math.pi / 2.0)
    assert aligned.objective_evaluations == 4
    assert aligned.reverse_energy is None
    assert aligned.external_revision == "fake-revision"
    assert aligned.radius_used_in_energy is False
    assert aligned.tree_a_nodes == aligned.tree_b_nodes == 5
    assert aligned.tree_a_terminal_leaves == aligned.tree_b_terminal_leaves == 2


def test_so2_quotient_does_not_remove_tilts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api = _fake_api(tmp_path)
    monkeypatch.setattr(adapter, "_load_external_api", lambda _checkout=None: api)
    tree = _asymmetric_tree()
    tilted = copy.deepcopy(tree)
    origin = np.asarray(tilted.nodes[tilted.graph["root"]]["pos"])
    tilt = adapter.rotation_matrix_about_axis(math.pi / 2.0, (1.0, 0.0, 0.0))
    for node in tilted.nodes:
        relative = np.asarray(tilted.nodes[node]["pos"]) - origin
        tilted.nodes[node]["pos"] = origin + tilt @ relative

    result = adapter.elastic_srvft_distance(
        tree,
        tilted,
        quotient_so2=True,
        lam_m=0.3,
        lam_s=1.2,
        lam_p=0.4,
        grid_size=24,
        refine=False,
    )

    assert result.value > 1.0
    assert result.quotient_so2 is True


def test_energy_is_invariant_to_joint_allowed_rotation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api = _fake_api(tmp_path)
    monkeypatch.setattr(adapter, "_load_external_api", lambda _checkout=None: api)
    tree_a = _asymmetric_tree()
    tree_b = copy.deepcopy(tree_a)
    tree_b.nodes["tip_a"]["pos"] += np.array([0.25, -0.4, 0.3])

    original = adapter.elastic_srvft_distance(
        tree_a,
        tree_b,
        quotient_so2=False,
        lam_m=0.3,
        lam_s=1.2,
        lam_p=0.4,
    )
    jointly_rotated = adapter.elastic_srvft_distance(
        _rotate_tree_about_root(tree_a, 0.73),
        _rotate_tree_about_root(tree_b, 0.73),
        quotient_so2=False,
        lam_m=0.3,
        lam_s=1.2,
        lam_p=0.4,
    )

    assert jointly_rotated.value == pytest.approx(original.value, abs=1e-12)


def test_so2_quotient_does_not_include_reflections(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api = _fake_api(tmp_path)
    monkeypatch.setattr(adapter, "_load_external_api", lambda _checkout=None: api)
    tree = _asymmetric_tree()
    reflected = copy.deepcopy(tree)
    origin = np.asarray(reflected.nodes[reflected.graph["root"]]["pos"])
    reflection = np.diag([-1.0, 1.0, 1.0])
    for node in reflected.nodes:
        relative = np.asarray(reflected.nodes[node]["pos"]) - origin
        reflected.nodes[node]["pos"] = origin + reflection @ relative

    result = adapter.elastic_srvft_distance(
        tree,
        reflected,
        quotient_so2=True,
        lam_m=0.3,
        lam_s=1.2,
        lam_p=0.4,
        grid_size=72,
        refine=False,
    )

    assert result.value > 0.1


def test_four_layer_truncation_raises_by_default_and_warns_when_requested(
    tmp_path: Path,
) -> None:
    def truncated_comp(raw: np.ndarray) -> dict[str, Any]:
        third = {"K_sideNum": 2, "beta_children": []}
        second = {"K_sideNum": 1, "beta_children": [third]}
        first = {"K_sideNum": 1, "beta_children": [second]}
        return {"raw": raw.copy(), "beta_children": [first]}

    api = _fake_api(tmp_path, comp_builder=truncated_comp)
    tree = _asymmetric_tree()

    with pytest.raises(adapter.ElasticSRVFTUnsupportedTree, match="silently omitted"):
        adapter._prepare_external_tree(
            tree,
            api,
            name="tree",
            default_radius=1.0,
            depth_policy="raise",
        )

    with pytest.warns(RuntimeWarning, match="fixed four branch layers"):
        prepared = adapter._prepare_external_tree(
            tree,
            api,
            name="tree",
            default_radius=1.0,
            depth_policy="warn",
        )
    assert prepared.represented_branch_count == 4
    assert prepared.omitted_frontier_branches == 2


def test_public_tree_diagnostics_supports_non_truncating_screening(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def truncated_comp(raw: np.ndarray) -> dict[str, Any]:
        third = {"K_sideNum": 2, "beta_children": []}
        second = {"K_sideNum": 1, "beta_children": [third]}
        first = {"K_sideNum": 1, "beta_children": [second]}
        return {"raw": raw.copy(), "beta_children": [first]}

    api = _fake_api(tmp_path, comp_builder=truncated_comp)
    monkeypatch.setattr(adapter, "_load_external_api", lambda _checkout=None: api)

    diagnostics = adapter.elastic_srvft_tree_diagnostics(
        _asymmetric_tree(), depth_policy="allow"
    )

    assert diagnostics.node_count == 5
    assert diagnostics.terminal_leaf_count == 2
    assert diagnostics.represented_branch_count == 4
    assert diagnostics.omitted_frontier_branches == 2
    assert diagnostics.depth_policy == "allow"
    assert diagnostics.external_revision == "fake-revision"
    assert diagnostics.external_checkout == str(tmp_path)


def test_missing_checkout_has_actionable_error(tmp_path: Path) -> None:
    missing = tmp_path / "not-cloned"

    with pytest.raises(
        adapter.ElasticSRVFTNotConfigured,
        match=r"Clone the repository.*or pass checkout",
    ):
        adapter._checkout_path(missing)


def test_invalid_options_fail_before_loading_external_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_loaded(_checkout: object = None) -> adapter._ExternalAPI:
        raise AssertionError("external checkout should not be loaded")

    monkeypatch.setattr(adapter, "_load_external_api", fail_if_loaded)
    with pytest.raises(ValueError, match="lam_m"):
        adapter.elastic_srvft_distance(
            _asymmetric_tree(),
            _asymmetric_tree(),
            lam_m=-1.0,
        )
