"""Unit tests for validation.dist_metrics.compute_distribution_metrics."""

import numpy as np
import networkx as nx
import pytest

from validation.dist_metrics import compute_distribution_metrics
from validation.plot import align_uhat_to_z, _rotation_align, plot_graph_grid_angles


def _toy_tree(scale: float = 1.0, seed: int = 0) -> nx.Graph:
    """A small rooted binary-ish tree with 3D positions."""
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.graph["root"] = 0
    jitter = lambda: rng.normal(0, 0.05, size=3)
    G.add_node(1, pos=(np.array([1.0, 1.0, 1.0]) + jitter()) * scale)
    G.add_node(2, pos=(np.array([-1.0, 1.0, 1.0]) + jitter()) * scale)
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_node(3, pos=(np.array([2.0, 2.0, 3.0]) + jitter()) * scale)
    G.add_node(4, pos=(np.array([0.0, 2.0, 3.0]) + jitter()) * scale)
    G.add_edge(1, 3)
    G.add_edge(1, 4)
    return G


def test_returns_expected_keys_and_finite():
    gen = [_toy_tree(1.0, i) for i in range(4)]
    gt = [_toy_tree(1.1, i + 100) for i in range(5)]
    m = compute_distribution_metrics(gen, gt)

    expected = {
        # pooled marginals (W1)
        "branch_length_w1",
        "bifurcation_angle_w1",
        "tmd_barlen_w1",
        "path_to_root_w1",
        "radial_to_root_w1",
        "contraction_w1",
        "branch_order_w1",
        # per-tree marginals (W1)
        "node_count_w1",
        "leaf_count_w1",
        "bifurcation_count_w1",
        "axial_extent_w1",
        "radial_span_w1",
        "total_extent_w1",
        "strahler_w1",
        "partition_asymmetry_w1",
        "sholl_peak_w1",
        "sholl_critical_radius_w1",
        "sholl_auc_w1",
        # joint distribution
        "mmd_morpho",
        "density_morpho",
        "coverage_morpho",
        "mmd_tmd",
        "density_tmd",
        "coverage_tmd",
        # topology
        "tree_edit_dist_mean",
        "tree_edit_skipped_frac",
        "tree_edit_n_pairs",
    }
    assert expected.issubset(set(m.keys()))
    for k in ("branch_length_w1", "tmd_barlen_w1", "node_count_w1", "contraction_w1", "mmd_morpho"):
        assert np.isfinite(m[k]), f"{k} should be finite, got {m[k]}"


def test_ks_emitted_for_continuous_not_discrete():
    gen = [_toy_tree(1.0, i) for i in range(4)]
    gt = [_toy_tree(1.1, i + 100) for i in range(5)]
    m = compute_distribution_metrics(gen, gt, ged_enabled=False)
    # continuous features get a KS statistic alongside W1
    for k in ("branch_length_ks", "bifurcation_angle_ks", "contraction_ks", "axial_extent_ks"):
        assert k in m, f"missing {k}"
    # discrete (integer/heavily-tied) features are W1-only in-loop
    for k in ("branch_order_ks", "node_count_ks", "leaf_count_ks", "bifurcation_count_ks", "strahler_ks"):
        assert k not in m, f"{k} should not be emitted for a discrete feature"


def test_enable_ks_false_omits_ks_keys():
    gen = [_toy_tree(1.0, i) for i in range(4)]
    gt = [_toy_tree(1.1, i + 100) for i in range(5)]
    m = compute_distribution_metrics(gen, gt, ged_enabled=False, enable_ks=False)
    assert not any(k.endswith("_ks") for k in m), [k for k in m if k.endswith("_ks")]


def test_enable_morphometrics_false_omits_new_marginals():
    gen = [_toy_tree(1.0, i) for i in range(4)]
    gt = [_toy_tree(1.1, i + 100) for i in range(5)]
    m = compute_distribution_metrics(gen, gt, ged_enabled=False, enable_morphometrics=False)
    for k in ("contraction_w1", "strahler_w1", "sholl_peak_w1", "path_to_root_w1"):
        assert k not in m, f"{k} should be omitted when morphometrics disabled"
    assert "branch_length_w1" in m  # base marginals stay


def test_enable_light_joint_false_omits_joint_keys():
    gen = [_toy_tree(1.0, i) for i in range(4)]
    gt = [_toy_tree(1.1, i + 100) for i in range(5)]
    m = compute_distribution_metrics(gen, gt, ged_enabled=False, enable_light_joint=False)
    for k in ("mmd_morpho", "mmd_tmd", "density_morpho", "coverage_tmd"):
        assert k not in m


def test_joint_mmd_larger_for_shifted_than_matched():
    # GT spans a range of sizes so the standardization has real variance; a set drawn
    # from the same range should give MMD ~ 0, a shifted set clearly more.
    scales = np.linspace(0.8, 1.2, 10)
    gt = [_toy_tree(float(s), i) for i, s in enumerate(scales)]
    matched = [_toy_tree(float(s), i + 100) for i, s in enumerate(scales)]
    shifted = [_toy_tree(float(s) + 1.0, i + 200) for i, s in enumerate(scales)]
    m_matched = compute_distribution_metrics(matched, gt, ged_enabled=False)["mmd_morpho"]
    m_shifted = compute_distribution_metrics(shifted, gt, ged_enabled=False)["mmd_morpho"]
    assert m_shifted > m_matched, (m_matched, m_shifted)


def test_identical_sets_give_zero_w1():
    gt = [_toy_tree(1.0, i) for i in range(5)]
    m = compute_distribution_metrics(gt, gt)
    for k, v in m.items():
        if k.endswith("_w1"):
            assert abs(v) < 1e-6, f"{k} should be ~0 for identical sets, got {v}"
    assert m["tree_edit_dist_mean"] == 0.0


def test_larger_scale_gap_increases_branch_length_w1():
    gt = [_toy_tree(1.0, i) for i in range(5)]
    close = [_toy_tree(1.05, i + 10) for i in range(5)]
    far = [_toy_tree(3.0, i + 20) for i in range(5)]
    m_close = compute_distribution_metrics(close, gt)["branch_length_w1"]
    m_far = compute_distribution_metrics(far, gt)["branch_length_w1"]
    assert m_far > m_close


def test_ged_disabled_skips_tree_edit_keys():
    gen = [_toy_tree(1.0, i) for i in range(3)]
    gt = [_toy_tree(1.1, i + 50) for i in range(3)]
    m = compute_distribution_metrics(gen, gt, ged_enabled=False)
    assert "tree_edit_dist_mean" not in m
    assert "branch_length_w1" in m


def test_empty_inputs_do_not_crash():
    m = compute_distribution_metrics([], [])
    assert np.isnan(m["branch_length_w1"])


# --- plotting helpers -----------------------------------------------------------------


def test_rotation_aligns_uhat_to_z():
    for axis in ([0, 1, 0], [1, 0, 0], [0, 0, -1], [1, 1, 1]):
        a = np.asarray(axis, dtype=float)
        R = _rotation_align(a, np.array([0.0, 0.0, 1.0]))
        out = R @ (a / np.linalg.norm(a))
        assert np.allclose(out, [0, 0, 1], atol=1e-6)


def test_align_uhat_does_not_mutate_original():
    G = _toy_tree(1.0, 0)
    orig = G.nodes[1]["pos"].copy()
    H = align_uhat_to_z(G, np.array([0.0, 1.0, 0.0]))
    assert np.allclose(G.nodes[1]["pos"], orig)  # untouched
    assert H.graph["root"] == 0
    # uhat=y -> y component maps onto z
    assert abs(H.nodes[1]["pos"][2] - orig[1]) < 1e-6


def test_grid_figure_has_expected_axes(tmp_path):
    import matplotlib.pyplot as plt

    graphs = [_toy_tree(1.0, i) for i in range(3)]
    angles = [(20, 30), (20, 120)]
    fig, path = plot_graph_grid_angles(
        graphs, out_dir=tmp_path, stem="s", file_tag="gen3d",
        angles=angles, uhat=np.array([0.0, 0.0, 1.0]),
    )
    assert len(fig.axes) == len(graphs) * len(angles)
    assert path.exists()
    plt.close("all")


# --- uhat-frame extent metrics --------------------------------------------------------


def _rotate_about_axis(G: nx.Graph, axis: np.ndarray, angle: float) -> nx.Graph:
    """Return a copy of G with positions rotated by `angle` (rad) about `axis` (Rodrigues)."""
    u = np.asarray(axis, dtype=float).reshape(3)
    u = u / np.linalg.norm(u)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]], dtype=float)
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    H = G.copy()
    for n in H.nodes():
        H.nodes[n]["pos"] = R @ np.asarray(H.nodes[n]["pos"], dtype=float).reshape(3)
    H.graph["root"] = G.graph.get("root")
    return H


def _scale_along(G: nx.Graph, axis: np.ndarray, factor: float) -> nx.Graph:
    """Stretch positions by `factor` along `axis`, leaving the perpendicular plane fixed."""
    u = np.asarray(axis, dtype=float).reshape(3)
    u = u / np.linalg.norm(u)
    H = G.copy()
    for n in H.nodes():
        p = np.asarray(H.nodes[n]["pos"], dtype=float).reshape(3)
        comp = (p @ u) * u
        H.nodes[n]["pos"] = p + (factor - 1.0) * comp
    H.graph["root"] = G.graph.get("root")
    return H


def test_extent_metrics_invariant_to_rotation_about_uhat():
    uhat = np.array([0.0, 1.0, 0.0])  # neuron axis
    gt = [_toy_tree(1.0, i) for i in range(5)]
    # rotate each generated tree by a different angle about uhat
    rotated = [_rotate_about_axis(G, uhat, 0.3 * (i + 1)) for i, G in enumerate(gt)]
    m = compute_distribution_metrics(rotated, gt, uhat=uhat, ged_enabled=False)
    for k in ("axial_extent_w1", "radial_span_w1", "total_extent_w1"):
        assert abs(m[k]) < 1e-6, f"{k} should be invariant to rotation about uhat, got {m[k]}"


def test_axial_and_radial_separate_correctly():
    uhat = np.array([0.0, 1.0, 0.0])
    gt = [_toy_tree(1.0, i) for i in range(5)]

    # stretch ALONG uhat -> axial grows, radial unchanged
    tall = [_scale_along(G, uhat, 3.0) for G in gt]
    m_tall = compute_distribution_metrics(tall, gt, uhat=uhat, ged_enabled=False)
    assert m_tall["axial_extent_w1"] > 1e-3
    assert m_tall["radial_span_w1"] < 1e-6

    # stretch the WHOLE cloud isotropically in the perpendicular plane (x and z)
    # -> radial grows, axial (y) unchanged
    perp = np.array([1.0, 0.0, 0.0])
    wide = [_scale_along(_scale_along(G, perp, 3.0), np.array([0.0, 0.0, 1.0]), 3.0) for G in gt]
    m_wide = compute_distribution_metrics(wide, gt, uhat=uhat, ged_enabled=False)
    assert m_wide["radial_span_w1"] > 1e-3
    assert m_wide["axial_extent_w1"] < 1e-6


# --- new morphometric extractors ------------------------------------------------------


def test_strahler_and_asymmetry_on_known_tree():
    from validation.structural_metrics import strahler_number, partition_asymmetry

    # _toy_tree: root 0 -> {1,2}; 1 -> {3,4}; leaves {2,3,4}.
    G = _toy_tree(1.0, 0)
    # Subtree leaf counts: leaves(3)=leaves(4)=leaves(2)=1, leaves(1)=2, leaves(0)=3.
    # Strahler: node1 has two order-1 children -> order 2; node0 children orders {2,1}
    # -> max=2, not tied -> order 2.
    assert strahler_number(G) == 2.0
    # Asymmetry: node0 partition (2,1) -> |2-1|/(2+1-2)=1; node1 partition (1,1) -> 0.
    # mean over the two branch points = 0.5.
    assert abs(partition_asymmetry(G) - 0.5) < 1e-9


def test_contraction_ratio_in_unit_interval():
    from validation.structural_metrics import contraction_ratio_values

    vals = contraction_ratio_values(_toy_tree(1.0, 3))
    assert vals.size > 0
    assert np.all(vals > 0.0) and np.all(vals <= 1.0 + 1e-9)


def test_sholl_summary_keys_and_peak_positive():
    from validation.structural_metrics import sholl_summary

    s = sholl_summary(_toy_tree(1.0, 1))
    assert set(s.keys()) == {"sholl_peak", "sholl_critical_radius", "sholl_auc"}
    assert s["sholl_peak"] >= 1.0


def test_degenerate_single_node_is_nan_safe():
    from validation.structural_metrics import (
        strahler_number,
        partition_asymmetry,
        contraction_ratio_values,
        path_length_to_root_values,
        sholl_summary,
    )

    G = nx.Graph()
    G.add_node(0, pos=np.zeros(3))
    G.graph["root"] = 0
    # per-tree scalars: strahler defined (=1), asymmetry undefined (nan)
    assert strahler_number(G) == 1.0
    assert np.isnan(partition_asymmetry(G))
    # pooled extractors return empty arrays, not exceptions
    assert contraction_ratio_values(G).size == 0
    assert path_length_to_root_values(G).size == 0
    assert all(np.isnan(v) for v in sholl_summary(G).values())
    # and the full pipeline tolerates a degenerate graph in the set
    m = compute_distribution_metrics([G], [_toy_tree(1.0, 0)], ged_enabled=False)
    assert "branch_length_w1" in m
