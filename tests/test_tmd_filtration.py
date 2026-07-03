"""Unit tests for the Euclidean-from-root (radial_root) TMD filtration + embedding."""

import numpy as np
import networkx as nx

from utils.tmd_conditioning_utils import (
    filtration_radial_root,
    filtration_height_z,
    filtration_radial_rho,
)
from utils.tmd import compute_tmd_embedding, compute_tmd_mixed, tmd_conditioning_dim
from validation.geometric_metric import height_z_range, span_xy_diameter


def _toy_tree(seed: int = 0) -> nx.Graph:
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.graph["root"] = 0
    G.add_node(1, pos=np.array([1.0, 0.5, 0.2]) + rng.normal(0, 0.02, 3))
    G.add_node(2, pos=np.array([-0.7, 1.0, 0.3]) + rng.normal(0, 0.02, 3))
    G.add_node(3, pos=np.array([2.0, 1.5, 1.0]) + rng.normal(0, 0.02, 3))
    G.add_node(4, pos=np.array([0.5, 2.0, 1.2]) + rng.normal(0, 0.02, 3))
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4)])
    return G


def _rotate_about_root(G: nx.Graph, axis, angle: float) -> nx.Graph:
    u = np.asarray(axis, dtype=float)
    u = u / np.linalg.norm(u)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]], dtype=float)
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    root_pos = np.asarray(G.nodes[G.graph["root"]]["pos"], dtype=float)
    H = G.copy()
    for n in H.nodes():
        p = np.asarray(H.nodes[n]["pos"], dtype=float)
        H.nodes[n]["pos"] = root_pos + R @ (p - root_pos)
    return H


def test_radial_root_values_equal_distance_from_root():
    G = _toy_tree(0)
    f = filtration_radial_root(G)
    root_pos = np.asarray(G.nodes[0]["pos"], dtype=float)
    for n in G.nodes():
        expected = float(np.linalg.norm(np.asarray(G.nodes[n]["pos"], dtype=float) - root_pos))
        assert abs(f[n] - expected) < 1e-9
    assert f[0] == 0.0


def test_embedding_shape_and_determinism():
    G = _toy_tree(1)
    e1 = compute_tmd_embedding(G, n_bins=16)
    e2 = compute_tmd_embedding(G, n_bins=16)
    assert e1.shape == (16 * 16,)
    assert np.allclose(e1, e2)


def test_embedding_rotation_invariant_about_root():
    G = _toy_tree(2)
    H = _rotate_about_root(G, axis=[0.3, 1.0, -0.4], angle=0.9)
    eG = compute_tmd_embedding(G, n_bins=16)
    eH = compute_tmd_embedding(H, n_bins=16)
    # radial_root depends only on distances from the root -> invariant to rotation.
    assert np.allclose(eG, eH, atol=1e-6)


# --- axis-aware filtrations (height / rho key off uhat, not a hardcoded z) ---

def test_height_filtration_axis_backcompat_and_y():
    G = _toy_tree(3)
    h_default = filtration_height_z(G)                      # default z
    h_z = filtration_height_z(G, uhat=(0.0, 0.0, 1.0))
    h_y = filtration_height_z(G, uhat=(0.0, 1.0, 0.0))
    for n in G.nodes():
        p = np.asarray(G.nodes[n]["pos"], dtype=float)
        assert abs(h_default[n] - p[2]) < 1e-9              # back-compat: z coordinate
        assert abs(h_default[n] - h_z[n]) < 1e-12
        assert abs(h_y[n] - p[1]) < 1e-9                    # projection onto y


def test_rho_filtration_axis_backcompat_and_y():
    G = _toy_tree(4)
    rho_z = filtration_radial_rho(G)                        # default z -> sqrt(x^2+y^2)
    rho_y = filtration_radial_rho(G, uhat=(0.0, 1.0, 0.0))  # perp to y -> sqrt(x^2+z^2)
    for n in G.nodes():
        x, y, z = np.asarray(G.nodes[n]["pos"], dtype=float)
        assert abs(rho_z[n] - float(np.hypot(x, y))) < 1e-9
        assert abs(rho_y[n] - float(np.hypot(x, z))) < 1e-9


def test_conditioning_dim_matches_embedding_length():
    G = _toy_tree(5)
    cases = [(("radial_root",), 16), (("path", "height", "radial_root"), 16), (("path", "height", "rho"), 8)]
    for fils, bins in cases:
        e = compute_tmd_mixed(G, filtrations=fils, n_bins=bins)
        assert tmd_conditioning_dim(fils, bins) == len(fils) * bins * bins
        assert e.shape == (tmd_conditioning_dim(fils, bins),)


def _tree_with_axis_variation() -> nx.Graph:
    """Tree whose z- and y-coordinates each have >=2 local minima (so the axis-dependent
    0d filtrations produce non-degenerate, axis-specific barcodes)."""
    G = nx.Graph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.graph["root"] = 0
    G.add_node(1, pos=np.array([1.0, 2.0, -1.0]))
    G.add_node(2, pos=np.array([-1.0, -2.0, 1.0]))
    G.add_node(3, pos=np.array([2.0, 3.0, -0.5]))
    G.add_node(4, pos=np.array([1.5, 1.0, -2.0]))
    G.add_node(5, pos=np.array([-2.0, -3.0, 2.0]))
    G.add_node(6, pos=np.array([-1.5, -1.0, 0.5]))
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    return G


def test_mixed_uhat_affects_only_axis_dependent_filtrations():
    G = _tree_with_axis_variation()
    fils = ("path", "height", "radial_root", "rho")
    b = 16 * 16
    e_z = compute_tmd_mixed(G, filtrations=fils, n_bins=16, uhat=(0.0, 0.0, 1.0))
    e_y = compute_tmd_mixed(G, filtrations=fils, n_bins=16, uhat=(0.0, 1.0, 0.0))
    # path (geodesic) and radial_root (soma distance) are axis-agnostic -> identical across uhat
    assert np.allclose(e_z[:b], e_y[:b], atol=1e-8)               # path channel
    assert np.allclose(e_z[2 * b:3 * b], e_y[2 * b:3 * b], atol=1e-8)  # radial_root channel
    # the conditioning vector as a whole changes with the axis (height/rho channels key off uhat)
    assert not np.allclose(e_z, e_y, atol=1e-6)


def test_geometric_metric_axis_backcompat_and_y():
    rng = np.random.default_rng(7)
    pts = rng.normal(0.0, 1.0, (20, 3))

    def _maxpair(a):
        d = a[:, None, :] - a[None, :, :]
        return float(np.sqrt(np.max(np.sum(d ** 2, axis=-1))))

    # height: default == z-range; explicit z matches; y-axis == y-range
    assert abs(height_z_range(pts) - float(pts[:, 2].max() - pts[:, 2].min())) < 1e-9
    assert abs(height_z_range(pts, uhat=(0.0, 0.0, 1.0)) - height_z_range(pts)) < 1e-12
    assert abs(height_z_range(pts, uhat=(0.0, 1.0, 0.0)) - float(pts[:, 1].max() - pts[:, 1].min())) < 1e-9

    # span: default == max pairwise in XY plane; y-axis == max pairwise in XZ plane
    assert abs(span_xy_diameter(pts) - _maxpair(pts[:, [0, 1]])) < 1e-9
    assert abs(span_xy_diameter(pts, uhat=(0.0, 1.0, 0.0)) - _maxpair(pts[:, [0, 2]])) < 1e-9
