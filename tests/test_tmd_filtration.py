"""Unit tests for the Euclidean-from-root (radial_root) TMD filtration + embedding."""

import numpy as np
import networkx as nx

from utils.tmd_conditioning_utils import filtration_radial_root
from utils.tmd import compute_tmd_embedding


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
