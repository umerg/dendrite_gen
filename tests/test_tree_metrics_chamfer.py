import math

import networkx as nx
import numpy as np

from metrics.chamfer import sample_tree_points, tree_chamfer_distance
from metrics.so2 import rotate_points_about_axis


def _tree() -> nx.Graph:
    graph = nx.Graph()
    positions = {
        "root": np.asarray([0.0, 0.0, 0.0]),
        "a": np.asarray([2.0, 0.0, 0.5]),
        "b": np.asarray([3.0, 1.0, 1.5]),
        "c": np.asarray([-0.5, 1.5, 2.0]),
    }
    for node, pos in positions.items():
        graph.add_node(node, pos=pos)
    graph.add_edges_from([("root", "a"), ("a", "b"), ("root", "c")])
    graph.graph["root"] = "root"
    return graph


def _rotated_tree(graph: nx.Graph, angle: float, axis=(0.0, 0.0, 1.0)) -> nx.Graph:
    out = nx.Graph()
    nodes = list(graph.nodes)
    points = np.stack([graph.nodes[node]["pos"] for node in nodes])
    rotated = rotate_points_about_axis(points, angle, axis)
    for node, pos in zip(nodes, rotated):
        out.add_node(node, pos=pos)
    out.add_edges_from(graph.edges)
    out.graph["root"] = graph.graph["root"]
    return out


def test_so2_quotient_removes_relative_azimuth() -> None:
    tree = _tree()
    rotated = _rotated_tree(tree, 0.83)
    raw = tree_chamfer_distance(tree, rotated, spacing=0.2, quotient_so2=False)
    quotient = tree_chamfer_distance(
        tree,
        rotated,
        spacing=0.2,
        quotient_so2=True,
        grid_size=36,
        refine=True,
    )
    assert raw.value > 0.1
    assert quotient.value < 1e-4


def test_so2_quotient_does_not_remove_axis_tilt() -> None:
    tree = _tree()
    tilted = _rotated_tree(tree, math.pi / 4.0, axis=(1.0, 0.0, 0.0))
    result = tree_chamfer_distance(tree, tilted, spacing=0.2, quotient_so2=True)
    assert result.value > 0.05


def test_sampling_is_stable_to_collinear_degree_two_subdivision() -> None:
    tree = nx.Graph()
    tree.add_node(0, pos=np.asarray([0.0, 0.0, 0.0]))
    tree.add_node(1, pos=np.asarray([2.0, 0.0, 0.0]))
    tree.add_edge(0, 1)
    tree.graph["root"] = 0

    subdivided = nx.Graph()
    subdivided.add_node("r", pos=np.asarray([0.0, 0.0, 0.0]))
    subdivided.add_node("m", pos=np.asarray([0.7, 0.0, 0.0]))
    subdivided.add_node("tip", pos=np.asarray([2.0, 0.0, 0.0]))
    subdivided.add_edges_from([("r", "m"), ("m", "tip")])
    subdivided.graph["root"] = "r"

    points_a = sample_tree_points(tree, spacing=0.25)
    points_b = sample_tree_points(subdivided, spacing=0.25)
    np.testing.assert_allclose(points_a, points_b)
    result = tree_chamfer_distance(tree, subdivided, spacing=0.25)
    assert result.value < 1e-12
