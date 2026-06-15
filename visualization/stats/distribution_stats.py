"""Within-tree distribution stats computed directly from NetworkX graphs."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from dendrite_gen.validation.structural_metrics import (
    bifurcation_angle_values,
    branch_length_values,
)

from ._graph import branch_order_map, pos_to_xyz, resolve_root, weighted_graph_with_lengths


GRAPH_DISTRIBUTION_KEYS = (
    "branch_length",
    "bifurcation_angle_deg",
    "path_dist",
    "radial_dist",
    "branch_order",
)


def graph_distribution_values(graph: nx.Graph, metric: str) -> np.ndarray:
    """Return raw within-tree values for one graph-based distribution metric."""
    if graph.number_of_nodes() == 0:
        return np.zeros((0,), dtype=np.float64)

    root = resolve_root(graph)

    if metric == "branch_length":
        return branch_length_values(graph)
    if metric == "bifurcation_angle_deg":
        return bifurcation_angle_values(graph, root=root, degrees=True)

    weighted_graph = weighted_graph_with_lengths(graph)
    if metric == "path_dist":
        path_map = nx.single_source_dijkstra_path_length(
            weighted_graph, root, weight="euclidean_dist"
        )
        values = [float(path_map[n]) for n in graph.nodes() if n != root]
        return np.asarray(values, dtype=np.float64)

    if metric == "radial_dist":
        root_pos = pos_to_xyz(graph.nodes[root].get("pos", np.zeros(3)))
        values = [
            float(np.linalg.norm(pos_to_xyz(graph.nodes[n].get("pos", np.zeros(3))) - root_pos))
            for n in graph.nodes()
            if n != root
        ]
        return np.asarray(values, dtype=np.float64)

    if metric == "branch_order":
        order_map = branch_order_map(graph, root)
        values = [float(order_map[n]) for n in graph.nodes() if n != root]
        return np.asarray(values, dtype=np.float64)

    raise ValueError(f"Unsupported graph distribution key: {metric}")


def graph_distribution_rows(
    graph: nx.Graph,
    *,
    tree_name: str,
    source: str,
    pair_index: int | None = None,
    metrics: tuple[str, ...] = GRAPH_DISTRIBUTION_KEYS,
) -> pd.DataFrame:
    """Return long-form raw distribution values for one graph."""
    rows: list[dict[str, object]] = []
    for metric in metrics:
        values = graph_distribution_values(graph, metric)
        for idx, value in enumerate(values):
            rows.append(
                {
                    "tree_name": tree_name,
                    "source": source,
                    "pair_index": pair_index,
                    "metric": metric,
                    "value": float(value),
                    "value_index": idx,
                }
            )
    return pd.DataFrame(rows)
