"""Internal graph helpers for figure statistics."""

from __future__ import annotations

import networkx as nx
import numpy as np


def pos_to_xyz(pos) -> np.ndarray:
    arr = np.asarray(pos, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size), mode="constant", constant_values=0.0)
    return arr[:3]


def resolve_root(graph: nx.Graph) -> int:
    root = graph.graph.get("root")
    if root in graph.nodes:
        return int(root)
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot resolve root of an empty graph.")

    best_node = None
    best_norm = None
    for nid in graph.nodes:
        pos = pos_to_xyz(graph.nodes[nid].get("pos", np.zeros(3)))
        norm = float(np.linalg.norm(pos))
        if best_norm is None or norm < best_norm:
            best_node = nid
            best_norm = norm
    if best_node is None:
        raise ValueError("Could not resolve graph root.")
    return int(best_node)


def edge_length(graph: nx.Graph, u: int, v: int) -> float:
    pu = pos_to_xyz(graph.nodes[u].get("pos", np.zeros(3)))
    pv = pos_to_xyz(graph.nodes[v].get("pos", np.zeros(3)))
    return float(np.linalg.norm(pu - pv))


def weighted_graph_with_lengths(graph: nx.Graph) -> nx.Graph:
    weighted = graph.copy()
    for u, v in weighted.edges():
        weighted.edges[u, v]["euclidean_dist"] = edge_length(weighted, u, v)
    return weighted


def branch_order_map(graph: nx.Graph, root: int) -> dict[int, int]:
    order = {root: 0}
    stack = [root]
    seen = {root}
    while stack:
        u = stack.pop()
        parent_order = order[u]
        for v in graph.neighbors(u):
            if v in seen:
                continue
            seen.add(v)
            next_order = parent_order + 1 if graph.degree(u) >= 3 and u != root else parent_order
            order[v] = next_order
            stack.append(v)
    return order
