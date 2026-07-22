"""Tree-level scalar stats computed directly from NetworkX graphs."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from dendrite_gen.validation.geometric_metric import (
    bbox_diag_length,
    height_z_range,
    span_xy_diameter,
)
from dendrite_gen.validation.structural_metrics import (
    mean_branch_amplitude,
    mean_branch_length,
)

from ._graph import branch_order_map, pos_to_xyz, resolve_root, weighted_graph_with_lengths


def graph_tree_scalar_stats(graph: nx.Graph, uhat=(0.0, 0.0, 1.0)) -> dict[str, float]:
    """Return a tree-level scalar summary for one rooted graph.

    ``uhat`` is the equivariance/growth axis used for the ``height`` (along-axis) and
    ``span_xy`` (perpendicular-to-axis) scalars; defaults to z for back-compat, pass
    the dataset's ``so2_axis`` (e.g. ``(0,1,0)``) for neurons.
    """
    if graph.number_of_nodes() == 0:
        return {
            "num_nodes": float("nan"),
            "num_edges": float("nan"),
            "num_tips": float("nan"),
            "num_branchpoints": float("nan"),
            "height": float("nan"),
            "span_xy": float("nan"),
            "bbox_diag": float("nan"),
            "max_path_dist": float("nan"),
            "max_radial_dist": float("nan"),
            "mean_branch_length": float("nan"),
            "mean_bifurcation_angle_deg": float("nan"),
            "max_branch_order": float("nan"),
        }

    root = resolve_root(graph)
    pts = np.stack(
        [pos_to_xyz(graph.nodes[n].get("pos", np.zeros(3))) for n in graph.nodes()],
        axis=0,
    )

    weighted_graph = weighted_graph_with_lengths(graph)
    path_lengths = nx.single_source_dijkstra_path_length(
        weighted_graph, root, weight="euclidean_dist"
    )
    root_pos = pos_to_xyz(graph.nodes[root].get("pos", np.zeros(3)))
    radial_dist = [
        float(np.linalg.norm(pos_to_xyz(graph.nodes[n].get("pos", np.zeros(3))) - root_pos))
        for n in graph.nodes()
    ]
    branch_order = branch_order_map(graph, root)

    return {
        "num_nodes": float(graph.number_of_nodes()),
        "num_edges": float(graph.number_of_edges()),
        "num_tips": float(sum(1 for n in graph.nodes() if n != root and graph.degree(n) == 1)),
        "num_branchpoints": float(
            sum(1 for n in graph.nodes() if n != root and graph.degree(n) >= 3)
        ),
        "height": float(height_z_range(pts, uhat=uhat)),
        "span_xy": float(span_xy_diameter(pts, uhat=uhat)),
        "bbox_diag": float(bbox_diag_length(pts)),
        "max_path_dist": float(max(path_lengths.values()) if path_lengths else 0.0),
        "max_radial_dist": float(max(radial_dist) if radial_dist else 0.0),
        "mean_branch_length": float(mean_branch_length(graph)),
        "mean_bifurcation_angle_deg": float(mean_branch_amplitude(graph, root=root, degrees=True)),
        "max_branch_order": float(max(branch_order.values()) if branch_order else 0.0),
    }


def graph_tree_scalar_row(
    graph: nx.Graph,
    *,
    tree_name: str,
    source: str,
    pair_index: int | None = None,
    uhat=(0.0, 0.0, 1.0),
) -> pd.DataFrame:
    """Return one-row dataframe for a graph-level scalar summary."""
    row = graph_tree_scalar_stats(graph, uhat=uhat)
    row["tree_name"] = tree_name
    row["source"] = source
    row["pair_index"] = pair_index
    return pd.DataFrame([row])
