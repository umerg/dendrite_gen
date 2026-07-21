"""Coordinate-frame helpers shared by neuron metric studies."""

from __future__ import annotations

import networkx as nx
import numpy as np


SCIENTIFIC_AXIS = (0.0, 1.0, 0.0)
INTERNAL_AXIS = (0.0, 0.0, 1.0)
SCIENTIFIC_Y_TO_INTERNAL_Z = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def transform_scientific_y_to_internal_z(graph: nx.Graph) -> nx.Graph:
    """Return a copy in the z-axis frame assumed by the current metric APIs."""

    transformed = graph.copy()
    for node in transformed.nodes:
        position = np.asarray(transformed.nodes[node].get("pos"), dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"Node {node!r} must have a finite 3-D 'pos' attribute.")
        transformed.nodes[node]["pos"] = np.dot(
            SCIENTIFIC_Y_TO_INTERNAL_Z,
            position,
        )
    return transformed


__all__ = [
    "INTERNAL_AXIS",
    "SCIENTIFIC_AXIS",
    "SCIENTIFIC_Y_TO_INTERNAL_Z",
    "transform_scientific_y_to_internal_z",
]
