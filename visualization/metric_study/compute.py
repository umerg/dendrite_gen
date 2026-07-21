"""Metric-agnostic pairwise distance-matrix computation."""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

import networkx as nx
import numpy as np

from .metric_registry import PairwiseDissimilarity, get_metric_variant


def _checked_value(
    value: object,
    *,
    metric_name: str,
    pair: tuple[int, int],
) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Metric {metric_name!r} returned a non-scalar value for pair {pair}."
        ) from exc
    if not math.isfinite(scalar):
        raise ValueError(
            f"Metric {metric_name!r} returned a non-finite value for pair {pair}."
        )
    if scalar < 0.0:
        raise ValueError(
            f"Metric {metric_name!r} returned a negative value for pair {pair}."
        )
    return scalar


def compute_symmetric_distance_matrix(
    graphs: Iterable[nx.Graph],
    metric: PairwiseDissimilarity[Any],
) -> np.ndarray:
    """Prepare every graph once and compute a symmetric distance matrix.

    The diagonal is evaluated rather than filled with an assumed zero, keeping
    identity failures visible during a metric study.  Only the upper triangle
    is evaluated; values are mirrored into the lower triangle.
    """
    if not metric.symmetric:
        raise ValueError(
            f"Metric {metric.name!r} is not declared symmetric and cannot be "
            "used by the symmetric distance-matrix engine."
        )

    graph_list = list(graphs)
    prepared = [metric.prepare(graph) for graph in graph_list]
    count = len(prepared)
    distances = np.empty((count, count), dtype=np.float64)

    for index_a, prepared_a in enumerate(prepared):
        for index_b in range(index_a, count):
            value = _checked_value(
                metric.compare(prepared_a, prepared[index_b]),
                metric_name=metric.name,
                pair=(index_a, index_b),
            )
            distances[index_a, index_b] = value
            distances[index_b, index_a] = value

    return distances


def compute_registered_distance_matrix(
    graphs: Iterable[nx.Graph],
    metric_name: str,
) -> np.ndarray:
    """Resolve a registered variant and compute its symmetric matrix."""
    return compute_symmetric_distance_matrix(
        graphs,
        get_metric_variant(metric_name),
    )


__all__ = [
    "compute_registered_distance_matrix",
    "compute_symmetric_distance_matrix",
]
