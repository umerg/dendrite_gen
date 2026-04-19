"""Compatibility re-exports for graph-based paper-figure statistics helpers."""

from .stats.distribution_stats import (
    GRAPH_DISTRIBUTION_KEYS,
    graph_distribution_rows,
    graph_distribution_values,
)
from .stats.tree_stats import graph_tree_scalar_row, graph_tree_scalar_stats

__all__ = [
    "GRAPH_DISTRIBUTION_KEYS",
    "graph_distribution_rows",
    "graph_distribution_values",
    "graph_tree_scalar_row",
    "graph_tree_scalar_stats",
]
