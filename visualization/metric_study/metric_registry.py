"""Metric-variant registry for ground-truth tree studies.

The registry is intentionally defined at the level of one scalar
dissimilarity.  A metric family may eventually contribute several variants,
but each registered entry has one fixed configuration and returns one value.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import networkx as nx

try:
    from dendrite_gen.metrics.persistence import compute_tmd_diagrams
    from dendrite_gen.visualization.tmd.distances import (
        persistence_diagram_wasserstein_distance,
    )
except ModuleNotFoundError as exc:
    if exc.name != "dendrite_gen":
        raise
    # Support imports from the repository root, where ``visualization`` and
    # ``metrics`` are top-level packages.
    from metrics.persistence import compute_tmd_diagrams  # type: ignore
    from visualization.tmd.distances import (  # type: ignore
        persistence_diagram_wasserstein_distance,
    )


PreparedTree = TypeVar("PreparedTree")


class PairwiseDissimilarity(Protocol[PreparedTree]):
    """Common interface for one configured scalar tree dissimilarity."""

    name: str
    symmetric: bool

    def prepare(self, graph: nx.Graph) -> PreparedTree:
        """Build the reusable representation of one input tree."""

    def compare(self, prepared_a: PreparedTree, prepared_b: PreparedTree) -> float:
        """Compare two prepared tree representations."""


TMD_PATH_WASSERSTEIN = "tmd_path_wasserstein"


@dataclass(frozen=True)
class TMDPathWasserstein:
    """Wasserstein distance between path-filtration TMD diagrams."""

    name: str = field(default=TMD_PATH_WASSERSTEIN, init=False)
    display_name: str = field(
        default="Path-filtration TMD Wasserstein",
        init=False,
    )
    symmetric: bool = field(default=True, init=False)

    @property
    def configuration(self) -> Mapping[str, object]:
        """Return the fixed scientific choices defining this variant."""
        return {
            "filtration": "path",
            "normalize_mode": "none",
            "wasserstein_order": 1.0,
            "ground_norm": "chebyshev",
            "weight_edges_by_euclidean": True,
            "simplify_to_critical_tree": True,
        }

    def prepare(self, graph: nx.Graph) -> object:
        """Compute the path-filtration diagram once for one tree."""
        return compute_tmd_diagrams(
            graph,
            normalize_mode="none",
            filtrations=("path",),
            weight_edges_by_euclidean=True,
            simplify_to_critical_tree=True,
        )["path"]

    def compare(self, prepared_a: object, prepared_b: object) -> float:
        """Compare two cached path-filtration diagrams."""
        return float(
            persistence_diagram_wasserstein_distance(
                prepared_a,
                prepared_b,
                order=1.0,
                ground_norm="chebyshev",
            )
        )


MetricFactory = Callable[[], PairwiseDissimilarity[Any]]

_METRIC_FACTORIES: dict[str, MetricFactory] = {
    TMD_PATH_WASSERSTEIN: TMDPathWasserstein,
}


def available_metric_variants() -> tuple[str, ...]:
    """Return registered variant names in deterministic order."""
    return tuple(sorted(_METRIC_FACTORIES))


def get_metric_variant(name: str) -> PairwiseDissimilarity[Any]:
    """Construct one registered metric variant by name."""
    try:
        factory = _METRIC_FACTORIES[name]
    except KeyError as exc:
        available = ", ".join(available_metric_variants())
        raise KeyError(
            f"Unknown metric variant {name!r}. Available variants: {available}."
        ) from exc
    return factory()


def register_metric_variant(
    name: str,
    factory: MetricFactory,
    *,
    replace: bool = False,
) -> None:
    """Register a factory for an additional metric variant.

    Registration is explicit so study code can add a variant without changing
    the distance-matrix engine.  Replacing an existing entry requires an
    opt-in to avoid silently changing a named experiment.
    """
    if not name:
        raise ValueError("Metric variant names must be non-empty.")
    if name in _METRIC_FACTORIES and not replace:
        raise ValueError(f"Metric variant {name!r} is already registered.")

    metric = factory()
    if metric.name != name:
        raise ValueError(
            f"Registry key {name!r} does not match metric.name {metric.name!r}."
        )
    _METRIC_FACTORIES[name] = factory


__all__ = [
    "PairwiseDissimilarity",
    "TMD_PATH_WASSERSTEIN",
    "TMDPathWasserstein",
    "available_metric_variants",
    "get_metric_variant",
    "register_metric_variant",
]
