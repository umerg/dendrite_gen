"""Programmatic comparison of one pair of rooted geometric trees.

This is intentionally a small orchestration layer.  Individual metric
implementations remain independently callable and testable in their own
modules, while this function gives the first metric-study milestone one stable
entry point.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal, Sequence

import networkx as nx

from .chamfer import Reduction, tree_chamfer_distance
from .distributions import (
    DEFAULT_DISTRIBUTIONS,
    EmptyPolicy,
    distribution_wasserstein_result,
)
from .fused_gw import FeatureMode, MassMode, fused_gromov_wasserstein_distance
from .persistence import (
    DEFAULT_FILTRATIONS,
    FiltrationName,
    GroundNorm,
    NormalizeMode,
    tmd_persistence_distances,
)


MetricFamily = Literal["chamfer", "persistence", "distributions", "fgw"]
AVAILABLE_METRIC_FAMILIES: tuple[MetricFamily, ...] = (
    "chamfer",
    "persistence",
    "distributions",
    "fgw",
)
DEFAULT_METRIC_FAMILIES: tuple[MetricFamily, ...] = (
    "chamfer",
    "persistence",
    "distributions",
)


def _metric_families(names: Sequence[str]) -> tuple[MetricFamily, ...]:
    supported = set(AVAILABLE_METRIC_FAMILIES)
    unknown = sorted(set(names) - supported)
    if unknown:
        raise ValueError(
            f"Unknown metric families {unknown!r}; choose from "
            f"{list(AVAILABLE_METRIC_FAMILIES)!r}."
        )
    # Deduplicate names; results are emitted in the canonical implementation order.
    return tuple(dict.fromkeys(names))  # type: ignore[return-value]


def compare_tree_pair(
    tree_a: nx.Graph,
    tree_b: nx.Graph,
    *,
    metric_families: Sequence[MetricFamily] = DEFAULT_METRIC_FAMILIES,
    quotient_so2: bool = True,
    so2_grid_size: int = 72,
    so2_refine: bool = True,
    chamfer_spacing: float = 1.0,
    chamfer_squared: bool = False,
    chamfer_reduction: Reduction = "sum",
    persistence_normalize_mode: NormalizeMode = "none",
    persistence_filtrations: Sequence[FiltrationName] = DEFAULT_FILTRATIONS,
    persistence_order: float = 1.0,
    persistence_ground_norm: GroundNorm = "chebyshev",
    distribution_names: Sequence[str] = DEFAULT_DISTRIBUTIONS,
    distribution_spacing: float = 1.0,
    distribution_empty_policy: EmptyPolicy = "nan",
    fgw_feature_mode: FeatureMode = "xyz",
    fgw_alpha: float = 0.5,
    fgw_mass_mode: MassMode = "cable_length",
    fgw_normalize: bool = True,
) -> dict[str, object]:
    """Evaluate selected metric families for exactly one tree pair.

    ``quotient_so2`` forms the relative-rotation quotient for Chamfer and for
    FGW with ``xyz`` node features.  TMD filtrations and the default morphology
    distributions are already invariant to rotations around the preferred
    z-axis.  FGW's optional ``axis`` features ``(z, rho)`` are invariant as
    well, so no redundant angular search is performed in that mode.
    """

    selected = _metric_families(metric_families)
    results: dict[str, object] = {}

    if "chamfer" in selected:
        chamfer = tree_chamfer_distance(
            tree_a,
            tree_b,
            spacing=chamfer_spacing,
            squared=chamfer_squared,
            reduction=chamfer_reduction,
            quotient_so2=quotient_so2,
            grid_size=so2_grid_size,
            refine=so2_refine,
        )
        chamfer_result = asdict(chamfer)
        chamfer_result.update(
            {
                "so2_handling": (
                    "relative_minimum" if quotient_so2 else "absolute_azimuth_retained"
                ),
                "relative_reflection_invariant": False,
            }
        )
        results["chamfer"] = chamfer_result

    if "persistence" in selected:
        distances = tmd_persistence_distances(
            tree_a,
            tree_b,
            normalize_mode=persistence_normalize_mode,
            filtrations=persistence_filtrations,
            order=persistence_order,
            ground_norm=persistence_ground_norm,
        )
        results["persistence"] = {
            "distances": distances,
            "filtrations": list(persistence_filtrations),
            "normalize_mode": persistence_normalize_mode,
            "wasserstein_order": float(persistence_order),
            "ground_norm": persistence_ground_norm,
            "nonfinite_policy": "raise",
            "weight_edges_by_euclidean": True,
            "simplify_to_critical_tree": True,
            "so2_handling": "intrinsically_invariant",
            "relative_reflection_invariant": True,
            "invariance_note": (
                "The selected scalar filtrations discard azimuthal handedness; "
                "path length is invariant to still larger rigid groups."
            ),
        }

    if "distributions" in selected:
        comparisons = {
            name: distribution_wasserstein_result(
                tree_a,
                tree_b,
                name,
                sample_spacing=distribution_spacing,
                empty_policy=distribution_empty_policy,
            )
            for name in distribution_names
        }
        results["distributions"] = {
            "distances": {
                name: comparison.value for name, comparison in comparisons.items()
            },
            "diagnostics": {
                name: {
                    "status": comparison.status,
                    "sample_count_a": comparison.sample_count_a,
                    "sample_count_b": comparison.sample_count_b,
                    "empty_a": comparison.empty_a,
                    "empty_b": comparison.empty_b,
                }
                for name, comparison in comparisons.items()
            },
            "sample_spacing": float(distribution_spacing),
            "empty_policy": distribution_empty_policy,
            "so2_handling": "intrinsically_invariant",
            "relative_reflection_invariant": True,
            "invariance_note": (
                "These summary distributions deliberately discard more than "
                "SO(2), including relative vertical-plane reflections."
            ),
        }

    if "fgw" in selected:
        fgw = fused_gromov_wasserstein_distance(
            tree_a,
            tree_b,
            feature_mode=fgw_feature_mode,
            alpha=fgw_alpha,
            mass_mode=fgw_mass_mode,
            normalize=fgw_normalize,
            quotient_so2=quotient_so2,
            grid_size=so2_grid_size,
            refine=so2_refine,
        )
        fgw_result = asdict(fgw)
        fgw_result.update(
            {
                "so2_handling": (
                    "relative_minimum"
                    if quotient_so2 and fgw_feature_mode == "xyz" and fgw_alpha < 1.0
                    else "intrinsically_invariant_structure_only"
                    if fgw_alpha == 1.0
                    else "intrinsically_invariant"
                    if fgw_feature_mode == "axis"
                    else "absolute_azimuth_retained"
                ),
                "relative_reflection_invariant": bool(
                    fgw_feature_mode == "axis" or fgw_alpha == 1.0
                ),
                "invariance_note": (
                    "At alpha=1 only tree-path structure remains, so no "
                    "angular search is needed."
                    if fgw_alpha == 1.0
                    else "The (z, rho) feature ablation discards azimuthal "
                    "handedness."
                    if fgw_feature_mode == "axis"
                    else "xyz features retain handedness unless the trees "
                    "themselves are symmetric."
                ),
                "structure_cost": "euclidean_weighted_tree_shortest_path",
                "feature_cost": "squared_euclidean",
            }
        )
        results["fgw"] = fgw_result

    return results


__all__ = [
    "AVAILABLE_METRIC_FAMILIES",
    "DEFAULT_METRIC_FAMILIES",
    "MetricFamily",
    "compare_tree_pair",
]
