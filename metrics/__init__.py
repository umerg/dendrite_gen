"""Standalone tree-to-tree dissimilarities used by metric studies.

The package intentionally keeps plotting and dataset orchestration elsewhere.
Optional third-party backends, such as POT's solver, are loaded only when used.
"""

from .chamfer import (
    ChamferResult,
    point_chamfer_components,
    point_chamfer_distance,
    sample_tree_points,
    tree_chamfer_distance,
)
from .distributions import (
    DEFAULT_DISTRIBUTIONS,
    DistributionWassersteinResult,
    EmpiricalTreeDistribution,
    all_default_distribution_wasserstein_distances,
    distribution_wasserstein_distance,
    distribution_wasserstein_result,
    tree_distribution,
)
from .fused_gw import (
    FusedGWResult,
    MassMode,
    fused_gromov_wasserstein_distance,
    fused_gw_distance,
)
from .pair import (
    AVAILABLE_METRIC_FAMILIES,
    DEFAULT_METRIC_FAMILIES,
    compare_tree_pair,
)
from .persistence import (
    DEFAULT_FILTRATIONS,
    compute_tmd_diagrams,
    tmd_persistence_distances,
)
from .so2 import (
    SO2Minimum,
    minimize_over_so2,
    rotate_points_about_axis,
    rotation_matrix_about_axis,
)

__all__ = [
    "ChamferResult",
    "AVAILABLE_METRIC_FAMILIES",
    "DEFAULT_DISTRIBUTIONS",
    "DEFAULT_FILTRATIONS",
    "DEFAULT_METRIC_FAMILIES",
    "DistributionWassersteinResult",
    "EmpiricalTreeDistribution",
    "FusedGWResult",
    "MassMode",
    "SO2Minimum",
    "all_default_distribution_wasserstein_distances",
    "compare_tree_pair",
    "compute_tmd_diagrams",
    "distribution_wasserstein_distance",
    "distribution_wasserstein_result",
    "fused_gromov_wasserstein_distance",
    "fused_gw_distance",
    "minimize_over_so2",
    "point_chamfer_components",
    "point_chamfer_distance",
    "rotate_points_about_axis",
    "rotation_matrix_about_axis",
    "sample_tree_points",
    "tmd_persistence_distances",
    "tree_distribution",
    "tree_chamfer_distance",
]
