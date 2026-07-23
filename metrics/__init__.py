"""Standalone tree-to-tree dissimilarities used by metric studies.

The package intentionally keeps plotting and dataset orchestration elsewhere.
Optional third-party backends, such as POT's solver, are loaded only when used.
"""

from .adapters.elastic_srvft import (
    ElasticSRVFTResult,
    elastic_srvft_distance,
)
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
    PreparedFusedGWTree,
    fused_gromov_wasserstein_distance,
    fused_gromov_wasserstein_distance_prepared,
    fused_gw_distance,
    prepare_fused_gw_tree,
)
from .morphometrics import (
    DEFAULT_SHOLL_SHELLS,
    MORPHOMETRIC_FEATURES,
    MorphometricReference,
    fit_morphometric_reference,
    fit_shared_sholl_radii,
    morphometric_euclidean_distance,
    morphometric_euclidean_distance_prepared,
    prepare_morphometric_tree,
    standardize_morphometric_vector,
    tree_morphometric_vector,
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
    "DEFAULT_SHOLL_SHELLS",
    "DistributionWassersteinResult",
    "EmpiricalTreeDistribution",
    "ElasticSRVFTResult",
    "FusedGWResult",
    "MassMode",
    "MORPHOMETRIC_FEATURES",
    "MorphometricReference",
    "PreparedFusedGWTree",
    "SO2Minimum",
    "all_default_distribution_wasserstein_distances",
    "compare_tree_pair",
    "compute_tmd_diagrams",
    "distribution_wasserstein_distance",
    "distribution_wasserstein_result",
    "elastic_srvft_distance",
    "fused_gromov_wasserstein_distance",
    "fused_gromov_wasserstein_distance_prepared",
    "fused_gw_distance",
    "fit_morphometric_reference",
    "fit_shared_sholl_radii",
    "minimize_over_so2",
    "morphometric_euclidean_distance",
    "morphometric_euclidean_distance_prepared",
    "point_chamfer_components",
    "point_chamfer_distance",
    "prepare_fused_gw_tree",
    "prepare_morphometric_tree",
    "rotate_points_about_axis",
    "rotation_matrix_about_axis",
    "sample_tree_points",
    "standardize_morphometric_vector",
    "tmd_persistence_distances",
    "tree_distribution",
    "tree_morphometric_vector",
    "tree_chamfer_distance",
]
