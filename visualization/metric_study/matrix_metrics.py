"""Prepared scalar metrics for resumable distance-matrix studies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

try:
    from dendrite_gen.metrics.chamfer import sample_tree_points
    from dendrite_gen.metrics.distributions import (
        CRITICAL_BRANCH_CABLE_LENGTH,
        CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG,
        CRITICAL_NODE_BRANCH_ORDER,
        CRITICAL_NODE_ROOT_PATH_LENGTH,
        UNIFORM_CABLE_HEIGHT_Z,
        UNIFORM_CABLE_RADIAL_XY,
        UNIFORM_CABLE_ROOT_EUCLIDEAN,
        EmpiricalTreeDistribution,
        tree_distribution,
    )
    from dendrite_gen.metrics.fused_gw import (
        PreparedFusedGWTree,
        fused_gromov_wasserstein_distance_prepared,
        prepare_fused_gw_tree,
    )
    from dendrite_gen.metrics.persistence import compute_tmd_diagrams
    from dendrite_gen.metrics.so2 import minimize_over_so2, rotate_points_about_axis
    from dendrite_gen.visualization.tmd.distances import (
        persistence_diagram_wasserstein_distance,
    )
except ModuleNotFoundError as exc:
    if exc.name != "dendrite_gen":
        raise
    from metrics.chamfer import sample_tree_points  # type: ignore
    from metrics.distributions import (  # type: ignore
        CRITICAL_BRANCH_CABLE_LENGTH,
        CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG,
        CRITICAL_NODE_BRANCH_ORDER,
        CRITICAL_NODE_ROOT_PATH_LENGTH,
        UNIFORM_CABLE_HEIGHT_Z,
        UNIFORM_CABLE_RADIAL_XY,
        UNIFORM_CABLE_ROOT_EUCLIDEAN,
        EmpiricalTreeDistribution,
        tree_distribution,
    )
    from metrics.fused_gw import (  # type: ignore
        PreparedFusedGWTree,
        fused_gromov_wasserstein_distance_prepared,
        prepare_fused_gw_tree,
    )
    from metrics.persistence import compute_tmd_diagrams  # type: ignore
    from metrics.so2 import (  # type: ignore
        minimize_over_so2,
        rotate_points_about_axis,
    )
    from visualization.tmd.distances import (  # type: ignore
        persistence_diagram_wasserstein_distance,
    )


CHAMFER = "chamfer"
TMD_PATH_WASSERSTEIN = "tmd_path_wasserstein"
TMD_HEIGHT_WASSERSTEIN = "tmd_height_wasserstein"
TMD_RHO_WASSERSTEIN = "tmd_rho_wasserstein"
DISTRIBUTION_BRANCH_LENGTH_WASSERSTEIN = (
    "distribution_branch_length_wasserstein"
)
DISTRIBUTION_SIBLING_ANGLE_WASSERSTEIN = (
    "distribution_sibling_angle_wasserstein"
)
DISTRIBUTION_ROOT_PATH_WASSERSTEIN = "distribution_root_path_wasserstein"
DISTRIBUTION_RADIAL_WASSERSTEIN = "distribution_radial_wasserstein"
DISTRIBUTION_HEIGHT_WASSERSTEIN = "distribution_height_wasserstein"
DISTRIBUTION_ROOT_EUCLIDEAN_WASSERSTEIN = (
    "distribution_root_euclidean_wasserstein"
)
DISTRIBUTION_BRANCH_ORDER_WASSERSTEIN = (
    "distribution_branch_order_wasserstein"
)
FUSED_GROMOV_WASSERSTEIN = "fused_gromov_wasserstein"

PERSISTENCE_VARIANTS = (
    TMD_PATH_WASSERSTEIN,
    TMD_HEIGHT_WASSERSTEIN,
    TMD_RHO_WASSERSTEIN,
)
DISTRIBUTION_VARIANTS = (
    DISTRIBUTION_BRANCH_LENGTH_WASSERSTEIN,
    DISTRIBUTION_SIBLING_ANGLE_WASSERSTEIN,
    DISTRIBUTION_ROOT_PATH_WASSERSTEIN,
    DISTRIBUTION_RADIAL_WASSERSTEIN,
    DISTRIBUTION_HEIGHT_WASSERSTEIN,
    DISTRIBUTION_ROOT_EUCLIDEAN_WASSERSTEIN,
    DISTRIBUTION_BRANCH_ORDER_WASSERSTEIN,
)
ALL_MATRIX_METRICS = (
    CHAMFER,
    *PERSISTENCE_VARIANTS,
    *DISTRIBUTION_VARIANTS,
    FUSED_GROMOV_WASSERSTEIN,
)
METRIC_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "chamfer": (CHAMFER,),
    "persistence": PERSISTENCE_VARIANTS,
    "distributions": DISTRIBUTION_VARIANTS,
    "fgw": (FUSED_GROMOV_WASSERSTEIN,),
    "all": ALL_MATRIX_METRICS,
}
METRIC_SELECTORS = tuple(dict.fromkeys((*METRIC_FAMILIES, *ALL_MATRIX_METRICS)))


class PreparedMatrixMetric(Protocol):
    """One symmetric scalar dissimilarity with reusable tree preparation."""

    name: str
    display_name: str
    family: str
    symmetric: bool
    allows_undefined: bool

    @property
    def configuration(self) -> Mapping[str, object]: ...

    def prepare(self, graph: nx.Graph) -> object: ...

    def compare(self, prepared_a: object, prepared_b: object) -> float: ...


def expand_metric_selection(selectors: Sequence[str]) -> tuple[str, ...]:
    """Expand family aliases into deterministic non-Elastic scalar variants."""

    if not selectors:
        raise ValueError("At least one metric selector is required.")
    if "elastic_srvft" in selectors or "elastic" in selectors:
        raise ValueError(
            "Elastic SRVFT is intentionally excluded from the matrix runner."
        )
    unknown = sorted(
        set(selectors) - set(METRIC_FAMILIES) - set(ALL_MATRIX_METRICS)
    )
    if unknown:
        raise ValueError(
            f"Unknown metric selectors {unknown!r}. Available selectors: "
            f"{list(METRIC_SELECTORS)!r}."
        )

    requested: set[str] = set()
    for selector in selectors:
        requested.update(METRIC_FAMILIES.get(selector, (selector,)))
    return tuple(name for name in ALL_MATRIX_METRICS if name in requested)


@dataclass(frozen=True)
class _PreparedChamfer:
    points: np.ndarray
    tree: cKDTree


@dataclass(frozen=True)
class ChamferMatrixMetric:
    """Arc-length-sampled Chamfer with a relative SO(2) quotient."""

    grid_size: int = 72
    refine: bool = True
    refinement_tolerance: float = 1e-8
    spacing: float = 1.0
    name: str = CHAMFER
    display_name: str = "Arc-length-sampled Chamfer"
    family: str = "chamfer"
    symmetric: bool = True
    allows_undefined: bool = False

    @property
    def configuration(self) -> Mapping[str, object]:
        return {
            "spacing": self.spacing,
            "squared": False,
            "reduction": "sum",
            "quotient_so2": True,
            "internal_axis": [0.0, 0.0, 1.0],
            "grid_size": self.grid_size,
            "refine": self.refine,
            "refinement_tolerance": self.refinement_tolerance,
        }

    def prepare(self, graph: nx.Graph) -> _PreparedChamfer:
        points = sample_tree_points(graph, spacing=self.spacing, center_root=True)
        return _PreparedChamfer(points=points, tree=cKDTree(points))

    def compare(
        self,
        prepared_a: object,
        prepared_b: object,
    ) -> float:
        if not isinstance(prepared_a, _PreparedChamfer) or not isinstance(
            prepared_b, _PreparedChamfer
        ):
            raise TypeError("Chamfer comparison requires prepared point clouds.")

        def objective(angle: float) -> float:
            rotated_b = rotate_points_about_axis(
                prepared_b.points,
                angle,
                (0.0, 0.0, 1.0),
            )
            a_in_b_frame = rotate_points_about_axis(
                prepared_a.points,
                -angle,
                (0.0, 0.0, 1.0),
            )
            a_to_b = prepared_b.tree.query(a_in_b_frame, k=1)[0]
            b_to_a = prepared_a.tree.query(rotated_b, k=1)[0]
            return float(np.mean(a_to_b) + np.mean(b_to_a))

        return float(
            minimize_over_so2(
                objective,
                grid_size=self.grid_size,
                refine=self.refine,
                refinement_tolerance=self.refinement_tolerance,
            ).value
        )


_PERSISTENCE_CONFIG = {
    TMD_PATH_WASSERSTEIN: ("path", "Path-filtration persistence W1"),
    TMD_HEIGHT_WASSERSTEIN: ("height", "Height-filtration persistence W1"),
    TMD_RHO_WASSERSTEIN: ("rho", "Radial-filtration persistence W1"),
}


@dataclass(frozen=True)
class PersistenceMatrixMetric:
    name: str
    filtration: str
    display_name: str
    family: str = "persistence"
    symmetric: bool = True
    allows_undefined: bool = False

    @property
    def configuration(self) -> Mapping[str, object]:
        return {
            "filtration": self.filtration,
            "normalize_mode": "none",
            "wasserstein_order": 1.0,
            "ground_norm": "chebyshev",
            "weight_edges_by_euclidean": True,
            "simplify_to_critical_tree": True,
            "grid_size": 0,
            "refine": False,
        }

    def prepare(self, graph: nx.Graph) -> object:
        return compute_tmd_diagrams(
            graph,
            normalize_mode="none",
            filtrations=(self.filtration,),
            weight_edges_by_euclidean=True,
            simplify_to_critical_tree=True,
        )[self.filtration]

    def compare(self, prepared_a: object, prepared_b: object) -> float:
        return float(
            persistence_diagram_wasserstein_distance(
                prepared_a,
                prepared_b,
                order=1.0,
                ground_norm="chebyshev",
            )
        )


_DISTRIBUTION_CONFIG = {
    DISTRIBUTION_BRANCH_LENGTH_WASSERSTEIN: (
        CRITICAL_BRANCH_CABLE_LENGTH,
        "Maximal-branch length W1",
    ),
    DISTRIBUTION_SIBLING_ANGLE_WASSERSTEIN: (
        CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG,
        "Sibling-branch angle W1",
    ),
    DISTRIBUTION_ROOT_PATH_WASSERSTEIN: (
        CRITICAL_NODE_ROOT_PATH_LENGTH,
        "Critical-node root-path W1",
    ),
    DISTRIBUTION_RADIAL_WASSERSTEIN: (
        UNIFORM_CABLE_RADIAL_XY,
        "Length-weighted radial-coordinate W1",
    ),
    DISTRIBUTION_HEIGHT_WASSERSTEIN: (
        UNIFORM_CABLE_HEIGHT_Z,
        "Length-weighted axial-coordinate W1",
    ),
    DISTRIBUTION_ROOT_EUCLIDEAN_WASSERSTEIN: (
        UNIFORM_CABLE_ROOT_EUCLIDEAN,
        "Length-weighted root-Euclidean W1",
    ),
    DISTRIBUTION_BRANCH_ORDER_WASSERSTEIN: (
        CRITICAL_NODE_BRANCH_ORDER,
        "Critical-node branch-order W1",
    ),
}


@dataclass(frozen=True)
class DistributionMatrixMetric:
    name: str
    distribution_name: str
    display_name: str
    spacing: float = 1.0
    family: str = "distribution_wasserstein"
    symmetric: bool = True
    allows_undefined: bool = True

    @property
    def configuration(self) -> Mapping[str, object]:
        return {
            "distribution_name": self.distribution_name,
            "sample_spacing": self.spacing,
            "wasserstein_order": 1.0,
            "empty_policy": "nan",
            "grid_size": 0,
            "refine": False,
        }

    def prepare(self, graph: nx.Graph) -> EmpiricalTreeDistribution:
        return tree_distribution(
            graph,
            self.distribution_name,
            sample_spacing=self.spacing,
        )

    def compare(self, prepared_a: object, prepared_b: object) -> float:
        if not isinstance(
            prepared_a, EmpiricalTreeDistribution
        ) or not isinstance(prepared_b, EmpiricalTreeDistribution):
            raise TypeError(
                "Distribution comparison requires prepared empirical distributions."
            )
        empty_a = prepared_a.values.size == 0
        empty_b = prepared_b.values.size == 0
        if empty_a and empty_b:
            return 0.0
        if empty_a or empty_b:
            return float("nan")
        value = float(
            wasserstein_distance(
                prepared_a.values,
                prepared_b.values,
                u_weights=prepared_a.weights,
                v_weights=prepared_b.weights,
            )
        )
        if not np.isfinite(value):
            raise RuntimeError(
                "Wasserstein solver returned a non-finite value for two "
                "non-empty distributions."
            )
        return value


@dataclass(frozen=True)
class FusedGWMatrixMetric:
    grid_size: int = 72
    refine: bool = True
    refinement_tolerance: float = 1e-8
    max_nodes: int = 1_000
    name: str = FUSED_GROMOV_WASSERSTEIN
    display_name: str = "Fused Gromov-Wasserstein"
    family: str = "fused_gromov_wasserstein"
    symmetric: bool = True
    allows_undefined: bool = False

    @property
    def configuration(self) -> Mapping[str, object]:
        return {
            "feature_mode": "xyz",
            "alpha": 0.5,
            "mass_mode": "cable_length",
            "normalize": True,
            "quotient_so2": True,
            "internal_axis": [0.0, 0.0, 1.0],
            "grid_size": self.grid_size,
            "refine": self.refine,
            "refinement_tolerance": self.refinement_tolerance,
            "max_iter": 1_000,
            "solver_tolerance": 1e-9,
            "max_nodes": self.max_nodes,
        }

    def prepare(self, graph: nx.Graph) -> PreparedFusedGWTree:
        if self.max_nodes > 0 and graph.number_of_nodes() > self.max_nodes:
            raise ValueError(
                f"FGW node guard rejected a {graph.number_of_nodes()}-node tree; "
                f"configured limit is {self.max_nodes}."
            )
        return prepare_fused_gw_tree(graph, mass_mode="cable_length")

    def compare(self, prepared_a: object, prepared_b: object) -> float:
        if not isinstance(
            prepared_a, PreparedFusedGWTree
        ) or not isinstance(prepared_b, PreparedFusedGWTree):
            raise TypeError("FGW comparison requires PreparedFusedGWTree inputs.")
        return float(
            fused_gromov_wasserstein_distance_prepared(
                prepared_a,
                prepared_b,
                feature_mode="xyz",
                alpha=0.5,
                normalize=True,
                quotient_so2=True,
                grid_size=self.grid_size,
                refine=self.refine,
                refinement_tolerance=self.refinement_tolerance,
                max_iter=1_000,
                tol=1e-9,
            ).value
        )


def build_matrix_metric(
    name: str,
    *,
    so2_grid_size: int,
    so2_refine: bool,
    so2_refinement_tolerance: float,
    fgw_max_nodes: int,
) -> PreparedMatrixMetric:
    """Build one fixed scalar metric configuration by canonical name."""

    if name == CHAMFER:
        return ChamferMatrixMetric(
            grid_size=so2_grid_size,
            refine=so2_refine,
            refinement_tolerance=so2_refinement_tolerance,
        )
    if name in _PERSISTENCE_CONFIG:
        filtration, display_name = _PERSISTENCE_CONFIG[name]
        return PersistenceMatrixMetric(
            name=name,
            filtration=filtration,
            display_name=display_name,
        )
    if name in _DISTRIBUTION_CONFIG:
        distribution_name, display_name = _DISTRIBUTION_CONFIG[name]
        return DistributionMatrixMetric(
            name=name,
            distribution_name=distribution_name,
            display_name=display_name,
        )
    if name == FUSED_GROMOV_WASSERSTEIN:
        return FusedGWMatrixMetric(
            grid_size=so2_grid_size,
            refine=so2_refine,
            refinement_tolerance=so2_refinement_tolerance,
            max_nodes=fgw_max_nodes,
        )
    if name in {"elastic", "elastic_srvft"}:
        raise ValueError(
            "Elastic SRVFT is intentionally excluded from the matrix runner."
        )
    raise KeyError(f"Unknown matrix metric: {name!r}")


__all__ = [
    "ALL_MATRIX_METRICS",
    "CHAMFER",
    "DISTRIBUTION_VARIANTS",
    "FUSED_GROMOV_WASSERSTEIN",
    "METRIC_FAMILIES",
    "METRIC_SELECTORS",
    "PERSISTENCE_VARIANTS",
    "PreparedMatrixMetric",
    "build_matrix_metric",
    "expand_metric_selection",
]
