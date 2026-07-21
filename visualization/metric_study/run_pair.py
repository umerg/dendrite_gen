"""CLI for comparing exactly two ground-truth SWC trees.

No prediction pickle, model output, or dataset pairing convention is involved.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np

try:
    from dendrite_gen.metrics.adapters.elastic_srvft import ElasticSRVFTError
    from dendrite_gen.metrics.distributions import DEFAULT_DISTRIBUTIONS
    from dendrite_gen.metrics.pair import (
        AVAILABLE_METRIC_FAMILIES,
        DEFAULT_METRIC_FAMILIES,
        compare_tree_pair,
    )
    from dendrite_gen.metrics.persistence import DEFAULT_FILTRATIONS
    from dendrite_gen.utils.data_loading import load_swc_graph
except ModuleNotFoundError as exc:
    if exc.name != "dendrite_gen":
        raise
    # Support ``python -m visualization.metric_study.run_pair`` from the repo.
    from metrics.adapters.elastic_srvft import ElasticSRVFTError  # type: ignore
    from metrics.distributions import DEFAULT_DISTRIBUTIONS  # type: ignore
    from metrics.pair import (  # type: ignore
        AVAILABLE_METRIC_FAMILIES,
        DEFAULT_METRIC_FAMILIES,
        compare_tree_pair,
    )
    from metrics.persistence import DEFAULT_FILTRATIONS  # type: ignore
    from utils.data_loading import load_swc_graph  # type: ignore


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _nonnegative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return value


def _wasserstein_order(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 1.0:
        raise argparse.ArgumentTypeError("must be finite and at least 1")
    return value


def _alpha(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must lie in [0, 1]")
    return value


def _grid_size(text: str) -> int:
    value = int(text)
    if value < 3:
        raise argparse.ArgumentTypeError("must be at least 3")
    return value


def _nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one pair of rooted ground-truth SWC trees using standalone "
            "geometry, persistence, distribution, FGW, and optional Elastic "
            "SRVFT dissimilarities."
        )
    )
    parser.add_argument("--tree-a", required=True, type=Path, help="First SWC file.")
    parser.add_argument("--tree-b", required=True, type=Path, help="Second SWC file.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=AVAILABLE_METRIC_FAMILIES,
        default=list(DEFAULT_METRIC_FAMILIES),
        help=(
            "Metric families to evaluate. The default runs Chamfer, persistence, "
            "and distributions; dense FGW and Elastic SRVFT are opt-in."
        ),
    )
    parser.add_argument(
        "--no-so2-quotient",
        dest="quotient_so2",
        action="store_false",
        help=(
            "Retain relative azimuth for Chamfer, xyz-feature FGW, and Elastic "
            "SRVFT. By default they are minimized over rotations around z."
        ),
    )
    parser.add_argument(
        "--so2-grid-size",
        type=_grid_size,
        default=72,
        help=(
            "Number of deterministic angles for Chamfer and FGW. Elastic has "
            "a dedicated coarse-grid option."
        ),
    )
    parser.add_argument(
        "--no-so2-refine",
        dest="so2_refine",
        action="store_false",
        help=(
            "Use only the Chamfer/FGW SO(2) grid. Elastic refinement has a "
            "separate opt-in switch."
        ),
    )

    chamfer = parser.add_argument_group("Chamfer")
    chamfer.add_argument(
        "--chamfer-spacing",
        type=_positive_float,
        default=1.0,
        help="Arc-length sampling spacing in SWC coordinate units.",
    )
    chamfer.add_argument(
        "--chamfer-squared",
        action="store_true",
        help="Square nearest-neighbour distances before reduction.",
    )
    chamfer.add_argument(
        "--chamfer-reduction",
        choices=("sum", "mean"),
        default="sum",
        help="Combine the two directional means by their sum or mean.",
    )

    persistence = parser.add_argument_group("TMD persistence")
    persistence.add_argument(
        "--persistence-normalization",
        choices=("none", "minmax", "max"),
        default="none",
        help="Filtration normalization; the default retains physical units.",
    )
    persistence.add_argument(
        "--filtrations",
        nargs="+",
        choices=DEFAULT_FILTRATIONS,
        default=list(DEFAULT_FILTRATIONS),
        help="TMD filtrations to compare independently.",
    )
    persistence.add_argument(
        "--persistence-order",
        type=_wasserstein_order,
        default=1.0,
        help="Wasserstein order for persistence-diagram matching.",
    )
    persistence.add_argument(
        "--persistence-ground-norm",
        choices=("euclidean", "chebyshev"),
        default="chebyshev",
        help=(
            "Ground norm in the persistence plane; Chebyshev/L-infinity is "
            "the conventional persistence-diagram choice."
        ),
    )

    distributions = parser.add_argument_group("Morphology distributions")
    distributions.add_argument(
        "--distributions",
        nargs="+",
        choices=DEFAULT_DISTRIBUTIONS,
        default=list(DEFAULT_DISTRIBUTIONS),
        help="Named per-tree distributions to compare with 1-Wasserstein.",
    )
    distributions.add_argument(
        "--distribution-spacing",
        type=_positive_float,
        default=1.0,
        help="Midpoint-quadrature spacing for cable distributions.",
    )
    distributions.add_argument(
        "--distribution-empty-policy",
        choices=("nan", "raise"),
        default="nan",
        help="How to handle a feature distribution present in only one tree.",
    )

    fgw = parser.add_argument_group("Fused Gromov-Wasserstein")
    fgw.add_argument(
        "--fgw-feature-mode",
        choices=("axis", "xyz"),
        default="xyz",
        help=(
            "Use azimuth-retaining xyz with the SO(2) quotient (default), or "
            "the cheaper invariant (z, rho) ablation."
        ),
    )

    fgw.add_argument(
        "--fgw-alpha",
        type=_alpha,
        default=0.5,
        help="POT structure-versus-feature tradeoff in [0, 1].",
    )
    fgw.add_argument(
        "--fgw-mass-mode",
        choices=("cable_length", "uniform_nodes"),
        default="cable_length",
        help="Assign mass by represented cable length, or equally per raw SWC node.",
    )
    fgw.add_argument(
        "--fgw-normalization",
        choices=("shared", "none"),
        default="shared",
        help="Use symmetric pairwise shared scales, or retain physical units.",
    )
    fgw.add_argument(
        "--fgw-max-nodes",
        type=_nonnegative_int,
        default=1000,
        help=(
            "Refuse dense FGW above this node count per tree (default: 1000); "
            "set 0 only to opt into a larger computation."
        ),
    )

    elastic = parser.add_argument_group("Elastic SRVFT (optional external backend)")
    elastic.add_argument(
        "--elastic-checkout",
        type=Path,
        help=(
            "External repository checkout. Default: "
            "metrics/external/elastic_srvft."
        ),
    )
    elastic.add_argument(
        "--elastic-lam-m",
        type=_nonnegative_float,
        default=0.2,
        help="Upstream trunk/morphology energy weight (default: 0.2).",
    )
    elastic.add_argument(
        "--elastic-lam-s",
        type=_nonnegative_float,
        default=1.0,
        help="Upstream side-branch/shape energy weight (default: 1.0).",
    )
    elastic.add_argument(
        "--elastic-lam-p",
        type=_nonnegative_float,
        default=0.2,
        help="Upstream attachment-position energy weight (default: 0.2).",
    )
    elastic.add_argument(
        "--elastic-so2-grid-size",
        type=_grid_size,
        default=8,
        help=(
            "Dedicated coarse SO(2) grid (default: 8). Each angle invokes the "
            "slow external alignment energy."
        ),
    )
    elastic.add_argument(
        "--elastic-so2-refine",
        action="store_true",
        help="Opt into bounded local refinement after the Elastic SO(2) grid.",
    )
    elastic.add_argument(
        "--elastic-refinement-tolerance",
        type=_positive_float,
        default=1e-3,
        help="Angular tolerance in radians for Elastic local refinement.",
    )
    elastic.add_argument(
        "--elastic-symmetrization",
        choices=("none", "mean"),
        default="none",
        help=(
            "Use the directional upstream energy, or average both directions "
            "at roughly twice the runtime."
        ),
    )
    elastic.add_argument(
        "--elastic-depth-policy",
        choices=("raise", "warn", "allow"),
        default="raise",
        help=(
            "Policy when the backend's fixed four branch layers would truncate "
            "structure (default: raise)."
        ),
    )
    elastic.add_argument(
        "--elastic-default-radius",
        type=_positive_float,
        default=1.0,
        help="Fallback radius for graph nodes lacking one (not used in energy).",
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional destination; the same JSON is always printed to stdout.",
    )
    return parser


def _tree_summary(path: Path, tree: nx.Graph) -> dict[str, object]:
    total_cable_length = sum(
        float(
            np.linalg.norm(
                np.asarray(tree.nodes[u]["pos"], dtype=np.float64)
                - np.asarray(tree.nodes[v]["pos"], dtype=np.float64)
            )
        )
        for u, v in tree.edges
    )
    return {
        "path": str(path.resolve()),
        "nodes": tree.number_of_nodes(),
        "edges": tree.number_of_edges(),
        "root": tree.graph.get("root"),
        "total_cable_length": total_cable_length,
    }


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_json is not None:
        output_path = args.output_json.resolve()
        input_paths = {args.tree_a.resolve(), args.tree_b.resolve()}
        if output_path in input_paths:
            parser.error("--output-json must not resolve to either input SWC path")
    tree_a = load_swc_graph(args.tree_a)
    tree_b = load_swc_graph(args.tree_b)
    if "fgw" in args.metrics and args.fgw_max_nodes > 0:
        largest_tree = max(tree_a.number_of_nodes(), tree_b.number_of_nodes())
        if largest_tree > args.fgw_max_nodes:
            parser.error(
                "FGW uses dense pairwise matrices and was requested for a tree "
                f"with {largest_tree} nodes, above --fgw-max-nodes="
                f"{args.fgw_max_nodes}. Increase the limit deliberately or "
                "omit FGW."
            )

    try:
        metric_results = compare_tree_pair(
            tree_a,
            tree_b,
            metric_families=args.metrics,
            quotient_so2=args.quotient_so2,
            so2_grid_size=args.so2_grid_size,
            so2_refine=args.so2_refine,
            chamfer_spacing=args.chamfer_spacing,
            chamfer_squared=args.chamfer_squared,
            chamfer_reduction=args.chamfer_reduction,
            persistence_normalize_mode=args.persistence_normalization,
            persistence_filtrations=args.filtrations,
            persistence_order=args.persistence_order,
            persistence_ground_norm=args.persistence_ground_norm,
            distribution_names=args.distributions,
            distribution_spacing=args.distribution_spacing,
            distribution_empty_policy=args.distribution_empty_policy,
            fgw_feature_mode=args.fgw_feature_mode,
            fgw_alpha=args.fgw_alpha,
            fgw_mass_mode=args.fgw_mass_mode,
            fgw_normalize=args.fgw_normalization == "shared",
            elastic_checkout=args.elastic_checkout,
            elastic_lam_m=args.elastic_lam_m,
            elastic_lam_s=args.elastic_lam_s,
            elastic_lam_p=args.elastic_lam_p,
            elastic_so2_grid_size=args.elastic_so2_grid_size,
            elastic_so2_refine=args.elastic_so2_refine,
            elastic_refinement_tolerance=args.elastic_refinement_tolerance,
            elastic_symmetrization=args.elastic_symmetrization,
            elastic_depth_policy=args.elastic_depth_policy,
            elastic_default_radius=args.elastic_default_radius,
        )
    except ElasticSRVFTError as exc:
        parser.error(str(exc))

    payload = _json_safe(
        {
            "schema_version": 1,
            "tree_a": _tree_summary(args.tree_a, tree_a),
            "tree_b": _tree_summary(args.tree_b, tree_b),
            "quotient_group": {
                "name": "SO(2)",
                "preferred_axis": "z",
                "enabled_for_azimuth_retaining_metrics": args.quotient_so2,
                "includes_tilts": False,
                "includes_axis_flips": False,
                "includes_reflections": False,
                "grid_size": args.so2_grid_size,
                "local_refinement": args.so2_refine,
                "refinement_angle_tolerance_rad": 1e-8,
            },
            "results": metric_results,
        }
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
