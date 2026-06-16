"""Main orchestrator for generating the current figure set."""

from __future__ import annotations

import argparse

from .common import add_shared_arguments, load_plot_context
from .run_cylinder_trees import (
    DEFAULT_PLOTLY_LEAF_COUNT,
    DEFAULT_PLOTLY_LEAF_OPACITY,
    DEFAULT_PLOTLY_LEAF_SCALE,
    DEFAULT_PLOTLY_LEAF_SEED,
    DEFAULT_PLOTLY_LEAVES,
    DEFAULT_PLOTLY_TEXTURE,
    DEFAULT_PLOTLY_TEXTURE_MAX_AXIAL_SEGMENTS,
    DEFAULT_PLOTLY_TEXTURE_STRENGTH,
    DEFAULT_PLOTLY_TEXTURE_TARGET_LENGTH_SCALE,
    run_cylinder_trees,
)
from .run_distribution_stats import run_distribution_stats
from .run_qualitative import ALL_QUALITATIVE_PROJECTIONS, QUALITATIVE_FIGURES, run_qualitative
from .run_tmd_figures import (
    DEFAULT_FILTRATIONS,
    TREE_SCALAR_ATTRIBUTES,
    run_tmd_figures,
)
from .run_tree_stats import run_tree_stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the current figure/visualization set.")
    add_shared_arguments(parser, default_max_pairs=1)
    parser.add_argument(
        "--projections",
        nargs="+",
        default=list(ALL_QUALITATIVE_PROJECTIONS),
        help="2D projections to render for qualitative plots.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=3,
        help="Number of columns for grid-style plots.",
    )
    parser.add_argument(
        "--stats-max-pairs",
        type=int,
        default=None,
        help="Optional limit for tree/distribution statistics. Defaults to all paired trees.",
    )
    parser.add_argument(
        "--skip-tmd",
        action="store_true",
        help="Skip TMD persistence-diagram visualizations.",
    )
    parser.add_argument(
        "--include-cylinders",
        action="store_true",
        help="Also render selected GT/pred trees as 3D cylinder models.",
    )
    parser.add_argument(
        "--cylinder-angle",
        nargs=2,
        type=float,
        metavar=("ELEV", "AZIM"),
        default=(20.0, 30.0),
        help="Camera angle for optional cylinder renderings.",
    )
    parser.add_argument(
        "--cylinder-segments",
        type=int,
        default=8,
        help="Number of radial segments per branch cylinder.",
    )
    parser.add_argument(
        "--cylinder-backend",
        choices=["matplotlib", "pyvista", "plotly"],
        default="matplotlib",
        help="Rendering backend for optional cylinder renderings.",
    )
    parser.add_argument(
        "--cylinder-curve-branches",
        action="store_true",
        help="Render optional cylinder branches as endpoint-preserving random curves.",
    )
    parser.add_argument(
        "--cylinder-curve-subsegments",
        type=int,
        default=5,
        help="Number of curved centerline subsegments per original branch edge.",
    )
    parser.add_argument(
        "--cylinder-curve-wiggle-scale",
        type=float,
        default=0.02,
        help="Curve wiggle amplitude as a fraction of branch path length.",
    )
    parser.add_argument(
        "--cylinder-curve-momentum",
        type=float,
        default=0.75,
        help="Memory factor for smooth optional cylinder curve noise.",
    )
    parser.add_argument(
        "--cylinder-curve-seed",
        type=int,
        default=0,
        help="Seed for deterministic optional cylinder branch curves.",
    )
    parser.add_argument(
        "--cylinder-radius-attr",
        type=str,
        default="radius",
        help="Node attribute containing per-node radii for optional cylinder renderings.",
    )
    parser.add_argument(
        "--cylinder-radius-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to existing or default cylinder radii.",
    )
    parser.add_argument(
        "--cylinder-default-radius",
        type=float,
        default=1.0,
        help="Radius used for nodes without a valid cylinder radius attribute.",
    )
    parser.add_argument(
        "--cylinder-synthesize-radii",
        action="store_true",
        help=(
            "Synthesize visual radii for optional cylinder renderings instead of "
            "reading the radius attribute."
        ),
    )
    parser.add_argument(
        "--cylinder-twig-radius",
        type=float,
        default=None,
        help="Terminal twig radius for synthesized cylinder radii.",
    )
    parser.add_argument(
        "--cylinder-twig-radius-scale",
        type=float,
        default=0.002,
        help="Graph bounding-box diagonal fraction used as synthesized cylinder twig radius when omitted.",
    )
    parser.add_argument(
        "--cylinder-pipe-exponent",
        type=float,
        default=0.35,
        help="Subtree tip-count exponent for synthesized cylinder radii.",
    )
    parser.add_argument(
        "--cylinder-length-exponent",
        type=float,
        default=0.12,
        help="Downstream-length exponent for synthesized cylinder radii.",
    )
    parser.add_argument(
        "--cylinder-radius-smoothing-passes",
        type=int,
        default=1,
        help="Number of parent-child monotonicity passes after synthesized cylinder path smoothing.",
    )
    parser.add_argument(
        "--cylinder-no-joints",
        action="store_true",
        help="Disable Plotly joint spheres for optional cylinder renderings.",
    )
    parser.add_argument(
        "--cylinder-joint-scale",
        type=float,
        default=1.05,
        help="Plotly joint sphere radius multiplier for optional cylinder renderings.",
    )
    parser.add_argument(
        "--cylinder-joint-segments",
        type=int,
        default=10,
        help="Plotly joint sphere mesh resolution for optional cylinder renderings.",
    )
    parser.add_argument(
        "--cylinder-plotly-texture",
        choices=["none", "bark"],
        default=DEFAULT_PLOTLY_TEXTURE,
        help="Procedural Plotly branch texture for optional cylinder renderings.",
    )
    parser.add_argument(
        "--cylinder-plotly-texture-strength",
        type=float,
        default=DEFAULT_PLOTLY_TEXTURE_STRENGTH,
        help="Strength of the procedural Plotly branch texture for optional cylinder renderings.",
    )
    parser.add_argument(
        "--cylinder-plotly-texture-target-length-scale",
        type=float,
        default=DEFAULT_PLOTLY_TEXTURE_TARGET_LENGTH_SCALE,
        help=(
            "Target optional Plotly bark cell length as a multiple of the "
            "graph's median edge length."
        ),
    )
    parser.add_argument(
        "--cylinder-plotly-texture-target-aspect",
        type=float,
        default=argparse.SUPPRESS,
        dest="cylinder_plotly_texture_target_length_scale",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cylinder-plotly-texture-max-axial-segments",
        type=int,
        default=DEFAULT_PLOTLY_TEXTURE_MAX_AXIAL_SEGMENTS,
        help="Maximum axial bark texture cells per original branch edge.",
    )
    parser.add_argument(
        "--cylinder-plotly-leaves",
        action="store_true",
        default=DEFAULT_PLOTLY_LEAVES,
        help="Add translucent low-poly leaf canopy polyhedra to optional Plotly cylinder renderings.",
    )
    parser.add_argument(
        "--cylinder-plotly-leaf-count",
        type=int,
        default=DEFAULT_PLOTLY_LEAF_COUNT,
        help=(
            "Requested number of low-poly leaf blobs for optional Plotly cylinder renderings. "
            "Terminal tips are partitioned into this many coverage groups when possible."
        ),
    )
    parser.add_argument(
        "--cylinder-plotly-leaf-opacity",
        type=float,
        default=DEFAULT_PLOTLY_LEAF_OPACITY,
        help="Opacity of the translucent Plotly leaf canopy polyhedra.",
    )
    parser.add_argument(
        "--cylinder-plotly-leaf-scale",
        type=float,
        default=DEFAULT_PLOTLY_LEAF_SCALE,
        help="Size multiplier for optional Plotly leaf canopy polyhedra.",
    )
    parser.add_argument(
        "--cylinder-plotly-leaf-seed",
        type=int,
        default=DEFAULT_PLOTLY_LEAF_SEED,
        help="Seed for deterministic optional Plotly leaf polyhedron placement.",
    )
    parser.add_argument(
        "--tmd-filtrations",
        nargs="+",
        default=list(DEFAULT_FILTRATIONS),
        help="TMD filtrations to include when TMD visualizations are enabled.",
    )
    parser.add_argument(
        "--tmd-normalize-mode",
        choices=["minmax", "max", "none"],
        default="minmax",
        help="Normalization mode used before computing TMD persistence diagrams.",
    )
    parser.add_argument(
        "--tmd-embedding-bins",
        type=int,
        default=16,
        help="Number of bins per axis for TMD persistence-image embedding vectors.",
    )
    parser.add_argument(
        "--tmd-embedding-sigma",
        type=float,
        default=0.05,
        help="Gaussian sigma for TMD persistence-image embedding vectors.",
    )
    parser.add_argument(
        "--tmd-embedding-weighting",
        choices=["none", "persistence"],
        default="persistence",
        help="Point weighting used for TMD persistence-image embedding vectors.",
    )
    parser.add_argument(
        "--tmd-embedding-reducer",
        choices=["auto", "umap", "pca"],
        default="auto",
        help="2D reducer for the joint TMD embedding. Auto uses UMAP when available, otherwise PCA.",
    )
    parser.add_argument(
        "--tmd-embedding-random-state",
        type=int,
        default=0,
        help="Random seed for stochastic TMD embedding reducers.",
    )
    parser.add_argument(
        "--tmd-umap-n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors used for the TMD embedding when UMAP is selected.",
    )
    parser.add_argument(
        "--tmd-umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist used for the TMD embedding when UMAP is selected.",
    )
    parser.add_argument(
        "--tmd-embedding-connect-pairs",
        action="store_true",
        help="Draw GT-to-pred connection lines in the TMD embedding scatter. Best for small subsets.",
    )
    parser.add_argument(
        "--tmd-embedding-point-alpha",
        type=float,
        default=0.35,
        help="Point opacity for the TMD embedding scatter.",
    )
    parser.add_argument(
        "--tmd-embedding-max-pairs",
        type=int,
        default=None,
        help="Optional limit for the TMD embedding. Defaults to all paired trees.",
    )
    parser.add_argument(
        "--tmd-embedding-combine-filtrations",
        action="store_true",
        help="Also write a combined TMD embedding that concatenates all selected filtrations.",
    )
    parser.add_argument(
        "--tmd-embedding-color-attributes",
        nargs="+",
        choices=list(TREE_SCALAR_ATTRIBUTES),
        default=list(TREE_SCALAR_ATTRIBUTES),
        help=(
            "Tree-level attributes used to color TMD embedding scatter points. "
            "One colored plot is written for each TMD embedding filtration and attribute. "
            "Defaults to all available tree-level scalar attributes."
        ),
    )
    parser.add_argument(
        "--tmd-diagram-distance-attributes",
        "--tmd-pi-distance-attributes",
        nargs="+",
        choices=list(TREE_SCALAR_ATTRIBUTES),
        default=None,
        dest="tmd_diagram_distance_attributes",
        help=(
            "GT tree-level attributes used on the y axis of TMD persistence-diagram "
            "distance scatter plots. One plot is written for each TMD embedding "
            "filtration and attribute. Defaults to the selected TMD embedding "
            "color attributes."
        ),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    context = load_plot_context(args)
    if args.stats_max_pairs is None:
        print(f"Tree and distribution statistics will use all {len(context.pairs)} paired trees.")
    else:
        print(
            "Tree and distribution statistics will use "
            f"{min(len(context.pairs), max(0, args.stats_max_pairs))} paired trees."
        )

    for projection in args.projections:
        run_qualitative(
            context,
            out_root=args.out_dir,
            projection=projection,
            figures=QUALITATIVE_FIGURES,
        )
    if args.include_cylinders:
        run_cylinder_trees(
            context,
            out_root=args.out_dir,
            angles=(tuple(args.cylinder_angle),),
            segments=args.cylinder_segments,
            radius_attr=args.cylinder_radius_attr,
            radius_scale=args.cylinder_radius_scale,
            default_radius=args.cylinder_default_radius,
            synthesize_radii=args.cylinder_synthesize_radii,
            twig_radius=args.cylinder_twig_radius,
            twig_radius_scale=args.cylinder_twig_radius_scale,
            pipe_exponent=args.cylinder_pipe_exponent,
            length_exponent=args.cylinder_length_exponent,
            radius_smoothing_passes=args.cylinder_radius_smoothing_passes,
            curve_branches=args.cylinder_curve_branches,
            curve_subsegments=args.cylinder_curve_subsegments,
            curve_wiggle_scale=args.cylinder_curve_wiggle_scale,
            curve_momentum=args.cylinder_curve_momentum,
            curve_seed=args.cylinder_curve_seed,
            backend=args.cylinder_backend,
            show_joints=not args.cylinder_no_joints,
            joint_scale=args.cylinder_joint_scale,
            joint_segments=args.cylinder_joint_segments,
            plotly_texture=args.cylinder_plotly_texture,
            plotly_texture_strength=args.cylinder_plotly_texture_strength,
            plotly_texture_target_length_scale=args.cylinder_plotly_texture_target_length_scale,
            plotly_texture_max_axial_segments=args.cylinder_plotly_texture_max_axial_segments,
            plotly_leaves=args.cylinder_plotly_leaves,
            plotly_leaf_count=args.cylinder_plotly_leaf_count,
            plotly_leaf_opacity=args.cylinder_plotly_leaf_opacity,
            plotly_leaf_scale=args.cylinder_plotly_leaf_scale,
            plotly_leaf_seed=args.cylinder_plotly_leaf_seed,
        )
    run_tree_stats(
        context,
        out_root=args.out_dir,
        ncols=args.ncols,
        max_pairs=args.stats_max_pairs,
    )
    run_distribution_stats(
        context,
        out_root=args.out_dir,
        ncols=args.ncols,
        modes=("single",),
        max_pairs=args.max_pairs,
    )
    run_distribution_stats(
        context,
        out_root=args.out_dir,
        ncols=args.ncols,
        modes=("pooled",),
        max_pairs=args.stats_max_pairs,
    )
    if not args.skip_tmd:
        run_tmd_figures(
            context,
            out_root=args.out_dir,
            filtrations=tuple(args.tmd_filtrations),
            normalize_mode=args.tmd_normalize_mode,
            ncols=args.ncols,
            embedding_bins=args.tmd_embedding_bins,
            embedding_sigma=args.tmd_embedding_sigma,
            embedding_weighting=args.tmd_embedding_weighting,
            embedding_reducer=args.tmd_embedding_reducer,
            embedding_random_state=args.tmd_embedding_random_state,
            umap_n_neighbors=args.tmd_umap_n_neighbors,
            umap_min_dist=args.tmd_umap_min_dist,
            embedding_connect_pairs=args.tmd_embedding_connect_pairs,
            embedding_point_alpha=args.tmd_embedding_point_alpha,
            embedding_max_pairs=args.tmd_embedding_max_pairs,
            embedding_combine_filtrations=args.tmd_embedding_combine_filtrations,
            embedding_color_attributes=tuple(args.tmd_embedding_color_attributes),
            pair_distance_attributes=(
                None
                if args.tmd_diagram_distance_attributes is None
                else tuple(args.tmd_diagram_distance_attributes)
            ),
        )


if __name__ == "__main__":
    main()
