"""Runner for TMD persistence-diagram figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import PlotContext, add_shared_arguments, ensure_runner_out_dir, load_plot_context


DEFAULT_FILTRATIONS = ("path", "height", "rho")
DEFAULT_EMBEDDING_COLOR_ATTRIBUTES = ("height",)
TREE_SCALAR_ATTRIBUTES = (
    "num_nodes",
    "num_edges",
    "num_tips",
    "num_branchpoints",
    "height",
    "span_xy",
    "bbox_diag",
    "max_path_dist",
    "max_radial_dist",
    "mean_branch_length",
    "mean_bifurcation_angle_deg",
    "max_branch_order",
)


def _compute_pair_tmd_diagrams(
    gt_graph,
    pred_graph,
    *,
    filtrations: tuple[str, ...],
    normalize_mode: str,
    uhat=(0.0, 0.0, 1.0),
) -> tuple[dict[str, object], dict[str, object]]:
    from dendrite_gen.utils.tmd import compute_tmd_barcode_diagram

    gt_diagrams: dict[str, object] = {}
    pred_diagrams: dict[str, object] = {}
    for filtration in filtrations:
        _, gt_diag = compute_tmd_barcode_diagram(
            gt_graph,
            filtration=filtration,
            normalize_mode=normalize_mode,
            weight_edges_by_euclidean=True,
            simplify_to_critical_tree=True,
            uhat=uhat,
        )
        _, pred_diag = compute_tmd_barcode_diagram(
            pred_graph,
            filtration=filtration,
            normalize_mode=normalize_mode,
            weight_edges_by_euclidean=True,
            simplify_to_critical_tree=True,
            uhat=uhat,
        )
        gt_diagrams[filtration] = gt_diag
        pred_diagrams[filtration] = pred_diag
    return gt_diagrams, pred_diagrams


def run_tmd_figures(
    context: PlotContext,
    *,
    out_root: Path,
    filtrations: tuple[str, ...] = DEFAULT_FILTRATIONS,
    normalize_mode: str = "minmax",
    ncols: int = 3,
    point_alpha: float = 0.75,
    embedding_bins: int = 16,
    embedding_sigma: float = 0.05,
    embedding_weighting: str = "persistence",
    embedding_reducer: str = "auto",
    embedding_random_state: int = 0,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    embedding_connect_pairs: bool = False,
    embedding_point_alpha: float = 0.35,
    embedding_max_pairs: int | None = None,
    embedding_combine_filtrations: bool = False,
    embedding_color_attributes: tuple[str, ...] = DEFAULT_EMBEDDING_COLOR_ATTRIBUTES,
    pair_distance_attributes: tuple[str, ...] | None = None,
    uhat=(0.0, 0.0, 1.0),
) -> None:
    """Render TMD persistence-diagram grids and a full-dataset embedding.

    ``uhat`` is the equivariance/growth axis for the axis-dependent ``height``/``rho``
    filtrations and the height/span tree scalars (default z; pass ``0 1 0`` for neurons).
    """
    from .stats.tree_stats import graph_tree_scalar_stats
    from .tmd.embedding import (
        TmdDiagramRecord,
        TmdEmbeddingRecord,
        diagrams_to_persistence_image_vector,
        pair_persistence_diagram_distances,
        persistence_image_ranges,
        reduce_tmd_embedding_records,
        write_tmd_embedding_points_csv,
    )
    from .tmd.plots import plot_tmd_embedding_scatter
    from .tmd.plots import plot_tmd_mean_persistence_images
    from .tmd.plots import plot_tmd_pair_distance_attribute_scatter
    from .tmd.plots import plot_tmd_persistence_grid

    unknown_attributes = sorted(set(embedding_color_attributes) - set(TREE_SCALAR_ATTRIBUTES))
    if unknown_attributes:
        raise ValueError(
            "Unknown embedding color attribute(s): "
            f"{', '.join(unknown_attributes)}. "
            f"Available attributes are: {', '.join(TREE_SCALAR_ATTRIBUTES)}"
        )
    if pair_distance_attributes is None:
        pair_distance_attributes = embedding_color_attributes
    unknown_distance_attributes = sorted(set(pair_distance_attributes) - set(TREE_SCALAR_ATTRIBUTES))
    if unknown_distance_attributes:
        raise ValueError(
            "Unknown pair distance attribute(s): "
            f"{', '.join(unknown_distance_attributes)}. "
            f"Available attributes are: {', '.join(TREE_SCALAR_ATTRIBUTES)}"
        )

    out_dir = ensure_runner_out_dir(out_root, "tmd_figures")

    for pair in context.selected_pairs:
        gt_idx = int(pair["gt_idx"])
        pred_idx = int(pair["pred_idx"])
        gt_path = context.gt_files[gt_idx]
        gt_graph = context.gt_graphs[gt_idx]
        pred_graph = context.pred_graphs[pred_idx]

        gt_diagrams, pred_diagrams = _compute_pair_tmd_diagrams(
            gt_graph,
            pred_graph,
            filtrations=filtrations,
            normalize_mode=normalize_mode,
            uhat=uhat,
        )

        out_path = out_dir / f"{gt_path.stem}_tmd_grid.png"
        plot_tmd_persistence_grid(
            gt_diagrams,
            pred_diagrams,
            filtrations=filtrations,
            out_path=out_path,
            normalize_mode=normalize_mode,
            ncols=ncols,
            point_alpha=point_alpha,
        )
        print(f"Wrote {out_path}")

    if embedding_max_pairs is None:
        embedding_pairs = list(context.pairs)
    else:
        embedding_pairs = context.pairs[: max(0, embedding_max_pairs)]

    embedding_inputs = []
    print(f"Building TMD embedding from {len(embedding_pairs)} paired tree(s).")
    for pair_idx, pair in enumerate(embedding_pairs):
        gt_idx = int(pair["gt_idx"])
        pred_idx = int(pair["pred_idx"])
        gt_path = context.gt_files[gt_idx]
        gt_graph = context.gt_graphs[gt_idx]
        pred_graph = context.pred_graphs[pred_idx]

        gt_diagrams, pred_diagrams = _compute_pair_tmd_diagrams(
            gt_graph,
            pred_graph,
            filtrations=filtrations,
            normalize_mode=normalize_mode,
            uhat=uhat,
        )

        embedding_inputs.append(
            TmdDiagramRecord(
                source="gt",
                pair_index=pair_idx,
                tree_name=gt_path.name,
                diagrams=gt_diagrams,
                attributes=graph_tree_scalar_stats(gt_graph, uhat=uhat),
            )
        )
        embedding_inputs.append(
            TmdDiagramRecord(
                source="pred",
                pair_index=pair_idx,
                tree_name=gt_path.name,
                diagrams=pred_diagrams,
                attributes=graph_tree_scalar_stats(pred_graph, uhat=uhat),
            )
        )

    if embedding_inputs:
        embedding_jobs = [(filtration, (filtration,)) for filtration in filtrations]
        if embedding_combine_filtrations and len(filtrations) > 1:
            embedding_jobs.append(("combined", filtrations))

        for embedding_name, embedding_filtrations in embedding_jobs:
            ranges = persistence_image_ranges(
                [item.diagrams for item in embedding_inputs],
                filtrations=embedding_filtrations,
                normalize_mode=normalize_mode,
            )
            embedding_records = []
            for item in embedding_inputs:
                vector = diagrams_to_persistence_image_vector(
                    item.diagrams,
                    filtrations=embedding_filtrations,
                    n_bins=embedding_bins,
                    sigma=embedding_sigma,
                    weighting=embedding_weighting,
                    birth_range=ranges["birth"],
                    persistence_range=ranges["persistence"],
                )
                embedding_records.append(
                    TmdEmbeddingRecord(
                        source=item.source,
                        pair_index=int(item.pair_index),
                        tree_name=item.tree_name,
                        vector=vector,
                        attributes=item.attributes,
                    )
                )

            if len(embedding_filtrations) == 1:
                filtration = embedding_filtrations[0]
                out_path = out_dir / f"tmd_mean_pi_{embedding_name}.png"
                plot_tmd_mean_persistence_images(
                    [record.vector for record in embedding_records if record.source == "gt"],
                    [record.vector for record in embedding_records if record.source == "pred"],
                    out_path=out_path,
                    filtration=filtration,
                    n_bins=embedding_bins,
                    birth_range=ranges["birth"],
                    persistence_range=ranges["persistence"],
                )
                print(f"Wrote {out_path}")

            for distance_attribute in pair_distance_attributes:
                pair_distance_records = pair_persistence_diagram_distances(
                    embedding_inputs,
                    attribute=distance_attribute,
                    embedding_name=embedding_name,
                    filtrations=embedding_filtrations,
                )
                out_path = (
                    out_dir
                    / f"tmd_diagram_distance_{embedding_name}_by_gt_{distance_attribute}.png"
                )
                plot_tmd_pair_distance_attribute_scatter(
                    pair_distance_records,
                    out_path=out_path,
                    attribute=distance_attribute,
                    embedding_name=embedding_name,
                )
                print(f"Wrote {out_path}")

            embedding = reduce_tmd_embedding_records(
                embedding_records,
                reducer=embedding_reducer,
                random_state=embedding_random_state,
                umap_n_neighbors=umap_n_neighbors,
                umap_min_dist=umap_min_dist,
            )
            out_path = out_dir / f"tmd_embedding_{embedding_name}.png"
            plot_tmd_embedding_scatter(
                embedding.records,
                embedding.coords,
                out_path=out_path,
                reducer=embedding.reducer,
                title=f"TMD Embedding: {embedding_name}",
                connect_pairs=embedding_connect_pairs,
                point_alpha=embedding_point_alpha,
            )
            print(f"Wrote {out_path}")

            for color_attribute in embedding_color_attributes:
                out_path = out_dir / f"tmd_embedding_{embedding_name}_color_{color_attribute}.png"
                plot_tmd_embedding_scatter(
                    embedding.records,
                    embedding.coords,
                    out_path=out_path,
                    reducer=embedding.reducer,
                    title=f"TMD Embedding: {embedding_name} colored by {color_attribute}",
                    connect_pairs=embedding_connect_pairs,
                    point_alpha=embedding_point_alpha,
                    color_attribute=color_attribute,
                )
                print(f"Wrote {out_path}")

            out_path = out_dir / f"tmd_embedding_{embedding_name}_points.csv"
            write_tmd_embedding_points_csv(embedding, out_path)
            print(f"Wrote {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TMD persistence-diagram figures.")
    add_shared_arguments(parser)
    parser.add_argument(
        "--filtrations",
        nargs="+",
        default=list(DEFAULT_FILTRATIONS),
        help="Filtrations to include (e.g. path height rho).",
    )
    parser.add_argument(
        "--normalize-mode",
        choices=["minmax", "max", "none"],
        default="minmax",
        help="Normalization mode used before computing persistence diagrams.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=3,
        help="Number of columns for the persistence-diagram grid.",
    )
    parser.add_argument(
        "--point-alpha",
        type=float,
        default=0.75,
        help="Opacity of GT/pred persistence-diagram points.",
    )
    parser.add_argument(
        "--embedding-bins",
        type=int,
        default=16,
        help="Number of bins per axis for persistence-image embedding vectors.",
    )
    parser.add_argument(
        "--embedding-sigma",
        type=float,
        default=0.05,
        help="Gaussian sigma for persistence-image embedding vectors.",
    )
    parser.add_argument(
        "--embedding-weighting",
        choices=["none", "persistence"],
        default="persistence",
        help="Point weighting used when building persistence-image embedding vectors.",
    )
    parser.add_argument(
        "--embedding-reducer",
        choices=["auto", "umap", "pca"],
        default="auto",
        help="2D reducer for the joint TMD embedding. Auto uses UMAP when available, otherwise PCA.",
    )
    parser.add_argument(
        "--embedding-random-state",
        type=int,
        default=0,
        help="Random seed for stochastic embedding reducers.",
    )
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors used when UMAP is selected.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist used when UMAP is selected.",
    )
    parser.add_argument(
        "--embedding-connect-pairs",
        action="store_true",
        help="Draw GT-to-pred connection lines in the embedding scatter. Best for small subsets.",
    )
    parser.add_argument(
        "--embedding-point-alpha",
        type=float,
        default=0.35,
        help="Point opacity for the TMD embedding scatter.",
    )
    parser.add_argument(
        "--embedding-max-pairs",
        type=int,
        default=None,
        help="Optional limit for the TMD embedding. Defaults to all paired trees.",
    )
    parser.add_argument(
        "--embedding-combine-filtrations",
        action="store_true",
        help="Also write a combined embedding that concatenates all selected filtrations.",
    )
    parser.add_argument(
        "--embedding-color-attributes",
        nargs="+",
        choices=list(TREE_SCALAR_ATTRIBUTES),
        default=list(DEFAULT_EMBEDDING_COLOR_ATTRIBUTES),
        help=(
            "Tree-level attributes used to color embedding scatter points. "
            "One colored plot is written for each embedding filtration and attribute."
        ),
    )
    parser.add_argument(
        "--diagram-distance-attributes",
        "--pi-distance-attributes",
        nargs="+",
        choices=list(TREE_SCALAR_ATTRIBUTES),
        default=None,
        dest="diagram_distance_attributes",
        help=(
            "GT tree-level attributes to place on the y axis of the "
            "persistence-diagram distance scatter. One plot is written for each "
            "embedding filtration and attribute. Defaults to the selected "
            "embedding color attributes."
        ),
    )
    parser.add_argument(
        "--so2-axis",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 1.0],
        metavar=("X", "Y", "Z"),
        help="Equivariance/growth axis for height & rho filtrations (default z; use 0 1 0 for neurons).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    context = load_plot_context(args)
    run_tmd_figures(
        context,
        out_root=args.out_dir,
        filtrations=tuple(args.filtrations),
        normalize_mode=args.normalize_mode,
        ncols=args.ncols,
        point_alpha=args.point_alpha,
        embedding_bins=args.embedding_bins,
        embedding_sigma=args.embedding_sigma,
        embedding_weighting=args.embedding_weighting,
        embedding_reducer=args.embedding_reducer,
        embedding_random_state=args.embedding_random_state,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        embedding_connect_pairs=args.embedding_connect_pairs,
        embedding_point_alpha=args.embedding_point_alpha,
        embedding_max_pairs=args.embedding_max_pairs,
        embedding_combine_filtrations=args.embedding_combine_filtrations,
        embedding_color_attributes=tuple(args.embedding_color_attributes),
        pair_distance_attributes=(
            None
            if args.diagram_distance_attributes is None
            else tuple(args.diagram_distance_attributes)
        ),
        uhat=tuple(args.so2_axis),
    )


if __name__ == "__main__":
    main()
