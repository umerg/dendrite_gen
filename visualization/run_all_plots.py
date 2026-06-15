"""Main orchestrator for generating the current figure set."""

from __future__ import annotations

import argparse

from .common import add_shared_arguments, load_plot_context
from .run_distribution_stats import run_distribution_stats
from .run_qualitative import ALL_QUALITATIVE_PROJECTIONS, QUALITATIVE_FIGURES, run_qualitative
from .run_tmd_figures import DEFAULT_FILTRATIONS, run_tmd_figures
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
        )


if __name__ == "__main__":
    main()
