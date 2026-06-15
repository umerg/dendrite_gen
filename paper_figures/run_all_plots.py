"""Simple orchestrator for generating the current paper-figure set."""

from __future__ import annotations

import argparse

from .common import add_shared_arguments, load_plot_context
from .run_distribution_stats import run_distribution_stats
from .run_qualitative import ALL_QUALITATIVE_PROJECTIONS, QUALITATIVE_FIGURES, run_qualitative
from .run_tree_stats import run_tree_stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the current full paper-figure set.")
    add_shared_arguments(parser, default_max_pairs=1)
    parser.add_argument(
        "--projection",
        default="xy",
        help="Reserved for future use. The current all-plots runner emits qualitative plots for xy, xz, and yz.",
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

    for projection in ALL_QUALITATIVE_PROJECTIONS:
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


if __name__ == "__main__":
    main()
