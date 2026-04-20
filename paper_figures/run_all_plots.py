"""Simple orchestrator for generating the current paper-figure set."""

from __future__ import annotations

import argparse

from .common import add_shared_arguments, load_plot_context
from .run_distribution_stats import run_distribution_stats
from .run_qualitative import QUALITATIVE_FIGURES, run_qualitative
from .run_tree_stats import run_tree_stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the current full paper-figure set.")
    add_shared_arguments(parser)
    parser.add_argument(
        "--projection",
        default="xy",
        help="2D projection to use for qualitative plots.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=3,
        help="Number of columns for grid-style plots.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    context = load_plot_context(args)

    run_qualitative(
        context,
        out_root=args.out_dir,
        projection=args.projection,
        figures=QUALITATIVE_FIGURES,
    )
    run_tree_stats(
        context,
        out_root=args.out_dir,
        ncols=args.ncols,
    )
    run_distribution_stats(
        context,
        out_root=args.out_dir,
        ncols=args.ncols,
        modes=("single", "pooled"),
    )


if __name__ == "__main__":
    main()
