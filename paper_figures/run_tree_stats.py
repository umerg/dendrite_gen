"""Runner for tree-level statistics figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import PlotContext, add_shared_arguments, ensure_runner_out_dir, load_plot_context


TREE_LEVEL_METRICS = (
    "height",
    "span_xy",
    "bbox_diag",
    "max_path_dist",
    "mean_branch_length",
    "mean_bifurcation_angle_deg",
)


def run_tree_stats(
    context: PlotContext,
    *,
    out_root: Path,
    ncols: int = 3,
) -> None:
    """Render tree-level statistics figures into the tree_stats subfolder."""
    import pandas as pd

    from .stats.plotting import plot_tree_level_hist_grid
    from .stats.tree_stats import graph_tree_scalar_row

    out_dir = ensure_runner_out_dir(out_root, "tree_stats")

    scalar_frames = []
    for pair_idx, pair in enumerate(context.selected_pairs):
        gt_idx = int(pair["gt_idx"])
        pred_idx = int(pair["pred_idx"])
        gt_path = context.gt_files[gt_idx]
        scalar_frames.append(
            graph_tree_scalar_row(
                context.gt_graphs[gt_idx],
                tree_name=gt_path.name,
                source="gt",
                pair_index=pair_idx,
            )
        )
        scalar_frames.append(
            graph_tree_scalar_row(
                context.pred_graphs[pred_idx],
                tree_name=gt_path.name,
                source="pred",
                pair_index=pair_idx,
            )
        )

    scalar_df = pd.concat(scalar_frames, ignore_index=True)
    out_path = out_dir / "treelevel_hist.png"
    plot_tree_level_hist_grid(
        scalar_df,
        metrics=TREE_LEVEL_METRICS,
        out_path=out_path,
        ncols=ncols,
    )
    print(f"Wrote {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate tree-level statistics figures.")
    add_shared_arguments(parser)
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
    run_tree_stats(context, out_root=args.out_dir, ncols=args.ncols)


if __name__ == "__main__":
    main()
