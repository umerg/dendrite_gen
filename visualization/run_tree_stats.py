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
    max_pairs: int | None = None,
    uhat=(0.0, 0.0, 1.0),
) -> None:
    """Render tree-level statistics figures into the tree_stats subfolder."""
    import pandas as pd

    from .stats.plotting import plot_tree_level_hist_grid, plot_tree_level_scatter_grid
    from .stats.tree_stats import graph_tree_scalar_row

    out_dir = ensure_runner_out_dir(out_root, "tree_stats")
    stats_pairs = context.pairs if max_pairs is None else context.pairs[: max(0, max_pairs)]

    scalar_frames = []
    for pair_idx, pair in enumerate(stats_pairs):
        gt_idx = int(pair["gt_idx"])
        pred_idx = int(pair["pred_idx"])
        gt_path = context.gt_files[gt_idx]
        scalar_frames.append(
            graph_tree_scalar_row(
                context.gt_graphs[gt_idx],
                tree_name=gt_path.name,
                source="gt",
                pair_index=pair_idx,
                uhat=uhat,
            )
        )
        scalar_frames.append(
            graph_tree_scalar_row(
                context.pred_graphs[pred_idx],
                tree_name=gt_path.name,
                source="pred",
                pair_index=pair_idx,
                uhat=uhat,
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

    out_path = out_dir / "treelevel_scatter.png"
    plot_tree_level_scatter_grid(
        scalar_df,
        metrics=TREE_LEVEL_METRICS,
        out_path=out_path,
        ncols=ncols,
    )
    print(f"Wrote {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate tree-level statistics figures.")
    add_shared_arguments(parser, default_max_pairs=None)
    parser.add_argument(
        "--ncols",
        type=int,
        default=3,
        help="Number of columns for grid-style plots.",
    )
    parser.add_argument(
        "--so2-axis",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 1.0],
        metavar=("X", "Y", "Z"),
        help="Equivariance/growth axis for height & span_xy scalars (default z; use 0 1 0 for neurons).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    context = load_plot_context(args)
    run_tree_stats(context, out_root=args.out_dir, ncols=args.ncols, max_pairs=args.max_pairs,
                   uhat=tuple(args.so2_axis))


if __name__ == "__main__":
    main()
