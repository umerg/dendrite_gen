"""Runner for within-tree distribution statistics figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .common import PlotContext, add_shared_arguments, ensure_runner_out_dir, load_plot_context


def run_distribution_stats(
    context: PlotContext,
    *,
    out_root: Path,
    ncols: int = 3,
    modes: Sequence[str] = ("pooled",),
) -> None:
    """Render within-tree distribution figures into the distribution_stats subfolder."""
    import pandas as pd

    from .stats.distribution_stats import GRAPH_DISTRIBUTION_KEYS, graph_distribution_rows
    from .stats.plotting import plot_distribution_hist_grid

    out_dir = ensure_runner_out_dir(out_root, "distribution_stats")
    distribution_metrics = list(GRAPH_DISTRIBUTION_KEYS)

    if "single" in modes:
        for pair_idx, pair in enumerate(context.selected_pairs):
            gt_idx = int(pair["gt_idx"])
            pred_idx = int(pair["pred_idx"])
            gt_path = context.gt_files[gt_idx]
            dist_frames = [
                graph_distribution_rows(
                    context.gt_graphs[gt_idx],
                    tree_name=gt_path.name,
                    source="gt",
                    pair_index=pair_idx,
                ),
                graph_distribution_rows(
                    context.pred_graphs[pred_idx],
                    tree_name=gt_path.name,
                    source="pred",
                    pair_index=pair_idx,
                ),
            ]
            dist_df = pd.concat(dist_frames, ignore_index=True)
            out_path = out_dir / f"{gt_path.stem}_distribution_hist.png"
            plot_distribution_hist_grid(
                dist_df,
                metrics=distribution_metrics,
                out_path=out_path,
                ncols=ncols,
                aggregation="pooled",
            )
            print(f"Wrote {out_path}")

    dataset_modes = [mode for mode in modes if mode != "single"]
    if dataset_modes:
        dist_frames = []
        for pair_idx, pair in enumerate(context.selected_pairs):
            gt_idx = int(pair["gt_idx"])
            pred_idx = int(pair["pred_idx"])
            gt_path = context.gt_files[gt_idx]
            dist_frames.append(
                graph_distribution_rows(
                    context.gt_graphs[gt_idx],
                    tree_name=gt_path.name,
                    source="gt",
                    pair_index=pair_idx,
                )
            )
            dist_frames.append(
                graph_distribution_rows(
                    context.pred_graphs[pred_idx],
                    tree_name=gt_path.name,
                    source="pred",
                    pair_index=pair_idx,
                )
            )
        dist_df = pd.concat(dist_frames, ignore_index=True)
        for mode in dataset_modes:
            out_path = out_dir / f"distribution_hist_{mode}.png"
            plot_distribution_hist_grid(
                dist_df,
                metrics=distribution_metrics,
                out_path=out_path,
                ncols=ncols,
                aggregation=mode,
            )
            print(f"Wrote {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate within-tree distribution figures.")
    add_shared_arguments(parser)
    parser.add_argument(
        "--distribution-mode",
        choices=["single", "pooled", "tree_average"],
        default="pooled",
        help="How to aggregate within-tree distribution plots.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate pooled dataset plots and per-tree single plots.",
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
    modes = ("single", "pooled") if args.all else (args.distribution_mode,)
    run_distribution_stats(
        context,
        out_root=args.out_dir,
        ncols=args.ncols,
        modes=modes,
    )


if __name__ == "__main__":
    main()
