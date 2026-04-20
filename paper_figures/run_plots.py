"""






************************************************
************************************************
DEPRECIATED
************************************************
************************************************







Entry point for generating paper figures.

The intent is to keep this script data-source driven so figures can be
regenerated when new SWC examples or evaluation outputs become available.

Planned CLI growth / roadmap:
    This file is intended to remain the single top-level orchestrator for
    paper figure generation. The plotting logic itself should live in modules
    such as ``qualitative.py``, ``distributions.py``, ``topology.py``, and
    ``compose.py``. ``run_plots.py`` should stay focused on:

      1. parsing args / config
      2. discovering inputs
      3. pairing GT / predicted examples
      4. deciding which figure targets to build
      5. dispatching to the appropriate plotting modules

    The long-term goal is to support three user intents with one entrypoint:

      1. Build one specific figure
         Example:
             ``--figure simple2d``
             ``--figure overlay2d``
             ``--figure gallery2d``
             ``--figure treelevel_hist``
             ``--figure main_figure``

      2. Build one figure family
         Example:
             ``--family qualitative``
             ``--family distributions``
             ``--family topology``

      3. Build everything for a dataset / run
         Example:
             ``--all --dataset dendrite --split test``

    Candidate argument groups for the fuller version of this script:

      A. Target selection
         Purpose:
             Choose the scope of the plotting run.
         Candidate flags:
             ``--figure <name>``
             ``--family <name>``
             ``--all``
             ``--list-figures``
             ``--list-families``

      B. Dataset / domain resolution
         Purpose:
             Decide which dataset, domain, and source directories are being
             plotted.
         Candidate flags:
             ``--dataset <name>``
             ``--domain <dendrite|wood_tree>``
             ``--split <train|val|test>``
             ``--run-name <name>``
             ``--gt-dir <path>``
             ``--pred-dir <path>``
             ``--metadata <path>``
             ``--config <path>``

      C. Pairing / example selection
         Purpose:
             Control how GT and predicted examples are matched and which subset
             is plotted.
         Candidate flags:
             ``--pairing <exact_name|closest_size|index>``
             ``--example-names <name1> <name2> ...``
             ``--example-indices <i1> <i2> ...``
             ``--max-pairs <int>``
             ``--seed <int>``

      D. Rendering options
         Purpose:
             Control visual defaults for a given rendering pass.
         Candidate flags:
             ``--projection <xy|xz|yz|yx|zx|zy>``
             ``--view-angle <elev> <azim>``
             ``--dpi <int>``
             ``--format <png|pdf|svg>``
             ``--transparent``
             ``--style <draft|paper>``

      E. Output control
         Purpose:
             Control where outputs go and whether existing outputs should be
             reused or replaced.
         Candidate flags:
             ``--out-dir <path>``
             ``--artifact-dir <path>``
             ``--overwrite``
             ``--skip-existing``
             ``--export-manifest``

      F. Execution / inspection helpers
         Purpose:
             Make the script easier to debug and automate.
         Candidate flags:
             ``--dry-run``
             ``--verbose``

    A likely example command for a more mature version would be:

        python -m dendrite_gen.paper_figures.run_plots \
            --dataset dendrite \
            --split test \
            --all \
            --pairing exact_name \
            --out-dir outputs/paper_figures/dendrite \
            --artifact-dir outputs/paper_figures/dendrite/artifacts \
            --format png \
            --format pdf \
            --dpi 300 \
            --style paper \
            --overwrite

    For now this script supports a small but growing set of figure targets:
    qualitative 2D GT/pred plots and a first tree-level histogram grid.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .utils.io import (
    list_swc_files,
    load_gt_file_graphs,
    load_pred_graphs_from_pickle,
    pair_graphs_by_index,
)


def _preview_names(items: Sequence[str], *, max_items: int = 5) -> str:
    """Return a short printable preview of names."""
    if not items:
        return "(none)"
    shown = list(items[:max_items])
    preview = ", ".join(shown)
    if len(items) > max_items:
        preview += f", ... (+{len(items) - max_items} more)"
    return preview


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate paper figures from SWC inputs.")
    parser.add_argument("--gt-dir", type=Path, help="Directory containing ground-truth SWC files.")
    parser.add_argument(
        "--pred-pkl",
        type=Path,
        help="Validation pickle containing predicted graphs under `pred_graphs`.",
    )
    parser.add_argument(
        "--ema-key",
        type=str,
        default=None,
        help="Optional EMA key inside the prediction pickle (e.g. `ema_1`, `ema_0.999`).",
    )
    parser.add_argument(
        "--figure",
        choices=["simple2d", "overlay2d", "gallery2d", "treelevel_hist", "distribution_hist"],
        default="simple2d",
        help="Figure family to generate.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all currently implemented figure targets for the given inputs.",
    )
    parser.add_argument(
        "--projection",
        default="xy",
        help="2D projection to use for simple 2D plots (e.g. xy, xz, zy).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=1,
        help="Maximum number of GT/pred pairs to render.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=3,
        help="Number of columns for grid-style plots such as histogram panels.",
    )
    parser.add_argument(
        "--distribution-mode",
        choices=["single", "pooled", "tree_average"],
        default="pooled",
        help="How to aggregate within-tree distribution plots.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dendrite_gen/outputs/paper_figures"),
        help="Directory where paper figure outputs will be written.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt_files: list[Path] = []
    if args.gt_dir is not None:
        gt_files = list_swc_files(args.gt_dir)
        print(f"Found {len(gt_files)} GT SWC files in {args.gt_dir}")
        print(f"GT file preview: {_preview_names([p.name for p in gt_files])}")
    if args.pred_pkl is not None:
        print(f"Using prediction pickle {args.pred_pkl}")
        if args.ema_key is not None:
            print(f"Using EMA key {args.ema_key}")
        else:
            print("Using prediction pickle without explicit EMA key selection.")
    print(f"Requested figure target: {args.figure}")
    print(f"Run all implemented figures: {args.all}")
    print(f"Projection: {args.projection}")
    print(f"Max pairs to render: {args.max_pairs}")
    print(f"Distribution mode: {args.distribution_mode}")
    print(f"Output directory: {args.out_dir}")

    if args.figure in {"simple2d", "overlay2d", "gallery2d", "treelevel_hist", "distribution_hist"} or args.all:
        from .qualitative.plots_2d import (
            plot_tree_gallery_2d,
            plot_tree_overlay_2d,
            plot_tree_pair_2d,
        )
        from .stats.distribution_stats import GRAPH_DISTRIBUTION_KEYS, graph_distribution_rows
        from .stats.tree_stats import graph_tree_scalar_row
        from .stats.plotting import plot_distribution_hist_grid, plot_tree_level_hist_grid

        if args.gt_dir is None or args.pred_pkl is None:
            raise ValueError("--gt-dir and --pred-pkl are required for the current figure targets.")

        gt_files, gt_graphs = load_gt_file_graphs(args.gt_dir)
        pred_graphs = load_pred_graphs_from_pickle(args.pred_pkl, ema_key=args.ema_key)
        pairs, unmatched = pair_graphs_by_index(gt_files, gt_graphs, pred_graphs)
        print(f"Loaded {len(pred_graphs)} predicted graph(s) from pickle.")
        if unmatched:
            print(f"Warning: GT/pred count mismatch: {unmatched}")
        if not pairs:
            raise ValueError("No GT/pred graph pairs could be formed.")
        print(f"Formed {len(pairs)} GT/pred pair(s) by index.")

        selected_pairs = pairs[: max(0, args.max_pairs)]
        selected_gt_names = [
            gt_files[int(pair["gt_idx"])].name
            for pair in selected_pairs
        ]
        print(
            f"Will render {'all implemented figures' if args.all else args.figure} for: "
            f"{_preview_names(selected_gt_names)}"
        )

        figures_to_make = (
            ["simple2d", "overlay2d", "gallery2d", "treelevel_hist", "distribution_hist"]
            if args.all
            else [args.figure]
        )

        if "gallery2d" in figures_to_make:
            gallery_gt = [gt_graphs[int(pair["gt_idx"])] for pair in selected_pairs]
            gallery_pred = [pred_graphs[int(pair["pred_idx"])] for pair in selected_pairs]
            gallery_labels = [gt_files[int(pair["gt_idx"])].name for pair in selected_pairs]
            out_path = args.out_dir / f"gallery2d_{args.projection}.png"
            plot_tree_gallery_2d(
                gallery_gt,
                gallery_pred,
                gallery_labels,
                projection=args.projection,
                out_path=out_path,
                max_examples=args.max_pairs,
                overlay=True,
            )
            print(f"Wrote {out_path}")

        per_pair_figures = [name for name in figures_to_make if name in {"simple2d", "overlay2d"}]
        if per_pair_figures:
            for pair in selected_pairs:
                gt_idx = int(pair["gt_idx"])
                pred_idx = int(pair["pred_idx"])
                gt_path = gt_files[gt_idx]
                gt_graph = gt_graphs[gt_idx]
                pred_graph = pred_graphs[pred_idx]
                stem = gt_path.stem
                if "simple2d" in per_pair_figures:
                    out_path = args.out_dir / f"{stem}_pair_{args.projection}.png"
                    plot_tree_pair_2d(
                        gt_graph,
                        pred_graph,
                        projection=args.projection,
                        out_path=out_path,
                        title_gt=f"GT: {gt_path.name}",
                        title_pred=f"Pred idx {pred_idx}",
                    )
                    print(f"Wrote {out_path}")
                if "overlay2d" in per_pair_figures:
                    out_path = args.out_dir / f"{stem}_overlay_{args.projection}.png"
                    plot_tree_overlay_2d(
                        gt_graph,
                        pred_graph,
                        projection=args.projection,
                        out_path=out_path,
                        title=f"{gt_path.name}: GT vs Pred",
                    )
                    print(f"Wrote {out_path}")

        if "treelevel_hist" in figures_to_make:
            import pandas as pd

            scalar_frames = []
            for pair_idx, pair in enumerate(selected_pairs):
                gt_idx = int(pair["gt_idx"])
                pred_idx = int(pair["pred_idx"])
                gt_path = gt_files[gt_idx]
                scalar_frames.append(
                    graph_tree_scalar_row(
                        gt_graphs[gt_idx],
                        tree_name=gt_path.name,
                        source="gt",
                        pair_index=pair_idx,
                    )
                )
                scalar_frames.append(
                    graph_tree_scalar_row(
                        pred_graphs[pred_idx],
                        tree_name=gt_path.name,
                        source="pred",
                        pair_index=pair_idx,
                    )
                )
            scalar_df = pd.concat(scalar_frames, ignore_index=True)
            metrics = [
                "height",
                "span_xy",
                "bbox_diag",
                "max_path_dist",
                "mean_branch_length",
                "mean_bifurcation_angle_deg",
            ]
            out_path = args.out_dir / "treelevel_hist.png"
            plot_tree_level_hist_grid(
                scalar_df,
                metrics=metrics,
                out_path=out_path,
                ncols=args.ncols,
            )
            print(f"Wrote {out_path}")

        if "distribution_hist" in figures_to_make:
            import pandas as pd

            distribution_metrics = list(GRAPH_DISTRIBUTION_KEYS)
            distribution_modes = (
                ["single", "pooled"]
                if args.all
                else [args.distribution_mode]
            )

            if "single" in distribution_modes:
                for pair_idx, pair in enumerate(selected_pairs):
                    gt_idx = int(pair["gt_idx"])
                    pred_idx = int(pair["pred_idx"])
                    gt_path = gt_files[gt_idx]
                    dist_frames = [
                        graph_distribution_rows(
                            gt_graphs[gt_idx],
                            tree_name=gt_path.name,
                            source="gt",
                            pair_index=pair_idx,
                        ),
                        graph_distribution_rows(
                            pred_graphs[pred_idx],
                            tree_name=gt_path.name,
                            source="pred",
                            pair_index=pair_idx,
                        ),
                    ]
                    dist_df = pd.concat(dist_frames, ignore_index=True)
                    out_path = args.out_dir / f"{gt_path.stem}_distribution_hist.png"
                    plot_distribution_hist_grid(
                        dist_df,
                        metrics=distribution_metrics,
                        out_path=out_path,
                        ncols=args.ncols,
                        aggregation="pooled",
                    )
                    print(f"Wrote {out_path}")

            dataset_modes = [mode for mode in distribution_modes if mode != "single"]
            if dataset_modes:
                dist_frames = []
                for pair_idx, pair in enumerate(selected_pairs):
                    gt_idx = int(pair["gt_idx"])
                    pred_idx = int(pair["pred_idx"])
                    gt_path = gt_files[gt_idx]
                    dist_frames.append(
                        graph_distribution_rows(
                            gt_graphs[gt_idx],
                            tree_name=gt_path.name,
                            source="gt",
                            pair_index=pair_idx,
                        )
                    )
                    dist_frames.append(
                        graph_distribution_rows(
                            pred_graphs[pred_idx],
                            tree_name=gt_path.name,
                            source="pred",
                            pair_index=pair_idx,
                        )
                    )
                dist_df = pd.concat(dist_frames, ignore_index=True)
                for mode in dataset_modes:
                    out_path = args.out_dir / f"distribution_hist_{mode}.png"
                    plot_distribution_hist_grid(
                        dist_df,
                        metrics=distribution_metrics,
                        out_path=out_path,
                        ncols=args.ncols,
                        aggregation=mode,
                    )
                    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
