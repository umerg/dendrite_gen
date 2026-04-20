"""Runner for qualitative paper figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .common import PlotContext, add_shared_arguments, ensure_runner_out_dir, load_plot_context


QUALITATIVE_FIGURES = ("simple2d", "overlay2d", "gallery2d")


def run_qualitative(
    context: PlotContext,
    *,
    out_root: Path,
    projection: str = "xy",
    figures: Sequence[str] = QUALITATIVE_FIGURES,
) -> None:
    """Render qualitative figures into the qualitative subfolder."""
    from .qualitative.plots_2d import (
        plot_tree_gallery_2d,
        plot_tree_overlay_2d,
        plot_tree_pair_2d,
    )

    out_dir = ensure_runner_out_dir(out_root, "qualitative")
    figures_to_make = list(figures)

    if "gallery2d" in figures_to_make:
        gallery_gt = [context.gt_graphs[int(pair["gt_idx"])] for pair in context.selected_pairs]
        gallery_pred = [context.pred_graphs[int(pair["pred_idx"])] for pair in context.selected_pairs]
        gallery_labels = [context.gt_files[int(pair["gt_idx"])].name for pair in context.selected_pairs]
        out_path = out_dir / f"gallery2d_{projection}.png"
        plot_tree_gallery_2d(
            gallery_gt,
            gallery_pred,
            gallery_labels,
            projection=projection,
            out_path=out_path,
            max_examples=len(context.selected_pairs),
            overlay=True,
        )
        print(f"Wrote {out_path}")

    per_pair_figures = [name for name in figures_to_make if name in {"simple2d", "overlay2d"}]
    if per_pair_figures:
        for pair in context.selected_pairs:
            gt_idx = int(pair["gt_idx"])
            pred_idx = int(pair["pred_idx"])
            gt_path = context.gt_files[gt_idx]
            gt_graph = context.gt_graphs[gt_idx]
            pred_graph = context.pred_graphs[pred_idx]
            stem = gt_path.stem

            if "simple2d" in per_pair_figures:
                out_path = out_dir / f"{stem}_pair_{projection}.png"
                plot_tree_pair_2d(
                    gt_graph,
                    pred_graph,
                    projection=projection,
                    out_path=out_path,
                    title_gt=f"GT: {gt_path.name}",
                    title_pred=f"Pred idx {pred_idx}",
                )
                print(f"Wrote {out_path}")

            if "overlay2d" in per_pair_figures:
                out_path = out_dir / f"{stem}_overlay_{projection}.png"
                plot_tree_overlay_2d(
                    gt_graph,
                    pred_graph,
                    projection=projection,
                    out_path=out_path,
                    title=f"{gt_path.name}: GT vs Pred",
                )
                print(f"Wrote {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate qualitative paper figures.")
    add_shared_arguments(parser)
    parser.add_argument(
        "--figure",
        choices=QUALITATIVE_FIGURES,
        default="simple2d",
        help="Qualitative figure to generate.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all implemented qualitative figure targets.",
    )
    parser.add_argument(
        "--projection",
        default="xy",
        help="2D projection to use (e.g. xy, xz, zy).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    context = load_plot_context(args)
    figures = QUALITATIVE_FIGURES if args.all else (args.figure,)
    run_qualitative(
        context,
        out_root=args.out_dir,
        projection=args.projection,
        figures=figures,
    )


if __name__ == "__main__":
    main()
