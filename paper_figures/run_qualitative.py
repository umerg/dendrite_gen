"""Runner for qualitative paper figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .common import PlotContext, add_shared_arguments, ensure_runner_out_dir, load_plot_context


QUALITATIVE_FIGURES = ("simple2d", "offset2d", "overlay2d", "gallery2d", "offset_gallery2d")
ALL_QUALITATIVE_PROJECTIONS = ("xy", "xz", "yz")


def run_qualitative(
    context: PlotContext,
    *,
    out_root: Path,
    projection: str = "xy",
    figures: Sequence[str] = QUALITATIVE_FIGURES,
    x_gap_scale: float = 0.05,
    y_offset_scale: float = 0.2,
    show_nodes: bool = False,
    nonroot_node_color: str | None = None,
    root_node_color: str | None = None,
    nonroot_node_size: float = 10.0,
    root_node_size: float = 18.0,
) -> None:
    """Render qualitative figures into the qualitative subfolder."""
    from .qualitative.plots_2d import (
        plot_tree_gallery_2d,
        plot_tree_offset_gallery_2d,
        plot_tree_offset_pair_2d,
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
            show_nodes=show_nodes,
            nonroot_node_color=nonroot_node_color,
            root_node_color=root_node_color,
            nonroot_node_size=nonroot_node_size,
            root_node_size=root_node_size,
        )
        print(f"Wrote {out_path}")

    if "offset_gallery2d" in figures_to_make:
        gallery_gt = [context.gt_graphs[int(pair["gt_idx"])] for pair in context.selected_pairs]
        gallery_pred = [context.pred_graphs[int(pair["pred_idx"])] for pair in context.selected_pairs]
        gallery_labels = [context.gt_files[int(pair["gt_idx"])].name for pair in context.selected_pairs]
        out_path = out_dir / f"offset_gallery2d_{projection}.png"
        plot_tree_offset_gallery_2d(
            gallery_gt,
            gallery_pred,
            gallery_labels,
            projection=projection,
            out_path=out_path,
            max_examples=len(context.selected_pairs),
            x_gap_scale=x_gap_scale,
            y_offset_scale=y_offset_scale,
            show_nodes=show_nodes,
            nonroot_node_color=nonroot_node_color,
            root_node_color=root_node_color,
            nonroot_node_size=nonroot_node_size,
            root_node_size=root_node_size,
        )
        print(f"Wrote {out_path}")

    per_pair_figures = [name for name in figures_to_make if name in {"simple2d", "offset2d", "overlay2d"}]
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
                    show_nodes=show_nodes,
                    nonroot_node_color=nonroot_node_color,
                    root_node_color=root_node_color,
                    nonroot_node_size=nonroot_node_size,
                    root_node_size=root_node_size,
                )
                print(f"Wrote {out_path}")

            if "offset2d" in per_pair_figures:
                out_path = out_dir / f"{stem}_offset_{projection}.png"
                plot_tree_offset_pair_2d(
                    gt_graph,
                    pred_graph,
                    projection=projection,
                    out_path=out_path,
                    title=f"{gt_path.name}: GT and Pred",
                    x_gap_scale=x_gap_scale,
                    y_offset_scale=y_offset_scale,
                    show_nodes=show_nodes,
                    nonroot_node_color=nonroot_node_color,
                    root_node_color=root_node_color,
                    nonroot_node_size=nonroot_node_size,
                    root_node_size=root_node_size,
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
                    show_nodes=show_nodes,
                    nonroot_node_color=nonroot_node_color,
                    root_node_color=root_node_color,
                    nonroot_node_size=nonroot_node_size,
                    root_node_size=root_node_size,
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
        help="Generate all implemented qualitative figure targets for xy, xz, and yz.",
    )
    parser.add_argument(
        "--projection",
        default="xy",
        help="2D projection to use for single-figure runs (e.g. xy, xz, zy).",
    )
    parser.add_argument(
        "--x-gap-scale",
        type=float,
        default=0.05,
        help="Horizontal gap between GT and prediction, expressed in tree widths.",
    )
    parser.add_argument(
        "--y-offset-scale",
        type=float,
        default=0.2,
        help="Vertical prediction offset, expressed in tree heights.",
    )
    parser.add_argument(
        "--show-nodes",
        action="store_true",
        help="Render nodes on top of the tree edges.",
    )
    parser.add_argument(
        "--nonroot-node-color",
        type=str,
        default=None,
        help="Color for non-root nodes. Defaults to the tree edge color.",
    )
    parser.add_argument(
        "--root-node-color",
        type=str,
        default=None,
        help="Color for the root node. Defaults to the tree edge color.",
    )
    parser.add_argument(
        "--nonroot-node-size",
        type=float,
        default=10.0,
        help="Marker size for non-root nodes.",
    )
    parser.add_argument(
        "--root-node-size",
        type=float,
        default=18.0,
        help="Marker size for the root node.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    context = load_plot_context(args)
    figures = QUALITATIVE_FIGURES if args.all else (args.figure,)
    projections = ALL_QUALITATIVE_PROJECTIONS if args.all else (args.projection,)
    for projection in projections:
        run_qualitative(
            context,
            out_root=args.out_dir,
            projection=projection,
            figures=figures,
            x_gap_scale=args.x_gap_scale,
            y_offset_scale=args.y_offset_scale,
            show_nodes=args.show_nodes,
            nonroot_node_color=args.nonroot_node_color,
            root_node_color=args.root_node_color,
            nonroot_node_size=args.nonroot_node_size,
            root_node_size=args.root_node_size,
        )


if __name__ == "__main__":
    main()
