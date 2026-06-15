"""Runner for 3D cylinder tree renderings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .common import PlotContext, add_shared_arguments, ensure_runner_out_dir, load_plot_context
from .utils.styles import DEFAULT_3D_ANGLES


CYLINDER_PLOT_MODES = ("pair", "gt", "pred", "all")


def _angle_tag(elev: float, azim: float) -> str:
    return f"e{int(round(elev))}_a{int(round(azim))}"


def run_cylinder_trees(
    context: PlotContext,
    *,
    out_root: Path,
    plot_mode: str = "pair",
    angles: Sequence[tuple[float, float]] = (DEFAULT_3D_ANGLES[0],),
    segments: int = 12,
    radius_attr: str = "radius",
    radius_scale: float = 1.0,
    default_radius: float = 1.0,
    show_axes: bool = False,
    cap_ends: bool = False,
) -> None:
    """Render selected GT/pred pairs as cylinder models."""
    from .qualitative.plots_3d import (
        plot_tree_cylinder_pair_3d,
        plot_tree_cylinder_single_3d,
    )

    if plot_mode not in CYLINDER_PLOT_MODES:
        raise ValueError(f"Unsupported cylinder plot mode '{plot_mode}'.")

    out_dir = ensure_runner_out_dir(out_root, "cylinders")
    for pair in context.selected_pairs:
        gt_idx = int(pair["gt_idx"])
        pred_idx = int(pair["pred_idx"])
        gt_path = context.gt_files[gt_idx]
        gt_graph = context.gt_graphs[gt_idx]
        pred_graph = context.pred_graphs[pred_idx]
        stem = gt_path.stem

        for elev, azim in angles:
            tag = _angle_tag(float(elev), float(azim))
            common_kwargs = dict(
                elev=float(elev),
                azim=float(azim),
                segments=segments,
                radius_attr=radius_attr,
                radius_scale=radius_scale,
                default_radius=default_radius,
                show_axes=show_axes,
                cap_ends=cap_ends,
            )

            if plot_mode in {"pair", "all"}:
                out_path = out_dir / f"{stem}_cylinder_pair_{tag}.png"
                plot_tree_cylinder_pair_3d(
                    gt_graph,
                    pred_graph,
                    out_path=out_path,
                    title_gt=f"GT: {gt_path.name}",
                    title_pred=f"Pred idx {pred_idx}",
                    **common_kwargs,
                )
                print(f"Wrote {out_path}")

            if plot_mode in {"gt", "all"}:
                out_path = out_dir / f"{stem}_gt_cylinder_{tag}.png"
                plot_tree_cylinder_single_3d(
                    gt_graph,
                    out_path=out_path,
                    title=f"GT: {gt_path.name}",
                    **common_kwargs,
                )
                print(f"Wrote {out_path}")

            if plot_mode in {"pred", "all"}:
                out_path = out_dir / f"{stem}_pred{pred_idx}_cylinder_{tag}.png"
                plot_tree_cylinder_single_3d(
                    pred_graph,
                    out_path=out_path,
                    title=f"Pred idx {pred_idx}",
                    **common_kwargs,
                )
                print(f"Wrote {out_path}")


def _parse_angle(text: str) -> tuple[float, float]:
    normalized = text.replace(":", ",")
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Angles must be formatted as ELEV,AZIM, e.g. 20,30.")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Angles must contain numeric elev and azim values.") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 3D cylinder renderings of tree graphs.")
    add_shared_arguments(parser, default_max_pairs=1)
    parser.add_argument(
        "--plot-mode",
        choices=CYLINDER_PLOT_MODES,
        default="pair",
        help="Which cylinder views to write.",
    )
    parser.add_argument(
        "--angle",
        action="append",
        type=_parse_angle,
        default=None,
        help="Camera angle as ELEV,AZIM. May be passed multiple times.",
    )
    parser.add_argument(
        "--all-default-angles",
        action="store_true",
        help="Render the shared default 3D angle set instead of one angle.",
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=12,
        help="Number of radial segments per branch cylinder.",
    )
    parser.add_argument(
        "--radius-attr",
        type=str,
        default="radius",
        help="Node attribute containing per-node radii. Missing values use --default-radius.",
    )
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to all existing or default radii.",
    )
    parser.add_argument(
        "--default-radius",
        type=float,
        default=1.0,
        help="Radius used for nodes without a valid radius attribute.",
    )
    parser.add_argument(
        "--show-axes",
        action="store_true",
        help="Keep 3D axes visible.",
    )
    parser.add_argument(
        "--cap-ends",
        action="store_true",
        help="Close cylinder ends. Usually off to avoid visible internal seams.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    context = load_plot_context(args)
    if args.all_default_angles:
        angles = tuple((float(elev), float(azim)) for elev, azim in DEFAULT_3D_ANGLES)
    elif args.angle:
        angles = tuple(args.angle)
    else:
        angles = (DEFAULT_3D_ANGLES[0],)
    run_cylinder_trees(
        context,
        out_root=args.out_dir,
        plot_mode=args.plot_mode,
        angles=angles,
        segments=args.segments,
        radius_attr=args.radius_attr,
        radius_scale=args.radius_scale,
        default_radius=args.default_radius,
        show_axes=args.show_axes,
        cap_ends=args.cap_ends,
    )


if __name__ == "__main__":
    main()
