"""Runner for 3D cylinder tree renderings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .common import PlotContext, add_shared_arguments, ensure_runner_out_dir, load_plot_context
from .utils.styles import DEFAULT_3D_ANGLES


CYLINDER_PLOT_MODES = ("pair", "gt", "pred", "all")
CYLINDER_BACKENDS = ("matplotlib", "pyvista", "plotly")


def _angle_tag(elev: float, azim: float) -> str:
    return f"e{int(round(elev))}_a{int(round(azim))}"


def run_cylinder_trees(
    context: PlotContext,
    *,
    out_root: Path,
    plot_mode: str = "pair",
    angles: Sequence[tuple[float, float]] = (DEFAULT_3D_ANGLES[0],),
    segments: int = 16,
    radius_attr: str = "radius",
    radius_scale: float = 1.0,
    default_radius: float = 1.0,
    synthesize_radii: bool = False,
    twig_radius: float | None = None,
    twig_radius_scale: float = 0.002,
    pipe_exponent: float = 0.35,
    length_exponent: float = 0.12,
    radius_smoothing_passes: int = 1,
    curve_branches: bool = False,
    curve_subsegments: int = 5,
    curve_wiggle_scale: float = 0.02,
    curve_momentum: float = 0.75,
    curve_seed: int = 0,
    backend: str = "matplotlib",
    show_axes: bool = False,
    cap_ends: bool = False,
    show_joints: bool = True,
    joint_scale: float = 1.05,
    joint_segments: int = 10,
) -> None:
    """Render selected GT/pred pairs as cylinder models."""
    from .geometry.curves import with_curved_branches
    from .geometry.radii import SYNTHESIZED_RADIUS_ATTR, with_synthesized_radii

    if plot_mode not in CYLINDER_PLOT_MODES:
        raise ValueError(f"Unsupported cylinder plot mode '{plot_mode}'.")
    if backend not in CYLINDER_BACKENDS:
        raise ValueError(f"Unsupported cylinder backend '{backend}'.")

    if backend == "plotly":
        from .qualitative.plots_3d_plotly import (
            plot_tree_cylinder_pair_plotly as plot_tree_cylinder_pair,
            plot_tree_cylinder_single_plotly as plot_tree_cylinder_single,
        )

        out_dir_name = "cylinders_plotly"
        file_ext = "html"
    elif backend == "pyvista":
        from .qualitative.plots_3d_pyvista import (
            plot_tree_cylinder_pair_pyvista as plot_tree_cylinder_pair,
            plot_tree_cylinder_single_pyvista as plot_tree_cylinder_single,
        )

        out_dir_name = "cylinders_pyvista"
        file_ext = "png"
    else:
        from .qualitative.plots_3d import (
            plot_tree_cylinder_pair_3d as plot_tree_cylinder_pair,
            plot_tree_cylinder_single_3d as plot_tree_cylinder_single,
        )

        out_dir_name = "cylinders"
        file_ext = "png"

    if curve_branches:
        out_dir_name = f"{out_dir_name}_curved"
    out_dir = ensure_runner_out_dir(out_root, out_dir_name)

    for pair in context.selected_pairs:
        gt_idx = int(pair["gt_idx"])
        pred_idx = int(pair["pred_idx"])
        gt_path = context.gt_files[gt_idx]
        gt_graph = context.gt_graphs[gt_idx]
        pred_graph = context.pred_graphs[pred_idx]
        stem = gt_path.stem
        render_radius_attr = radius_attr

        if curve_branches:
            curve_kwargs = dict(
                subsegments=curve_subsegments,
                wiggle_scale=curve_wiggle_scale,
                momentum=curve_momentum,
                seed=curve_seed,
                radius_attrs=(radius_attr,),
            )
            gt_graph = with_curved_branches(gt_graph, **curve_kwargs)
            pred_graph = with_curved_branches(pred_graph, **curve_kwargs)

        if synthesize_radii:
            synthesis_kwargs = dict(
                twig_radius=twig_radius,
                twig_radius_scale=twig_radius_scale,
                pipe_exponent=pipe_exponent,
                length_exponent=length_exponent,
                smoothing_passes=radius_smoothing_passes,
            )
            gt_graph = with_synthesized_radii(
                gt_graph,
                radius_attr=SYNTHESIZED_RADIUS_ATTR,
                **synthesis_kwargs,
            )
            pred_graph = with_synthesized_radii(
                pred_graph,
                radius_attr=SYNTHESIZED_RADIUS_ATTR,
                **synthesis_kwargs,
            )
            render_radius_attr = SYNTHESIZED_RADIUS_ATTR

        for elev, azim in angles:
            tag = _angle_tag(float(elev), float(azim))
            common_kwargs = dict(
                elev=float(elev),
                azim=float(azim),
                segments=segments,
                radius_attr=render_radius_attr,
                radius_scale=radius_scale,
                default_radius=default_radius,
                show_axes=show_axes,
                cap_ends=cap_ends,
            )
            if backend == "plotly":
                common_kwargs.update(
                    show_joints=show_joints,
                    joint_scale=joint_scale,
                    joint_segments=joint_segments,
                )

            if plot_mode in {"pair", "all"}:
                out_path = out_dir / f"{stem}_cylinder_pair_{tag}.{file_ext}"
                plot_tree_cylinder_pair(
                    gt_graph,
                    pred_graph,
                    out_path=out_path,
                    title_gt=f"GT: {gt_path.name}",
                    title_pred=f"Pred idx {pred_idx}",
                    **common_kwargs,
                )
                print(f"Wrote {out_path}")

            if plot_mode in {"gt", "all"}:
                out_path = out_dir / f"{stem}_gt_cylinder_{tag}.{file_ext}"
                plot_tree_cylinder_single(
                    gt_graph,
                    out_path=out_path,
                    title=f"GT: {gt_path.name}",
                    **common_kwargs,
                )
                print(f"Wrote {out_path}")

            if plot_mode in {"pred", "all"}:
                out_path = out_dir / f"{stem}_pred{pred_idx}_cylinder_{tag}.{file_ext}"
                plot_tree_cylinder_single(
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
        default=16,
        help="Number of radial segments per branch cylinder.",
    )
    parser.add_argument(
        "--backend",
        choices=CYLINDER_BACKENDS,
        default="matplotlib",
        help="Rendering backend. Non-default backends write to backend-specific subfolders.",
    )
    parser.add_argument(
        "--curve-branches",
        action="store_true",
        help="Render branch paths as smooth endpoint-preserving random curves.",
    )
    parser.add_argument(
        "--curve-subsegments",
        type=int,
        default=5,
        help="Number of curved centerline subsegments per original branch edge.",
    )
    parser.add_argument(
        "--curve-wiggle-scale",
        type=float,
        default=0.02,
        help="Curve wiggle amplitude as a fraction of branch path length.",
    )
    parser.add_argument(
        "--curve-momentum",
        type=float,
        default=0.75,
        help="Memory factor for smooth curve noise; larger values bend more coherently.",
    )
    parser.add_argument(
        "--curve-seed",
        type=int,
        default=0,
        help="Seed for deterministic branch curves.",
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
        "--synthesize-radii",
        action="store_true",
        help="Synthesize visual radii before rendering instead of reading --radius-attr.",
    )
    parser.add_argument(
        "--twig-radius",
        type=float,
        default=None,
        help="Terminal twig radius for synthesized radii. Defaults to graph size times --twig-radius-scale.",
    )
    parser.add_argument(
        "--twig-radius-scale",
        type=float,
        default=0.002,
        help=(
            "Graph bounding-box diagonal fraction used as synthesized twig radius "
            "when --twig-radius is omitted."
        ),
    )
    parser.add_argument(
        "--pipe-exponent",
        type=float,
        default=0.35,
        help="Subtree tip-count exponent for synthesized radii.",
    )
    parser.add_argument(
        "--length-exponent",
        type=float,
        default=0.12,
        help="Downstream-length exponent for synthesized radii.",
    )
    parser.add_argument(
        "--radius-smoothing-passes",
        type=int,
        default=1,
        help="Number of parent-child monotonicity passes after synthesized path smoothing.",
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
    parser.add_argument(
        "--no-joints",
        action="store_true",
        help="Disable Plotly joint spheres at tree endpoints and branchpoints.",
    )
    parser.add_argument(
        "--joint-scale",
        type=float,
        default=1.05,
        help="Plotly joint sphere radius multiplier relative to the node radius.",
    )
    parser.add_argument(
        "--joint-segments",
        type=int,
        default=10,
        help="Plotly joint sphere mesh resolution.",
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
        synthesize_radii=args.synthesize_radii,
        twig_radius=args.twig_radius,
        twig_radius_scale=args.twig_radius_scale,
        pipe_exponent=args.pipe_exponent,
        length_exponent=args.length_exponent,
        radius_smoothing_passes=args.radius_smoothing_passes,
        curve_branches=args.curve_branches,
        curve_subsegments=args.curve_subsegments,
        curve_wiggle_scale=args.curve_wiggle_scale,
        curve_momentum=args.curve_momentum,
        curve_seed=args.curve_seed,
        backend=args.backend,
        show_axes=args.show_axes,
        cap_ends=args.cap_ends,
        show_joints=not args.no_joints,
        joint_scale=args.joint_scale,
        joint_segments=args.joint_segments,
    )


if __name__ == "__main__":
    main()
