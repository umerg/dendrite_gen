"""Runner for TMD persistence-diagram paper figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import PlotContext, add_shared_arguments, ensure_runner_out_dir, load_plot_context


DEFAULT_FILTRATIONS = ("path", "height", "rho")


def run_tmd_figures(
    context: PlotContext,
    *,
    out_root: Path,
    filtrations: tuple[str, ...] = DEFAULT_FILTRATIONS,
    normalize_mode: str = "minmax",
    ncols: int = 3,
    point_alpha: float = 0.75,
) -> None:
    """Render one TMD persistence-diagram grid per selected GT/pred pair."""
    from dendrite_gen.utils.tmd import compute_tmd_barcode_diagram

    from .tmd.plots import plot_tmd_persistence_grid

    out_dir = ensure_runner_out_dir(out_root, "tmd_figures")

    for pair in context.selected_pairs:
        gt_idx = int(pair["gt_idx"])
        pred_idx = int(pair["pred_idx"])
        gt_path = context.gt_files[gt_idx]
        gt_graph = context.gt_graphs[gt_idx]
        pred_graph = context.pred_graphs[pred_idx]

        gt_diagrams: dict[str, object] = {}
        pred_diagrams: dict[str, object] = {}
        for filtration in filtrations:
            _, gt_diag = compute_tmd_barcode_diagram(
                gt_graph,
                filtration=filtration,
                normalize_mode=normalize_mode,
                weight_edges_by_euclidean=True,
                simplify_to_critical_tree=True,
            )
            _, pred_diag = compute_tmd_barcode_diagram(
                pred_graph,
                filtration=filtration,
                normalize_mode=normalize_mode,
                weight_edges_by_euclidean=True,
                simplify_to_critical_tree=True,
            )
            gt_diagrams[filtration] = gt_diag
            pred_diagrams[filtration] = pred_diag

        out_path = out_dir / f"{gt_path.stem}_tmd_grid.png"
        plot_tmd_persistence_grid(
            gt_diagrams,
            pred_diagrams,
            filtrations=filtrations,
            out_path=out_path,
            normalize_mode=normalize_mode,
            ncols=ncols,
            point_alpha=point_alpha,
        )
        print(f"Wrote {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TMD persistence-diagram figures.")
    add_shared_arguments(parser)
    parser.add_argument(
        "--filtrations",
        nargs="+",
        default=list(DEFAULT_FILTRATIONS),
        help="Filtrations to include (e.g. path height rho).",
    )
    parser.add_argument(
        "--normalize-mode",
        choices=["minmax", "max", "none"],
        default="minmax",
        help="Normalization mode used before computing persistence diagrams.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=3,
        help="Number of columns for the persistence-diagram grid.",
    )
    parser.add_argument(
        "--point-alpha",
        type=float,
        default=0.75,
        help="Opacity of GT/pred persistence-diagram points.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    context = load_plot_context(args)
    run_tmd_figures(
        context,
        out_root=args.out_dir,
        filtrations=tuple(args.filtrations),
        normalize_mode=args.normalize_mode,
        ncols=args.ncols,
        point_alpha=args.point_alpha,
    )


if __name__ == "__main__":
    main()
