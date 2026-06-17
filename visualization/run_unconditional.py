"""Runner for unconditioned GT/pred distribution diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import PlotContext, add_shared_arguments, ensure_runner_out_dir, load_plot_context
from .stats.unconditional import (
    compute_unconditional_pca,
    graph_unconditional_feature_row,
    plot_unconditional_pca,
    plot_unconditional_pca_by_feature,
    unconditional_feature_columns,
)


def run_unconditional(
    context: PlotContext,
    *,
    out_root: Path,
    max_pairs: int | None = None,
    point_alpha: float = 0.55,
) -> None:
    """Render unconditioned population-level figures into the unconditional subfolder."""
    out_dir = ensure_runner_out_dir(out_root, "unconditional")
    stats_pairs = context.pairs if max_pairs is None else context.pairs[: max(0, max_pairs)]
    if max_pairs is None:
        print(f"Unconditional PCA will use all {len(stats_pairs)} paired tree slots.")
    else:
        print(f"Unconditional PCA will use {len(stats_pairs)} paired tree slots.")

    feature_rows: list[dict[str, object]] = []
    for pair_idx, pair in enumerate(stats_pairs):
        gt_idx = int(pair["gt_idx"])
        pred_idx = int(pair["pred_idx"])
        gt_path = context.gt_files[gt_idx]
        feature_rows.append(
            graph_unconditional_feature_row(
                context.gt_graphs[gt_idx],
                tree_name=gt_path.name,
                source="gt",
                pair_index=pair_idx,
            )
        )
        feature_rows.append(
            graph_unconditional_feature_row(
                context.pred_graphs[pred_idx],
                tree_name=f"pred_{pred_idx}",
                source="pred",
                pair_index=pair_idx,
            )
        )

    feature_df = pd.DataFrame(feature_rows)
    feature_columns = unconditional_feature_columns()
    coord_df, loadings_df, metadata_df = compute_unconditional_pca(
        feature_df,
        feature_columns=feature_columns,
    )
    out_path = out_dir / "tree_feature_pca.png"
    plot_unconditional_pca(
        coord_df,
        loadings_df,
        metadata_df,
        out_path=out_path,
        point_alpha=point_alpha,
    )
    print(f"Wrote {out_path}")

    feature_paths = plot_unconditional_pca_by_feature(
        coord_df,
        feature_df,
        metadata_df,
        feature_columns=feature_columns,
        out_dir=out_dir / "tree_feature_pca_by_feature",
        point_alpha=min(1.0, max(0.0, point_alpha + 0.17)),
    )
    print(
        f"Wrote {len(feature_paths)} feature-colored PCA plot(s) to "
        f"{out_dir / 'tree_feature_pca_by_feature'}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate unconditioned population-level GT/pred diagnostics. "
            "The GT and predicted samples are compared as sets, not as matched pairs."
        )
    )
    add_shared_arguments(parser, default_max_pairs=None)
    parser.add_argument(
        "--point-alpha",
        type=float,
        default=0.55,
        help="Point opacity for the unconditional PCA scatter.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    context = load_plot_context(args)
    run_unconditional(
        context,
        out_root=args.out_dir,
        max_pairs=args.max_pairs,
        point_alpha=args.point_alpha,
    )


if __name__ == "__main__":
    main()
