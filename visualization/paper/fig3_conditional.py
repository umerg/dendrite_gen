"""Figure 3: target-specific morphology comparisons."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from dendrite_gen.metrics.distributions import (
    CRITICAL_BRANCH_CABLE_LENGTH,
    CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG,
)
from dendrite_gen.utils.tmd import compute_tmd_barcode_diagram
from dendrite_gen.visualization.paper.common import (
    configure_paper_style,
    density_bandwidth,
    density_domain,
    merge_density_samples,
    plot_density_comparison,
    plot_persistence_overlay,
    plot_tree_panel,
    projected_limits,
    save_png_pdf,
    single_tree_distribution_samples,
    symmetric_depth_norm,
)
from dendrite_gen.visualization.paper.colors import (
    NEUTRAL_COLOR,
    OURS_COLOR,
    REFERENCE_COLOR,
    TREE_COLOR,
    TREE_DEPTH_CMAP,
)
from dendrite_gen.visualization.utils.io import (
    load_gt_file_graphs,
    load_pred_graphs_from_pickle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "data" / "neurons_conditional" / "val"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "data" / "step_5500.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dendrite_gen" / "outputs" / "paper_figures"
SO2_AXIS = (0.0, 1.0, 0.0)

# Layout knobs for quick paper-figure iteration. Column widths are ordered as
# target tree, generated tree, persistence diagram, branch KDE, and angle KDE.
FIGURE_WIDTH_INCHES = 7.2
FIGURE_BASE_HEIGHT_INCHES = 0.70
FIGURE_ROW_HEIGHT_INCHES = 1.20
TREE_COLUMN_WIDTH = 1.20
PERSISTENCE_COLUMN_WIDTH = 0.68
DENSITY_COLUMN_WIDTH = 1.00
COLUMN_WIDTH_RATIOS = (
    TREE_COLUMN_WIDTH,
    TREE_COLUMN_WIDTH,
    PERSISTENCE_COLUMN_WIDTH,
    DENSITY_COLUMN_WIDTH,
    DENSITY_COLUMN_WIDTH,
)
GRID_LEFT = 0.055
GRID_RIGHT = 0.985
GRID_TOP = 0.86
GRID_BOTTOM = 0.12
COLUMN_SPACING = 0.16
ROW_SPACING = 0.03
DENSITY_BOX_ASPECT = 0.42


@dataclass(frozen=True)
class PairSpec:
    index: int
    expected_filename: str


PAIR_SPECS = (
    PairSpec(2, "864691134885587194_520228.swc"),
    PairSpec(4, "864691134885702394_63854.swc"),
    PairSpec(5, "864691134885703162_585631.swc"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--pred-pkl", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--ema-key", default="ema_1")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--projection",
        choices=("xy", "xz", "yz", "yx", "zx", "zy"),
        default="xy",
        help="Coordinate projection used for the target and generated tree panels.",
    )
    parser.add_argument(
        "--pair-indices",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Explicit aligned reference/prediction indices. Supplying these uses the "
            "given dataset instead of the legacy placeholder pairing assertion."
        ),
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=2,
        help=(
            "Number of configured target/generated pairs to show. The main-paper "
            "default is 2; increase this for the appendix."
        ),
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help=(
            "Output filename stem (without .png or .pdf). By default, the "
            "two-row figure uses fig3_conditional_placeholder and expanded "
            "figures use fig3_conditional_appendix."
        ),
    )
    return parser.parse_args()


def _assert_pairing(gt_files: list[Path], pred_graphs: list) -> None:
    if len(gt_files) != 2528 or len(pred_graphs) != 2528:
        raise ValueError(
            "Placeholder Figure 3 expects the exact 2,528-file neurons_conditional/val "
            "dataset and the aligned step_5500 prediction list."
        )
    for pair in PAIR_SPECS:
        actual = gt_files[pair.index].name
        if actual != pair.expected_filename:
            raise ValueError(
                f"Pair index {pair.index} resolved to {actual!r}, expected "
                f"{pair.expected_filename!r}; refusing to silently mispair trees."
            )


def _compute_path_diagram(graph):
    _, diagram = compute_tmd_barcode_diagram(
        graph,
        filtration="path",
        normalize_mode="minmax",
        weight_edges_by_euclidean=True,
        simplify_to_critical_tree=True,
        uhat=SO2_AXIS,
    )
    return diagram


def make_figure(args: argparse.Namespace) -> tuple[Path, Path]:
    configure_paper_style()
    gt_files, gt_graphs = load_gt_file_graphs(args.reference_dir)
    pred_graphs = load_pred_graphs_from_pickle(args.pred_pkl, ema_key=args.ema_key)
    if args.pair_indices is None:
        _assert_pairing(gt_files, pred_graphs)
        pair_specs = PAIR_SPECS
    else:
        if len(set(args.pair_indices)) != len(args.pair_indices):
            raise ValueError("--pair-indices must not contain duplicates.")
        invalid = [
            index
            for index in args.pair_indices
            if index < 0 or index >= len(gt_files) or index >= len(pred_graphs)
        ]
        if invalid:
            raise IndexError(
                "Pair indices must exist in both inputs "
                f"(reference={len(gt_files)}, predictions={len(pred_graphs)}); "
                f"invalid: {invalid}"
            )
        pair_specs = tuple(
            PairSpec(index=index, expected_filename=gt_files[index].name)
            for index in args.pair_indices
        )
    if not 1 <= args.num_examples <= len(pair_specs):
        raise ValueError(
            f"--num-examples must be between 1 and {len(pair_specs)}, "
            f"got {args.num_examples}."
        )

    # Compute the shared plotting scales from every curated pair, even when the
    # main-paper rendering shows only the first two. That way an example looks
    # identical when the same layout is expanded for the appendix.
    all_pair_data = []
    for pair in pair_specs:
        target = gt_graphs[pair.index]
        generated = pred_graphs[pair.index]
        all_pair_data.append(
            {
                "spec": pair,
                "target": target,
                "generated": generated,
                "target_diagram": _compute_path_diagram(target),
                "generated_diagram": _compute_path_diagram(generated),
                "branch_target": single_tree_distribution_samples(
                    target, CRITICAL_BRANCH_CABLE_LENGTH
                ),
                "branch_generated": single_tree_distribution_samples(
                    generated, CRITICAL_BRANCH_CABLE_LENGTH
                ),
                "angle_target": single_tree_distribution_samples(
                    target, CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG
                ),
                "angle_generated": single_tree_distribution_samples(
                    generated, CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG
                ),
            }
        )
    pair_data = all_pair_data[: args.num_examples]

    branch_reference = merge_density_samples(
        [row["branch_target"] for row in all_pair_data]
    )
    branch_all = [
        sample
        for row in all_pair_data
        for sample in (row["branch_target"], row["branch_generated"])
    ]
    branch_data_domain = density_domain(branch_all, lower=0.0)
    branch_bandwidth = density_bandwidth(
        branch_reference, branch_data_domain, scale_factor=1.10
    )
    # Let the smoothed curve close naturally just to the left of the physical
    # zero boundary. Only nonnegative ticks are shown because lengths remain
    # nonnegative; the extension is KDE support, not data support.
    branch_domain = (-2.5 * branch_bandwidth, branch_data_domain[1])

    angle_reference = merge_density_samples(
        [row["angle_target"] for row in all_pair_data]
    )
    angle_all = [
        sample
        for row in all_pair_data
        for sample in (row["angle_target"], row["angle_generated"])
    ]
    angle_domain = density_domain(angle_all, lower=0.0, upper=180.0)
    angle_bandwidth = density_bandwidth(angle_reference, angle_domain, scale_factor=1.05)

    depth_norm = symmetric_depth_norm(
        [
            graph
            for row in all_pair_data
            for graph in (row["target"], row["generated"])
        ],
        projection=args.projection,
        percentile=80.0,
        nonlinear_width=0.25,
    )

    figure_height = (
        FIGURE_BASE_HEIGHT_INCHES + FIGURE_ROW_HEIGHT_INCHES * len(pair_data)
    )
    figure = plt.figure(figsize=(FIGURE_WIDTH_INCHES, figure_height))
    grid = figure.add_gridspec(
        len(pair_data),
        5,
        width_ratios=COLUMN_WIDTH_RATIOS,
        left=GRID_LEFT,
        right=GRID_RIGHT,
        top=GRID_TOP,
        bottom=GRID_BOTTOM,
        hspace=ROW_SPACING,
        wspace=COLUMN_SPACING,
    )
    column_titles = (
        "Conditioning target",
        "Generated sample",
        "Path persistence",
        "Branch cable length",
        "Bifurcation angle",
    )
    colors = {"Target": REFERENCE_COLOR, "Generated": OURS_COLOR}

    axes = []
    for row_index, row in enumerate(pair_data):
        row_axes = [figure.add_subplot(grid[row_index, column]) for column in range(5)]
        axes.append(row_axes)
        pair_limits = projected_limits(
            [row["target"], row["generated"]],
            projection=args.projection,
            pad_fraction=0.025,
        )
        plot_tree_panel(
            row_axes[0],
            row["target"],
            color=TREE_COLOR,
            projection=args.projection,
            limits=pair_limits,
            linewidth=0.75,
            depth_coloring=True,
            depth_cmap=TREE_DEPTH_CMAP,
            depth_norm=depth_norm,
            depth_segment_length=1.0,
        )
        plot_tree_panel(
            row_axes[1],
            row["generated"],
            color=TREE_COLOR,
            projection=args.projection,
            limits=pair_limits,
            linewidth=0.75,
            depth_coloring=True,
            depth_cmap=TREE_DEPTH_CMAP,
            depth_norm=depth_norm,
            depth_segment_length=1.0,
        )
        row_axes[0].text(
            -0.11,
            0.5,
            f"Example {row_index + 1}",
            transform=row_axes[0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=7.0,
            color=NEUTRAL_COLOR,
        )

        plot_persistence_overlay(
            row_axes[2], row["target_diagram"], row["generated_diagram"]
        )
        plot_density_comparison(
            row_axes[3],
            {"Target": row["branch_target"], "Generated": row["branch_generated"]},
            colors=colors,
            domain=branch_domain,
            bandwidth=branch_bandwidth,
        )
        row_axes[3].set_xticks(
            [
                tick
                for tick in row_axes[3].get_xticks()
                if 0.0 <= tick <= branch_data_domain[1]
            ]
        )
        plot_density_comparison(
            row_axes[4],
            {"Target": row["angle_target"], "Generated": row["angle_generated"]},
            colors=colors,
            domain=angle_domain,
            bandwidth=angle_bandwidth,
        )
        for density_axis in row_axes[3:5]:
            density_axis.set_box_aspect(DENSITY_BOX_ASPECT)
            density_axis.set_anchor("C")

        if row_index == len(pair_data) - 1:
            row_axes[2].set_xlabel("Birth", labelpad=2.0)
            row_axes[2].set_ylabel("Death", labelpad=2.0)
            row_axes[3].set_xlabel(r"Length ($\mu$m)", labelpad=2.0)
            row_axes[4].set_xlabel("Angle (degrees)", labelpad=2.0)
        else:
            row_axes[2].set_xticklabels([])
            row_axes[2].set_yticklabels([])
            row_axes[3].set_xticklabels([])
            row_axes[4].set_xticklabels([])

    for column, title in enumerate(column_titles):
        slot = grid[0, column].get_position(figure)
        figure.text(
            0.5 * (slot.x0 + slot.x1),
            0.865,
            title,
            ha="center",
            va="bottom",
            fontsize=9.0,
        )

    handles, labels = axes[0][3].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.985),
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.8,
    )
    output_name = args.output_stem or (
        "fig3_conditional_placeholder"
        if args.num_examples == 2
        else "fig3_conditional_appendix"
    )
    output_stem = args.out_dir / output_name
    paths = save_png_pdf(figure, output_stem)
    plt.close(figure)
    return paths


def main() -> None:
    args = _parse_args()
    png_path, pdf_path = make_figure(args)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
