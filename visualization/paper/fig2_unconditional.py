"""Draft Figure 2: population-level unconditional distribution comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist

from dendrite_gen.metrics.distributions import (
    CRITICAL_BRANCH_CABLE_LENGTH,
    CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG,
)
from dendrite_gen.validation.structural_metrics import (
    contraction_ratio_values,
    partition_asymmetry,
)
from dendrite_gen.visualization.paper.common import (
    DensitySamples,
    configure_paper_style,
    density_bandwidth,
    density_domain,
    plot_density_comparison,
    save_png_pdf,
    scalar_density_samples,
    tree_balanced_distribution_samples,
)
from dendrite_gen.visualization.paper.colors import (
    OURS_COLOR,
    REFERENCE_COLOR,
    SEMLAFLOW_COLOR,
)
from dendrite_gen.visualization.utils.io import (
    load_gt_file_graphs,
    load_pred_graphs_from_pickle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "data2" / "neurons_conditional" / "test"
DEFAULT_OURS_DIR = (
    PROJECT_ROOT
    / "data2"
    / "trees_test_set_generations"
    / "parity_neurons_uncond"
)
DEFAULT_OURS_PKL = None
DEFAULT_SEMLA_PKL = (
    PROJECT_ROOT / "data2" / "epoch209" / "neuron_samples_sanitised.pkl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dendrite_gen" / "outputs" / "paper_figures"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument(
        "--ours-pkl",
        type=Path,
        default=DEFAULT_OURS_PKL,
        help="Optional prediction pickle for Ours; overrides the default SWC directory.",
    )
    parser.add_argument(
        "--ours-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory of generated SWC files for Ours. When neither "
            "Ours source option is supplied, the current parity_neurons_uncond "
            "directory is used."
        ),
    )
    parser.add_argument("--ours-ema-key", default="ema_1")
    parser.add_argument("--ours-label", default="Ours")
    parser.add_argument(
        "--ours-color",
        choices=("ours", "semlaflow"),
        default="ours",
    )
    parser.add_argument("--semla-pkl", type=Path, default=DEFAULT_SEMLA_PKL)
    parser.add_argument("--semla-ema-key", default="ema_1")
    parser.add_argument("--omit-semla", action="store_true")
    parser.add_argument(
        "--botanical-z-up",
        action="store_true",
        help="Map botanical z-up coordinates to this plotter's y-up metric convention.",
    )
    parser.add_argument("--output-stem", default="fig2_unconditional_distributions")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-trees",
        type=int,
        default=None,
        help="Optional quick-preview cap applied independently to every source.",
    )
    return parser.parse_args()


def _cap(items: list, maximum: int | None) -> list:
    return items if maximum is None else items[: max(maximum, 0)]


def _map_botanical_z_to_metric_y(graphs: list[nx.Graph]) -> None:
    for graph in graphs:
        for node in graph.nodes:
            position = np.asarray(graph.nodes[node]["pos"], dtype=float)
            if position.shape != (3,):
                raise ValueError(f"Node {node!r} has invalid position shape {position.shape}.")
            graph.nodes[node]["pos"] = position[[0, 2, 1]].copy()


def _tree_balanced_value_samples(graphs: list[nx.Graph], extractor) -> DensitySamples:
    """Give every tree equal mass for an array-valued morphology extractor."""
    per_tree = []
    for graph in graphs:
        values = np.asarray(extractor(graph), dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size:
            per_tree.append(values)
    if not per_tree:
        return DensitySamples(np.zeros(0), np.zeros(0), 0)

    tree_count = len(per_tree)
    values = np.concatenate(per_tree)
    weights = np.concatenate(
        [np.full(part.size, 1.0 / (tree_count * part.size)) for part in per_tree]
    )
    return DensitySamples(values, weights, tree_count)


def paper_tree_scalar_stats(graph: nx.Graph) -> dict[str, float]:
    """Compute Figure 2 scalars in each tree's intrinsic axial frame."""
    if graph.number_of_nodes() == 0:
        return {
            "axial_extent": np.nan,
            "lateral_span": np.nan,
            "max_path_length": np.nan,
            "total_cable_length": np.nan,
            "partition_asymmetry": np.nan,
        }

    root = graph.graph.get("root")
    if root not in graph:
        raise ValueError("Every paper tree must have graph.graph['root'] set.")

    positions = {
        node: np.asarray(graph.nodes[node]["pos"], dtype=np.float64) for node in graph.nodes
    }
    points = np.stack(list(positions.values()), axis=0)
    axis = graph.graph.get("axis")
    if axis is None:
        axis = points.mean(axis=0) - positions[root]
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis_norm) or axis_norm < 1e-12:
        axis = np.array([0.0, 1.0, 0.0])
    else:
        axis = axis / axis_norm
    # ``einsum`` avoids spurious floating-point warnings emitted by Accelerate's
    # narrow-matrix ``matmul`` path for a handful of otherwise finite trees.
    axial_coordinates = np.einsum("ij,j->i", points, axis)
    axial_extent = float(np.ptp(axial_coordinates))
    lateral_points = points - np.outer(axial_coordinates, axis)
    lateral_span = float(np.max(pdist(lateral_points))) if len(points) > 1 else 0.0

    def edge_length(u, v, _attributes) -> float:
        return float(np.linalg.norm(positions[u] - positions[v]))

    path_lengths = nx.single_source_dijkstra_path_length(graph, root, weight=edge_length)
    max_path_length = float(max(path_lengths.values())) if path_lengths else 0.0
    total_cable_length = float(
        sum(np.linalg.norm(positions[u] - positions[v]) for u, v in graph.edges)
    )
    return {
        "axial_extent": axial_extent,
        "lateral_span": lateral_span,
        "max_path_length": max_path_length,
        "total_cable_length": total_cable_length,
        "partition_asymmetry": float(partition_asymmetry(graph)),
    }


def make_figure(args: argparse.Namespace) -> tuple[Path, Path]:
    configure_paper_style()
    _, reference_graphs = load_gt_file_graphs(args.reference_dir)
    if args.ours_dir is not None and args.ours_pkl is not None:
        raise ValueError("Supply at most one of --ours-dir and --ours-pkl.")
    if args.ours_pkl is not None:
        ours_graphs = load_pred_graphs_from_pickle(
            args.ours_pkl, ema_key=args.ours_ema_key
        )
    else:
        ours_dir = args.ours_dir or DEFAULT_OURS_DIR
        _, ours_graphs = load_gt_file_graphs(ours_dir)
    semla_graphs = (
        []
        if args.omit_semla
        else load_pred_graphs_from_pickle(args.semla_pkl, ema_key=args.semla_ema_key)
    )
    if args.botanical_z_up:
        _map_botanical_z_to_metric_y(reference_graphs)
        _map_botanical_z_to_metric_y(ours_graphs)
        _map_botanical_z_to_metric_y(semla_graphs)

    reference_graphs = _cap(reference_graphs, args.max_trees)
    ours_graphs = _cap(ours_graphs, args.max_trees)
    semla_graphs = _cap(semla_graphs, args.max_trees)
    if not reference_graphs or not ours_graphs or (not args.omit_semla and not semla_graphs):
        raise ValueError("Every requested source must contain at least one tree.")

    sources = {"Reference": reference_graphs}
    colors = {"Reference": REFERENCE_COLOR}
    if not args.omit_semla:
        sources["SemlaFlow"] = semla_graphs
        colors["SemlaFlow"] = SEMLAFLOW_COLOR
    sources[args.ours_label] = ours_graphs
    colors[args.ours_label] = (
        SEMLAFLOW_COLOR if args.ours_color == "semlaflow" else OURS_COLOR
    )

    print("Computing tree-level summaries...")
    scalar_rows = {
        label: [paper_tree_scalar_stats(graph) for graph in graphs]
        for label, graphs in sources.items()
    }

    metric_panels = []
    for key, title, xlabel, bandwidth_scale in (
        (
            "max_path_length",
            "Maximum path length",
            r"Maximum path length ($\mu$m)",
            1.05,
        ),
        (
            "total_cable_length",
            "Total cable length",
            r"Total cable length ($\mu$m)",
            1.05,
        ),
    ):
        samples = {
            label: scalar_density_samples([row[key] for row in rows])
            for label, rows in scalar_rows.items()
        }
        domain = density_domain(list(samples.values()), lower=0.0)
        bandwidth = density_bandwidth(
            samples["Reference"], domain, scale_factor=bandwidth_scale
        )
        metric_panels.append((title, xlabel, samples, domain, bandwidth))

    print("Computing tree-balanced within-tree distributions...")
    for distribution_name, title, xlabel, domain_bounds, bandwidth_scale in (
        (
            CRITICAL_BRANCH_CABLE_LENGTH,
            "Branch cable length",
            r"Branch cable length ($\mu$m)",
            (0.0, None),
            1.15,
        ),
        (
            CRITICAL_BRANCH_CHORD_SIBLING_ANGLE_DEG,
            "Bifurcation angle",
            "Bifurcation angle (degrees)",
            (0.0, 180.0),
            1.10,
        ),
    ):
        samples = {
            label: tree_balanced_distribution_samples(graphs, distribution_name)
            for label, graphs in sources.items()
        }
        domain = density_domain(
            list(samples.values()), lower=domain_bounds[0], upper=domain_bounds[1]
        )
        bandwidth = density_bandwidth(
            samples["Reference"], domain, scale_factor=bandwidth_scale
        )
        if distribution_name == CRITICAL_BRANCH_CABLE_LENGTH:
            domain = (-2.5 * bandwidth, domain[1])
        metric_panels.append((title, xlabel, samples, domain, bandwidth))

    contraction_samples = {
        label: _tree_balanced_value_samples(graphs, contraction_ratio_values)
        for label, graphs in sources.items()
    }
    contraction_data_domain = density_domain(
        list(contraction_samples.values()), lower=0.0, upper=1.0
    )
    contraction_bandwidth = density_bandwidth(
        contraction_samples["Reference"],
        contraction_data_domain,
        scale_factor=1.05,
    )
    contraction_domain = (
        contraction_data_domain[0],
        contraction_data_domain[1] + 2.5 * contraction_bandwidth,
    )
    metric_panels.append(
        (
            "Contraction ratio",
            "Contraction ratio",
            contraction_samples,
            contraction_domain,
            contraction_bandwidth,
        )
    )

    asymmetry_samples = {
        label: scalar_density_samples([row["partition_asymmetry"] for row in rows])
        for label, rows in scalar_rows.items()
    }
    asymmetry_domain = density_domain(
        list(asymmetry_samples.values()), lower=0.0, upper=1.0
    )
    asymmetry_bandwidth = density_bandwidth(
        asymmetry_samples["Reference"], asymmetry_domain, scale_factor=1.05
    )
    metric_panels.append(
        (
            "Partition asymmetry",
            "Partition asymmetry",
            asymmetry_samples,
            asymmetry_domain,
            asymmetry_bandwidth,
        )
    )

    figure = plt.figure(figsize=(7.2, 2.25))
    grid = figure.add_gridspec(
        2,
        3,
        left=0.06,
        right=0.99,
        top=0.96,
        bottom=0.12,
        hspace=0.52,
        wspace=0.27,
    )
    axes = []
    for panel, axis in zip(metric_panels, grid.subplots().flat):
        title, xlabel, samples, domain, bandwidth = panel
        plot_density_comparison(
            axis,
            samples,
            colors=colors,
            domain=domain,
            bandwidth=bandwidth,
            xlabel=xlabel,
        )
        if title == "Branch cable length":
            axis.set_xticks([tick for tick in axis.get_xticks() if tick >= 0.0])
        elif title == "Contraction ratio":
            axis.set_xticks([tick for tick in axis.get_xticks() if tick <= 1.0])
        axes.append(axis)

    axes[0].legend(
        loc="upper right",
        bbox_to_anchor=(0.99, 0.98),
        frameon=False,
        handlelength=1.5,
        handletextpad=0.45,
        labelspacing=0.25,
        borderaxespad=0.0,
        fontsize=7.0,
    )
    output_stem = args.out_dir / args.output_stem
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
