"""Compare each selected tree with its own copy under axial rotations."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

try:
    from dendrite_gen.metrics.chamfer import (
        point_chamfer_distance,
        sample_tree_points,
    )
    from dendrite_gen.metrics.fused_gw import (
        fused_gromov_wasserstein_distance_prepared,
        prepare_fused_gw_tree,
    )
    from dendrite_gen.metrics.so2 import rotate_points_about_axis
    from dendrite_gen.utils.data_loading import load_swc_graph
except ModuleNotFoundError as exc:
    if exc.name != "dendrite_gen":
        raise
    from metrics.chamfer import (  # type: ignore
        point_chamfer_distance,
        sample_tree_points,
    )
    from metrics.fused_gw import (  # type: ignore
        fused_gromov_wasserstein_distance_prepared,
        prepare_fused_gw_tree,
    )
    from metrics.so2 import rotate_points_about_axis  # type: ignore
    from utils.data_loading import load_swc_graph  # type: ignore

from .dataset import TreeRecord, discover_tree_records
from .frame import transform_scientific_y_to_internal_z
from .matrix_metrics import (
    CHAMFER,
    DISTRIBUTION_SIBLING_ANGLE_WASSERSTEIN,
    FUSED_GROMOV_WASSERSTEIN,
    TMD_PATH_WASSERSTEIN,
    build_matrix_metric,
)


SELF_ROTATION_METRICS = (
    CHAMFER,
    FUSED_GROMOV_WASSERSTEIN,
    TMD_PATH_WASSERSTEIN,
    DISTRIBUTION_SIBLING_ANGLE_WASSERSTEIN,
)


@dataclass(frozen=True)
class MetricProfile:
    """Display and interpretation metadata for one angle profile."""

    name: str
    display_name: str
    rotation_treatment: str


METRIC_PROFILES = {
    CHAMFER: MetricProfile(
        name=CHAMFER,
        display_name="Chamfer",
        rotation_treatment="before SO(2) minimization",
    ),
    FUSED_GROMOV_WASSERSTEIN: MetricProfile(
        name=FUSED_GROMOV_WASSERSTEIN,
        display_name="Fused Gromov-Wasserstein",
        rotation_treatment="xyz features, before SO(2) minimization",
    ),
    TMD_PATH_WASSERSTEIN: MetricProfile(
        name=TMD_PATH_WASSERSTEIN,
        display_name="Path-barcode Wasserstein",
        rotation_treatment="intrinsically SO(2)-invariant",
    ),
    DISTRIBUTION_SIBLING_ANGLE_WASSERSTEIN: MetricProfile(
        name=DISTRIBUTION_SIBLING_ANGLE_WASSERSTEIN,
        display_name="Sibling-angle Wasserstein",
        rotation_treatment="intrinsically SO(2)-invariant",
    ),
}


def select_rotation_trees(
    records: Sequence[TreeRecord],
    tree_count: int,
    *,
    seed: int,
) -> tuple[TreeRecord, ...]:
    """Select one tree from each of several randomly selected classes."""

    if tree_count <= 0:
        raise ValueError("tree_count must be positive")
    groups: dict[int, list[TreeRecord]] = {}
    for record in sorted(
        records,
        key=lambda item: (item.cell_class, item.tree_id, item.swc_path.as_posix()),
    ):
        groups.setdefault(record.cell_class, []).append(record)
    if len(groups) < tree_count:
        raise ValueError(
            f"Requested {tree_count} different classes, but only "
            f"{len(groups)} classes are available."
        )

    rng = np.random.default_rng(seed)
    selected_classes = rng.choice(sorted(groups), size=tree_count, replace=False)
    selected: list[TreeRecord] = []
    for cell_class in selected_classes:
        group = groups[int(cell_class)]
        selected.append(group[int(rng.choice(len(group)))])
    return tuple(selected)


def rotate_graph_about_root_z(graph: nx.Graph, angle_rad: float) -> nx.Graph:
    """Return a copy rotated around the root and the internal z axis."""

    root = graph.graph.get("root")
    if root not in graph:
        raise ValueError("graph.graph['root'] must name an existing root node")
    root_position = np.asarray(graph.nodes[root].get("pos"), dtype=np.float64)
    if root_position.shape != (3,) or not np.all(np.isfinite(root_position)):
        raise ValueError("The root must have a finite 3-D 'pos' attribute")

    rotated = graph.copy()
    for node in graph.nodes:
        position = np.asarray(graph.nodes[node].get("pos"), dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"Node {node!r} must have a finite 3-D 'pos' attribute")
        centered = position - root_position
        rotated.nodes[node]["pos"] = root_position + rotate_points_about_axis(
            centered[None, :],
            angle_rad,
            (0.0, 0.0, 1.0),
        )[0]
    return rotated


def metric_angle_curve(
    graph: nx.Graph,
    metric_name: str,
    angles_deg: np.ndarray,
    *,
    chamfer_spacing: float,
    fgw_max_nodes: int,
) -> np.ndarray:
    """Evaluate one metric against rotated copies of the same tree."""

    angles_rad = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    if metric_name == CHAMFER:
        points = sample_tree_points(
            graph,
            spacing=chamfer_spacing,
            center_root=True,
        )
        return np.asarray(
            [
                point_chamfer_distance(
                    points,
                    rotate_points_about_axis(
                        points,
                        float(angle_rad),
                        (0.0, 0.0, 1.0),
                    ),
                    squared=False,
                    reduction="sum",
                )
                for angle_rad in angles_rad
            ],
            dtype=np.float64,
        )

    if metric_name == FUSED_GROMOV_WASSERSTEIN:
        if fgw_max_nodes > 0 and graph.number_of_nodes() > fgw_max_nodes:
            raise ValueError(
                f"FGW node guard rejected a {graph.number_of_nodes()}-node tree; "
                f"configured limit is {fgw_max_nodes}."
            )
        prepared = prepare_fused_gw_tree(graph, mass_mode="cable_length")
        values: list[float] = []
        for angle_rad in angles_rad:
            rotated = replace(
                prepared,
                centered_positions=rotate_points_about_axis(
                    prepared.centered_positions,
                    float(angle_rad),
                    (0.0, 0.0, 1.0),
                ),
            )
            result = fused_gromov_wasserstein_distance_prepared(
                prepared,
                rotated,
                feature_mode="xyz",
                alpha=0.5,
                normalize=True,
                quotient_so2=False,
                max_iter=1_000,
                tol=1e-9,
            )
            values.append(float(result.value))
        return np.asarray(values, dtype=np.float64)

    if metric_name not in {
        TMD_PATH_WASSERSTEIN,
        DISTRIBUTION_SIBLING_ANGLE_WASSERSTEIN,
    }:
        raise KeyError(f"Unsupported self-rotation metric: {metric_name!r}")

    metric = build_matrix_metric(
        metric_name,
        so2_grid_size=72,
        so2_refine=False,
        so2_refinement_tolerance=1e-8,
        fgw_max_nodes=fgw_max_nodes,
    )
    prepared = metric.prepare(graph)
    return np.asarray(
        [
            metric.compare(
                prepared,
                metric.prepare(rotate_graph_about_root_z(graph, float(angle_rad))),
            )
            for angle_rad in angles_rad
        ],
        dtype=np.float64,
    )


def _default_dataset_root() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "neurons_conditional"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare selected ground-truth trees with their own axially rotated "
            "copies."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_default_dataset_root(),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--trees", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=SELF_ROTATION_METRICS,
        default=list(SELF_ROTATION_METRICS),
    )
    parser.add_argument(
        "--angle-step-deg",
        type=float,
        default=2.0,
        help="Angular sampling interval; must divide 360 degrees.",
    )
    parser.add_argument("--chamfer-spacing", type=float, default=1.0)
    parser.add_argument("--fgw-max-nodes", type=int, default=1_000)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def _plot_profiles(
    rows: Sequence[dict[str, object]],
    records: Sequence[TreeRecord],
    metric_names: Sequence[str],
    output_dir: Path,
) -> tuple[Path, Path]:
    colors = ("#287B7A", "#C26D3A", "#7663A6", "#4F789C")
    line_styles = ("-", "--", "-.", ":")
    column_count = 2
    row_count = int(np.ceil(len(metric_names) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(7.15, 2.75 * row_count + 0.45),
        sharex=True,
        squeeze=False,
    )

    legend_handles = []
    legend_labels = []
    for metric_index, metric_name in enumerate(metric_names):
        axis = axes.flat[metric_index]
        profile = METRIC_PROFILES[metric_name]
        metric_rows = [row for row in rows if row["metric"] == metric_name]
        numerically_zero = max(
            abs(float(row["distance"])) for row in metric_rows
        ) < 1e-10
        for tree_index, record in enumerate(records, start=1):
            tree_rows = [
                row for row in metric_rows if row["tree_index"] == tree_index
            ]
            x = np.asarray(
                [row["angle_deg"] for row in tree_rows],
                dtype=np.float64,
            )
            y = np.asarray(
                [row["distance"] for row in tree_rows],
                dtype=np.float64,
            )
            plotted_y = np.zeros_like(y) if numerically_zero else y
            (line,) = axis.plot(
                x,
                plotted_y,
                color=colors[(tree_index - 1) % len(colors)],
                linestyle=line_styles[(tree_index - 1) % len(line_styles)],
                linewidth=1.65,
            )
            if metric_index == 0:
                legend_handles.append(line)
                legend_labels.append(
                    f"{record.cell_type}: {record.tree_id[-6:]}"
                )

        axis.set_title(
            profile.display_name,
            loc="left",
            fontsize=10,
            fontweight="bold",
            color="#19344D",
            pad=14,
        )
        axis.text(
            0.0,
            1.015,
            profile.rotation_treatment,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="#5C6B76",
        )
        axis.set_ylabel("Dissimilarity", fontsize=8.5)
        axis.set_xlim(0.0, 360.0)
        axis.set_xticks(np.arange(0.0, 361.0, 60.0))
        axis.grid(color="#CBD7DE", linewidth=0.55, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8.2, colors="#22313D")
        axis.margins(x=0)

        if numerically_zero:
            axis.set_ylim(-1e-12, 1e-12)
            axis.set_yticks([0.0])
            axis.text(
                0.98,
                0.88,
                "zero to numerical precision",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#5C6B76",
            )
        else:
            axis.set_ylim(bottom=0.0)

    for metric_index in range(len(metric_names), row_count * column_count):
        axes.flat[metric_index].set_visible(False)
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("Rotation of the copy (degrees)", fontsize=8.5)

    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(len(legend_labels), 3),
        frameon=False,
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.935), h_pad=1.5, w_pad=1.2)
    pdf_path = output_dir / "self_rotation_metric_profiles.pdf"
    png_path = output_dir / "self_rotation_metric_profiles.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.trees <= 0:
        raise ValueError("--trees must be positive")
    if (
        not np.isfinite(args.angle_step_deg)
        or args.angle_step_deg <= 0.0
        or not np.isclose(
            360.0 / args.angle_step_deg,
            round(360.0 / args.angle_step_deg),
        )
    ):
        raise ValueError("--angle-step-deg must be positive and divide 360")
    if not np.isfinite(args.chamfer_spacing) or args.chamfer_spacing <= 0.0:
        raise ValueError("--chamfer-spacing must be positive")
    if args.fgw_max_nodes < 0:
        raise ValueError("--fgw-max-nodes cannot be negative")

    metric_names = tuple(dict.fromkeys(args.metrics))
    dataset_root = args.dataset_root.expanduser().resolve()
    records = discover_tree_records(dataset_root, split_dirs=(args.split,))
    selected = select_rotation_trees(records, args.trees, seed=args.seed)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    angle_count = int(round(360.0 / args.angle_step_deg))
    angles_deg = np.linspace(0.0, 360.0, angle_count + 1)
    csv_rows: list[dict[str, object]] = []
    tree_metadata: list[dict[str, object]] = []
    curve_metadata: list[dict[str, object]] = []

    for tree_index, record in enumerate(selected, start=1):
        graph = transform_scientific_y_to_internal_z(
            load_swc_graph(record.swc_path)
        )
        tree_metadata.append(
            {
                "tree_index": tree_index,
                "tree_id": record.tree_id,
                "cell_class": record.cell_class,
                "cell_type": record.cell_type,
                "swc_path": str(record.swc_path),
                "node_count": graph.number_of_nodes(),
            }
        )
        for metric_name in metric_names:
            values = metric_angle_curve(
                graph,
                metric_name,
                angles_deg,
                chamfer_spacing=args.chamfer_spacing,
                fgw_max_nodes=args.fgw_max_nodes,
            )
            curve_metadata.append(
                {
                    "tree_index": tree_index,
                    "metric": metric_name,
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "value_at_zero": float(values[0]),
                    "value_at_360": float(values[-1]),
                }
            )
            for angle_deg, distance in zip(angles_deg, values, strict=True):
                csv_rows.append(
                    {
                        "tree_index": tree_index,
                        "cell_class": record.cell_class,
                        "cell_type": record.cell_type,
                        "tree_id": record.tree_id,
                        "metric": metric_name,
                        "angle_deg": float(angle_deg),
                        "distance": float(distance),
                    }
                )

    pdf_path, png_path = _plot_profiles(
        csv_rows,
        selected,
        metric_names,
        output_dir,
    )
    csv_path = output_dir / "self_rotation_metric_profiles.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    payload: dict[str, object] = {
        "dataset_root": str(dataset_root),
        "split": args.split,
        "tree_count": len(selected),
        "seed": args.seed,
        "metrics": [
            {
                "name": name,
                "display_name": METRIC_PROFILES[name].display_name,
                "rotation_treatment": METRIC_PROFILES[name].rotation_treatment,
            }
            for name in metric_names
        ],
        "angle_step_deg": float(args.angle_step_deg),
        "chamfer_spacing": float(args.chamfer_spacing),
        "fgw_max_nodes": int(args.fgw_max_nodes),
        "frame_contract": {
            "scientific_axis": "y",
            "coordinate_map": "(x, y, z) -> (x, -z, y)",
            "rotation_axis_after_transform": "z",
        },
        "trees": tree_metadata,
        "curves": curve_metadata,
        "artifacts": {
            "pdf": str(pdf_path),
            "png": str(png_path),
            "csv": str(csv_path),
        },
    }
    json_path = output_dir / "run.json"
    payload["artifacts"]["metadata"] = str(json_path)  # type: ignore[index]
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = run(args)
    except (FileNotFoundError, KeyError, NotADirectoryError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
