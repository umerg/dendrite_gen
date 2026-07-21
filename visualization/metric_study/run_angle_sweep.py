"""Plot a tree dissimilarity over one complete relative SO(2) rotation."""

from __future__ import annotations

import argparse
import csv
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
    from dendrite_gen.metrics.so2 import rotate_points_about_axis
    from dendrite_gen.utils.data_loading import load_swc_graph
except ModuleNotFoundError as exc:
    if exc.name != "dendrite_gen":
        raise
    from metrics.chamfer import (  # type: ignore
        point_chamfer_distance,
        sample_tree_points,
    )
    from metrics.so2 import rotate_points_about_axis  # type: ignore
    from utils.data_loading import load_swc_graph  # type: ignore

from .dataset import TreeRecord, discover_tree_records
from .frame import transform_scientific_y_to_internal_z


def select_same_class_pairs(
    records: Sequence[TreeRecord],
    pair_count: int,
    *,
    seed: int,
) -> tuple[tuple[TreeRecord, TreeRecord], ...]:
    """Select one disjoint pair from each of several random classes."""

    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    groups: dict[int, list[TreeRecord]] = {}
    for record in sorted(
        records,
        key=lambda item: (item.cell_class, item.tree_id, item.swc_path.as_posix()),
    ):
        groups.setdefault(record.cell_class, []).append(record)
    eligible = sorted(
        cell_class for cell_class, group in groups.items() if len(group) >= 2
    )
    if len(eligible) < pair_count:
        raise ValueError(
            f"Requested {pair_count} different classes, but only "
            f"{len(eligible)} contain at least two trees."
        )

    rng = np.random.default_rng(seed)
    selected_classes = rng.choice(eligible, size=pair_count, replace=False)
    pairs: list[tuple[TreeRecord, TreeRecord]] = []
    for cell_class in selected_classes:
        group = groups[int(cell_class)]
        indices = rng.choice(len(group), size=2, replace=False)
        pairs.append((group[int(indices[0])], group[int(indices[1])]))
    return tuple(pairs)


def chamfer_angle_curve(
    graph_a: nx.Graph,
    graph_b: nx.Graph,
    angles_deg: np.ndarray,
    *,
    spacing: float,
) -> tuple[np.ndarray, int, int]:
    """Evaluate Chamfer after rotating the second tree through the given angles."""

    points_a = sample_tree_points(graph_a, spacing=spacing, center_root=True)
    points_b = sample_tree_points(graph_b, spacing=spacing, center_root=True)
    values = np.asarray(
        [
            point_chamfer_distance(
                points_a,
                rotate_points_about_axis(
                    points_b,
                    np.deg2rad(float(angle_deg)),
                    (0.0, 0.0, 1.0),
                ),
                squared=False,
                reduction="sum",
            )
            for angle_deg in angles_deg
        ],
        dtype=np.float64,
    )
    return values, len(points_a), len(points_b)


def _default_dataset_root() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "neurons_conditional"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Chamfer distance against relative axial rotation."
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=_default_dataset_root()
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--angle-step-deg",
        type=float,
        default=2.0,
        help="Angular sampling interval; must divide 360 degrees.",
    )
    parser.add_argument("--chamfer-spacing", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.pairs <= 0:
        raise ValueError("--pairs must be positive")
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

    dataset_root = args.dataset_root.expanduser().resolve()
    records = discover_tree_records(dataset_root, split_dirs=(args.split,))
    pairs = select_same_class_pairs(records, args.pairs, seed=args.seed)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    angle_count = int(round(360.0 / args.angle_step_deg))
    angles_deg = np.linspace(0.0, 360.0, angle_count + 1)
    curves: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []

    for pair_index, (record_a, record_b) in enumerate(pairs, start=1):
        graph_a = transform_scientific_y_to_internal_z(
            load_swc_graph(record_a.swc_path)
        )
        graph_b = transform_scientific_y_to_internal_z(
            load_swc_graph(record_b.swc_path)
        )
        values, point_count_a, point_count_b = chamfer_angle_curve(
            graph_a,
            graph_b,
            angles_deg,
            spacing=args.chamfer_spacing,
        )
        minimum_index = int(np.argmin(values[:-1]))
        curve = {
            "pair_index": pair_index,
            "cell_class": record_a.cell_class,
            "cell_type": record_a.cell_type,
            "tree_a_id": record_a.tree_id,
            "tree_b_id": record_b.tree_id,
            "point_count_a": point_count_a,
            "point_count_b": point_count_b,
            "minimum_angle_deg": float(angles_deg[minimum_index]),
            "minimum_distance": float(values[minimum_index]),
            "maximum_distance": float(np.max(values)),
        }
        curves.append(curve)
        for angle_deg, distance in zip(angles_deg, values, strict=True):
            csv_rows.append(
                {
                    **{
                        key: value
                        for key, value in curve.items()
                        if key
                        not in {
                            "minimum_angle_deg",
                            "minimum_distance",
                            "maximum_distance",
                        }
                    },
                    "angle_deg": float(angle_deg),
                    "chamfer_distance": float(distance),
                }
            )

    figure, axes = plt.subplots(
        len(curves),
        1,
        figsize=(7.15, 2.25 * len(curves)),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    for axis, curve in zip(axes[:, 0], curves, strict=True):
        pair_rows = [
            row for row in csv_rows if row["pair_index"] == curve["pair_index"]
        ]
        x = np.asarray([row["angle_deg"] for row in pair_rows], dtype=np.float64)
        y = np.asarray(
            [row["chamfer_distance"] for row in pair_rows], dtype=np.float64
        )
        axis.plot(x, y, color="#287B7A", linewidth=1.8)
        axis.scatter(
            [curve["minimum_angle_deg"]],
            [curve["minimum_distance"]],
            color="#19344D",
            s=25,
            zorder=3,
        )
        axis.axvline(
            float(curve["minimum_angle_deg"]),
            color="#19344D",
            linewidth=0.8,
            alpha=0.5,
        )
        axis.set_title(
            f'{curve["cell_type"]}: {curve["tree_a_id"]} vs '
            f'{curve["tree_b_id"]}',
            loc="left",
            fontsize=9.5,
            fontweight="bold",
            color="#19344D",
        )
        axis.text(
            0.99,
            0.94,
            f'minimum {curve["minimum_angle_deg"]:.0f} deg  '
            f'd={curve["minimum_distance"]:.3g}',
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="#22313D",
        )
        axis.set_ylabel("Chamfer", fontsize=9)
        axis.grid(axis="both", color="#CBD7DE", linewidth=0.55, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8.5, colors="#22313D")
        axis.margins(x=0)

    axes[-1, 0].set_xlabel("Relative rotation angle (degrees)", fontsize=9)
    axes[-1, 0].set_xticks(np.arange(0.0, 361.0, 60.0))
    pdf_path = output_dir / "chamfer_angle_profiles.pdf"
    png_path = output_dir / "chamfer_angle_profiles.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    csv_path = output_dir / "chamfer_angle_profiles.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    payload: dict[str, object] = {
        "metric": "chamfer",
        "dataset_root": str(dataset_root),
        "split": args.split,
        "pair_count": len(pairs),
        "seed": args.seed,
        "angle_step_deg": float(args.angle_step_deg),
        "chamfer_spacing": float(args.chamfer_spacing),
        "scientific_axis": "y",
        "curves": curves,
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
    except (NotADirectoryError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
