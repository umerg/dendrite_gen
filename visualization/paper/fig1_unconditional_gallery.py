"""Free-layout gallery for unconditional reference and model samples."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any, Mapping, Sequence
import warnings

import matplotlib.pyplot as plt
import numpy as np

from dendrite_gen.visualization.paper.common import (
    configure_paper_style,
    plot_tree_panel,
    save_png_pdf,
    symmetric_depth_norm,
)
from dendrite_gen.visualization.paper.colors import (
    BACKGROUND_COLOR,
    GUIDE_ACCENT_COLOR,
    GUIDE_COLOR,
    TREE_COLOR,
    TREE_DEPTH_CMAP,
    resolve_paper_color,
)
from dendrite_gen.visualization.utils.io import (
    load_gt_file_graphs,
    load_pred_graphs_from_pickle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_suffix(".toml")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dendrite_gen" / "outputs" / "paper_figures"


@dataclass(frozen=True)
class ColumnSpec:
    key: str
    label: str
    origin: tuple[float, float]
    width: float
    color: str
    fontsize: float
    fontweight: str


@dataclass(frozen=True)
class SampleSpec:
    column: str
    source: str
    index: int
    offset: tuple[float, float]
    rotation_deg: float
    scale: float
    zorder: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--depth-normalization",
        choices=("figure", "tree"),
        default=None,
        help="Override figure.depth_normalization from the TOML config.",
    )
    parser.add_argument(
        "--show-guides",
        action="store_true",
        help="Overlay column boundaries, sample centers, and sample indices.",
    )
    return parser.parse_args()


def _pair(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain exactly two numbers.")
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two numbers.")
    pair = (float(value[0]), float(value[1]))
    if not np.all(np.isfinite(pair)):
        raise ValueError(f"{name} must contain finite numbers.")
    return pair


def _quadruple(value: object, name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain exactly four numbers.")
    if len(value) != 4:
        raise ValueError(f"{name} must contain exactly four numbers.")
    result = tuple(float(item) for item in value)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite numbers.")
    return result


def _load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Layout config does not exist: {path}")
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _project_path(raw_path: object) -> Path:
    path = Path(str(raw_path)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_sources(config: Mapping[str, Any]) -> dict[str, list]:
    sources: dict[str, list] = {}
    for name, source in config.items():
        if not isinstance(source, Mapping):
            raise ValueError(f"Source {name!r} must be a TOML table.")
        kind = str(source.get("kind", "prediction_pickle"))
        path = _project_path(source["path"])
        if kind == "prediction_pickle":
            ema_key = source.get("ema_key")
            sources[name] = load_pred_graphs_from_pickle(path, ema_key=ema_key)
        elif kind == "swc_directory":
            _, sources[name] = load_gt_file_graphs(path)
        else:
            raise ValueError(
                f"Unsupported source kind {kind!r} for {name!r}; expected "
                "'prediction_pickle' or 'swc_directory'."
            )
    if not sources:
        raise ValueError("The config must define at least one [sources.*] table.")
    return sources


def _sample_specs(raw_samples: object) -> list[SampleSpec]:
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("The config must contain at least one [[samples]] entry.")

    samples = []
    for position, raw in enumerate(raw_samples):
        if not isinstance(raw, Mapping):
            raise ValueError(f"samples[{position}] must be a TOML table.")
        scale = float(raw.get("scale", 1.0))
        rotation_deg = float(raw.get("rotation_deg", 0.0))
        zorder = float(raw.get("zorder", 2.0))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"samples[{position}].scale must be positive and finite.")
        if not np.isfinite(rotation_deg) or not np.isfinite(zorder):
            raise ValueError(
                f"samples[{position}] rotation and z-order must be finite."
            )
        samples.append(
            SampleSpec(
                column=str(raw["column"]),
                source=str(raw["source"]),
                index=int(raw["index"]),
                offset=_pair(raw["offset"], f"samples[{position}].offset"),
                rotation_deg=rotation_deg,
                scale=scale,
                zorder=zorder,
            )
        )
    return samples


def _column_specs(raw_columns: object) -> list[ColumnSpec]:
    if not isinstance(raw_columns, list) or not raw_columns:
        raise ValueError("The config must contain at least one [[columns]] entry.")

    columns = []
    for position, raw in enumerate(raw_columns):
        if not isinstance(raw, Mapping):
            raise ValueError(f"columns[{position}] must be a TOML table.")
        width = float(raw["width"])
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError(f"columns[{position}].width must be positive and finite.")
        columns.append(
            ColumnSpec(
                key=str(raw["key"]),
                label=str(raw["label"]),
                origin=_pair(raw["origin"], f"columns[{position}].origin"),
                width=width,
                color=resolve_paper_color(str(raw.get("color", "text"))),
                fontsize=float(raw.get("fontsize", 9.0)),
                fontweight=str(raw.get("fontweight", "semibold")),
            )
        )
    if len({column.key for column in columns}) != len(columns):
        raise ValueError("Every [[columns]] key must be unique.")
    return columns


def _sample_center(
    sample: SampleSpec,
    columns_by_key: Mapping[str, ColumnSpec],
) -> tuple[float, float]:
    column = columns_by_key[sample.column]
    return (
        column.origin[0] + sample.offset[0],
        column.origin[1] - sample.offset[1],
    )


def _draw_scale_bar(
    axis: plt.Axes,
    config: Mapping[str, Any],
    sample_scales: Sequence[float],
) -> None:
    if not bool(config.get("show", True)):
        return
    if not np.allclose(sample_scales, sample_scales[0]):
        warnings.warn(
            "The scale bar was hidden because samples use different display scales.",
            stacklevel=2,
        )
        return

    x_limits = axis.get_xlim()
    y_limits = axis.get_ylim()
    default_offset = (50.0, abs(y_limits[1] - y_limits[0]) - 35.0)
    offset = _pair(config.get("offset", default_offset), "scale_bar.offset")
    position = (
        min(x_limits) + offset[0],
        max(y_limits) - offset[1],
    )
    physical_length = float(config.get("length_um", 100.0))
    display_length = physical_length * sample_scales[0]
    color = resolve_paper_color(str(config.get("color", "neutral")))
    axis.plot(
        [position[0], position[0] + display_length],
        [position[1], position[1]],
        color=color,
        linewidth=float(config.get("linewidth", 1.4)),
        solid_capstyle="butt",
        zorder=100,
    )
    axis.text(
        position[0] + 0.5 * display_length,
        position[1] + float(config.get("label_offset", 14.0)),
        rf"{physical_length:g} $\mu$m",
        ha="center",
        va="bottom",
        color=color,
        fontsize=float(config.get("fontsize", 6.5)),
        zorder=100,
    )


def _draw_guides(
    axis: plt.Axes,
    figure_config: Mapping[str, Any],
    columns: Sequence[ColumnSpec],
    samples: Sequence[SampleSpec],
    sample_centers: Sequence[tuple[float, float]],
) -> None:
    x_limits = _pair(figure_config["x_limits"], "figure.x_limits")
    y_limits = _pair(figure_config["y_limits"], "figure.y_limits")
    for column in columns:
        left = column.origin[0]
        right = left + column.width
        top = column.origin[1]
        axis.plot(
            [left, right, right, left, left],
            [top, top, y_limits[0], y_limits[0], top],
            color=GUIDE_COLOR,
            linewidth=0.6,
            linestyle="--",
            zorder=0,
        )
    for sample, center in zip(samples, sample_centers):
        axis.scatter(
            [center[0]],
            [center[1]],
            s=7,
            facecolor=BACKGROUND_COLOR,
            edgecolor=GUIDE_ACCENT_COLOR,
            linewidth=0.6,
            zorder=200,
        )
        axis.text(
            center[0] + 0.008 * (x_limits[1] - x_limits[0]),
            center[1],
            f"{sample.column}:{sample.index}",
            color=GUIDE_ACCENT_COLOR,
            fontsize=5.0,
            ha="left",
            va="center",
            zorder=200,
        )


def make_figure(args: argparse.Namespace) -> tuple[Path, Path]:
    configure_paper_style()
    config = _load_config(args.config)
    figure_config = config["figure"]
    if not isinstance(figure_config, Mapping):
        raise ValueError("[figure] must be a TOML table.")

    sources = _load_sources(config.get("sources", {}))
    columns = _column_specs(config.get("columns", []))
    samples = _sample_specs(config.get("samples", []))
    columns_by_key = {column.key: column for column in columns}

    selected_graphs = []
    for sample in samples:
        if sample.column not in columns_by_key:
            raise ValueError(f"Unknown column {sample.column!r} in sample config.")
        if sample.source not in sources:
            raise ValueError(f"Unknown source {sample.source!r} in sample config.")
        graphs = sources[sample.source]
        if not 0 <= sample.index < len(graphs):
            raise IndexError(
                f"Sample index {sample.index} is outside source {sample.source!r} "
                f"with {len(graphs)} trees."
            )
        selected_graphs.append(graphs[sample.index])
    sample_centers = [
        _sample_center(sample, columns_by_key) for sample in samples
    ]

    projection = str(figure_config.get("projection", "xy"))
    depth_normalization = args.depth_normalization or str(
        figure_config.get("depth_normalization", "figure")
    )
    if depth_normalization not in {"figure", "tree"}:
        raise ValueError("figure.depth_normalization must be 'figure' or 'tree'.")
    depth_scale = str(figure_config.get("depth_scale", "asinh"))
    if depth_scale not in {"asinh", "linear"}:
        raise ValueError("figure.depth_scale must be 'asinh' or 'linear'.")
    nonlinear_width = (
        float(figure_config.get("depth_linear_width", 0.25))
        if depth_scale == "asinh"
        else None
    )
    depth_percentile = float(figure_config.get("depth_percentile", 80.0))
    shared_depth_norm = (
        symmetric_depth_norm(
            selected_graphs,
            projection=projection,
            percentile=depth_percentile,
            nonlinear_width=nonlinear_width,
        )
        if depth_normalization == "figure"
        else None
    )

    figure_size = _pair(
        figure_config.get("size_inches", (7.2, 3.6)), "figure.size_inches"
    )
    x_limits = _pair(figure_config["x_limits"], "figure.x_limits")
    y_limits = _pair(figure_config["y_limits"], "figure.y_limits")
    axes_rect = _quadruple(
        figure_config.get("axes_rect", (0.01, 0.01, 0.98, 0.98)),
        "figure.axes_rect",
    )

    figure = plt.figure(figsize=figure_size)
    axis = figure.add_axes(axes_rect)
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()

    for sample, graph, center in sorted(
        zip(samples, selected_graphs, sample_centers),
        key=lambda item: item[0].zorder,
    ):
        depth_norm = shared_depth_norm or symmetric_depth_norm(
            [graph],
            projection=projection,
            percentile=depth_percentile,
            nonlinear_width=nonlinear_width,
        )
        plot_tree_panel(
            axis,
            graph,
            color=TREE_COLOR,
            projection=projection,
            linewidth=float(figure_config.get("linewidth", 0.72)),
            limits=(x_limits, y_limits),
            depth_coloring=True,
            depth_cmap=TREE_DEPTH_CMAP,
            depth_norm=depth_norm,
            depth_segment_length=float(
                figure_config.get("depth_segment_length", 1.0)
            ),
            layout_center=center,
            layout_rotation_deg=sample.rotation_deg,
            layout_scale=sample.scale,
            layout_zorder=sample.zorder,
        )

    for column in columns:
        axis.text(
            column.origin[0] + 0.5 * column.width,
            column.origin[1],
            column.label,
            color=column.color,
            fontsize=column.fontsize,
            fontweight=column.fontweight,
            ha="center",
            va="top",
            zorder=100,
        )

    _draw_scale_bar(axis, config.get("scale_bar", {}), [s.scale for s in samples])
    if args.show_guides or bool(figure_config.get("show_guides", False)):
        _draw_guides(axis, figure_config, columns, samples, sample_centers)

    output_stem = args.out_dir / str(
        config.get("output", {}).get("stem", "fig1_unconditional_gallery")
    )
    paths = save_png_pdf(figure, output_stem)
    plt.close(figure)
    return paths


def main() -> None:
    args = _parse_args()
    print(f"Using layout config {args.config.resolve()}")
    png_path, pdf_path = make_figure(args)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
