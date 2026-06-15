"""TOML-driven runner for curated offset-gallery figures."""

from __future__ import annotations

import argparse
import tomllib
from argparse import Namespace
from pathlib import Path

from .common import ensure_runner_out_dir, load_plot_context, select_pairs_by_gt_names


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a curated offset gallery from a TOML spec.")
    parser.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="Path to the TOML spec describing the curated offset gallery.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output-root override. If omitted, the spec value is used.",
    )
    return parser


def _load_spec(spec_path: Path) -> dict:
    with Path(spec_path).open("rb") as f:
        return tomllib.load(f)


def _sample_specs_from_toml(spec: dict) -> list[dict]:
    samples = list(spec.get("samples", []))
    if samples:
        return samples
    neurons = list(spec.get("neurons", []))
    return [{"name": name} for name in neurons]


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    from .qualitative.plots_2d import plot_tree_offset_gallery_2d

    spec = _load_spec(args.spec)

    sample_specs = _sample_specs_from_toml(spec)
    tree_names = [sample["name"] for sample in sample_specs]
    if not tree_names:
        raise ValueError("Spec must contain a non-empty `samples` list or `neurons` list.")

    load_args = Namespace(
        gt_dir=Path(spec["gt_dir"]),
        pred_pkl=Path(spec["pred_pkl"]),
        ema_key=spec.get("ema_key"),
        max_pairs=10**9,
        out_dir=Path(spec.get("out_dir", "dendrite_gen/outputs/paper_figures")),
    )
    context = load_plot_context(load_args)
    context = select_pairs_by_gt_names(context, tree_names)

    out_root = Path(args.out_dir) if args.out_dir is not None else Path(spec.get("out_dir", "dendrite_gen/outputs/paper_figures"))
    out_dir = ensure_runner_out_dir(out_root, "curated_offset_gallery")

    projection = spec.get("projection", "xy")
    gallery_name = spec.get("gallery_name", "curated_offset_gallery")
    default_x_gap_scale = float(spec.get("x_gap_scale", 0.05))
    default_y_offset_scale = float(spec.get("y_offset_scale", 0.2))
    subplot_wspace = float(spec.get("subplot_wspace", 0.06))
    subplot_hspace = float(spec.get("subplot_hspace", 0.10))
    x_gap_scales = [
        float(sample.get("x_gap_scale", default_x_gap_scale))
        for sample in sample_specs
    ]
    y_offset_scales = [
        float(sample.get("y_offset_scale", default_y_offset_scale))
        for sample in sample_specs
    ]
    panel_dxs = [
        float(sample.get("panel_dx", 0.0))
        for sample in sample_specs
    ]
    panel_dys = [
        float(sample.get("panel_dy", 0.0))
        for sample in sample_specs
    ]

    gallery_gt = [context.gt_graphs[int(pair["gt_idx"])] for pair in context.selected_pairs]
    gallery_pred = [context.pred_graphs[int(pair["pred_idx"])] for pair in context.selected_pairs]
    gallery_labels = [context.gt_files[int(pair["gt_idx"])].name for pair in context.selected_pairs]
    out_path = out_dir / f"{gallery_name}.png"

    plot_tree_offset_gallery_2d(
        gallery_gt,
        gallery_pred,
        gallery_labels,
        projection=projection,
        out_path=out_path,
        max_examples=len(context.selected_pairs),
        x_gap_scale=default_x_gap_scale,
        y_offset_scale=default_y_offset_scale,
        x_gap_scales=x_gap_scales,
        y_offset_scales=y_offset_scales,
        panel_dxs=panel_dxs,
        panel_dys=panel_dys,
        subplot_wspace=subplot_wspace,
        subplot_hspace=subplot_hspace,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
