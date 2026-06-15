"""Shared orchestration helpers for paper-figure runners."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .utils.io import (
    load_gt_file_graphs,
    load_pred_graphs_from_pickle,
    pair_graphs_by_index,
)


def preview_names(items: Sequence[str], *, max_items: int = 5) -> str:
    """Return a short printable preview of names."""
    if not items:
        return "(none)"
    shown = list(items[:max_items])
    preview = ", ".join(shown)
    if len(items) > max_items:
        preview += f", ... (+{len(items) - max_items} more)"
    return preview


def add_shared_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_max_pairs: int | None = 1,
) -> None:
    """Add the shared data-selection arguments used by runner scripts."""
    parser.add_argument("--gt-dir", type=Path, required=True, help="Directory containing GT SWC files.")
    parser.add_argument(
        "--pred-pkl",
        type=Path,
        required=True,
        help="Validation pickle containing predicted graphs under `pred_graphs`.",
    )
    parser.add_argument(
        "--ema-key",
        type=str,
        default=None,
        help="Optional EMA key inside the prediction pickle (e.g. `ema_1`, `ema_0.999`).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=default_max_pairs,
        help=(
            "Maximum number of GT/pred pairs to render. "
            "Use all available pairs when omitted and the runner default is unrestricted."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dendrite_gen/outputs/paper_figures"),
        help="Root output directory. Each runner writes into its own subfolder here.",
    )


@dataclass
class PlotContext:
    """Loaded GT/pred data shared across paper-figure runners."""

    gt_files: list[Path]
    gt_graphs: list
    pred_graphs: list
    pairs: list[dict[str, int]]
    unmatched: dict[str, int]
    selected_pairs: list[dict[str, int]]

    @property
    def selected_gt_names(self) -> list[str]:
        return [self.gt_files[int(pair["gt_idx"])].name for pair in self.selected_pairs]


def context_with_selected_pairs(context: PlotContext, selected_pairs: list[dict[str, int]]) -> PlotContext:
    """Return a copy of the plot context with a different selected-pair subset."""
    return PlotContext(
        gt_files=context.gt_files,
        gt_graphs=context.gt_graphs,
        pred_graphs=context.pred_graphs,
        pairs=context.pairs,
        unmatched=context.unmatched,
        selected_pairs=selected_pairs,
    )


def select_pairs_by_gt_names(context: PlotContext, tree_names: Sequence[str]) -> PlotContext:
    """Return a new plot context containing only pairs for the requested GT filenames."""
    requested = list(tree_names)
    pair_by_name = {
        context.gt_files[int(pair["gt_idx"])].name: pair
        for pair in context.pairs
    }
    missing = [name for name in requested if name not in pair_by_name]
    if missing:
        raise ValueError(f"Requested GT trees were not found in paired data: {missing}")
    selected_pairs = [pair_by_name[name] for name in requested]
    return context_with_selected_pairs(context, selected_pairs)


def load_plot_context(args: argparse.Namespace, *, print_summary: bool = True) -> PlotContext:
    """Load GT/pred graphs and select the subset of pairs to render."""
    gt_files, gt_graphs = load_gt_file_graphs(args.gt_dir)
    pred_graphs = load_pred_graphs_from_pickle(args.pred_pkl, ema_key=args.ema_key)
    pairs, unmatched = pair_graphs_by_index(gt_files, gt_graphs, pred_graphs)
    if not pairs:
        raise ValueError("No GT/pred graph pairs could be formed.")

    if args.max_pairs is None:
        selected_pairs = list(pairs)
    else:
        selected_pairs = pairs[: max(0, args.max_pairs)]
    if print_summary:
        print(f"Found {len(gt_files)} GT SWC files in {args.gt_dir}")
        print(f"GT file preview: {preview_names([p.name for p in gt_files])}")
        print(f"Using prediction pickle {args.pred_pkl}")
        if args.ema_key is not None:
            print(f"Using EMA key {args.ema_key}")
        else:
            print("Using prediction pickle without explicit EMA key selection.")
        print(f"Loaded {len(pred_graphs)} predicted graph(s) from pickle.")
        if unmatched:
            print(f"Warning: GT/pred count mismatch: {unmatched}")
        print(f"Formed {len(pairs)} GT/pred pair(s) by index.")
        print(f"Selected {len(selected_pairs)} pair(s): {preview_names([gt_files[int(pair['gt_idx'])].name for pair in selected_pairs])}")

    return PlotContext(
        gt_files=gt_files,
        gt_graphs=gt_graphs,
        pred_graphs=pred_graphs,
        pairs=pairs,
        unmatched=unmatched,
        selected_pairs=selected_pairs,
    )


def ensure_runner_out_dir(root: Path, runner_name: str) -> Path:
    """Create and return the runner-specific output subfolder."""
    out_dir = Path(root) / runner_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
