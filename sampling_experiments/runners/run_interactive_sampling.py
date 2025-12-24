"""CLI runner that will orchestrate interactive sampling experiments."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

# Ensure the project root is importable when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
import torch as th
from omegaconf import DictConfig, OmegaConf

from sampling_experiments.interactive_methods import (
    InteractiveExpansionOneShot,
    InteractiveExpansionOneShotAugmented,
)
from sampling_experiments.loaders import load_sampling_items
from sampling_experiments.runners.eval_helpers import (
    chunk_target_sizes,
    load_graphs_for_split,
    target_sizes_from_graphs,
)
from sampling_experiments.utils import load_hydra_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run interactive expansion sampling")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a Hydra config (e.g., config/small_trees_run.yaml)",
    )
    parser.add_argument("--config-override", action="append", default=[], help="Optional Hydra-style overrides")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--ema-beta", type=float, default=None, help="EMA beta to load (matches Trainer naming)")
    parser.add_argument("--device", type=str, default="cpu", help="Device identifier (cpu / cuda:0 / mps)")
    parser.add_argument(
        "--method",
        type=str,
        choices=["expansion", "expansion_augmented"],
        default=None,
        help="Override method (defaults to config value)",
    )
    parser.add_argument("--target-sizes", type=int, nargs="*", default=None, help="Explicit per-graph target sizes")
    parser.add_argument(
        "--dataset-split",
        type=str,
        choices=["train", "val", "test"],
        default=None,
        help="If provided, derive target sizes from this dataset split.",
    )
    parser.add_argument("--max-graphs", type=int, default=None, help="When using dataset-split, cap number of graphs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override sampling batch size")
    parser.add_argument("--map-threshold", type=float, default=None, help="Override map threshold for expansion probs")
    parser.add_argument(
        "--ensure-progress",
        action="store_true",
        help="Force at least one leaf expansion per step when capacity allows (uses highest-probability leaf).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("sampling_experiments/artifacts"), help="Where to store traces")
    return parser


def _resolve_target_tensor(target_sizes: Sequence[int], *, device: th.device) -> th.Tensor:
    tensor = th.tensor(list(target_sizes), dtype=th.long)
    return tensor.to(device)


def _select_method_class(method_name: str):
    if method_name == "expansion":
        return InteractiveExpansionOneShot
    if method_name == "expansion_augmented":
        return InteractiveExpansionOneShotAugmented
    raise ValueError(f"Unsupported method for interactive sampling: {method_name}")


def _prepare_run_dir(base_dir: Path, method_name: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{method_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _save_run_artifacts(
    run_dir: Path,
    *,
    graphs: list[nx.Graph],
    traces,
    context,
    cfg: DictConfig,
    target_source: dict,
) -> None:
    summary_graphs = []
    for idx, (graph, graph_traces) in enumerate(zip(graphs, traces)):
        graph_path = run_dir / f"graph_{idx}_final.gpickle"
        trace_path = run_dir / f"graph_{idx}_trace.pkl"
        with open(graph_path, "wb") as f:
            pickle.dump(graph, f)
        with open(trace_path, "wb") as f:
            pickle.dump(graph_traces, f)
        summary_graphs.append(
            {
                "graph_idx": idx,
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "num_traces": len(graph_traces),
                "graph_path": graph_path.name,
                "trace_path": trace_path.name,
            }
        )

    metadata = {
        "checkpoint": str(context.checkpoint_path),
        "ema_beta": context.ema_beta,
        "method": cfg.method.name,
        "device": str(context.device),
        "num_graphs": len(graphs),
        "target_source": target_source,
        "graphs": summary_graphs,
    }
    with open(run_dir / "run_summary.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if OmegaConf is not None:
        with open(run_dir / "config_resolved.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg))


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(argv)
    cfg = load_hydra_config(parsed.config, parsed.config_override)
    if parsed.method is not None:
        cfg.method.name = parsed.method

    method_name = cfg.method.name
    method_cls = _select_method_class(method_name)

    context = load_sampling_items(
        cfg=cfg,
        checkpoint=parsed.checkpoint,
        ema_beta=parsed.ema_beta,
        device=parsed.device,
        method_cls=method_cls,
    )

    if parsed.target_sizes:
        target_sizes_list = list(parsed.target_sizes)
        target_source = {"type": "manual", "num_graphs": len(target_sizes_list)}
    elif parsed.dataset_split is not None:
        graphs_ref = load_graphs_for_split(cfg, parsed.dataset_split, max_graphs=parsed.max_graphs)
        target_sizes_list = target_sizes_from_graphs(graphs_ref)
        if not target_sizes_list:
            raise ValueError("No graphs loaded from dataset split to derive target sizes.")
        target_source = {
            "type": "dataset_split",
            "split": parsed.dataset_split,
            "num_graphs": len(target_sizes_list),
            "max_graphs": parsed.max_graphs,
        }
    else:
        raise ValueError("Provide --target-sizes or --dataset-split to specify sampling targets.")

    batch_size = parsed.batch_size
    if batch_size is None:
        batch_size = getattr(cfg.validation, "batch_size", None) or getattr(cfg.training, "batch_size", None)
    if batch_size is None:
        batch_size = len(target_sizes_list)
    target_batches = chunk_target_sizes(target_sizes_list, batch_size=batch_size)

    all_graphs: list[nx.Graph] = []
    all_traces: list[list] = []
    for batch_idx, target_batch in enumerate(target_batches):
        target_tensor = _resolve_target_tensor(target_batch.tolist(), device=context.device)
        graphs_batch, traces_batch = context.method.sample_graphs_with_trace(
            target_tensor,
            context.model,
            map_threshold=parsed.map_threshold,
            ensure_progress=parsed.ensure_progress,
        )
        all_graphs.extend(graphs_batch)
        all_traces.extend(traces_batch)
        print(f"Batch {batch_idx + 1}/{len(target_batches)}: generated {len(graphs_batch)} graphs.")

    run_dir = _prepare_run_dir(parsed.output_dir, method_name)
    _save_run_artifacts(
        run_dir,
        graphs=all_graphs,
        traces=all_traces,
        context=context,
        cfg=cfg,
        target_source={**target_source, "batch_size": batch_size, "num_batches": len(target_batches)},
    )
    print(f"Interactive sampling artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
