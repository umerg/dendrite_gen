"""Utility functions mirroring Trainer.evaluate batching for sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import networkx as nx
import torch as th

import graph_generation as gg
from utils.data_loading import load_swc_graphs_from_dir


SYNTHETIC_SPLIT_SEEDS = {
    "train": 0,
    "val": 1,
    "test": 2,
}


def chunk_target_sizes(target_sizes: Iterable[int], *, batch_size: int) -> List[th.Tensor]:
    """Split iterable of sizes into tensors respecting batch_size ordering."""
    tensor_sizes = list(target_sizes)
    if not tensor_sizes:
        return []
    batches: List[th.Tensor] = []
    for idx in range(0, len(tensor_sizes), batch_size):
        chunk = tensor_sizes[idx : idx + batch_size]
        batches.append(th.tensor(chunk, dtype=th.long))
    return batches


def _largest_connected_components(graphs: Sequence[nx.Graph]) -> list[nx.Graph]:
    """Keep only the largest connected component of each graph (matching Trainer)."""
    kept = []
    for G in graphs:
        if G.number_of_nodes() == 0:
            continue
        if nx.is_connected(G):
            kept.append(G.copy())
            continue
        largest = max(nx.connected_components(G), key=len)
        kept.append(G.subgraph(largest).copy())
    return kept


def load_graphs_for_split(cfg, split: str, *, max_graphs: int | None = None) -> list[nx.Graph]:
    """Load dataset graphs for the requested split using cfg.dataset settings."""
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split '{split}'. Expected train/val/test.")

    graphs: list[nx.Graph]
    if getattr(cfg.dataset, "load", False):
        data_root = Path(cfg.dataset.data_dir)
        split_dir = data_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")
        graphs = load_swc_graphs_from_dir(split_dir)
    elif getattr(cfg.dataset, "name", None) in ("tree_synthetic",):
        num_key = f"{split}_size"
        num_graphs = getattr(cfg.dataset, num_key, None)
        if num_graphs is None:
            raise ValueError(f"cfg.dataset.{num_key} must be set for synthetic dataset.")
        seed = SYNTHETIC_SPLIT_SEEDS.get(split, 0)
        graphs = gg.data.generate_tree_graphs(
            num_graphs=num_graphs,
            min_size=cfg.dataset.min_size,
            max_size=cfg.dataset.max_size,
            seed=seed,
        )
    else:
        raise ValueError(
            "Unsupported dataset configuration for sampling. "
            "Only SWC directory datasets (cfg.dataset.load=True) or 'tree_synthetic' are supported."
        )

    graphs = _largest_connected_components(graphs)
    if max_graphs is not None:
        graphs = graphs[:max_graphs]
    return graphs


def target_sizes_from_graphs(graphs: Sequence[nx.Graph]) -> list[int]:
    """Return node counts for each graph."""
    return [g.number_of_nodes() for g in graphs]
