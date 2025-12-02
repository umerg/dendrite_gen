"""
Utility script to inspect the distribution of ``leaf_expansion`` labels
across reduction levels, graph sizes, and remaining capacity
(total tree size - current nodes). Mirrors the dataset/reduction logic
from ``main.py`` so that the sampled states match training.

Example:
    python tests/expansion_debug_analysis.py --config-name small_trees_run
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch as th
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_OUTPUT_DIR = REPO_ROOT / "expansion_debug"

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

import graph_generation as gg
from utils.data_loading import load_swc_graphs_from_dir, nx_graph_to_adj_pos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze leaf expansion label balance.")
    parser.add_argument(
        "--config-name",
        type=str,
        default="small_trees_run",
        help="Hydra config name (e.g., small_trees_run, smoke_synthetic).",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=10_000,
        help="Number of dataloader iterations to sample (batch_size=4 => 40k samples).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for the reduction dataloader.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder where visualizations/logs are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for numpy/random/torch to make sampling deterministic-ish.",
    )
    return parser.parse_args()


def load_cfg(config_name: str):
    config_dir = REPO_ROOT / "config"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name=config_name)
    return cfg


def load_train_graphs(cfg) -> list[nx.Graph]:
    if cfg.dataset.load:
        data_root = Path(cfg.dataset.data_dir)
        if not data_root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_root}")
        train_graphs = load_swc_graphs_from_dir(data_root / "train")
    elif cfg.dataset.name in ("tree_synthetic",):
        graph_generator = gg.data.generate_tree_graphs
        train_graphs = graph_generator(
            num_graphs=cfg.dataset.train_size,
            min_size=cfg.dataset.min_size,
            max_size=cfg.dataset.max_size,
            seed=0,
        )
    else:
        raise ValueError(f"Unsupported dataset configuration: {cfg.dataset.name}")

    cleaned = []
    for G in train_graphs:
        largest_comp = max(nx.connected_components(G), key=len)
        cleaned.append(G.subgraph(largest_comp).copy())
    return cleaned


def build_dataloader(cfg, train_graphs, batch_size: int) -> DataLoader:
    red_factory = gg.reduction.ReductionFactory(
        mode=cfg.reduction.mode,
        cherry_p=cfg.reduction.cherry_p,
        ensure_progress=cfg.reduction.ensure_progress,
        root=cfg.reduction.root,
        contract_root=cfg.reduction.contract_root,
    )

    adjs, poses = [], []
    for G in train_graphs:
        adj, pos, _ = nx_graph_to_adj_pos(G)
        adjs.append(adj)
        poses.append(pos)

    dataset = gg.data.InfiniteRandRedDataset(adjs=adjs, poses=poses, red_factory=red_factory)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=Batch.from_data_list,
        num_workers=0,
    )
    return dataloader


def update_bins(bins, key, labels: th.Tensor) -> None:
    total = int(labels.numel())
    if total == 0:
        return
    num_twos = int((labels == 2).sum().item())
    bucket = bins[key]
    bucket["total"] += total
    bucket["two"] += num_twos


def extract_new_leaf_labels(data, leaf_labels: th.Tensor) -> th.Tensor | None:
    new_leaf_mask = getattr(data, "new_leaf_mask_from_next", None)
    if new_leaf_mask is None:
        return None

    if isinstance(new_leaf_mask, th.Tensor):
        node_mask = new_leaf_mask.to(dtype=th.bool)
    else:
        node_mask = th.as_tensor(new_leaf_mask, dtype=th.bool)

    if node_mask.numel() == 0 or not node_mask.any():
        return None

    leaf_idx = getattr(data, "leaf_idx", None)
    if leaf_idx is None:
        return None
    if isinstance(leaf_idx, th.Tensor):
        leaf_idx_long = leaf_idx.to(dtype=th.long)
    else:
        leaf_idx_long = th.as_tensor(leaf_idx, dtype=th.long)
    if leaf_idx_long.numel() == 0:
        return None

    max_idx = int(leaf_idx_long.max().item())
    if max_idx >= node_mask.numel():
        # Safety guard in case mask length mismatches node count.
        return None

    new_leaf_mask_for_leaves = node_mask[leaf_idx_long]
    if new_leaf_mask_for_leaves.numel() == 0 or not new_leaf_mask_for_leaves.any():
        return None
    return leaf_labels[new_leaf_mask_for_leaves]


def compute_stats(dataloader: DataLoader, num_iterations: int) -> dict:
    by_level = defaultdict(lambda: {"two": 0, "total": 0})
    by_size = defaultdict(lambda: {"two": 0, "total": 0})
    by_remaining_capacity = defaultdict(lambda: {"two": 0, "total": 0})
    new_by_level = defaultdict(lambda: {"two": 0, "total": 0})
    new_by_remaining_capacity = defaultdict(lambda: {"two": 0, "total": 0})
    total_samples = 0
    total_leaves = 0
    total_twos = 0
    total_new_leaves = 0
    total_new_twos = 0

    loader_iter = iter(dataloader)
    for idx in range(num_iterations):
        batch = next(loader_iter)
        data_list = batch.to_data_list()
        for data in data_list:
            total_samples += 1
            labels = getattr(data, "leaf_expansion", None)
            if labels is None or labels.numel() == 0:
                continue

            labels = labels.to(th.long)
            total_leaves += int(labels.numel())
            total_twos += int((labels == 2).sum().item())

            level = int(getattr(data, "reduction_level").item())
            current_n = int(getattr(data, "target_size").item())
            total_size_attr = getattr(data, "total_tree_size", None)
            remaining_capacity = None
            if total_size_attr is not None:
                total_size = int(total_size_attr.item())
                remaining_capacity = max(total_size - current_n, 0)

            new_labels = extract_new_leaf_labels(data, labels)

            update_bins(by_level, level, labels)
            update_bins(by_size, current_n, labels)
            if remaining_capacity is not None:
                update_bins(by_remaining_capacity, remaining_capacity, labels)
            if new_labels is not None and new_labels.numel() > 0:
                total_new_leaves += int(new_labels.numel())
                total_new_twos += int((new_labels == 2).sum().item())
                update_bins(new_by_level, level, new_labels)
                if remaining_capacity is not None:
                    update_bins(new_by_remaining_capacity, remaining_capacity, new_labels)

        if (idx + 1) % max(1, num_iterations // 10) == 0:
            print(f"[{idx + 1}/{num_iterations}] iterations processed...")

    return {
        "by_level": by_level,
        "by_size": by_size,
        "by_remaining_capacity": by_remaining_capacity,
        "new_by_level": new_by_level,
        "new_by_remaining_capacity": new_by_remaining_capacity,
        "total_samples": total_samples,
        "total_leaves": total_leaves,
        "total_twos": total_twos,
        "total_new_leaves": total_new_leaves,
        "total_new_twos": total_new_twos,
    }


def bins_to_sorted_list(bins: dict) -> list[dict]:
    items = []
    for key in sorted(bins.keys()):
        entry = bins[key]
        total = entry["total"]
        frac = entry["two"] / total if total else 0.0
        items.append(
            {
                "value": int(key),
                "total_leaves": total,
                "leaf_expansion_two": entry["two"],
                "fraction_leaf_expansion_two": frac,
            }
        )
    return items


def save_plot(stats_list: list[dict], xlabel: str, output_path: Path) -> None:
    if not stats_list:
        print(f"No data available for {output_path.name}, skipping plot.")
        return

    xs = [entry["value"] for entry in stats_list]
    fractions = [entry["fraction_leaf_expansion_two"] for entry in stats_list]
    totals = [entry["total_leaves"] for entry in stats_list]

    fig, ax_frac = plt.subplots(figsize=(8, 5))
    ax_frac.plot(xs, fractions, marker="o", color="tab:blue", label="fraction label=2")
    ax_frac.set_xlabel(xlabel)
    ax_frac.set_ylabel("Fraction leaf_expansion == 2", color="tab:blue")
    ax_frac.set_ylim(0.0, 1.0)

    ax_count = ax_frac.twinx()
    ax_count.bar(xs, totals, alpha=0.2, color="tab:gray", label="leaf count")
    ax_count.set_ylabel("Leaf samples")

    ax_frac.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_logs(output_dir: Path, results: dict, cfg) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    level_stats = bins_to_sorted_list(results["by_level"])
    size_stats = bins_to_sorted_list(results["by_size"])
    capacity_stats = bins_to_sorted_list(results["by_remaining_capacity"])
    new_level_stats = bins_to_sorted_list(results["new_by_level"])
    new_capacity_stats = bins_to_sorted_list(results["new_by_remaining_capacity"])

    summary = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "num_iterations": int(results.get("num_iterations", 0)),
        "batch_size": int(results.get("batch_size", 0)),
        "total_samples": results["total_samples"],
        "total_leaves": results["total_leaves"],
        "total_leaf_expansion_two": results["total_twos"],
        "total_new_leaves": results["total_new_leaves"],
        "total_new_leaf_expansion_two": results["total_new_twos"],
        "overall_fraction_two": (
            results["total_twos"] / results["total_leaves"] if results["total_leaves"] else 0.0
        ),
        "overall_fraction_two_new_leaves": (
            results["total_new_twos"] / results["total_new_leaves"]
            if results["total_new_leaves"]
            else 0.0
        ),
        "fraction_by_reduction_level": level_stats,
        "fraction_by_current_n": size_stats,
        "fraction_by_remaining_capacity": capacity_stats,
        "fraction_by_reduction_level_new_leaves": new_level_stats,
        "fraction_by_remaining_capacity_new_leaves": new_capacity_stats,
    }

    json_path = output_dir / "leaf_expansion_stats.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote JSON summary to {json_path}")

    txt_path = output_dir / "leaf_expansion_stats.txt"
    with txt_path.open("w") as f:
        f.write(
            f"Samples: {summary['total_samples']} (leaves={summary['total_leaves']}), "
            f"overall frac leaf_expansion=2: {summary['overall_fraction_two']:.4f}\n\n"
        )
        f.write(
            f"New leaves total={summary['total_new_leaves']} "
            f"overall frac leaf_expansion=2 in new leaves: "
            f"{summary['overall_fraction_two_new_leaves']:.4f}\n\n"
        )
        f.write("By reduction level:\n")
        for entry in level_stats:
            f.write(
                f"  level={entry['value']:4d} | leaves={entry['total_leaves']:6d} "
                f"| frac= {entry['fraction_leaf_expansion_two']:.4f}\n"
            )
        f.write("\nBy reduction level (new leaves only):\n")
        for entry in new_level_stats:
            f.write(
                f"  level={entry['value']:4d} | new_leaves={entry['total_leaves']:6d} "
                f"| frac= {entry['fraction_leaf_expansion_two']:.4f}\n"
            )
        f.write("\nBy current_n:\n")
        for entry in size_stats:
            f.write(
                f"  current_n={entry['value']:4d} | leaves={entry['total_leaves']:6d} "
                f"| frac= {entry['fraction_leaf_expansion_two']:.4f}\n"
            )
        f.write("\nBy remaining capacity (total_size - current_n):\n")
        for entry in capacity_stats:
            f.write(
                f"  remaining={entry['value']:4d} | leaves={entry['total_leaves']:6d} "
                f"| frac= {entry['fraction_leaf_expansion_two']:.4f}\n"
            )
        f.write("\nBy remaining capacity (new leaves only):\n")
        for entry in new_capacity_stats:
            f.write(
                f"  remaining={entry['value']:4d} | new_leaves={entry['total_leaves']:6d} "
                f"| frac= {entry['fraction_leaf_expansion_two']:.4f}\n"
            )
    print(f"Wrote text log to {txt_path}")

    save_plot(level_stats, "Reduction level", output_dir / "fraction_by_reduction_level.png")
    save_plot(size_stats, "Current number of nodes (current_n)", output_dir / "fraction_by_current_n.png")
    save_plot(
        capacity_stats,
        "Remaining capacity (total tree size - current_n)",
        output_dir / "fraction_by_remaining_capacity.png",
    )
    save_plot(
        new_level_stats,
        "Reduction level (new leaves only)",
        output_dir / "fraction_by_reduction_level_new_leaves.png",
    )
    save_plot(
        new_capacity_stats,
        "Remaining capacity (new leaves only)",
        output_dir / "fraction_by_remaining_capacity_new_leaves.png",
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    cfg = load_cfg(args.config_name)
    cfg.training.batch_size = args.batch_size

    train_graphs = load_train_graphs(cfg)
    print(f"Loaded {len(train_graphs)} training graphs.")

    dataloader = build_dataloader(cfg, train_graphs, batch_size=args.batch_size)
    print("Dataloader constructed. Beginning sampling...")

    stats = compute_stats(dataloader, num_iterations=args.num_iterations)
    stats["num_iterations"] = args.num_iterations
    stats["batch_size"] = args.batch_size

    output_dir = Path(args.output_dir)
    write_logs(output_dir, stats, cfg)


if __name__ == "__main__":
    main()
