"""Analyse K-root children generation quality.

Compares GT vs predicted trees focusing on:
1. Root children positions (angles, distances)
2. Per-subtree statistics (depth, node count, spread)
3. Whether subtrees are too similar (lack of diversity)

Usage:
    conda run -n NEURO2 python tests/analyse_k_root_generation.py \
        --pkl outputs/2026-04-01/17-02-21/validation/step_30001.pkl \
        --gt-dir /Volumes/Seagate/neurons_v1/train \
        --out /tmp/k_root_analysis
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.data_loading import load_swc_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_root(G: nx.Graph) -> int:
    """Return root node (highest degree, or node 0 if tied)."""
    return max(G.nodes(), key=lambda n: G.degree(n))


def root_tree(G: nx.Graph, root: int) -> dict:
    """BFS-root the tree, return {node: parent} with root: None."""
    parent = {root: None}
    queue = deque([root])
    while queue:
        u = queue.popleft()
        for v in G.neighbors(u):
            if v not in parent:
                parent[v] = u
                queue.append(v)
    return parent


def get_children(parent_map: dict) -> dict:
    """Invert parent map to children map."""
    children = {n: [] for n in parent_map}
    for node, par in parent_map.items():
        if par is not None:
            children[par].append(node)
    return children


def subtree_stats(G: nx.Graph, subtree_root: int, parent_map: dict, children_map: dict) -> dict:
    """Compute statistics for the subtree rooted at subtree_root."""
    # BFS to collect all nodes in subtree
    nodes = []
    queue = deque([subtree_root])
    while queue:
        u = queue.popleft()
        nodes.append(u)
        for c in children_map.get(u, []):
            queue.append(c)

    positions = np.array([G.nodes[n]["pos"] for n in nodes])

    # Depth of subtree
    depth_map = {subtree_root: 0}
    q = deque([subtree_root])
    while q:
        u = q.popleft()
        for c in children_map.get(u, []):
            if c in depth_map:
                continue
            depth_map[c] = depth_map[u] + 1
            q.append(c)
    max_depth = max(depth_map.values()) if depth_map else 0

    # Spread: std of positions
    spread = positions.std(axis=0).mean() if len(positions) > 1 else 0.0

    # Total path length
    total_length = 0.0
    for n in nodes:
        p = parent_map.get(n)
        if p is not None and p in {nn for nn in nodes}:
            pos_n = np.array(G.nodes[n]["pos"])
            pos_p = np.array(G.nodes[p]["pos"])
            total_length += np.linalg.norm(pos_n - pos_p)

    # Leaf count
    leaf_count = sum(1 for n in nodes if len(children_map.get(n, [])) == 0)

    return {
        "n_nodes": len(nodes),
        "max_depth": max_depth,
        "spread": float(spread),
        "total_length": float(total_length),
        "leaf_count": leaf_count,
        "positions": positions,
    }


def analyse_tree(G: nx.Graph, label: str) -> dict:
    """Full analysis of one tree."""
    root = find_root(G)
    parent_map = root_tree(G, root)
    children_map = get_children(parent_map)

    root_pos = np.array(G.nodes[root]["pos"])
    root_children = children_map[root]
    k = len(root_children)

    # Root children positions and angles
    child_positions = np.array([G.nodes[c]["pos"] for c in root_children])
    offsets = child_positions - root_pos[None, :]
    distances = np.linalg.norm(offsets, axis=1)

    # Pairwise angles between root children offset vectors
    angles = []
    for i in range(k):
        for j in range(i + 1, k):
            cos_a = np.dot(offsets[i], offsets[j]) / (
                np.linalg.norm(offsets[i]) * np.linalg.norm(offsets[j]) + 1e-12
            )
            angles.append(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

    # Per-subtree stats
    subtrees = {}
    for i, c in enumerate(root_children):
        subtrees[i] = subtree_stats(G, c, parent_map, children_map)

    return {
        "label": label,
        "root": root,
        "k": k,
        "n_nodes": G.number_of_nodes(),
        "root_pos": root_pos,
        "child_offsets": offsets,
        "child_distances": distances,
        "pairwise_angles": np.array(angles),
        "subtrees": subtrees,
    }


def print_analysis(info: dict):
    """Print summary for one tree."""
    label = info["label"]
    k = info["k"]
    print(f"\n{'='*60}")
    print(f"  {label}  |  N={info['n_nodes']}  |  K={k} root children")
    print(f"{'='*60}")

    print(f"\n  Root child distances from root:")
    for i, d in enumerate(info["child_distances"]):
        print(f"    child {i}: dist={d:.4f}")
    print(f"    mean={info['child_distances'].mean():.4f}  std={info['child_distances'].std():.4f}  "
          f"cv={info['child_distances'].std() / (info['child_distances'].mean() + 1e-12):.3f}")

    if len(info["pairwise_angles"]) > 0:
        pa = info["pairwise_angles"]
        print(f"\n  Pairwise angles between root children:")
        print(f"    min={pa.min():.1f}°  max={pa.max():.1f}°  mean={pa.mean():.1f}°  std={pa.std():.1f}°")
        # Ideal uniform spacing
        ideal = 360.0 / k if k > 1 else 0
        print(f"    (ideal uniform spacing ≈ {ideal:.1f}°)")

    print(f"\n  Per-subtree breakdown:")
    print(f"    {'idx':>3} {'nodes':>6} {'depth':>6} {'leaves':>6} {'spread':>8} {'length':>8}")
    print(f"    {'---':>3} {'-----':>6} {'-----':>6} {'------':>6} {'------':>8} {'------':>8}")
    subtrees = info["subtrees"]
    for i in sorted(subtrees):
        s = subtrees[i]
        print(f"    {i:>3} {s['n_nodes']:>6} {s['max_depth']:>6} {s['leaf_count']:>6} "
              f"{s['spread']:>8.4f} {s['total_length']:>8.4f}")

    # Diversity metrics across subtrees
    nodes_arr = np.array([subtrees[i]["n_nodes"] for i in sorted(subtrees)])
    depths_arr = np.array([subtrees[i]["max_depth"] for i in sorted(subtrees)])
    spreads_arr = np.array([subtrees[i]["spread"] for i in sorted(subtrees)])
    lengths_arr = np.array([subtrees[i]["total_length"] for i in sorted(subtrees)])

    print(f"\n  Subtree diversity (std / mean = coefficient of variation):")
    for name, arr in [("nodes", nodes_arr), ("depth", depths_arr),
                      ("spread", spreads_arr), ("length", lengths_arr)]:
        cv = arr.std() / (arr.mean() + 1e-12)
        print(f"    {name:>8}: mean={arr.mean():.2f}  std={arr.std():.2f}  cv={cv:.3f}")


def plot_comparison(gt_info: dict, pred_info: dict, out_path: Path):
    """Side-by-side 3D plot with subtrees colored differently."""
    fig = plt.figure(figsize=(16, 7))

    for idx, (info, title_prefix) in enumerate([(gt_info, "GT"), (pred_info, "Pred")]):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        root = info["root"]
        G_nodes = info["n_nodes"]
        k = info["k"]

        # Color each subtree differently
        cmap = plt.cm.Set1
        colors = [cmap(i / max(k, 1)) for i in range(k)]

        # Plot root
        rp = info["root_pos"]
        ax.scatter(*rp, c="gold", s=120, zorder=10, edgecolors="black", linewidths=0.5)

        # Plot each subtree
        for i, subtree in info["subtrees"].items():
            pos = subtree["positions"]
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                       c=[colors[i]], s=30, alpha=0.8, label=f"child_{i} (n={subtree['n_nodes']})")

            # Draw edges within subtree to root child
            # Connect root to child
            child_pos = pos[0]  # subtree root = first node
            ax.plot([rp[0], child_pos[0]], [rp[1], child_pos[1]], [rp[2], child_pos[2]],
                    c=colors[i], alpha=0.5, linewidth=1.5)

        ax.set_title(f"{title_prefix} | N={G_nodes} | K={k}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        if k <= 10:
            ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_subtree_bars(gt_info: dict, pred_info: dict, out_path: Path):
    """Bar chart comparing subtree stats between GT and pred."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ["n_nodes", "max_depth", "spread", "total_length"]
    titles = ["Node Count", "Max Depth", "Spatial Spread", "Total Path Length"]

    for ax, metric, title in zip(axes.flat, metrics, titles):
        gt_vals = [gt_info["subtrees"][i][metric] for i in sorted(gt_info["subtrees"])]
        pred_vals = [pred_info["subtrees"][i][metric] for i in sorted(pred_info["subtrees"])]

        k_gt = len(gt_vals)
        k_pred = len(pred_vals)
        k_max = max(k_gt, k_pred)

        x_gt = np.arange(k_gt)
        x_pred = np.arange(k_pred)
        width = 0.35

        ax.bar(x_gt - width / 2, gt_vals, width, label="GT", color="steelblue", alpha=0.8)
        ax.bar(x_pred + width / 2, pred_vals, width, label="Pred", color="darkorange", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Root child index")
        ax.set_xticks(np.arange(k_max))
        ax.legend()

        # Add CV annotation
        gt_cv = np.std(gt_vals) / (np.mean(gt_vals) + 1e-12)
        pred_cv = np.std(pred_vals) / (np.mean(pred_vals) + 1e-12)
        ax.annotate(f"CV: GT={gt_cv:.2f}, Pred={pred_cv:.2f}",
                    xy=(0.02, 0.95), xycoords="axes fraction", fontsize=8,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    fig.suptitle(f"Subtree Comparison: GT (K={gt_info['k']}) vs Pred (K={pred_info['k']})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_angular_distribution(gt_info: dict, pred_info: dict, out_path: Path):
    """Compare pairwise angle distributions between root children."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, info, title in zip(axes, [gt_info, pred_info], ["GT", "Pred"]):
        offsets = info["child_offsets"]
        k = info["k"]

        # Project onto XZ plane (perpendicular to Y if uhat=[0,1,0])
        angles_from_x = np.degrees(np.arctan2(offsets[:, 2], offsets[:, 0]))

        ax.set_title(f"{title} | K={k} | Angular spread of root children")

        # Polar-like visualization on unit circle
        theta_rad = np.radians(angles_from_x)
        dists = info["child_distances"]
        dists_norm = dists / (dists.max() + 1e-12)

        for i in range(k):
            ax.arrow(0, 0, np.cos(theta_rad[i]) * dists_norm[i],
                     np.sin(theta_rad[i]) * dists_norm[i],
                     head_width=0.05, head_length=0.03, fc=f"C{i}", ec=f"C{i}")
            ax.annotate(f"c{i}", (np.cos(theta_rad[i]) * (dists_norm[i] + 0.1),
                                   np.sin(theta_rad[i]) * (dists_norm[i] + 0.1)), fontsize=8)

        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal")
        ax.axhline(0, color="gray", linewidth=0.3)
        ax.axvline(0, color="gray", linewidth=0.3)
        circle = plt.Circle((0, 0), 1.0, fill=False, color="lightgray", linewidth=0.5)
        ax.add_patch(circle)

        # Annotate pairwise angles
        pa = info["pairwise_angles"]
        if len(pa) > 0:
            ax.text(0.02, 0.02, f"Pairwise angles:\nmin={pa.min():.0f}° max={pa.max():.0f}°\nmean={pa.mean():.0f}° std={pa.std():.0f}°",
                    transform=ax.transAxes, fontsize=7, verticalalignment="bottom",
                    bbox=dict(facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse K-root children generation")
    parser.add_argument("--pkl", required=True, help="Path to validation .pkl")
    parser.add_argument("--gt-dir", required=True, help="Directory with GT .swc files")
    parser.add_argument("--out", default="/tmp/k_root_analysis", help="Output directory")
    parser.add_argument("--ema-key", default="ema_1", help="EMA key in pkl")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    pred_graphs = data[args.ema_key]["pred_graphs"]
    print(f"Loaded {len(pred_graphs)} predicted graphs")

    # Load GT graphs (sorted to match eval order)
    gt_dir = Path(args.gt_dir)
    gt_files = sorted(gt_dir.glob("*.csv.swc"))
    # Filter out macOS resource fork files
    gt_files = [f for f in gt_files if not f.name.startswith("._")]
    gt_graphs = []
    for f in gt_files:
        G = load_swc_graph(f)
        gt_graphs.append(G)
        print(f"  GT: {f.name} N={G.number_of_nodes()}")

    if len(pred_graphs) != len(gt_graphs):
        print(f"WARNING: {len(pred_graphs)} preds vs {len(gt_graphs)} GT graphs")

    # Analyse each pair
    n_pairs = min(len(gt_graphs), len(pred_graphs))
    for i in range(n_pairs):
        print(f"\n{'#'*60}")
        print(f"  GRAPH {i}")
        print(f"{'#'*60}")

        gt_info = analyse_tree(gt_graphs[i], f"GT graph {i}")
        pred_info = analyse_tree(pred_graphs[i], f"Pred graph {i}")

        print_analysis(gt_info)
        print_analysis(pred_info)

        # Plots
        plot_comparison(gt_info, pred_info, out_dir / f"graph_{i}_3d_subtrees.png")
        plot_subtree_bars(gt_info, pred_info, out_dir / f"graph_{i}_subtree_bars.png")
        plot_angular_distribution(gt_info, pred_info, out_dir / f"graph_{i}_angular.png")

    print(f"\nPlots saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
