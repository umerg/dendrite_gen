#!/usr/bin/env python3
"""Analyze cleaned neuron dataset: size distribution, root branching, binary tree validation."""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.data_loading import load_swc_graphs_from_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze cleaned neuron dataset for size, root branching, and binary validation."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Users/umer/Documents/neurons_cleaned"),
        help="Directory containing cleaned SWC files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()

    print(f"Loading graphs from {data_dir} ...")
    graphs = load_swc_graphs_from_dir(data_dir)
    n_graphs = len(graphs)
    if n_graphs == 0:
        print("No graphs found.")
        return
    print(f"Loaded {n_graphs} graphs.\n")

    sizes = []
    depths = []
    root_children = []
    violations = []  # (filename_idx, node, degree)

    for i, G in enumerate(graphs):
        n = G.number_of_nodes()
        sizes.append(n)

        root = G.graph.get("root")
        if root is not None:
            root_children.append(G.degree(root))
        else:
            root_children.append(-1)

        # BFS depth from root
        if root is not None:
            max_depth = 0
            q = deque([(root, 0)])
            visited = {root}
            while q:
                u, d = q.popleft()
                if d > max_depth:
                    max_depth = d
                for v in G.neighbors(u):
                    if v not in visited:
                        visited.add(v)
                        q.append((v, d + 1))
            depths.append(max_depth)
        else:
            depths.append(-1)

        # Check binary constraint: every non-root node should have degree <= 3
        for node in G.nodes():
            if node == root:
                continue
            deg = G.degree(node)
            if deg > 3:
                violations.append((i, node, deg))

    sizes = np.array(sizes)
    depths = np.array(depths)
    root_children = np.array(root_children)

    # --- Size distribution ---
    print("=" * 60)
    print("NEURON SIZE DISTRIBUTION (number of nodes)")
    print("=" * 60)
    print(f"  count:  {len(sizes)}")
    print(f"  mean:   {sizes.mean():.1f}")
    print(f"  std:    {sizes.std():.1f}")
    print(f"  median: {np.median(sizes):.0f}")
    print(f"  min:    {sizes.min()}")
    print(f"  max:    {sizes.max()}")
    print(f"  p5:     {np.percentile(sizes, 5):.0f}")
    print(f"  p25:    {np.percentile(sizes, 25):.0f}")
    print(f"  p75:    {np.percentile(sizes, 75):.0f}")
    print(f"  p95:    {np.percentile(sizes, 95):.0f}")

    # Histogram buckets
    print("\n  Size histogram:")
    bin_edges = np.arange(0, sizes.max() + 20, 20)
    counts, _ = np.histogram(sizes, bins=bin_edges)
    for j in range(len(counts)):
        if counts[j] > 0:
            lo, hi = int(bin_edges[j]), int(bin_edges[j + 1])
            bar = "#" * max(1, int(counts[j] / max(counts) * 40))
            print(f"    [{lo:4d}-{hi:4d}) {counts[j]:5d}  {bar}")

    # --- Depth distribution ---
    print()
    print("=" * 60)
    print("TREE DEPTH DISTRIBUTION (max BFS depth from root)")
    print("=" * 60)
    print(f"  count:  {len(depths)}")
    print(f"  mean:   {depths.mean():.1f}")
    print(f"  std:    {depths.std():.1f}")
    print(f"  median: {np.median(depths):.0f}")
    print(f"  min:    {depths.min()}")
    print(f"  max:    {depths.max()}")
    print(f"  p5:     {np.percentile(depths, 5):.0f}")
    print(f"  p25:    {np.percentile(depths, 25):.0f}")
    print(f"  p75:    {np.percentile(depths, 75):.0f}")
    print(f"  p95:    {np.percentile(depths, 95):.0f}")

    print("\n  Depth histogram:")
    depth_bin_edges = np.arange(0, depths.max() + 5, 5)
    depth_counts, _ = np.histogram(depths, bins=depth_bin_edges)
    for j in range(len(depth_counts)):
        if depth_counts[j] > 0:
            lo, hi = int(depth_bin_edges[j]), int(depth_bin_edges[j + 1])
            bar = "#" * max(1, int(depth_counts[j] / max(depth_counts) * 40))
            print(f"    [{lo:3d}-{hi:3d}) {depth_counts[j]:5d}  {bar}")

    # --- Root children distribution ---
    print()
    print("=" * 60)
    print("ROOT/SOMA CHILDREN DISTRIBUTION (degree of root node)")
    print("=" * 60)
    unique, ucounts = np.unique(root_children, return_counts=True)
    for val, cnt in zip(unique, ucounts):
        pct = 100.0 * cnt / n_graphs
        print(f"  degree {val:3d}: {cnt:6d} graphs ({pct:5.1f}%)")
    print(f"\n  mean degree:   {root_children.mean():.2f}")
    print(f"  median degree: {np.median(root_children):.0f}")
    print(f"  max degree:    {root_children.max()}")

    # --- Binary tree validation ---
    print()
    print("=" * 60)
    print("BINARY TREE VALIDATION (non-root nodes with degree > 3)")
    print("=" * 60)
    if not violations:
        print("  PASS: All non-root nodes have degree <= 3 (binary branching).")
    else:
        n_violating_graphs = len(set(v[0] for v in violations))
        print(f"  FAIL: {len(violations)} violations in {n_violating_graphs} graphs.")
        # Show first few
        for idx, node, deg in violations[:20]:
            print(f"    graph {idx}: node {node} has degree {deg}")
        if len(violations) > 20:
            print(f"    ... and {len(violations) - 20} more")

    print()


if __name__ == "__main__":
    main()
