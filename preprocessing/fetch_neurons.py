#!/usr/bin/env python3
"""
Fetch neuron skeletons from MICrONS S3 bucket, extract dendrite branching
topology, and save as .csv.swc files compatible with utils/data_loading.py.

Usage:
    conda run -n NEURO2 python preprocessing/fetch_neurons.py \
        --output-dir /Volumes/Seagate/neurons_v1/train \
        --num-neurons 3 \
        --seed 42 \
        --max-depth 16
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# Import the cleaning pipeline from clean_trees.py (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_trees import clean_swc_tree, write_swc

SKEL_PATH = "s3://bossdb-open-data/iarpa_microns/minnie/minnie65/skeletons/v661/skeletons/"


def list_skeleton_files():
    """List all .swc skeleton files on S3."""
    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    return fs.ls(SKEL_PATH)


def load_skeleton(filename: str):
    """Load a single skeleton from S3 using skeleton_plot."""
    import skeleton_plot.skel_io as skio
    return skio.read_skeleton(SKEL_PATH, filename)


def build_branch_graph(skel, include_root=True, include_tips=True):
    """
    Reduce a meshparty skeleton to its branching topology.
    (Adapted from notebooks/skel_2.ipynb)

    Returns a NetworkX Graph where:
      - Nodes are skeleton vertex indices with 'pos' and 'kind' attrs
      - Edges have 'length_nm' and 'n_vertices' attrs
    """
    sk = getattr(skel, "skeleton", skel)

    topo = set(sk.branch_points)
    if include_root and sk.root is not None:
        topo.add(int(sk.root))
    if include_tips:
        topo.update(sk.end_points)

    topo = np.array(sorted(topo), dtype=int)

    G = nx.Graph()
    for v in topo:
        kind = "branch"
        if include_root and v == sk.root:
            kind = "root"
        elif include_tips and v in set(sk.end_points.tolist()):
            kind = "tip"
        G.add_node(int(v), pos=tuple(sk.vertices[v]), kind=kind)

    for seg in sk.segments_plus:
        u, v = int(seg[0]), int(seg[-1])
        if u in G and v in G and u != v:
            length = float(sk.path_length(seg))
            G.add_edge(u, v, length_nm=length, n_vertices=len(seg))

    if include_root and sk.root in G:
        cc = max(nx.connected_components(G), key=len)
        if sk.root not in cc:
            for c in nx.connected_components(G):
                if sk.root in c:
                    cc = c
                    break
        G = G.subgraph(cc).copy()

    return G


def graph_to_swc_df(G: nx.Graph, scale_nm_to_um: bool = True) -> pd.DataFrame:
    """
    Convert a branch topology NetworkX graph to an SWC DataFrame.

    BFS from the root node to assign parent pointers and sequential ids.
    Root gets id=1, parent=0.
    Coordinates are optionally converted from nm to micrometers (÷1000).
    """
    # Find root node
    root = None
    for n, data in G.nodes(data=True):
        if data.get("kind") == "root":
            root = n
            break
    if root is None:
        raise ValueError("No root node found in graph")

    # BFS to get ordering and parent map
    order = []
    parent_map = {root: None}
    visited = {root}
    queue = deque([root])
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in G.neighbors(u):
            if v not in visited:
                visited.add(v)
                parent_map[v] = u
                queue.append(v)

    # Assign sequential ids (root = 1)
    id_map = {orig: idx + 1 for idx, orig in enumerate(order)}

    rows = []
    for orig in order:
        new_id = id_map[orig]
        pos = G.nodes[orig]["pos"]
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        if scale_nm_to_um:
            x /= 1000.0
            y /= 1000.0
            z /= 1000.0
        parent_orig = parent_map[orig]
        parent_id = 0 if parent_orig is None else id_map[parent_orig]
        rows.append({
            "id": new_id,
            "type": 3,
            "x": x,
            "y": y,
            "z": z,
            "radius": 1.0,
            "parent": parent_id,
        })

    return pd.DataFrame(rows, columns=["id", "type", "x", "y", "z", "radius", "parent"])


def process_one_neuron(
    skel_file: str,
    output_dir: Path,
    max_depth: int | None = None,
) -> Path | None:
    """
    Download one neuron, extract dendrite topology, clean, and save as .csv.swc.
    Returns the output path on success, None on failure.
    """
    filename = skel_file.split("/")[-1]
    segment_id = filename.replace(".swc", "")
    print(f"  Loading skeleton: {filename}")

    try:
        sk = load_skeleton(filename)
    except Exception as e:
        print(f"  [SKIP] Failed to load {filename}: {e}")
        return None

    # Check that vertex_properties has compartment info
    if "compartment" not in sk.vertex_properties:
        print(f"  [SKIP] {filename}: no compartment info")
        return None

    # Mask to dendrites + soma
    comp = sk.vertex_properties["compartment"]
    dendrite_mask = (comp == 1) | (comp == 3) | (comp == 4)
    if dendrite_mask.sum() < 5:
        print(f"  [SKIP] {filename}: too few dendrite vertices ({dendrite_mask.sum()})")
        return None

    sk_dendrite = sk.apply_mask(dendrite_mask)

    # Build branch topology graph
    G = build_branch_graph(sk_dendrite, include_root=True, include_tips=True)
    if G.number_of_nodes() < 3:
        print(f"  [SKIP] {filename}: branch graph too small ({G.number_of_nodes()} nodes)")
        return None

    if not nx.is_connected(G):
        print(f"  [SKIP] {filename}: disconnected branch graph")
        return None

    print(f"  Branch graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Convert to SWC DataFrame (nm → µm)
    df = graph_to_swc_df(G, scale_nm_to_um=True)

    # Run through the cleaning pipeline (collapse deg-2, normalize high-degree, re-index)
    df_clean = clean_swc_tree(
        df,
        root_parent_value=0,
        keep_parent_value=0,
        max_depth=max_depth,
        keep_attrs=False,  # type=3, radius=1.0
    )

    # Save
    out_path = output_dir / f"neuron_{segment_id}.csv.swc"
    write_swc(df_clean, out_path, root_parent_value=0)
    print(f"  Saved: {out_path} ({len(df_clean)} nodes)")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Fetch neurons from MICrONS and save as .csv.swc")
    ap.add_argument("--output-dir", type=Path, default=Path("/Volumes/Seagate/neurons_v1/train"),
                    help="Directory to save .csv.swc files")
    ap.add_argument("--num-neurons", type=int, default=3, help="Number of neurons to fetch")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--max-depth", type=int, default=None, help="Max tree depth after cleaning")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Listing skeleton files on S3...")
    skel_files = list_skeleton_files()
    print(f"Found {len(skel_files)} skeleton files")

    random.seed(args.seed)
    # Shuffle and try candidates until we have enough successful conversions
    candidates = list(skel_files)
    random.shuffle(candidates)

    saved = []
    for skel_file in candidates:
        if len(saved) >= args.num_neurons:
            break
        print(f"\n[{len(saved)+1}/{args.num_neurons}] Processing {skel_file.split('/')[-1]}")
        result = process_one_neuron(skel_file, args.output_dir, max_depth=args.max_depth)
        if result is not None:
            saved.append(result)

    print(f"\nDone. Saved {len(saved)} neuron(s) to {args.output_dir}")
    for p in saved:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
