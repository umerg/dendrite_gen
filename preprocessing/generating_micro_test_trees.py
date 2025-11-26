import argparse
import random
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from preprocessing.clean_trees import clean_swc_tree, read_swc


NodeAttr = Dict[str, float]


def list_swc_files(input_dir: Path) -> List[Path]:
    """Return sorted SWC files filtered the same way as utils.data_loading."""
    swc_files = []
    for swc_file in sorted(input_dir.iterdir()):
        if swc_file.is_file() and not swc_file.name.startswith("._") and swc_file.name.endswith(".csv.swc"):
            swc_files.append(swc_file)
    return swc_files


def compute_tree_parents(
    G: nx.Graph, *, root: Optional[int] = None
) -> Dict[int, Optional[int]]:
    """Compute parent pointers for the undirected tree starting at ``root``."""
    if root is None:
        root = next(iter(G.nodes()))
    parents: Dict[int, Optional[int]] = {root: None}
    stack = [root]
    while stack:
        node = stack.pop()
        for nbr in G.neighbors(node):
            if nbr == parents[node]:
                continue
            parents[nbr] = node
            stack.append(nbr)
    return parents


def clean_swc_to_graph(
    swc_path: Path,
    *,
    max_depth: Optional[int],
    root_parent_value: int = 0,
) -> Tuple[nx.Graph, Dict[int, NodeAttr]]:
    """Load, clean, and convert an SWC file into a NetworkX tree."""
    df_raw = read_swc(swc_path)
    df_clean = clean_swc_tree(
        df_raw,
        root_parent_value=root_parent_value,
        keep_parent_value=root_parent_value,
        max_depth=max_depth,
        keep_attrs=True,
    )

    G = nx.Graph()
    node_attrs: Dict[int, NodeAttr] = {}
    root_candidates: List[int] = []
    for row in df_clean.itertuples(index=False):
        nid = int(row.id)
        parent = int(row.parent)
        pos = np.array([float(row.x), float(row.y), float(row.z)], dtype=np.float64)
        node_attrs[nid] = {"type": int(row.type), "radius": float(row.radius)}
        G.add_node(nid, pos=pos)
        if parent > root_parent_value:
            G.add_edge(parent, nid)
        else:
            root_candidates.append(nid)

    if not nx.is_tree(G):
        raise ValueError(f"Cleaned SWC at {swc_path} is not a tree")
    if root_candidates:
        G.graph["root"] = root_candidates[0]
    else:
        G.graph["root"] = int(df_clean.iloc[0]["id"])
    return G, node_attrs


def sample_complete_binary_subtree(
    G: nx.Graph,
    rng: random.Random,
    tree_parents: Dict[int, Optional[int]],
    min_nodes: int,
    max_nodes: int,
    stop_prob: float,
    max_attempts: int = 256,
) -> Optional[Tuple[List[int], Dict[int, Optional[int]], int]]:
    """Sample a rooted complete-binary subtree bounded by [min_nodes, max_nodes]."""
    all_nodes = list(G.nodes())
    if len(all_nodes) < min_nodes:
        return None

    for _ in range(max_attempts):
        root = rng.choice(all_nodes)
        parent: Dict[int, Optional[int]] = {root: tree_parents.get(root)}
        selected: List[int] = [root]
        q = deque([root])
        valid = True

        while q and len(selected) < max_nodes:
            node = q.popleft()
            children = [nbr for nbr in G.neighbors(node) if nbr != parent[node]]
            child_count = len(children)

            if child_count > 2:
                print(f"[WARN] Node {node} has {child_count} children; cannot form complete binary subtree.")
                valid = False
                break
            if child_count == 0:
                continue
            if child_count == 1:
                # Skip singletons to avoid lone leaves.
                continue

            rng.shuffle(children)
            should_expand = len(selected) + child_count <= max_nodes and (
                len(selected) < min_nodes or rng.random() > stop_prob
            )
            if not should_expand:
                continue
            for child in children:
                parent[child] = node
                selected.append(child)
                q.append(child)

        if not valid:
            continue
        if min_nodes <= len(selected) <= max_nodes:
            parent[root] = None
            return selected, parent, root

    return None


def write_swc_subtree(
    path: Path,
    node_order: List[int],
    parent_map: Dict[int, Optional[int]],
    node_attrs: Dict[int, NodeAttr],
    positions: Dict[int, np.ndarray],
) -> None:
    """Write the sampled subtree into a SWC file with sequential ids."""
    path.parent.mkdir(parents=True, exist_ok=True)
    id_map = {orig_id: idx + 1 for idx, orig_id in enumerate(node_order)}
    with path.open("w") as f:
        for orig_id in node_order:
            attr = node_attrs.get(orig_id, {})
            ntype = int(attr.get("type", 3))
            radius = float(attr.get("radius", 1.0))
            x, y, z = positions[orig_id]
            parent = parent_map.get(orig_id)
            parent_id = id_map[parent] if parent is not None else 0
            f.write(
                f"{id_map[orig_id]} {ntype} {x:.6f} {y:.6f} {z:.6f} {radius:.6f} {parent_id}\n"
            )


def generate_micro_trees(
    input_dir: Path,
    output_dir: Path,
    num_subgraphs: int,
    max_source_trees: int,
    min_nodes: int,
    max_nodes: int,
    stop_prob: float,
    seed: Optional[int],
    clean_max_depth: Optional[int],
    root_parent_value: int = 0,
) -> None:
    rng = random.Random(seed)
    swc_files = list_swc_files(input_dir)
    if not swc_files:
        raise FileNotFoundError(f"No SWC files found in {input_dir}")

    produced = 0
    considered = 0

    for swc_file in swc_files:
        if produced >= num_subgraphs or considered >= max_source_trees:
            break
        considered += 1
        try:
            G, node_attrs = clean_swc_to_graph(
                swc_file, max_depth=clean_max_depth, root_parent_value=root_parent_value
            )
        except Exception as exc:
            print(f"[WARN] Skipping {swc_file.name}: failed to clean/load graph ({exc})")
            continue

        base_root = G.graph.get("root")
        tree_parents = compute_tree_parents(G, root=base_root)

        sample = sample_complete_binary_subtree(
            G,
            rng,
            tree_parents=tree_parents,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            stop_prob=stop_prob,
        )
        if sample is None:
            print(f"[INFO] No valid subtree found for {swc_file.name} (skipping).")
            continue

        node_ids, parent_map, root = sample
        root_pos = G.nodes[root]["pos"]
        positions = {nid: (G.nodes[nid]["pos"] - root_pos) for nid in node_ids}

        output_name = f"{swc_file.stem}.swc"
        out_path = output_dir / output_name
        write_swc_subtree(out_path, node_ids, parent_map, node_attrs, positions)
        produced += 1
        print(f"[OK] Saved subtree from {swc_file.name} -> {out_path.name} ({len(node_ids)} nodes)")

    if produced < num_subgraphs:
        print(
            f"[WARN] Requested {num_subgraphs} subgraphs but only produced {produced}. "
            "Consider adjusting constraints or providing more source trees."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate bounded-size complete binary subtrees from SWC dendrite graphs."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing cleaned SWC files.")
    parser.add_argument("output_dir", type=Path, help="Destination directory for generated SWC trees.")
    parser.add_argument(
        "--num-subgraphs", type=int, default=100, help="Number of subgraphs to generate (default: 100)."
    )
    parser.add_argument(
        "--max-source-trees",
        type=int,
        default=100,
        help="Maximum number of original SWC files to sample from (default: 100).",
    )
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=8,
        help="Minimum number of nodes per sampled tree (default: 8).",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=35,
        help="Maximum number of nodes per sampled tree (default: 35).",
    )
    parser.add_argument(
        "--stop-prob",
        type=float,
        default=0.35,
        help="Probability of halting expansion after reaching the minimum size (default: 0.35).",
    )
    parser.add_argument(
        "--clean-max-depth",
        type=int,
        default=10000,
        help="Depth cutoff for the cleaning pipeline (<=0 disables; default: 10000).",
    )
    parser.add_argument(
        "--root-parent-value",
        type=int,
        default=0,
        choices=[0, -1],
        help="Parent value to treat as root when cleaning/exporting (default: 0).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    generate_micro_trees(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_subgraphs=args.num_subgraphs,
        max_source_trees=args.max_source_trees,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        stop_prob=args.stop_prob,
        seed=args.seed,
        clean_max_depth=args.clean_max_depth if args.clean_max_depth > 0 else None,
        root_parent_value=args.root_parent_value,
    )


if __name__ == "__main__":
    main()
