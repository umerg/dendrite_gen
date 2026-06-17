"""Convert `.smol` sample dumps into visualization-friendly prediction pickles.

The current `.smol` files we have explored are:
  - a top-level pickle containing a list of per-sample byte blobs
  - each blob unpickles to a dict with at least `coords` and `bond_indices`

This script turns those samples into one rooted NetworkX tree per entry and
writes a pickle payload that the visualization runners can already consume.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import pickle
from typing import Any

import networkx as nx
import numpy as np


ROOT_MODE_CHOICES = ("degree", "origin", "centroid", "first")
COMPONENT_MODE_CHOICES = ("largest", "root")
OUT_FORMAT_CHOICES = ("validation", "pred_graphs", "graph-list")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _node_sort_key(node: Any) -> tuple[int, Any]:
    try:
        return (0, int(node))
    except (TypeError, ValueError):
        return (1, str(node))


def _normalize_coords(coords: Any) -> np.ndarray:
    arr = np.asarray(_to_numpy(coords), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected coords to have shape (N, C), got {arr.shape}.")
    if arr.shape[1] < 3:
        arr = np.pad(arr, ((0, 0), (0, 3 - arr.shape[1])), mode="constant")
    return arr[:, :3]


def _normalize_edge_pairs(edge_pairs: Any) -> np.ndarray:
    arr = np.asarray(_to_numpy(edge_pairs), dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"Expected bond_indices to have rank 2, got {arr.shape}.")
    if arr.shape[1] == 2:
        pairs = arr
    elif arr.shape[0] == 2:
        pairs = arr.T
    else:
        raise ValueError(f"Could not interpret bond_indices with shape {arr.shape}.")
    return pairs.astype(np.int64, copy=False)


def _optional_int_array(values: Any, *, length: int) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(_to_numpy(values))
    if arr.size != length:
        return None
    return arr.reshape(length).astype(np.int64, copy=False)


def _component_subgraph(graph: nx.Graph, component_nodes: set[int]) -> nx.Graph:
    return graph.subgraph(component_nodes).copy()


def _select_root(graph: nx.Graph, positions: dict[int, np.ndarray], mode: str) -> int:
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot choose a root for an empty graph.")

    nodes = list(graph.nodes)
    if mode == "degree":
        return min(
            nodes,
            key=lambda n: (
                -graph.degree(n),
                float(np.linalg.norm(positions[n])),
                _node_sort_key(n),
            ),
        )
    if mode == "origin":
        return min(nodes, key=lambda n: (float(np.linalg.norm(positions[n])), _node_sort_key(n)))
    if mode == "centroid":
        pts = np.stack([positions[n] for n in nodes], axis=0)
        centroid = np.mean(pts, axis=0)
        return min(
            nodes,
            key=lambda n: (
                float(np.linalg.norm(positions[n] - centroid)),
                -graph.degree(n),
                _node_sort_key(n),
            ),
        )
    if mode == "first":
        return min(nodes, key=_node_sort_key)
    raise ValueError(f"Unsupported root mode: {mode!r}")


def _select_component(
    graph: nx.Graph,
    positions: dict[int, np.ndarray],
    *,
    component_mode: str,
    root_mode: str,
) -> tuple[nx.Graph, int]:
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot select a component from an empty graph.")

    components = [set(component) for component in nx.connected_components(graph)]
    if component_mode == "largest":
        component_nodes = max(
            components,
            key=lambda component: (
                len(component),
                max(graph.degree(n) for n in component),
                min(float(np.linalg.norm(positions[n])) for n in component),
            ),
        )
        subgraph = _component_subgraph(graph, component_nodes)
        root = _select_root(subgraph, positions, root_mode)
        return subgraph, root

    if component_mode == "root":
        root = _select_root(graph, positions, root_mode)
        component_nodes = next(component for component in components if root in component)
        subgraph = _component_subgraph(graph, component_nodes)
        return subgraph, root

    raise ValueError(f"Unsupported component mode: {component_mode!r}")


def _spanning_tree(component: nx.Graph, root: int) -> nx.Graph:
    if component.number_of_nodes() <= 1:
        tree = component.copy()
        tree.graph.update(component.graph)
        return tree
    bfs_tree = nx.bfs_tree(component, source=root)
    tree = nx.Graph()
    tree.graph.update(component.graph)
    tree.add_nodes_from(component.nodes(data=True))
    tree.add_edges_from(bfs_tree.edges())
    for u, v in tree.edges():
        if component.has_edge(u, v):
            tree.edges[u, v].update(component.edges[u, v])
    return tree


def _bfs_relabel_order(tree: nx.Graph, root: int) -> list[int]:
    if root not in tree:
        raise ValueError(f"Root {root!r} is not in the tree.")
    order: list[int] = []
    seen = {root}
    queue = [root]
    for node in queue:
        order.append(node)
        neighbors = sorted(
            (nbr for nbr in tree.neighbors(node) if nbr not in seen),
            key=lambda n: (-tree.degree(n), _node_sort_key(n)),
        )
        for neighbor in neighbors:
            seen.add(neighbor)
            queue.append(neighbor)
    return order


def _graph_from_sample(
    sample: dict[str, Any],
    *,
    sample_index: int,
) -> tuple[nx.Graph, dict[int, np.ndarray], dict[str, Any]]:
    coords = _normalize_coords(sample["coords"])
    bond_indices = _normalize_edge_pairs(sample["bond_indices"])
    node_count = coords.shape[0]

    atomics = _optional_int_array(sample.get("atomics"), length=node_count)
    charges = _optional_int_array(sample.get("charges"), length=node_count)
    bond_types_raw = sample.get("bond_types")
    bond_types = None
    if bond_types_raw is not None:
        bond_types = np.asarray(_to_numpy(bond_types_raw)).reshape(-1).astype(np.int64, copy=False)
        if bond_types.size != bond_indices.shape[0]:
            bond_types = None

    graph = nx.Graph()
    for node in range(node_count):
        graph.add_node(
            node,
            pos=coords[node].astype(np.float32, copy=False),
            atomic=int(atomics[node]) if atomics is not None else None,
            charge=int(charges[node]) if charges is not None else None,
            source_node_id=node,
        )

    for edge_idx, (u, v) in enumerate(bond_indices):
        if u < 0 or v < 0 or u >= node_count or v >= node_count or u == v:
            continue
        graph.add_edge(int(u), int(v))
        if bond_types is not None and graph.has_edge(int(u), int(v)):
            graph.edges[int(u), int(v)]["bond_type"] = int(bond_types[edge_idx])

    graph.graph["source_sample_index"] = int(sample_index)
    graph.graph["source_sample_id"] = sample.get("id")
    graph.graph["source_device"] = sample.get("device")
    graph.graph["source_node_count"] = int(node_count)
    graph.graph["source_edge_count"] = int(graph.number_of_edges())

    positions = {node: coords[node] for node in graph.nodes}
    meta = {
        "raw_connected": bool(nx.is_connected(graph)) if graph.number_of_nodes() else True,
        "raw_tree": bool(nx.is_tree(graph)) if graph.number_of_nodes() else True,
    }
    return graph, positions, meta


def _convert_one_sample(
    sample: dict[str, Any],
    *,
    sample_index: int,
    component_mode: str,
    root_mode: str,
) -> tuple[nx.Graph, Counter]:
    raw_graph, positions, meta = _graph_from_sample(sample, sample_index=sample_index)
    stats = Counter()
    stats["raw_samples"] += 1
    stats["raw_nodes_total"] += raw_graph.number_of_nodes()
    stats["raw_edges_total"] += raw_graph.number_of_edges()
    if meta["raw_connected"]:
        stats["raw_connected"] += 1
    else:
        stats["raw_disconnected"] += 1
    if meta["raw_tree"]:
        stats["raw_tree"] += 1
    else:
        stats["raw_not_tree"] += 1

    component, root = _select_component(
        raw_graph,
        positions,
        component_mode=component_mode,
        root_mode=root_mode,
    )
    stats["kept_nodes_total"] += component.number_of_nodes()
    stats["kept_edges_total"] += component.number_of_edges()
    if component_mode == "largest":
        stats["selected_largest_component"] += 1
    else:
        stats["selected_root_component"] += 1

    if not nx.is_tree(component):
        stats["treeified_samples"] += 1
        component = _spanning_tree(component, root)

    order = _bfs_relabel_order(component, root)
    mapping = {old_node: new_node for new_node, old_node in enumerate(order)}
    converted = nx.relabel_nodes(component, mapping, copy=True)

    root_pos = positions[root].astype(np.float64, copy=False)
    for old_node, new_node in mapping.items():
        pos = positions[old_node] - root_pos
        converted.nodes[new_node]["pos"] = pos.astype(np.float32, copy=False)
        converted.nodes[new_node]["source_node_id"] = int(old_node)

    converted.graph["root"] = 0
    converted.graph["source_format"] = "smol"
    converted.graph["source_sample_index"] = int(sample_index)
    converted.graph["source_sample_id"] = sample.get("id")
    converted.graph["source_device"] = sample.get("device")
    converted.graph["conversion_component_mode"] = component_mode
    converted.graph["conversion_root_mode"] = root_mode
    converted.graph["conversion_raw_connected"] = meta["raw_connected"]
    converted.graph["conversion_raw_tree"] = meta["raw_tree"]
    converted.graph["conversion_raw_node_count"] = raw_graph.number_of_nodes()
    converted.graph["conversion_raw_edge_count"] = raw_graph.number_of_edges()

    stats["emitted_graphs"] += 1
    stats["emitted_nodes_total"] += converted.number_of_nodes()
    stats["emitted_edges_total"] += converted.number_of_edges()
    return converted, stats


def _format_summary(stats: Counter) -> list[str]:
    raw_samples = stats.get("raw_samples", 0)
    emitted = stats.get("emitted_graphs", 0)
    avg_emitted_nodes = (
        stats.get("emitted_nodes_total", 0) / emitted if emitted else 0.0
    )
    avg_raw_nodes = stats.get("raw_nodes_total", 0) / raw_samples if raw_samples else 0.0
    return [
        f"raw samples: {raw_samples}",
        f"emitted graphs: {emitted}",
        f"raw connected: {stats.get('raw_connected', 0)}",
        f"raw disconnected: {stats.get('raw_disconnected', 0)}",
        f"raw trees: {stats.get('raw_tree', 0)}",
        f"raw non-trees: {stats.get('raw_not_tree', 0)}",
        f"treeified samples: {stats.get('treeified_samples', 0)}",
        f"avg raw nodes: {avg_raw_nodes:.1f}",
        f"avg emitted nodes: {avg_emitted_nodes:.1f}",
    ]


def build_output_payload(
    graphs: list[nx.Graph],
    *,
    out_format: str,
    ema_key: str,
    smol_path: Path,
    stats: Counter,
) -> Any:
    summary = {
        "source_path": str(smol_path),
        "raw_samples": int(stats.get("raw_samples", 0)),
        "emitted_graphs": int(stats.get("emitted_graphs", 0)),
        "raw_connected": int(stats.get("raw_connected", 0)),
        "raw_disconnected": int(stats.get("raw_disconnected", 0)),
        "raw_tree": int(stats.get("raw_tree", 0)),
        "raw_not_tree": int(stats.get("raw_not_tree", 0)),
        "treeified_samples": int(stats.get("treeified_samples", 0)),
    }

    if out_format == "graph-list":
        return graphs

    if out_format == "pred_graphs":
        return {
            "pred_graphs": graphs,
            "conversion_summary": summary,
        }

    if out_format == "validation":
        return {
            ema_key: {
                "pred_graphs": graphs,
                "conversion_summary": summary,
            }
        }

    raise ValueError(f"Unsupported output format: {out_format!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a `.smol` sample dump into a pickle containing rooted "
            "NetworkX trees for the visualization pipeline."
        )
    )
    parser.add_argument(
        "--smol-path",
        type=Path,
        required=True,
        help="Input `.smol` file.",
    )
    parser.add_argument(
        "--out-pkl",
        type=Path,
        required=True,
        help="Output pickle path.",
    )
    parser.add_argument(
        "--out-format",
        choices=OUT_FORMAT_CHOICES,
        default="validation",
        help=(
            "Output payload shape. `validation` writes `{ema_key: {pred_graphs: ...}}`, "
            "`pred_graphs` writes `{pred_graphs: ...}`, and `graph-list` writes just the list."
        ),
    )
    parser.add_argument(
        "--ema-key",
        default="ema_1",
        help="EMA key used when `--out-format validation` is selected.",
    )
    parser.add_argument(
        "--component-mode",
        choices=COMPONENT_MODE_CHOICES,
        default="largest",
        help=(
            "How to collapse disconnected samples to one graph. "
            "`largest` keeps the largest connected component; `root` keeps the component "
            "containing the chosen root candidate."
        ),
    )
    parser.add_argument(
        "--root-mode",
        choices=ROOT_MODE_CHOICES,
        default="degree",
        help=(
            "How to choose a root before recentering. "
            "`degree` picks the highest-degree node, `origin` picks the node closest to (0,0,0), "
            "`centroid` picks the node closest to the geometric centroid, and `first` picks the "
            "lowest node id."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of samples to convert.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smol_path = Path(args.smol_path)
    out_pkl = Path(args.out_pkl)

    if not smol_path.exists():
        raise FileNotFoundError(f"SMOL file does not exist: {smol_path}")
    if not smol_path.is_file():
        raise FileNotFoundError(f"SMOL path is not a file: {smol_path}")

    with smol_path.open("rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, list):
        raise TypeError(
            f"Expected top-level `.smol` payload to be a list, got {type(payload).__name__}."
        )

    graphs: list[nx.Graph] = []
    stats: Counter = Counter()
    limit = args.limit if args.limit is None or args.limit >= 0 else None
    sample_items = payload if limit is None else payload[:limit]

    for sample_index, item in enumerate(sample_items):
        sample = pickle.loads(item) if isinstance(item, (bytes, bytearray, memoryview)) else item
        if not isinstance(sample, dict):
            raise TypeError(
                f"Sample {sample_index} should decode to a dict, got {type(sample).__name__}."
            )
        if "coords" not in sample or "bond_indices" not in sample:
            raise KeyError(
                f"Sample {sample_index} is missing required keys `coords` and/or `bond_indices`."
            )
        graph, sample_stats = _convert_one_sample(
            sample,
            sample_index=sample_index,
            component_mode=args.component_mode,
            root_mode=args.root_mode,
        )
        graphs.append(graph)
        stats.update(sample_stats)

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    output_payload = build_output_payload(
        graphs,
        out_format=args.out_format,
        ema_key=args.ema_key,
        smol_path=smol_path,
        stats=stats,
    )
    with out_pkl.open("wb") as handle:
        pickle.dump(output_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote {out_pkl}")
    for line in _format_summary(stats):
        print(f"  - {line}")


if __name__ == "__main__":
    main()
