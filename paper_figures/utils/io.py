"""Input/output helpers for paper figure generation.

This module centralizes file discovery and loading for the paper-facing
plotting pipeline.

The existing validation scripts in this repository duplicate several small IO
helpers (SWC discovery, prediction-pickle parsing, GT/pred pairing). Here we
collect the useful parts into one place so newer plotting code can share a
single interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import pickle
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import networkx as nx


@dataclass(frozen=True)
class FigureTreeRecord:
    """A minimal record describing one tree input for plotting."""

    label: str
    path: Path
    domain: str


def list_swc_files(root: Path) -> list[Path]:
    """Return SWC files below ``root`` sorted by filename."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")
    return sorted(
        p
        for p in root.iterdir()
        if p.is_file() and not p.name.startswith("._") and p.name.endswith(".swc")
    )


def load_tree_graph(path: Path) -> "nx.Graph":
    """Load one SWC tree using the repo's standard SWC loader."""
    from dendrite_gen.utils.data_loading import load_swc_graph

    return load_swc_graph(Path(path))


def _pos_to_xyz(pos: Any) -> tuple[float, float, float]:
    if pos is None:
        return (0.0, 0.0, 0.0)
    try:
        vals = list(pos)
    except TypeError:
        vals = [pos]
    vals = [float(v) for v in vals]
    if len(vals) < 3:
        vals.extend([0.0] * (3 - len(vals)))
    return (vals[0], vals[1], vals[2])


def ensure_root_from_origin(G: "nx.Graph", *, tol: float = 1e-5) -> int | None:
    """Ensure ``G.graph['root']`` is set, defaulting to the node closest to the origin."""
    if "root" in G.graph and G.graph["root"] in G.nodes:
        return G.graph["root"]
    if G.number_of_nodes() == 0:
        return None

    best_node = None
    best_norm = None
    within_tol: list[tuple[int, float]] = []
    for nid in G.nodes:
        pos = _pos_to_xyz(G.nodes[nid].get("pos"))
        norm = math.sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2])
        if norm <= tol:
            within_tol.append((nid, norm))
        if best_norm is None or norm < best_norm:
            best_norm = norm
            best_node = nid

    if within_tol:
        within_tol.sort(key=lambda x: x[1])
        root = within_tol[0][0]
    else:
        root = best_node

    if root is not None:
        G.graph["root"] = root
    return root


def load_gt_file_graphs(gt_dir: Path) -> tuple[list[Path], list["nx.Graph"]]:
    """Load GT SWC files and graphs with matching ordering."""
    gt_files = list_swc_files(gt_dir)
    gt_graphs = [load_tree_graph(path) for path in gt_files]
    return gt_files, gt_graphs


def extract_pred_graphs(payload: Any, ema_key: str | None = None) -> list["nx.Graph"]:
    """Extract predicted graphs from a validation pickle payload."""
    if isinstance(payload, dict):
        if "pred_graphs" in payload:
            return payload["pred_graphs"]
        if ema_key is not None:
            if ema_key not in payload:
                available = ", ".join(sorted(str(k) for k in payload.keys()))
                raise KeyError(f"EMA key '{ema_key}' not in pickle. Available: {available}")
            inner = payload[ema_key]
            if isinstance(inner, dict) and "pred_graphs" in inner:
                return inner["pred_graphs"]
            raise KeyError(f"EMA entry '{ema_key}' missing 'pred_graphs'.")
        if len(payload) == 1:
            only_val = next(iter(payload.values()))
            if isinstance(only_val, dict) and "pred_graphs" in only_val:
                return only_val["pred_graphs"]
    raise ValueError("Unrecognized pickle format: could not find 'pred_graphs'.")


def load_pred_graphs_from_pickle(
    pred_pkl: Path,
    *,
    ema_key: str | None = None,
) -> list["nx.Graph"]:
    """Load predicted graphs from a validation pickle."""
    pred_pkl = Path(pred_pkl)
    if not pred_pkl.exists():
        raise FileNotFoundError(f"Prediction pickle does not exist: {pred_pkl}")
    if not pred_pkl.is_file():
        raise FileNotFoundError(f"Prediction pickle is not a file: {pred_pkl}")

    with pred_pkl.open("rb") as f:
        payload = pickle.load(f)
    pred_graphs = extract_pred_graphs(payload, ema_key=ema_key)
    if not pred_graphs:
        raise ValueError("No predicted graphs found in pickle.")
    for graph in pred_graphs:
        ensure_root_from_origin(graph)
    return pred_graphs


def pair_graphs_by_index(
    gt_files: list[Path],
    gt_graphs: list["nx.Graph"],
    pred_graphs: list["nx.Graph"],
) -> tuple[list[dict[str, int | str | None]], list[dict[str, int]]]:
    """Pair GT and predicted graphs 1:1 by list index."""
    n = min(len(gt_graphs), len(pred_graphs))
    pairs: list[dict[str, int | str | None]] = []
    for i in range(n):
        gt_size = gt_graphs[i].number_of_nodes()
        pred_size = pred_graphs[i].number_of_nodes()
        pairs.append(
            {
                "gt_idx": i,
                "pred_idx": i,
                "gt_name": gt_files[i].name if i < len(gt_files) else None,
                "match_type": "index",
                "size_diff": abs(gt_size - pred_size),
            }
        )

    unmatched: list[dict[str, int]] = []
    if len(gt_graphs) != len(pred_graphs):
        unmatched.append(
            {
                "gt_count": len(gt_graphs),
                "pred_count": len(pred_graphs),
                "matched": n,
            }
        )
    return pairs, unmatched
