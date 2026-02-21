"""
Interactive 3D overlay of ground-truth (SWC) and predicted trees.

Loads GT graphs from a directory (same selection rules as chamfer.py) and
predicted graphs from a validation pickle, then overlays a chosen GT/pred pair
in Plotly. Node visibility and per-tree opacity can be adjusted with widgets
when running in a notebook environment.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import plotly.graph_objects as go

try:
    from ipywidgets import Checkbox, FloatSlider, HBox, VBox
    from IPython.display import display

    _HAVE_WIDGETS = True
except Exception:
    _HAVE_WIDGETS = False


# Ensure repo root is on sys.path when running as a script from validation/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.data_loading import load_swc_graphs_from_dir


GT_COLOR = "#1f77b4"
PRED_COLOR = "#8b1e3f"
GT_EDGE_COLOR = "#9ecae1"
PRED_EDGE_COLOR = "#f0a6b5"
NODE_SIZE = 5
EDGE_WIDTH = 3


def _list_swc_files(dir_path: Path) -> list[Path]:
    """Mirror utils.data_loading.load_swc_graphs_from_dir file selection logic."""
    dir_path = Path(dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        raise NotADirectoryError(f"Provided path is not a directory: {dir_path}")
    files: list[Path] = []
    for swc_file in sorted(dir_path.iterdir()):
        if not swc_file.is_file():
            continue
        name = swc_file.name
        if name.startswith("._"):
            continue
        if not name.endswith(".csv.swc"):
            continue
        files.append(swc_file)
    return files


def _pos_to_xyz(pos: Any) -> np.ndarray:
    arr = np.asarray(pos, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size), mode="constant", constant_values=0.0)
    return arr[:3]


def _graph_positions(G: nx.Graph) -> dict[int, np.ndarray]:
    return {n: _pos_to_xyz(G.nodes[n].get("pos", np.zeros(3))) for n in G.nodes()}


def _edge_xyz(pos: dict[int, np.ndarray], edges: list[tuple[int, int]]) -> tuple[list[float], list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for u, v in edges:
        p0 = pos.get(u)
        p1 = pos.get(v)
        if p0 is None or p1 is None:
            continue
        xs.extend([float(p0[0]), float(p1[0]), None])
        ys.extend([float(p0[1]), float(p1[1]), None])
        zs.extend([float(p0[2]), float(p1[2]), None])
    return xs, ys, zs


def _node_xyz(pos: dict[int, np.ndarray]) -> tuple[list[float], list[float], list[float], list[str]]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    labels: list[str] = []
    for node, p in pos.items():
        xs.append(float(p[0]))
        ys.append(float(p[1]))
        zs.append(float(p[2]))
        labels.append(str(node))
    return xs, ys, zs, labels


def _extract_pred_graphs(payload: Any, ema_key: str | None) -> list[nx.Graph]:
    """Handle validation pickle formats (ema-keyed dict or direct dict)."""
    if isinstance(payload, dict):
        if "pred_graphs" in payload:
            return payload["pred_graphs"]
        if ema_key is not None:
            if ema_key not in payload:
                available = ", ".join(sorted(payload.keys()))
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


def _group_by_size(graphs: list[nx.Graph]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for idx, G in enumerate(graphs):
        groups.setdefault(G.number_of_nodes(), []).append(idx)
    return groups


def _match_by_size(
    gt_graphs: list[nx.Graph],
    pred_graphs: list[nx.Graph],
) -> list[dict[str, int]]:
    """Match indices by node count; fall back to closest-size matching."""
    gt_groups = _group_by_size(gt_graphs)
    pred_groups = _group_by_size(pred_graphs)
    pairs: list[dict[str, int]] = []

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    unmatched_gt: list[int] = []
    unmatched_pred: list[int] = []

    for size in sorted(set(gt_groups) | set(pred_groups)):
        g_list = gt_groups.get(size, [])
        p_list = pred_groups.get(size, [])
        n = min(len(g_list), len(p_list))
        for i in range(n):
            gt_idx = g_list[i]
            pred_idx = p_list[i]
            pairs.append(
                {
                    "gt_idx": gt_idx,
                    "pred_idx": pred_idx,
                    "match_type": "exact",
                    "size_diff": 0,
                }
            )
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
        if len(g_list) > n:
            unmatched_gt.extend(g_list[n:])
        if len(p_list) > n:
            unmatched_pred.extend(p_list[n:])

    if not pred_graphs:
        return pairs

    unused_pred = set(unmatched_pred)
    for gt_idx in unmatched_gt:
        gt_size = gt_graphs[gt_idx].number_of_nodes()
        candidate_pool = unused_pred if unused_pred else set(range(len(pred_graphs)))
        best_pred = None
        best_diff = None
        for pred_idx in candidate_pool:
            pred_size = pred_graphs[pred_idx].number_of_nodes()
            diff = abs(gt_size - pred_size)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_pred = pred_idx
        if best_pred is None:
            continue
        pairs.append(
            {
                "gt_idx": gt_idx,
                "pred_idx": best_pred,
                "match_type": "closest",
                "size_diff": int(best_diff) if best_diff is not None else 0,
            }
        )
        matched_gt.add(gt_idx)
        matched_pred.add(best_pred)
        if best_pred in unused_pred:
            unused_pred.remove(best_pred)

    return pairs


def _in_ipython() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        return get_ipython() is not None
    except Exception:
        return False


def load_graphs(gt_dir: Path, pred_pkl: Path, ema_key: str | None) -> tuple[list[Path], list[nx.Graph], list[nx.Graph]]:
    gt_dir = Path(gt_dir)
    pred_pkl = Path(pred_pkl)
    gt_files = _list_swc_files(gt_dir)
    gt_graphs = load_swc_graphs_from_dir(gt_dir)
    if len(gt_files) != len(gt_graphs):
        raise RuntimeError("GT file list and loaded graph count mismatch.")

    with pred_pkl.open("rb") as f:
        payload = pickle.load(f)
    pred_graphs = _extract_pred_graphs(payload, ema_key)
    if not pred_graphs:
        raise ValueError("No predicted graphs found in pickle.")

    return gt_files, gt_graphs, pred_graphs


def build_overlay_figure(
    gt: nx.Graph,
    pred: nx.Graph,
    *,
    show_gt_nodes: bool = True,
    show_pred_nodes: bool = True,
    gt_opacity: float = 0.9,
    pred_opacity: float = 0.7,
    gt_edge_color: str = GT_EDGE_COLOR,
    pred_edge_color: str = PRED_EDGE_COLOR,
    title: str = "GT vs Pred Overlay",
    as_widget: bool = False,
) -> tuple[go.Figure, dict[str, int]]:
    pos_gt = _graph_positions(gt)
    pos_pred = _graph_positions(pred)

    gt_edge_x, gt_edge_y, gt_edge_z = _edge_xyz(pos_gt, list(gt.edges()))
    pred_edge_x, pred_edge_y, pred_edge_z = _edge_xyz(pos_pred, list(pred.edges()))

    gt_node_x, gt_node_y, gt_node_z, gt_labels = _node_xyz(pos_gt)
    pred_node_x, pred_node_y, pred_node_z, pred_labels = _node_xyz(pos_pred)

    fig: go.Figure
    if as_widget:
        fig = go.FigureWidget()
    else:
        fig = go.Figure()

    trace_indices: dict[str, int] = {}

    fig.add_trace(
        go.Scatter3d(
            x=gt_edge_x,
            y=gt_edge_y,
            z=gt_edge_z,
            mode="lines",
            line=dict(color=gt_edge_color, width=EDGE_WIDTH),
            name="GT skeleton",
            opacity=gt_opacity,
            hoverinfo="skip",
        )
    )
    trace_indices["gt_edges"] = len(fig.data) - 1

    fig.add_trace(
        go.Scatter3d(
            x=gt_node_x,
            y=gt_node_y,
            z=gt_node_z,
            mode="markers",
            marker=dict(size=NODE_SIZE, color=GT_COLOR, line=dict(color="black", width=0.3)),
            name="GT nodes",
            opacity=gt_opacity,
            visible=show_gt_nodes,
            text=gt_labels,
        )
    )
    trace_indices["gt_nodes"] = len(fig.data) - 1

    fig.add_trace(
        go.Scatter3d(
            x=pred_edge_x,
            y=pred_edge_y,
            z=pred_edge_z,
            mode="lines",
            line=dict(color=pred_edge_color, width=EDGE_WIDTH),
            name="Pred skeleton",
            opacity=pred_opacity,
            hoverinfo="skip",
        )
    )
    trace_indices["pred_edges"] = len(fig.data) - 1

    fig.add_trace(
        go.Scatter3d(
            x=pred_node_x,
            y=pred_node_y,
            z=pred_node_z,
            mode="markers",
            marker=dict(size=NODE_SIZE, color=PRED_COLOR, line=dict(color="black", width=0.3)),
            name="Pred nodes",
            opacity=pred_opacity,
            visible=show_pred_nodes,
            text=pred_labels,
        )
    )
    trace_indices["pred_nodes"] = len(fig.data) - 1

    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(itemsizing="constant"),
    )
    return fig, trace_indices


def attach_widgets(
    fig: go.Figure,
    trace_indices: dict[str, int],
    *,
    gt_label: str,
    pred_label: str,
    initial_gt_opacity: float,
    initial_pred_opacity: float,
    show_gt_nodes: bool,
    show_pred_nodes: bool,
) -> None:
    if not _HAVE_WIDGETS:
        return

    gt_opacity_slider = FloatSlider(
        value=initial_gt_opacity,
        min=0.05,
        max=1.0,
        step=0.05,
        description="GT opacity",
        readout_format=".2f",
        continuous_update=True,
    )
    pred_opacity_slider = FloatSlider(
        value=initial_pred_opacity,
        min=0.05,
        max=1.0,
        step=0.05,
        description="Pred opacity",
        readout_format=".2f",
        continuous_update=True,
    )
    gt_nodes_checkbox = Checkbox(value=show_gt_nodes, description=f"Show {gt_label} nodes")
    pred_nodes_checkbox = Checkbox(value=show_pred_nodes, description=f"Show {pred_label} nodes")

    def _apply_state() -> None:
        gt_opacity = float(gt_opacity_slider.value)
        pred_opacity = float(pred_opacity_slider.value)
        fig.data[trace_indices["gt_edges"]].opacity = gt_opacity
        fig.data[trace_indices["gt_nodes"]].opacity = gt_opacity
        fig.data[trace_indices["pred_edges"]].opacity = pred_opacity
        fig.data[trace_indices["pred_nodes"]].opacity = pred_opacity
        fig.data[trace_indices["gt_nodes"]].visible = bool(gt_nodes_checkbox.value)
        fig.data[trace_indices["pred_nodes"]].visible = bool(pred_nodes_checkbox.value)

    gt_opacity_slider.observe(lambda _change: _apply_state(), names="value")
    pred_opacity_slider.observe(lambda _change: _apply_state(), names="value")
    gt_nodes_checkbox.observe(lambda _change: _apply_state(), names="value")
    pred_nodes_checkbox.observe(lambda _change: _apply_state(), names="value")

    _apply_state()
    display(VBox([HBox([gt_opacity_slider, pred_opacity_slider]), HBox([gt_nodes_checkbox, pred_nodes_checkbox]), fig]))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay GT and predicted trees in interactive 3D.")
    parser.add_argument("--gt-dir", type=Path, required=True, help="Directory containing GT .csv.swc files.")
    parser.add_argument("--pred-pkl", type=Path, required=True, help="Validation pickle with pred_graphs.")
    parser.add_argument("--ema-key", type=str, default=None, help="EMA key inside pickle (e.g., 'ema_0.999').")
    parser.add_argument("--gt-index", type=int, default=0, help="GT graph index (ignored if --match-by-size).")
    parser.add_argument("--pred-index", type=int, default=0, help="Pred graph index (ignored if --match-by-size).")
    parser.add_argument("--match-by-size", action="store_true", help="Match GT/pred by node count.")
    parser.add_argument("--pair-index", type=int, default=0, help="Which matched pair to select when using --match-by-size.")
    parser.add_argument("--gt-opacity", type=float, default=0.9, help="Initial GT opacity (0-1).")
    parser.add_argument("--pred-opacity", type=float, default=0.7, help="Initial pred opacity (0-1).")
    parser.add_argument("--gt-edge-color", type=str, default=GT_EDGE_COLOR, help="GT skeleton color.")
    parser.add_argument("--pred-edge-color", type=str, default=PRED_EDGE_COLOR, help="Pred skeleton color.")
    parser.add_argument("--no-gt-nodes", action="store_true", help="Hide GT nodes initially.")
    parser.add_argument("--no-pred-nodes", action="store_true", help="Hide pred nodes initially.")
    parser.add_argument("--widgets", action="store_true", help="Use ipywidgets controls when available.")
    parser.add_argument("--save-html", type=Path, default=None, help="Optional path to save the plot as HTML.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    gt_files, gt_graphs, pred_graphs = load_graphs(args.gt_dir, args.pred_pkl, args.ema_key)

    if args.match_by_size:
        pairs = _match_by_size(gt_graphs, pred_graphs)
        if not pairs:
            raise RuntimeError("No GT/pred pairs found to plot.")
        if args.pair_index < 0 or args.pair_index >= len(pairs):
            raise IndexError(f"pair-index {args.pair_index} out of range (0..{len(pairs)-1}).")
        pair = pairs[args.pair_index]
        gt_idx = pair["gt_idx"]
        pred_idx = pair["pred_idx"]
        match_note = f"{pair['match_type']} (size diff {pair['size_diff']})"
    else:
        gt_idx = args.gt_index
        pred_idx = args.pred_index
        match_note = "manual"

    if gt_idx < 0 or gt_idx >= len(gt_graphs):
        raise IndexError(f"GT index {gt_idx} out of range (0..{len(gt_graphs)-1}).")
    if pred_idx < 0 or pred_idx >= len(pred_graphs):
        raise IndexError(f"Pred index {pred_idx} out of range (0..{len(pred_graphs)-1}).")

    gt = gt_graphs[gt_idx]
    pred = pred_graphs[pred_idx]
    gt_name = gt_files[gt_idx].name if gt_idx < len(gt_files) else f"gt_{gt_idx}"
    pred_name = f"pred_{pred_idx}"
    title = (
        f"GT {gt_idx} ({gt_name}, n={gt.number_of_nodes()}) "
        f"vs Pred {pred_idx} (n={pred.number_of_nodes()}) | {match_note}"
    )

    use_widgets = args.widgets and _HAVE_WIDGETS and _in_ipython()
    fig, trace_indices = build_overlay_figure(
        gt,
        pred,
        show_gt_nodes=not args.no_gt_nodes,
        show_pred_nodes=not args.no_pred_nodes,
        gt_opacity=float(args.gt_opacity),
        pred_opacity=float(args.pred_opacity),
        gt_edge_color=str(args.gt_edge_color),
        pred_edge_color=str(args.pred_edge_color),
        title=title,
        as_widget=use_widgets,
    )

    if args.save_html is not None:
        fig.write_html(str(args.save_html))
        print(f"Saved HTML to {args.save_html}")

    if use_widgets:
        attach_widgets(
            fig,
            trace_indices,
            gt_label="GT",
            pred_label="Pred",
            initial_gt_opacity=float(args.gt_opacity),
            initial_pred_opacity=float(args.pred_opacity),
            show_gt_nodes=not args.no_gt_nodes,
            show_pred_nodes=not args.no_pred_nodes,
        )
    else:
        if args.widgets and not _HAVE_WIDGETS:
            print("ipywidgets not available; falling back to static Plotly figure.")
        elif args.widgets and not _in_ipython():
            print("Widgets requested but not in IPython; falling back to static Plotly figure.")
        fig.show()


if __name__ == "__main__":
    main()
