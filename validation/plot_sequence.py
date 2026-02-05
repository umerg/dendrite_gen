"""
Plot a depth-wise cherry reduction sequence for a predicted graph.

Produces a single panel image with 4 plots per row. Deepest-level nodes are
highlighted in orange; all nodes/edges are otherwise dark blue.
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"axes.labelsize": 16, "axes.titlesize": 20})

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

# Ensure repo root is on sys.path when running as a script from validation/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.data_loading import nx_graph_to_adj_pos
from graph_generation.depth_reduction import DepthReductionFactory


EDGE_COLOR = "#1b2a4a"
NODE_COLOR = "#1b2a4a"
DEEP_COLOR = "#f28e2b"
NODE_SIZE = 18
DEEP_NODE_SIZE = 28
EDGE_WIDTH = 1.2


def _pos_to_xyz(pos: Any) -> np.ndarray:
    arr = np.asarray(pos, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size), mode="constant", constant_values=0.0)
    return arr[:3]


def _ensure_root_from_origin(G, *, tol: float = 1e-3) -> int | None:
    if "root" in G.graph and G.graph["root"] in G.nodes:
        return G.graph["root"]
    if G.number_of_nodes() == 0:
        return None

    best_node = None
    best_norm = None
    within_tol: list[tuple[int, float]] = []
    for nid in G.nodes:
        pos = _pos_to_xyz(G.nodes[nid].get("pos", np.zeros(3)))
        norm = float(np.linalg.norm(pos))
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


def _extract_pred_graphs(payload: Any, ema_key: str | None) -> list:
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


def _compute_depths(adj: sp.csr_matrix, root: int) -> np.ndarray:
    n = adj.shape[0]
    depth = -np.ones(n, dtype=np.int64)
    if n == 0:
        return depth
    root = int(root)
    depth[root] = 0
    queue = [root]
    indptr = adj.indptr
    indices = adj.indices
    while queue:
        u = queue.pop(0)
        for v in indices[indptr[u] : indptr[u + 1]]:
            if depth[v] >= 0:
                continue
            depth[v] = depth[u] + 1
            queue.append(int(v))
    return depth


def _set_axes_tight(
    ax,
    pts: np.ndarray,
    pad_frac: float = 0.04,
    *,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    z_lim: tuple[float, float] | None = None,
) -> None:
    if pts.size == 0:
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ranges = maxs - mins
    pad = np.maximum(ranges * pad_frac, 1e-3)
    mins = mins - pad
    maxs = maxs + pad
    if x_lim is None:
        ax.set_xlim(mins[0], maxs[0])
        x_range = maxs[0] - mins[0]
    else:
        ax.set_xlim(x_lim[0], x_lim[1])
        x_range = x_lim[1] - x_lim[0]
    if y_lim is None:
        ax.set_ylim(mins[1], maxs[1])
        y_range = maxs[1] - mins[1]
    else:
        ax.set_ylim(y_lim[0], y_lim[1])
        y_range = y_lim[1] - y_lim[0]
    if z_lim is None:
        ax.set_zlim(mins[2], maxs[2])
        z_range = maxs[2] - mins[2]
    else:
        ax.set_zlim(z_lim[0], z_lim[1])
        z_range = z_lim[1] - z_lim[0]
    ax.set_box_aspect(
        [
            max(abs(x_range), 1e-3),
            max(abs(y_range), 1e-3),
            max(abs(z_range), 1e-3),
        ]
    )


def _plot_graph_panel(
    ax,
    adj: sp.csr_matrix,
    pos: np.ndarray,
    deepest: np.ndarray,
    *,
    elev: float,
    azim: float,
    title: str,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    z_lim: tuple[float, float] | None = None,
) -> None:
    if pos.size == 0:
        ax.set_title(title)
        ax.set_axis_off()
        return

    deepest_set = set(int(i) for i in deepest.tolist()) if deepest.size else set()
    coo = adj.tocoo()
    for u, v in zip(coo.row, coo.col):
        if u >= v:
            continue
        p0 = pos[u]
        p1 = pos[v]
        edge_color = DEEP_COLOR if (u in deepest_set or v in deepest_set) else EDGE_COLOR
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color=edge_color,
            linewidth=EDGE_WIDTH,
        )

    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        s=NODE_SIZE,
        c=NODE_COLOR,
        edgecolors="k",
        linewidths=0.3,
    )
    if deepest.size:
        deep_pts = pos[deepest]
        ax.scatter(
            deep_pts[:, 0],
            deep_pts[:, 1],
            deep_pts[:, 2],
            s=DEEP_NODE_SIZE,
            c=DEEP_COLOR,
            edgecolors="k",
            linewidths=0.3,
        )

    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    _set_axes_tight(ax, pos, x_lim=x_lim, y_lim=y_lim, z_lim=z_lim)


def build_reduction_sequence(adj: sp.csr_matrix, pos: np.ndarray) -> list[dict[str, Any]]:
    reducer = DepthReductionFactory(mode="deterministic", contract_root=True, root=0)(adj)
    cur_pos = pos
    sequence: list[dict[str, Any]] = []

    while True:
        root = int(reducer.root)
        depths = _compute_depths(reducer.adj, root)
        max_depth = int(depths.max()) if depths.size else 0
        deepest = np.where(depths == max_depth)[0] if depths.size else np.array([], dtype=int)
        sequence.append(
            {
                "adj": reducer.adj,
                "pos": cur_pos,
                "root": root,
                "deepest": deepest,
                "n": int(reducer.n),
                "level": int(reducer.level),
            }
        )

        if reducer.n <= 1:
            break
        next_reducer = reducer.get_reduced_graph()
        if not next_reducer.did_contract:
            break
        cur_pos = cur_pos[next_reducer.survivor_mask]
        reducer = next_reducer

    return sequence


def plot_sequence(
    sequence: Iterable[dict[str, Any]],
    *,
    out_path: Path,
    elev: float = 20.0,
    azim: float = 30.0,
    ncols: int = 5,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    z_lim: tuple[float, float] | None = None,
) -> Path:
    seq = list(sequence)
    seq.reverse()
    n_plots = len(seq)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n_plots / ncols)) if n_plots else 1

    fig = plt.figure(figsize=(4.2 * ncols, 4.0 * nrows))
    for i, item in enumerate(seq):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        title = f"Level {i} | n={item['n']}"
        _plot_graph_panel(
            ax,
            item["adj"],
            item["pos"],
            item["deepest"],
            elev=elev,
            azim=azim,
            title=title,
            x_lim=x_lim,
            y_lim=y_lim,
            z_lim=z_lim,
        )

    for j in range(n_plots, nrows * ncols):
        ax = fig.add_subplot(nrows, ncols, j + 1, projection="3d")
        ax.set_axis_off()

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot depth-wise cherry reduction sequence for a prediction.")
    parser.add_argument("--pred-pkl", type=Path, required=True, help="Pickle file with predicted graphs.")
    parser.add_argument("--pred-index", type=int, default=0, help="Index of prediction to plot.")
    parser.add_argument("--ema-key", type=str, default=None, help="EMA key inside pickle (e.g., 'ema_0.999').")
    parser.add_argument("--out", type=Path, default=None, help="Output image path.")
    parser.add_argument("--elev", type=float, default=20.0, help="Elevation angle for 3D view.")
    parser.add_argument("--azim", type=float, default=30.0, help="Azimuth angle for 3D view.")
    parser.add_argument("--ncols", type=int, default=5, help="Number of panels per row.")
    parser.add_argument("--x-lim", type=float, nargs=2, default=None, help="Fixed x-axis limits (min max).")
    parser.add_argument("--y-lim", type=float, nargs=2, default=None, help="Fixed y-axis limits (min max).")
    parser.add_argument("--z-lim", type=float, nargs=2, default=None, help="Fixed z-axis limits (min max).")
    parser.add_argument("--root-tol", type=float, default=1e-3, help="Tolerance for origin-based root selection.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    with Path(args.pred_pkl).open("rb") as f:
        payload = pickle.load(f)
    pred_graphs = _extract_pred_graphs(payload, args.ema_key)
    if not pred_graphs:
        raise ValueError("No predicted graphs found in pickle.")
    if args.pred_index < 0 or args.pred_index >= len(pred_graphs):
        raise IndexError(f"pred-index {args.pred_index} out of range (0..{len(pred_graphs)-1}).")

    G = pred_graphs[args.pred_index]
    _ensure_root_from_origin(G, tol=args.root_tol)

    adj, pos, _node_order = nx_graph_to_adj_pos(G)
    seq = build_reduction_sequence(adj, pos)

    if args.out is None:
        out_path = Path(f"pred{args.pred_index:04d}_depth_sequence.png")
    else:
        out_path = args.out

    x_lim = tuple(args.x_lim) if args.x_lim is not None else None
    y_lim = tuple(args.y_lim) if args.y_lim is not None else None
    z_lim = tuple(args.z_lim) if args.z_lim is not None else None
    plot_sequence(
        seq,
        out_path=out_path,
        elev=args.elev,
        azim=args.azim,
        ncols=args.ncols,
        x_lim=x_lim,
        y_lim=y_lim,
        z_lim=z_lim,
    )
    print(f"Saved sequence plot to {out_path}")


if __name__ == "__main__":
    main()
