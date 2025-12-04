"""Visualize the full reduction of a random SWC tree.

This script wires together the existing SWC loading helpers, the
CherryReducer/ReductionFactory, and the RandRedDataset logic to produce a
sequence of ``ReducedGraphData`` objects and save a plot for each step with
"new leaves" highlighted.

Usage (run from repo root):

    python tests/plot_reduction_sequence.py --swc /path/to/tree_dir --out plots/

If ``--swc`` points to a directory, a random ``*.swc`` file inside it is chosen.
The reducer configuration mirrors the defaults from ``ReductionFactory`` but
can be tweaked via CLI flags.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch_sparse import SparseTensor

# Ensure project root import resolution
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.data_loading import load_swc_graph, nx_graph_to_adj_pos  # noqa: E402
from graph_generation.reduction import ReductionFactory  # noqa: E402
from graph_generation.data.reduction_dataset import RandRedDataset  # noqa: E402


def _load_matplotlib():
    import importlib
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    importlib.import_module("mpl_toolkits.mplot3d")  # ensure 3D backend
    return plt


def _pick_swc(path_like: Path, rng: np.random.Generator) -> Path:
    path_like = path_like.expanduser()
    if path_like.is_file():
        return path_like
    if path_like.is_dir():
        swc_files = sorted([p for p in path_like.rglob("*.swc") if p.is_file()])
        if not swc_files:
            raise FileNotFoundError(f"No *.swc files found under directory: {path_like}")
        idx = int(rng.integers(len(swc_files)))
        return swc_files[idx]
    raise FileNotFoundError(f"Provided path is neither file nor directory: {path_like}")


def _to_numpy(tensor) -> np.ndarray:
    if tensor is None:
        return np.array([], dtype=np.int64)
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _edges_from_adj(adj) -> np.ndarray:
    if isinstance(adj, SparseTensor):
        row, col, _ = adj.coo()
        row = row.cpu().numpy()
        col = col.cpu().numpy()
    elif isinstance(adj, torch.Tensor):
        src, dst = torch.nonzero(adj > 0, as_tuple=True)
        row = src.cpu().numpy()
        col = dst.cpu().numpy()
    else:
        raise TypeError(f"Unsupported adjacency type: {type(adj)}")
    mask = row < col
    return np.stack([row[mask], col[mask]], axis=1)


class _SeqRunner(RandRedDataset):
    """Concrete wrapper just to expose get_random_reduction_sequence."""

    def __iter__(self):  # pragma: no cover - not used in this script
        raise NotImplementedError("Iteration not supported in plotting helper")


def _run_reduction(adj, pos, seed: int, contract_root: bool, cherry_p: float, mode: str):
    red_factory = ReductionFactory(mode=mode, cherry_p=cherry_p, contract_root=contract_root)
    dataset = _SeqRunner([adj], [pos], red_factory)
    rng = np.random.default_rng(seed)
    reducer = red_factory(adj.copy(), rng=rng)
    seq = dataset.get_random_reduction_sequence(reducer, pos.copy(), rng)
    return seq


def _plot_sequence(sequence: Sequence, out_dir: Path, plt) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, data in enumerate(sequence):
        positions = _to_numpy(data.pos)
        new_leaf_mask = _to_numpy(data.new_leaf_mask_from_next).astype(bool)
        parent_idx = _to_numpy(data.parent_idx_1b).astype(np.int64) - 1
        root_mask = parent_idx == -1
        edges = _edges_from_adj(data.adj)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")

        for u, v in edges:
            pu = positions[u]
            pv = positions[v]
            ax.plot([pu[0], pv[0]], [pu[1], pv[1]], [pu[2], pv[2]], color="lightgray", linewidth=0.8, alpha=0.8)

        colors = np.full(positions.shape[0], "steelblue", dtype=object)
        sizes = np.full(positions.shape[0], 25, dtype=np.float64)
        colors[new_leaf_mask] = "crimson"
        sizes[new_leaf_mask] = 60
        colors[root_mask] = "gold"
        sizes[root_mask] = 90
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, s=sizes)

        ax.set_title(
            f"Level {int(_to_numpy(data.reduction_level))} | N={positions.shape[0]} | new={int(new_leaf_mask.sum())}"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=20, azim=30)
        fig.tight_layout()
        fig.savefig(out_dir / f"reduction_step_{idx:02d}.png", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot reduction sequence for an SWC tree")
    parser.add_argument("--swc", required=True, help="Path to an SWC file or directory containing SWC files")
    parser.add_argument("--out", default="reduction_plots", help="Directory to store per-step plots")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reducer randomness")
    parser.add_argument("--mode", choices=["stochastic", "deterministic"], default="stochastic")
    parser.add_argument("--cherry-p", type=float, default=0.8, dest="cherry_p", help="Contraction probability")
    parser.add_argument("--contract-root", action="store_true", help="Allow contracting the root as well")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    swc_path = _pick_swc(Path(args.swc), rng)
    print(f"Using SWC file: {swc_path}")

    graph = load_swc_graph(swc_path)
    adj, pos, _ = nx_graph_to_adj_pos(graph)

    sequence = _run_reduction(adj, pos, args.seed, args.contract_root, args.cherry_p, args.mode)
    print(f"Generated {len(sequence)} reduction steps")

    plt = _load_matplotlib()
    out_dir = Path(args.out)
    _plot_sequence(sequence, out_dir, plt)
    print(f"Saved plots to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
