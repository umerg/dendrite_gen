"""SWC loading smoke test.

Loads all .swc files from a directory provided via env var `SWC_TEST_DIR`,
verifies they form trees, converts each to adjacency/pos arrays, and (if
`matplotlib` is available) saves a simple 3D scatter + edge plot per graph
into `dataloading_tests/` under the project root.

To run manually:
    SWC_TEST_DIR=/absolute/path/to/swc_dir pytest -q tests/test_swc_loading.py

Optional override for output directory via env var `SWC_TEST_OUTPUT_DIR`.
If the env var `SWC_TEST_DIR` is not set the test is skipped cleanly.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import pytest
import networkx as nx

# Ensure project root on path (so 'utils' is importable when running directly)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.data_loading import load_swc_graphs_from_dir, nx_graph_to_adj_pos


def _maybe_get_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused but validates 3D backend
        return plt
    except Exception:  # broad: any ImportError or backend issue -> skip plotting
        return None


@pytest.mark.skipif("SWC_TEST_DIR" not in os.environ, reason="SWC_TEST_DIR env var not set")
def test_load_and_plot_swc_dir():
    swc_dir = Path(os.environ["SWC_TEST_DIR"]).expanduser()
    assert swc_dir.is_dir(), f"Provided SWC_TEST_DIR is not a directory: {swc_dir}" 

    graphs = load_swc_graphs_from_dir(swc_dir)
    assert graphs, f"No .swc files found in {swc_dir}" 

    # Output directory
    out_root = Path(os.environ.get("SWC_TEST_OUTPUT_DIR", "dataloading_tests"))
    out_root.mkdir(parents=True, exist_ok=True)

    plt = _maybe_get_matplotlib()
    plotted = 0

    for idx, G in enumerate(graphs):
        # Basic structural assertions
        assert nx.is_tree(G), f"Graph {idx} not a tree"
        assert len(G) > 0, f"Graph {idx} empty"

        # Derive adjacency & positions (smoke path)
        A, P, order = nx_graph_to_adj_pos(G)
        assert P.shape[0] == len(G), "Position array length mismatch"
        assert A.shape[0] == len(G), "Adjacency rows mismatch"

        # Always store raw arrays for inspection
        npz_path = out_root / f"graph_{idx:03d}.npz"
        import numpy as np
        np.savez_compressed(npz_path, adjacency=A.data, indices=A.indices, indptr=A.indptr, shape=A.shape, pos=P, order=order)

        if plt is not None:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection="3d")
            xs, ys, zs = P[:, 0], P[:, 1], P[:, 2]
            ax.scatter(xs, ys, zs, s=8, c='k')
            # Draw edges
            for u, v in G.edges():
                pu = G.nodes[u]['pos']
                pv = G.nodes[v]['pos']
                ax.plot([pu[0], pv[0]], [pu[1], pv[1]], [pu[2], pv[2]], color='blue', linewidth=0.5)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_title(f"SWC Graph {idx}")
            fig.tight_layout()
            fig_path = out_root / f"graph_{idx:03d}.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            plotted += 1

    # If matplotlib available ensure at least one plot was created
    if plt is not None:
        assert plotted == len(graphs), "Not all graphs plotted despite matplotlib availability"
    else:
        # Without matplotlib we still consider the test successful if arrays saved
        assert (out_root / "graph_000.npz").exists(), "Expected npz output missing"
