#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test the geometric coarsening dataloader on 4 random MICrONS neurons.

What this script does:
  1) Picks 4 random skeleton files from `skel_files`
  2) Loads + masks dendrites (your exact code)
  3) Builds branch graphs with tips (so pruning happens)
  4) Converts to (adj, pos) with a consistent node order
  5) Creates InfiniteRandRedDataset and a DataLoader
  6) Prints VERY verbose info for a handful of yielded samples
  7) (Optional) Plots the full coarsening sequence in 3D

Usage:
  # Display plots in GUI (default)
  python test_geometric_loader.py --plot-sequences --num-batches 2 --batch-size 2 --seed 2
  
  # Save plots to files
  python test_geometric_loader.py --plot-sequences --save-plots --output-dir ./my_plots

Dependencies: numpy, scipy, networkx, matplotlib, torch, torch_geometric
"""

import argparse
import random
import sys
from pathlib import Path

# Add project root to Python path to ensure imports work from any directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D proj)

import torch as th
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

# --- Skeleton data loading ---
import skeleton_plot as sklpt
import skeleton_plot.skel_io as skio
import s3fs

# --- import your project modules ---
from graph_generation.data import InfiniteRandRedDataset
from graph_generation.reduction import CherryReducer, ReductionFactory


def load_skeleton_data(num_skeletons=4, seed=42):
    """
    Load skeleton data from MICrONS dataset.
    
    Parameters:
    -----------
    num_skeletons : int
        Number of random skeletons to load
    seed : int
        Random seed for reproducible skeleton selection
        
    Returns:
    --------
    skeletons : list
        List of loaded skeleton objects
    filenames : list
        List of corresponding filenames
    """
    print("Loading skeleton data from MICrONS dataset...")
    
    # Setup paths
    skel_path = "s3://bossdb-open-data/iarpa_microns/minnie/minnie65/skeletons/v661/skeletons/"
    
    try:
        # Initialize S3 filesystem
        fs = s3fs.S3FileSystem(anon=True)
        
        # List all skeleton files
        print("Fetching skeleton file list from S3...")
        skel_files = fs.ls(skel_path)
        print(f"Found {len(skel_files)} skeleton files")
        
        if len(skel_files) == 0:
            raise ValueError("No skeleton files found in S3 bucket")
        
        # Set random seed and pick files
        random.seed(seed)
        num_to_pick = min(num_skeletons, len(skel_files))
        picked_files = random.sample(skel_files, k=num_to_pick)
        
        print(f"Selected {num_to_pick} random skeleton files:")
        for i, file_path in enumerate(picked_files, 1):
            filename = file_path.split('/')[-1]
            print(f"  {i}. {filename}")
        
        # Load skeletons
        skeletons = []
        filenames = []
        
        for file_path in picked_files:
            filename = file_path.split('/')[-1]
            print(f"\nLoading skeleton: {filename}")
            
            try:
                # Load the skeleton
                sk = skio.read_skeleton(skel_path, filename)
                
                # Extract dendrites only (compartments 1=soma, 3=basal dendrite, 4=apical dendrite)
                dendrite_inds = (
                    (sk.vertex_properties['compartment'] == 3) | 
                    (sk.vertex_properties['compartment'] == 4) | 
                    (sk.vertex_properties['compartment'] == 1)  # soma included to connect dendrite graphs
                )
                sk_dendrite = sk.apply_mask(dendrite_inds)
                
                print(f"  Original vertices: {sk.vertices.shape[0]}")
                print(f"  Dendrite vertices: {sk_dendrite.vertices.shape[0]}")
                print(f"  Branch points: {len(sk_dendrite.branch_points)}")
                print(f"  End points: {len(sk_dendrite.end_points)}")
                
                skeletons.append(sk_dendrite)
                filenames.append(filename)
                
            except Exception as e:
                print(f"  ⚠️  Failed to load {filename}: {e}")
                continue
        
        if len(skeletons) == 0:
            raise ValueError("Failed to load any skeleton files")
        
        print(f"\n✅ Successfully loaded {len(skeletons)} skeletons")
        return skeletons, filenames
        
    except Exception as e:
        print(f"\n❌ Error loading skeleton data: {e}")
        print("\nFalling back to synthetic data...")
        return create_synthetic_skeletons(num_skeletons, seed)


def create_synthetic_skeletons(num_skeletons=4, seed=42):
    """
    Create synthetic skeleton-like data as fallback.
    
    Returns:
    --------
    skeletons : list
        List of synthetic skeleton-like objects with required attributes
    filenames : list
        List of synthetic filenames
    """
    print("Creating synthetic skeleton data...")
    
    random.seed(seed)
    np.random.seed(seed)
    
    skeletons = []
    filenames = []
    
    for i in range(num_skeletons):
        # Create a simple tree structure
        n_nodes = random.randint(50, 150)
        
        # Generate tree structure (ensuring it's connected)
        edges = []
        vertices = np.random.randn(n_nodes, 3) * 1000  # positions in nm
        
        # Create a tree by connecting each node to a previous one
        for j in range(1, n_nodes):
            parent = random.randint(0, j-1)
            edges.append([parent, j])
        
        edges = np.array(edges)
        
        # Find branch points (nodes with degree > 2)
        degrees = np.zeros(n_nodes)
        for edge in edges:
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1
        
        branch_points = np.where(degrees > 2)[0]
        end_points = np.where(degrees == 1)[0]
        root = 0  # First node is root
        
        # Create a simple object with required attributes
        class SyntheticSkeleton:
            def __init__(self):
                self.vertices = vertices
                self.edges = edges
                self.branch_points = branch_points
                self.end_points = end_points
                self.root = root
                self.vertex_properties = {'compartment': np.ones(n_nodes, dtype=int)}
                
            def apply_mask(self, mask):
                # For synthetic data, just return self
                return self
        
        skeleton = SyntheticSkeleton()
        filename = f"synthetic_neuron_{i+1}.swc"
        
        print(f"  Generated synthetic skeleton {i+1}: {n_nodes} vertices, {len(branch_points)} branch points")
        
        skeletons.append(skeleton)
        filenames.append(filename)
    
    print(f"✅ Created {len(skeletons)} synthetic skeletons")
    return skeletons, filenames


# Minimal dependencies
import numpy as np
import networkx as nx

def build_branch_graph(skel_or_meshwork, include_root=True, include_tips=False):
    """
    Reduce a meshparty skeleton to its branching topology.

    Parameters
    ----------
    skel_or_meshwork : meshparty.skeleton.Skeleton or meshparty.meshwork.Meshwork
        Either the Skeleton directly (nrn.skeleton) or a Meshwork (nrn).
    include_root : bool
        Include the soma/root as a topological node.
    include_tips : bool
        If True, include end points (tips) as topo nodes; otherwise only branch points (+ root).

    Returns
    -------
    G : networkx.Graph
        Nodes are skeleton-vertex indices of the selected topo points.
        Node attrs:
            - pos: (x,y,z) coordinates (same units as the skeleton, typically nm)
            - kind: 'branch' | 'root' | 'tip'
        Edge attrs:
            - length_nm: path length along the original skeleton between the two topo nodes
            - n_vertices: number of skeleton vertices along that segment/path
            - path: numpy array of the skeleton-vertex indices comprising the segment
    """
    # Accept either a Meshwork or a Skeleton
    sk = getattr(skel_or_meshwork, "skeleton", skel_or_meshwork)

    # --- Choose which topological points to keep
    topo = set(sk.branch_points)  # branch points
    if include_root and sk.root is not None:
        topo.add(int(sk.root))
    if include_tips:
        topo.update(sk.end_points)

    topo = np.array(sorted(topo), dtype=int)

    # --- Build the reduced graph by walking segments between topo nodes
    # Use segments_plus so each segment includes the parent/topo node on the rootward end.
    # (segments are defined from a branch-or-tip to the next rootward branch/root) 
    G = nx.Graph()

    # Add nodes with attributes
    for v in topo:
        kind = "branch"
        if include_root and v == sk.root:
            kind = "root"
        elif include_tips and v in set(sk.end_points.tolist()):
            kind = "tip"
        G.add_node(int(v), pos=tuple(sk.vertices[v]), kind=kind)

    # Traverse segments and keep only those whose endpoints are both selected topo nodes
    for seg in sk.segments_plus:
        u, v = int(seg[0]), int(seg[-1])  # distal topo node -> rootward topo node
        if u in G and v in G and u != v:
            # Length along the original skeleton path
            length = float(sk.path_length(seg))
            G.add_edge(u, v, length_nm=length, n_vertices=len(seg), path=seg.copy())

    # Optional: keep only the connected component containing the root (if present)
    if include_root and sk.root in G:
        cc = max(nx.connected_components(G), key=len)  # largest CC (usually the soma tree)
        if sk.root not in cc:
            # fall back to the component with the root explicitly
            for c in nx.connected_components(G):
                if sk.root in c:
                    cc = c
                    break
        G = G.subgraph(cc).copy()

    return G


# -----------------------------
# Utilities
# -----------------------------
def verbose_graph_print(tag, A: sp.csr_matrix, pos: np.ndarray):
    n = A.shape[0]
    nnz = int(A.nnz)
    degrees = np.asarray(A.sum(1)).ravel().astype(int)
    print(f"\n[{tag}] n={n}, pos={tuple(pos.shape)}, nnz={nnz} (undirected edges ~ {nnz//2})")
    print(f"  deg(min/med/max) = {degrees.min()}/{int(np.median(degrees))}/{degrees.max()}")
    if n <= 20:
        print("  degrees:", degrees.tolist())


def nx_graph_to_adj_pos(G: nx.Graph):
    """
    Convert a NetworkX graph with node attribute 'pos' -> (CSR adjacency, pos[N,3]),
    using a frozen node order so adj rows match pos rows.
    """
    # Node order as an array of ints (skeleton vertex ids are ints)
    node_order = np.array(list(G.nodes()), dtype=int)
    # Ensure connectivity and no self-loops
    assert nx.is_tree(G), "CherryReducer expects a single rooted tree; this G is not a tree."

    # Build adjacency in a fixed order
    A_arr = nx.to_scipy_sparse_array(G, nodelist=node_order, dtype=np.float64, format="csr")
    # Force to SciPy CSR MATRIX (not sparse array) and normalize
    A = sp.csr_matrix(A_arr)               # ensures .indices/.indptr exist
    A.data[:] = 1.0                        # unweighted
    A.eliminate_zeros()
    A.sort_indices()

    # Positions stacked in the SAME order
    P = np.stack([np.asarray(G.nodes[i]["pos"], dtype=np.float64) for i in node_order], axis=0)

    # Cast
    A = A.astype(np.float64, copy=False)
    P = P.astype(np.float32, copy=False)
    return A, P, node_order


def plot_graph_3d(A: sp.csr_matrix, pos: np.ndarray, leaf_mask=None, title: str = "", ax=None):
    """Simple 3D line plot of a graph from CSR adjacency and Nx3 positions. Highlights leaves if mask provided."""
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")

    # plot edges (use upper triangle to avoid duplicates)
    A_coo = A.tocoo()
    for u, v in zip(A_coo.row, A_coo.col):
        if u < v:
            xs, ys, zs = zip(pos[u], pos[v])
            ax.plot(xs, ys, zs, linewidth=0.6, alpha=0.7, color="gray")

    # plot nodes (highlight leaves if provided)
    if leaf_mask is not None and getattr(leaf_mask, "size", 0) == pos.shape[0]:
        leaf_mask = np.asarray(leaf_mask, dtype=bool)
        nl = ~leaf_mask
        ax.scatter(pos[nl, 0], pos[nl, 1], pos[nl, 2], s=8, depthshade=True, color="#4C78A8", alpha=0.9)
        ax.scatter(pos[leaf_mask, 0], pos[leaf_mask, 1], pos[leaf_mask, 2], s=20, depthshade=True, color="#F58518", alpha=0.95)
    else:
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=6, depthshade=True, color="#4C78A8")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    return ax


def plot_graph_2d(A: sp.csr_matrix, pos: np.ndarray, axes=(2, 1), leaf_mask=None, title: str = "", ax=None):
    """
    2D plot of graph (default projection z vs y). Highlights leaves if mask provided.
    axes is a tuple of column indices into pos: e.g., (0,1)->xy, (0,2)->xz, (1,2)->yz, (2,1)->zy.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ai, aj = axes
    # Edges
    A_coo = A.tocoo()
    for u, v in zip(A_coo.row, A_coo.col):
        if u < v:
            x = [pos[u, ai], pos[v, ai]]
            y = [pos[u, aj], pos[v, aj]]
            ax.plot(x, y, linewidth=0.6, alpha=0.7, color="gray")

    # Nodes (highlight leaves if provided)
    if leaf_mask is not None and getattr(leaf_mask, "size", 0) == pos.shape[0]:
        leaf_mask = np.asarray(leaf_mask, dtype=bool)
        nl = ~leaf_mask
        ax.scatter(pos[nl, ai], pos[nl, aj], s=8, color="#4C78A8", alpha=0.9, label="internal")
        ax.scatter(pos[leaf_mask, ai], pos[leaf_mask, aj], s=20, color="#F58518", alpha=0.95, label="leaves")
        # Show legend only if both are present
        if leaf_mask.any() and (~leaf_mask).any():
            ax.legend(loc="best", fontsize=8)
    else:
        ax.scatter(pos[:, ai], pos[:, aj], s=6, color="#4C78A8")

    # Labels
    labels = ["x", "y", "z"]
    ax.set_xlabel(labels[ai]); ax.set_ylabel(labels[aj])
    ax.set_title(title, fontsize=10)
    return ax


def collect_full_sequence(adj: sp.csr_matrix, pos: np.ndarray, red_factory: ReductionFactory, rng=None):
    """
    Build the entire coarsening sequence starting at (adj,pos), returning a list of dicts:
    [{adj, pos, n, leaf_idx, leaf_mask, level}, ...]
    Uses your CherryReducer directly to mirror the dataset logic.
    """
    rng = np.random.default_rng() if rng is None else rng
    seq = []
    cr = red_factory(adj)

    # Level 0
    leaf0_idx = np.array(sorted(cr._state.leaves - {cr._state.root}), dtype=np.int64)
    leaf_mask = np.zeros(cr.n, dtype=bool)
    if len(leaf0_idx) > 0:
        leaf_mask[leaf0_idx] = True
    seq.append(dict(
        level=cr.level,
        adj=cr.adj.copy(),
        pos=pos.copy(),
        n=cr.n,
        leaf_idx=leaf0_idx.copy(),
        leaf_mask=leaf_mask.copy(),
    ))

    # Keep reducing until no contraction
    pos_curr = pos.copy()
    while True:
        nxt = cr.get_reduced_graph()
        if not nxt.did_contract:
            break
        pos_curr = pos_curr[nxt.survivor_mask]
        seq.append(dict(
            level=nxt.level,
            adj=nxt.adj.copy(),
            pos=pos_curr.copy(),
            n=nxt.n,
            leaf_idx=nxt.leaf_idx.copy(),
            leaf_mask=nxt.leaf_mask.copy(),
        ))
        cr = nxt

    return seq


def plot_sequence(seq, max_panels=6, suptitle="Reduction sequence", save_path=None, plot_2d=False, projection="zy"):
    # Map projection string to axes indices
    proj_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2), "zy": (2, 1)}
    axes = proj_map.get(projection, (2, 1))

    k = min(len(seq), max_panels)
    cols = min(3, k)
    rows = int(np.ceil(k / cols))
    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    for i in range(k):
        step = seq[i]
        title = f"Level {step['level']} | n={step['n']} | leaves={int(step['leaf_mask'].sum())}"
        if plot_2d:
            ax = fig.add_subplot(rows, cols, i + 1)
            plot_graph_2d(step["adj"], step["pos"], axes=axes, leaf_mask=step["leaf_mask"], title=title, ax=ax)
        else:
            ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
            plot_graph_3d(step["adj"], step["pos"], leaf_mask=step["leaf_mask"], title=title, ax=ax)
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()

    # Save plot if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")

    return fig


# -----------------------------
# Main
# -----------------------------
def main(args):
    # 0) Repro
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    # 1) Load skeleton data from MICrONS dataset
    print("=" * 60)
    print("LOADING SKELETON DATA")
    print("=" * 60)
    
    try:
        skeletons, filenames = load_skeleton_data(num_skeletons=4, seed=args.seed)
    except Exception as e:
        print(f"Failed to load real skeleton data: {e}")
        print("Exiting...")
        sys.exit(1)

    # 2) Build NX branch graphs, then (adj, pos)
    print("\n" + "=" * 60)
    print("BUILDING BRANCH GRAPHS")
    print("=" * 60)
    
    adjs, poses, graphs = [], [], []
    for i, (sk, fname) in enumerate(zip(skeletons, filenames)):
        print(f"\nProcessing skeleton {i+1}/{len(skeletons)}: {fname}")

        # Build branch graph with tips (so pruning happens)
        G = build_branch_graph(sk, include_root=True, include_tips=True)
        print(f"  Branch graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        if G.number_of_nodes() < 3:
            print(f"  ⚠️  Skipping - too few nodes for meaningful reduction")
            continue

        # Convert to (adj, pos) with frozen order
        A, P, node_order = nx_graph_to_adj_pos(G)
        # Basic checks
        assert A.shape[0] == P.shape[0], "adjacency and pos must agree in node count"
        assert P.shape[1] == 3, "pos must be N x 3"
        verbose_graph_print(f"Graph from {fname}", A, P)

        adjs.append(A)
        poses.append(P)
        graphs.append(G)

    if len(adjs) == 0:
        print("❌ No valid graphs were created. Exiting...")
        sys.exit(1)

    print(f"\n✅ Successfully processed {len(adjs)} skeleton graphs")

    # 3) Make a ReductionFactory (same defaults as your training, adjust as needed)
    print("\n" + "=" * 60)
    print("SETTING UP DATALOADER")
    print("=" * 60)
    
    red_factory = ReductionFactory(mode="stochastic", cherry_p=0.8, ensure_progress=True, root="argmax_degree")

    # 4) Build the dataset (Infinite stream) and dataloader
    train_dataset = InfiniteRandRedDataset(adjs=adjs, poses=poses, red_factory=red_factory)

    num_workers = args.workers if args.workers >= 0 else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False,
        collate_fn=Batch.from_data_list,
        num_workers=num_workers,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )

    # 5) Pull a few batches and print VERY verbose information
    print("\n" + "=" * 60)
    print("TESTING DATALOADER")
    print("=" * 60)
    
    batches_seen = 0
    for batch in train_loader:
        batches_seen += 1
        # batch is a PyG Batch: attributes merged; adj is block-diagonal SparseTensor
        print(f"\n--- Batch {batches_seen} ---")
        print(f" nodes total: {batch.num_nodes}")
        print(f" x shape:     {tuple(batch.x.shape)}  (should be [sumN, 3])")
        if hasattr(batch, "pos"):
            print(f" pos shape:   {tuple(batch.pos.shape)} (if present)")
        # SparseTensor -> show nnz
        if hasattr(batch, "adj"):
            try:
                nnz = int(batch.adj.nnz())
            except Exception:
                nnz = "n/a"
            print(f" adj nnz:     {nnz} (block-diagonal)")
        # Leaves
        if hasattr(batch, "leaf_idx"):
            print(f" leaf_idx:    total {batch.leaf_idx.numel()} (batched indices)")
        if hasattr(batch, "leaf_mask"):
            print(f" leaf_mask:   {tuple(batch.leaf_mask.shape)} (bool)")
        if hasattr(batch, "leaf_expansion"):
            # Check values in {1,2}
            uniq = th.unique(batch.leaf_expansion).tolist()
            print(f" leaf_expansion values: {uniq}")

        # A couple of assertions for safety
        assert batch.x.shape[1] == 3, "x must be 3D positions"
        if hasattr(batch, "pos"):
            assert th.allclose(batch.pos, batch.x), "pos should equal x under current design"

        # Stop if we've hit requested number of batches
        if batches_seen >= args.num_batches:
            break

    # 6) (Optional) Visualize full reduction sequences for each skeleton
    if args.plot_sequences:
        print("\n" + "=" * 60)
        print("PLOTTING COARSENING SEQUENCES")
        print("=" * 60)
        
        rng = np.random.default_rng(args.seed)
        
        # Create output directory for plots if saving
        if args.save_plots:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving plots to: {output_dir}")
        
        for i, (A, P, fname) in enumerate(zip(adjs, poses, filenames)):
            print(f"\nProcessing neuron {i+1}/{len(adjs)}: {fname}")
            seq = collect_full_sequence(A, P, red_factory, rng=rng)
            
            # Console summary
            print(f"  Sequence length: {len(seq)} levels")
            for step in seq[:5]:  # Show first 5 levels
                print(f"    Level {step['level']:2d}: {step['n']:4d} nodes, {int(step['leaf_mask'].sum()):4d} leaves")
            if len(seq) > 5:
                print(f"    ... ({len(seq)-5} more levels)")
            
            # Generate plot
            suptitle = f"Neuron {i+1}: {fname} (first {args.max_panels} levels)"
            save_path = None
            if args.save_plots:
                # Clean filename for saving
                clean_name = fname.replace('.swc', '').replace('synthetic_', '')
                save_path = output_dir / f"neuron_{i+1}_{clean_name}_coarsening.png"
            
            fig = plot_sequence(
                seq,
                max_panels=args.max_panels,
                suptitle=suptitle,
                save_path=save_path,
                plot_2d=args.plot_2d,
                projection=args.projection,
            )
        
        if not args.save_plots:
            plt.show()
        else:
            plt.close('all')  # Close figures to save memory

    print("\n" + "=" * 60)
    print("✅ COMPLETE - Geometric dataloader test successful!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2, help="set >0 to test multiprocessing")
    parser.add_argument("--plot-sequences", action="store_true")
    parser.add_argument("--max-panels", type=int, default=10, help="max levels to plot per neuron")
    parser.add_argument("--save-plots", action="store_true", help="save plots to files instead of displaying")
    parser.add_argument("--output-dir", type=str, default="./plots", help="directory to save plots")
    parser.add_argument("--plot-2d", action="store_true", help="use 2D plots instead of 3D")
    parser.add_argument(
        "--projection",
        type=str,
        default="zy",
        choices=["xy", "xz", "yz", "zy"],
        help="2D projection axes when using --plot-2d",
    )
    args = parser.parse_args()
    main(args)
