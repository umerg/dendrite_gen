# tests/test_cherry_dataset.py
import os
import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
import torch

# --- adjust these imports to your package layout ---
from yourpkg.reduction import ReductionFactory, CherryReducer
from yourpkg.datasets import RandRedDataset
from yourpkg.data import ReducedGraphData
# ---------------------------------------------------

# ---------------- graph builders ----------------
def make_balanced_binary(height: int = 3):
    G = nx.balanced_tree(r=2, h=height, create_using=nx.Graph)
    return nx.to_scipy_sparse_array(G, dtype=np.float64, format="csr")

def make_caterpillar(length: int = 8):
    # spine 0..L
    edges = [(i, i + 1) for i in range(length)]
    # attach one leaf to each internal spine vertex
    leaf_id = length + 1
    for i in range(1, length):
        edges.append((i, leaf_id))
        leaf_id += 1
    n = leaf_id
    rows, cols = zip(*edges)
    data = np.ones(len(edges), dtype=np.float64)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    A = A + A.T
    return A.tocsr()

def make_random_tree(n: int = 32, seed: int = 0):
    G = nx.random_tree(n=n, seed=seed)
    return nx.to_scipy_sparse_array(G, dtype=np.float64, format="csr")

# --------------- helpers -----------------------
def scipy_from_sparse_tensor(st):
    """Convert PyG SparseTensor -> SciPy CSR for plotting."""
    # If it's already SciPy, pass through
    if sp.issparse(st):
        return st.tocsr()
    import torch
    from torch_geometric.typing import SparseTensor as PYG_ST
    if isinstance(st, PYG_ST):
        row, col, val = st.coo()
        row = row.cpu().numpy()
        col = col.cpu().numpy()
        val = val.cpu().numpy()
        A = sp.coo_matrix((val, (row, col)), shape=st.sizes())
        return A.tocsr()
    raise TypeError(f"Unsupported sparse type: {type(st)}")

def draw_graph(A, title, out_path, layout_seed=0):
    """Draw SciPy CSR adjacency as a simple NetworkX plot."""
    A = A.tocsr()
    G = nx.from_scipy_sparse_array(A)
    if G.number_of_nodes() == 0:
        return
    pos = nx.spring_layout(G, seed=layout_seed)  # deterministic-ish
    plt.figure(figsize=(5, 4))
    nx.draw_networkx(G, pos, node_size=400, font_size=9, with_labels=True)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def mapping_from_P(P_inv):
    """
    Given fine->coarse binary membership (n x m), return a dict {fine: coarse}.
    """
    if sp.issparse(P_inv):
        P_inv = P_inv.tocsr()
        n = P_inv.shape[0]
        out = {}
        indptr = P_inv.indptr
        indices = P_inv.indices
        for i in range(n):
            s, e = indptr[i], indptr[i + 1]
            if e > s:
                out[i] = int(indices[s])  # exactly one 1 per row
        return out
    else:
        # PyG SparseTensor path
        row, col, _ = P_inv.coo()
        row = row.cpu().numpy()
        col = col.cpu().numpy()
        out = {}
        for r, c in zip(row, col):
            out[int(r)] = int(c)
        return out

# --------------- main test ---------------------
def main():
    ap = argparse.ArgumentParser("CherryDataset sanity test")
    ap.add_argument("--graph", type=str, default="balanced",
                    choices=["balanced", "caterpillar", "random_tree"])
    ap.add_argument("--height", type=int, default=3, help="balanced tree height")
    ap.add_argument("--length", type=int, default=8, help="caterpillar spine length")
    ap.add_argument("--n", type=int, default=32, help="random_tree n")
    ap.add_argument("--mode", type=str, default="stochastic",
                    choices=["stochastic", "deterministic"])
    ap.add_argument("--p", type=float, default=0.8, help="Bernoulli p for cherries (stochastic)")
    ap.add_argument("--outdir", type=str, default="out_cherry_test", help="output directory")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Build the test graph
    if args.graph == "balanced":
        A0 = make_balanced_binary(args.height)
        name = f"balanced_h{args.height}"
    elif args.graph == "caterpillar":
        A0 = make_caterpillar(args.length)
        name = f"caterpillar_L{args.length}"
    else:
        A0 = make_random_tree(args.n, seed=0)
        name = f"random_n{args.n}"

    # Factory & reducer
    red_factory = ReductionFactory(
        mode=args.mode,
        cherry_p=args.p,
        ensure_progress=True,
        root="argmax_degree",
        weighted_reduction=False,
    )

    # We’ll use the dataset’s sequence generator to mirror training
    ds = RandRedDataset([A0], red_factory)
    rng = np.random.default_rng(42)
    reducer = red_factory(A0.copy())
    seq = ds.get_random_reduction_sequence(reducer, rng)

    if not seq:
        print("No reduction steps produced (graph is already terminal).")
        return

    # Visualise and print metadata
    print(f"\n=== Cherry reduction sequence for {name} ({args.mode}, p={args.p}) ===")
    # Also draw original graph for reference
    draw_graph(A0, f"Level 0 (n={A0.shape[0]}) — original", os.path.join(args.outdir, f"{name}_level0.png"))

    prev_adj = A0
    for k, rgd in enumerate(seq, start=1):
        # rgd.adj is fine (level k-1), rgd.adj_reduced is coarse (level k)
        A_fine = scipy_from_sparse_tensor(rgd.adj)
        A_coarse = scipy_from_sparse_tensor(rgd.adj_reduced)
        P_inv = rgd.expansion_matrix  # fine->coarse (n x m)
        n_f, n_c = A_fine.shape[0], A_coarse.shape[0]

        # Visuals
        draw_graph(A_fine, f"Level {k-1} (n={n_f})", os.path.join(args.outdir, f"{name}_L{k-1}_fine.png"))
        draw_graph(A_coarse, f"Level {k} (n={n_c})", os.path.join(args.outdir, f"{name}_L{k}_coarse.png"))

        # Metadata
        mapping = mapping_from_P(P_inv)
        node_exp = rgd.node_expansion.cpu().numpy() if isinstance(rgd.node_expansion, torch.Tensor) else rgd.node_expansion
        nnz = P_inv.nnz if sp.issparse(P_inv) else P_inv.nnz()

        print(f"\n-- Step {k} --")
        print(f"fine n={n_f} → coarse n={n_c}")
        print(f"expansion_matrix shape={P_inv.shape}, nnz={nnz}")
        print("node_expansion (per coarse node):", node_exp.tolist())
        # show first 20 mappings to keep log concise
        items = sorted(mapping.items())
        preview = items[:min(20, len(items))]
        print("fine→coarse mapping preview (i -> j):", preview)
        if len(items) > len(preview):
            print(f"... ({len(items)-len(preview)} more)")

        # Optional: check consistency A_coarse ≈ Pᵀ A_fine P
        Af_check = (P_inv.T @ A_fine @ P_inv).tocoo()
        Af_check.setdiag(0)
        Af_check.eliminate_zeros()
        equal = (sp.csr_matrix(Af_check.shape) + Af_check).astype(bool).astype(int).nnz == A_coarse.astype(bool).astype(int).nnz \
                and (Af_check != A_coarse).nnz == 0
        print(f"coarsening consistency Pᵀ A P == adj_reduced? {'OK' if equal else 'MISMATCH'}")

        prev_adj = A_coarse

    print(f"\nSaved figures to: {Path(args.outdir).resolve()}")
    print("Done.")

if __name__ == "__main__":
    main()
