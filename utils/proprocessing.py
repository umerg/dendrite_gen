import numpy as np
import scipy.sparse as sp
import networkx as nx

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