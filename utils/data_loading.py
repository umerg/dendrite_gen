import numpy as np
import scipy.sparse as sp
import networkx as nx
from pathlib import Path


def load_swc_graph(path):
    """
    Load a single cleaned SWC file into an undirected NetworkX tree graph.

    Expected SWC columns (whitespace separated):
        id  type  x  y  z  radius  parent_id

    Root selection (conditional):
        * If original SWC root (id 1 has parent_id==0) has >=2 children, keep id 1 as root.
        * Else if id 1 has exactly 1 child AND id 2 exists with >=2 children, re-root at id 2 and drop id 1 (redundant upstream leaf).
        * Else if id 2 exists with >=2 children (even if id 1 absent), use id 2.
        * Otherwise fall back to the first node encountered with parent_id == 0.

    Post-processing adjustments:
        * Positions are recentered so the chosen root node is at the origin (0,0,0).
        * Root id stored as G.graph['root'].

    Node attributes:
        pos: np.ndarray shape (3,) float64 (x,y,z) (after recentering)

    Returns:
        G (nx.Graph) with integer node ids matching SWC ids and 'pos' attribute.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SWC file not found: {path}")

    G = nx.Graph()
    parent_links = []  # (child, parent)
    fallback_root = None  # first parent_id==0 encountered (usually id 1)
    root_id = None

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                raise ValueError(f"Malformed SWC line (expected >=7 cols): '{line}'")
            nid = int(parts[0])
            # type_col = int(parts[1])  # currently unused
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
            # we ignore radius now per spec (parts[5])
            parent = int(parts[6])

            if parent == 0 and fallback_root is None:
                # Record original SWC root candidate for fallback logic
                fallback_root = nid

            G.add_node(nid, pos=np.array([x, y, z], dtype=np.float64))
            if parent > 0:
                parent_links.append((nid, parent))

    # Build edges
    for child, parent in parent_links:
        if parent not in G:
            raise ValueError(f"Parent id {parent} referenced before definition in {path}")
        G.add_edge(parent, child)

    # Derive children map from parent_links (original directed structure)
    children_map = {}
    for child, parent in parent_links:
        children_map.setdefault(parent, []).append(child)

    # Determine root according to conditional rules
    orig_root_children = len(children_map.get(1, [])) if 1 in G else 0
    id2_children = len(children_map.get(2, [])) if 2 in G else 0

    if 1 in G and orig_root_children >= 2:
        root_id = 1
    elif 1 in G and orig_root_children == 1 and 2 in G and id2_children >= 2:
        root_id = 2
        # Drop id 1 if it's a simple upstream leaf only connected to 2
        if G.degree[1] == 1 and 2 in G.neighbors(1):
            G.remove_node(1)
    elif 2 in G and id2_children >= 2:
        root_id = 2
    elif fallback_root is not None:
        root_id = fallback_root
    else:
        raise ValueError(f"No suitable root found in SWC file: {path}")


    if not nx.is_tree(G):
        raise AssertionError(f"Loaded graph from {path} is not a tree.")

    # Recenter positions so root is at origin
    root_pos = G.nodes[root_id]["pos"].copy()
    for nid in G.nodes:
        G.nodes[nid]["pos"] = G.nodes[nid]["pos"] - root_pos

    # Store root id on graph for downstream ordering logic
    G.graph["root"] = root_id
    return G


def load_swc_graphs_from_dir(dir_path):
    """Load all cleaned SWC files in a directory into a list of graphs.

    Inclusion criteria:
        * Regular files whose names end with '.csv.swc'
        * Not starting with '._' (macOS metadata files)

    Files failing these criteria are ignored.
    Returned list is sorted by filename.
    """
    dir_path = Path(dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        raise NotADirectoryError(f"Provided path is not a directory: {dir_path}")
    graphs = []
    for swc_file in sorted(dir_path.iterdir()):
        if not swc_file.is_file():
            continue
        name = swc_file.name
        if name.startswith("._"):
            continue
        if not name.endswith(".csv.swc"):
            continue
        graphs.append(load_swc_graph(swc_file))
    return graphs

def nx_graph_to_adj_pos(G: nx.Graph):
    """
    Convert a NetworkX graph with node attribute 'pos' -> (CSR adjacency, pos[N,3]).

    Ordering guarantees:
        * Root node (G.graph['root'] if present) appears first.
        * Remaining nodes follow their insertion order excluding the root.
    """
    assert nx.is_tree(G), "Expected a single tree graph."

    root_id = G.graph.get("root")
    if root_id is not None:
        # Preserve original insertion order for non-root nodes
        ordered_nodes = [root_id] + [n for n in G.nodes() if n != root_id]
    else:
        ordered_nodes = list(G.nodes())
    node_order = np.array(ordered_nodes, dtype=int)

    A_arr = nx.to_scipy_sparse_array(G, nodelist=node_order, dtype=np.float64, format="csr")
    A = sp.csr_matrix(A_arr)
    A.data[:] = 1.0
    A.eliminate_zeros()
    A.sort_indices()

    P = np.stack([np.asarray(G.nodes[i]["pos"], dtype=np.float64) for i in node_order], axis=0)

    A = A.astype(np.float64, copy=False)
    P = P.astype(np.float32, copy=False)
    return A, P, node_order