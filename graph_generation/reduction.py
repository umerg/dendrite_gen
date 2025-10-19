# reduction.py
from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Callable
from collections import defaultdict, deque
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp

# Use scipy.sparse.csr_matrix for widest compatibility
csr = sp.csr_matrix
coo = sp.coo_matrix

Node = int

@dataclass
class _CherryState:
    parent: Dict[Node, Optional[Node]]
    children: Dict[Node, List[Node]]
    leaves: Set[Node]
    num_children: Dict[Node, int]
    leaf_child_count: Dict[Node, int]
    current_cherries: Set[Node]
    root: Node

class CherryReducer:
    """
    Event-driven cherry contraction for rooted trees (root may be k-ary).
    - 'Cherry' = parent whose ALL children are leaves (>=1 child).
    - Per level, prune:
        * deterministic: all cherries
        * stochastic: each cherry with prob p (ensuring at least one)
    - Returns a NEW CherryReducer with updated state, new adj, and
      expansion_matrix (fine->coarse membership) for THIS step.

    Attributes exposed for the dataset:
      - adj:             scipy.sparse CSR (fine adjacency at this step)
      - n:               int, number of fine nodes
      - level:           current reduction level (int)
      - expansion_matrix: fine->coarse membership (n x m), binary COO
      - node_expansion:  sizes of clusters that formed THIS graph
                         (col sums of previous expansion; = ones at level 0)
    """

    def __init__(
        self,
        adj,                           # scipy sparse (CSR/COO), unweighted is fine
        root: int = 0,
        mode: str = "stochastic",      # "deterministic" | "stochastic"
        cherry_p: float = 0.8,
        ensure_progress: bool = True,
        state: Optional[_CherryState] = None,
        level: int = 0,
        weighted_reduction: bool = False,  # if True, coarsen via Laplacian
    ):
        # Normalize adjacency to CSR
        self.adj: csr = adj.tocsr() if not isinstance(adj, csr) else adj.copy()
        self.n: int = self.adj.shape[0]
        self.level = level
        self.weighted_reduction = weighted_reduction

        # Keep Laplacian only if weighted_reduction requested
        self._lap = None
        if self.weighted_reduction:
            deg = np.asarray(self.adj.sum(1)).ravel()
            self._lap = sp.diags(deg) - self.adj

        # Policy
        self.mode = mode
        self.cherry_p = float(cherry_p)
        self.ensure_progress = ensure_progress

        # Root/state
        self.root = root
        self._state = state if state is not None else self._build_initial_state()

        # Outputs for dataset consumption
        self.survivor_mask = None      # np.ndarray[int64], shape (m,), indices of surviving nodes
        self.leaf_idx = None          # np.ndarray[int64], shape (L,)
        self.leaf_mask = None         # np.ndarray[bool], shape (N,)
        self.leaf_expansion = None    # np.ndarray[int32], shape (L,), values {1,2}
    # parent_idx_1b will be derived externally per full node list; leaf-specific parents no longer stored
        self.did_contract = False

    # ------------------------------------------------------------------
    # Public API used by your datasets
    # ------------------------------------------------------------------
    def get_reduced_graph(self, rng=np.random.default_rng()) -> "CherryReducer":
        """
        Execute one cherry-prune level and return the next CherryReducer:
          - choose cherries (all or Bernoulli(p))
          - build P (coarse x fine) and P_inv = (fine x coarse) binary
          - compute adj_reduced = P_inv^T @ adj @ P_inv (drop self-loops)
          - update state incrementally and remap indices
        """
        S = self._state
        cherries_all = list(S.current_cherries)

        if not cherries_all or self.n <= 1:
            # No further contraction; return a "no-op" next level
            cr = CherryReducer(
                adj=self.adj,
                root=S.root,
                mode=self.mode,
                cherry_p=self.cherry_p,
                ensure_progress=self.ensure_progress,
                state=S,
                level=self.level + 1,
                weighted_reduction=self.weighted_reduction,
            )
            cr.did_contract = False
            # expose current leaves for completeness (labels default to 1)
            leaf_idx = np.array(sorted(S.leaves - {S.root}), dtype=np.int64)
            
            leaf_mask = np.zeros(self.n, dtype=bool)
            if len(leaf_idx) > 0:
                leaf_mask[leaf_idx] = True
            
            cr.survivor_mask = np.arange(self.n, dtype=np.int64)
            cr.leaf_idx = leaf_idx
            cr.leaf_mask = leaf_mask
            cr.leaf_expansion = np.ones_like(leaf_idx, dtype=np.int32)
            # parent indices for current leaves (use -1 for root / None)
            # parent_idx_1b is constructed externally; omit leaf_parent_idx
            return cr

        # Choose cherries for this level
        if self.mode == "deterministic":
            chosen_parents = cherries_all
        else:
            chosen_parents = [u for u in cherries_all if rng.random() < self.cherry_p]
            if self.ensure_progress and not chosen_parents:
                chosen_parents = [rng.choice(cherries_all)]

        chosen_set = set(chosen_parents)

        # Leaves removed this level = union of children of chosen parents
        removed_leaves: Set[Node] = set()
        for u in chosen_parents:
            removed_leaves.update(S.children[u])

        # Survivors = all nodes except removed leaves (parents survive)
        keep = np.ones(self.n, dtype=bool)
        if removed_leaves:
            keep[np.fromiter(removed_leaves, dtype=int, count=len(removed_leaves))] = False
        survivors = np.nonzero(keep)[0]
        m = survivors.size  # coarse size

        # Map old -> new indices
        new_index = -np.ones(self.n, dtype=int)
        new_index[survivors] = np.arange(m, dtype=int)

        # Build P (m x n) and P_inv (n x m)
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        # a) For chosen parents: one coarse row aggregates parent + its (removed) children
        for u in chosen_parents:
            r = new_index[u]  # parent survives
            rows.append(r); cols.append(u); data.append(1.0)
            for v in S.children[u]:
                rows.append(r); cols.append(v); data.append(1.0)

        # b) For other survivors that aren't chosen parents: identity
        for s in survivors:
            if s in chosen_set:
                continue
            rows.append(new_index[s]); cols.append(s); data.append(1.0)

        P = coo((data, (rows, cols)), shape=(m, self.n), dtype=np.float64)   # coarse x fine
        P_inv = P.transpose().tocsr()                                       # fine x coarse
        P_inv.data[:] = 1.0  # binary membership

        # Compute reduced adjacency (drop self-loops, keep unweighted)
        if self.weighted_reduction:
            Lr = P_inv.T @ self._lap @ P_inv
            Ar = -Lr + sp.diags(Lr.diagonal())
        else:
            M = (P_inv.T @ self.adj @ P_inv).tocoo()
            r, c = M.row, M.col
            msk = r != c
            Ar = coo((np.ones(msk.sum(), dtype=np.float64), (r[msk], c[msk])), shape=(m, m)).tocsr()

        # -------- Event-driven state update (local only), then remap --------
        parent = S.parent.copy()
        children = {u: ch[:] for u, ch in S.children.items()}
        leaves = set(S.leaves)
        num_children = S.num_children.copy()
        leaf_child_count = S.leaf_child_count.copy()
        current_cherries = set(S.current_cherries)

        current_cherries.difference_update(chosen_set)  # chosen parents won't stay cherries
        maybe_new_cherries: Set[Node] = set()

        for u in chosen_parents:
            ch = children[u]
            # remove child leaves
            for v in ch:
                leaves.discard(v)
                parent.pop(v, None)
            # parent becomes a leaf
            children[u] = []
            leaf_child_count[u] = 0
            num_children[u] = 0
            if u != S.root:
                leaves.add(u)
                gp = parent.get(u, None)
                if gp is not None:
                    leaf_child_count[gp] += 1
                    if num_children[gp] > 0 and leaf_child_count[gp] == num_children[gp]:
                        maybe_new_cherries.add(gp)

        current_cherries.update(maybe_new_cherries)

        # Remap state to new indices
        def remap(x: Optional[Node]) -> Optional[Node]:
            if x is None: return None
            ix = new_index[x]
            return None if ix < 0 else int(ix)

        new_parent: Dict[Node, Optional[Node]] = {}
        new_children: Dict[Node, List[Node]] = defaultdict(list)

        for u in survivors:
            ru = remap(u)
            new_parent[ru] = remap(parent.get(u, None) if u in parent else None)
            new_children[ru] = [remap(v) for v in children.get(u, []) if new_index[v] >= 0]

        new_leaves = {remap(v) for v in leaves if new_index[v] >= 0}
        new_num_children = {u: len(ch) for u, ch in new_children.items()}
        new_leaf_child_count = {u: sum((v in new_leaves) for v in ch) for u, ch in new_children.items()}
        new_current_cherries = {
            u for u, cnt in new_leaf_child_count.items()
            if new_num_children[u] > 0 and cnt == new_num_children[u]
        }

        new_state = _CherryState(
            parent=new_parent,
            children=new_children,
            leaves=new_leaves - {remap(S.root)},
            num_children=new_num_children,
            leaf_child_count=new_leaf_child_count,
            current_cherries=new_current_cherries,
            root=remap(S.root),
        )

        next_cr = CherryReducer(
            adj=Ar,
            root=new_state.root if new_state.root is not None else 0,
            mode=self.mode,
            cherry_p=self.cherry_p,
            ensure_progress=self.ensure_progress,
            state=new_state,
            level=self.level + 1,
            weighted_reduction=self.weighted_reduction,
        )
        next_cr.did_contract = True

        # Leaf labels for NEXT graph (Ar / next_cr)
        node_sizes = np.asarray(P_inv.sum(axis=0)).ravel().astype(np.int32)  # len m
        leaf_idx = np.array(sorted(next_cr._state.leaves - {next_cr._state.root}), dtype=np.int64)
        leaf_y = np.where(node_sizes[leaf_idx] > 1, 2, 1).astype(np.int32)   # binary {1,2}

        # Create survivor mask to extract positions/features only of nodes of coarse graph
        next_cr.survivor_mask = survivors.astype(np.int64)

        # Create leaf mask
        leaf_mask = np.zeros(m, dtype=bool)
        if len(leaf_idx) > 0:
            leaf_mask[leaf_idx] = True

        next_cr.leaf_idx = leaf_idx
        next_cr.leaf_mask = leaf_mask
        next_cr.leaf_expansion = leaf_y
        # parent indices for leaves in coarse graph (already remapped)
        # parent_idx_1b constructed externally when building ReducedGraphData

        return next_cr

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_initial_state(self) -> _CherryState:
        # BFS root the tree
        parent: Dict[Node, Optional[Node]] = {self.root: None}
        children: Dict[Node, List[Node]] = defaultdict(list)

        seen = {self.root}
        q = deque([self.root])
        # Use CSR row access
        while q:
            u = q.popleft()
            row = self.adj[u]
            nbrs = row.indices
            for v in nbrs:
                if v in seen:
                    continue
                seen.add(v)
                parent[v] = u
                children[u].append(v)
                q.append(v)

        if len(seen) != self.n:
            missing = set(range(self.n)) - seen
            raise ValueError(f"Graph is not a single rooted tree. Unreached: {sorted(list(missing))[:10]} ...")

        leaves: Set[Node] = {u for u in range(self.n) if len(children[u]) == 0}
        leaves.discard(self.root)

        num_children = {u: len(children[u]) for u in range(self.n)}
        leaf_child_count = {u: sum((v in leaves) for v in children[u]) for u in range(self.n)}
        current_cherries = {
            u for u in range(self.n) if num_children[u] > 0 and leaf_child_count[u] == num_children[u]
        }

        return _CherryState(
            parent=parent,
            children=children,
            leaves=leaves,
            num_children=num_children,
            leaf_child_count=leaf_child_count,
            current_cherries=current_cherries,
            root=self.root,
        )

RootSpec = Union[int, str, Callable[[sp.spmatrix], int]]

class ReductionFactory:
    """
    Minimal factory that instantiates a CherryReducer for a given adjacency.
    - root: int index, or "auto" / "argmax_degree", or a callable adj -> root_index
    - mode: "stochastic" | "deterministic"
    """
    def __init__(
        self,
        *,
        mode: str = "stochastic",
        cherry_p: float = 0.8,
        ensure_progress: bool = True,
        root: RootSpec = "argmax_degree",
        weighted_reduction: bool = False,
    ):
        self.mode = mode
        self.cherry_p = float(cherry_p)
        self.ensure_progress = ensure_progress
        self.root = root
        self.weighted_reduction = weighted_reduction

    def _resolve_root(self, adj: sp.spmatrix) -> int:
        if isinstance(self.root, int):
            return self.root
        if callable(self.root):
            return int(self.root(adj))
        # default heuristics for trees
        if isinstance(adj, sp.csr_matrix):
            deg = np.asarray(adj.sum(1)).ravel()
        else:
            deg = np.asarray(adj.tocsr().sum(1)).ravel()
        if self.root in ("argmax_degree", "auto"):
            return int(np.argmax(deg))  # often a good proxy for soma/root
        raise ValueError(f"Unrecognized root spec: {self.root}")

    def __call__(self, adj: sp.spmatrix) -> CherryReducer:
        root = self._resolve_root(adj)
        return CherryReducer(
            adj=adj,
            root=root,
            mode=self.mode,
            cherry_p=self.cherry_p,
            ensure_progress=self.ensure_progress,
            weighted_reduction=self.weighted_reduction,
        )