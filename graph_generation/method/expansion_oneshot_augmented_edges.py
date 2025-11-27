import networkx as nx
import torch as th
from torch.nn import Module
import torch.nn.functional as F
from torch_scatter import scatter
from torch_sparse import SparseTensor
import logging

logger = logging.getLogger(__name__)

from .method import Method

class Expansion_OneShot_Augmented(Method):
    """Graph generation method generating graphs by local expansion."""

    EDGE_PARENT_TO_CHILD = 0
    EDGE_CHILD_TO_PARENT = 1
    EDGE_SIBLING = 2

    def __init__(
        self,
        deterministic_expansion: bool = False,      # just sets seeding for reproducibility
        red_threshold: int = 0,
        leaf_noise_sigma: float = 0.05,             # <-- stddev of Gaussian around parent (same units as pos)
        leaf_noise_clip: float | None = None,       # <-- optional radius clamp (float) or None
        sibling_loss_weight: float = 0.8,           # weight for sibling distance regularizer
        use_sibling_matching: bool = False,         # if True, use per-parent matching for positional loss
        debug: bool = False,
        debug_max_batches: int = 2,
        debug_dir: str | None = None,
    ):
        super().__init__(diffusion=None)
        self.deterministic_expansion = deterministic_expansion
        self.red_threshold = red_threshold
        self.leaf_noise_sigma = float(leaf_noise_sigma)
        self.leaf_noise_clip = leaf_noise_clip
        self.sibling_loss_weight = float(sibling_loss_weight)
        self.use_sibling_matching = bool(use_sibling_matching)
        self.debug = debug
        self._debug_step = 0
        from pathlib import Path as _P
        self.debug_dir = _P(debug_dir) if debug_dir is not None else _P.cwd() / "debug_graphs"
        self.debug_max_batches = debug_max_batches
        if self.debug:
            logger.info(f"Expansion_OneShot debug enabled; plots will be saved under: {self.debug_dir}")
    
    # ---------------------------------------------------------
    # Shared noise sampler to guarantee identical stochastic policy
    # for training-time masking and generation-time branching.
    # ---------------------------------------------------------
    def _sample_noise(self, shape: tuple, device: th.device, *, sigma: float, clip: float | None):
        """Return Gaussian noise with optional norm clipping to a ball of radius `clip`.

        Clipping logic matches training masking so distributions align.
        shape: (N,3)
        """
        noise = th.randn(shape, device=device) * sigma
        if clip is not None and clip > 0:
            norms = noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            scale = th.minimum(th.ones_like(norms), clip / norms)
            noise = noise * scale
        return noise

    def _build_augmented_edge_index(
        self, parent_idx: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        """
        Build directed edges with categorical labels for parent/child/sibling relations.

        Returns:
            edge_index: LongTensor [2, E]
            edge_types: LongTensor [E] with labels:
                0 -> parent -> child
                1 -> child -> parent
                2 -> sibling <-> sibling
        """
        device = parent_idx.device
        parent_to_children: dict[int, list[int]] = {}
        src_list: list[int] = []
        dst_list: list[int] = []
        type_list: list[int] = []

        for child, parent in enumerate(parent_idx.tolist()):
            if parent >= 0:
                # parent -> child
                src_list.append(parent)
                dst_list.append(child)
                type_list.append(self.EDGE_PARENT_TO_CHILD)
                # child -> parent
                src_list.append(child)
                dst_list.append(parent)
                type_list.append(self.EDGE_CHILD_TO_PARENT)
                parent_to_children.setdefault(parent, []).append(child)

        for siblings in parent_to_children.values():
            if len(siblings) < 2:
                continue
            for i in range(len(siblings)):
                for j in range(len(siblings)):
                    if i == j:
                        continue
                    src_list.append(siblings[i])
                    dst_list.append(siblings[j])
                    type_list.append(self.EDGE_SIBLING)

        if src_list:
            edge_index = th.tensor([src_list, dst_list], device=device, dtype=parent_idx.dtype)
            edge_types = th.tensor(type_list, device=device, dtype=parent_idx.dtype)
        else:
            edge_index = parent_idx.new_zeros((2, 0))
            edge_types = parent_idx.new_zeros((0,))
        return edge_index, edge_types
    
    def sample_graphs(self, target_size: th.Tensor, model: Module):
        """Generate a batch of graphs starting from one root node per graph.

        Process:
          1. Initialize one root node per graph (position optionally noisy).
          2. Initialize all roots as leaves with an expansion label (default branching '2').
          3. Iteratively call `expand` until termination (no capacity for another 2-child expansion
             across all graphs or no leaves remain) or a safety `max_steps` bound.
          4. Unbatch the final adjacency into per-graph NetworkX graphs and return them.

        Args:
          target_size: Long/Int tensor [G] desired node count per graph (capacity upper bound).
          model: module providing 'rel_pred' & 'expansion_pred'.

        Optional behavior (tunable via attributes):
          - Deterministic mode uses seeded shuffles for fair capacity trimming & root noise.
          - Noise distribution matches training masking via `_sample_noise`.

        Returns:
          graphs: list[networkx.Graph] of length G. Each node carries:
              - 'pos': np.ndarray (3,) geometric position
        """
        if target_size.dim() != 1:
            raise ValueError("target_size must be 1D tensor of per-graph capacities.")
        if (target_size < 1).any():
            raise ValueError("All target_size entries must be >=1 to allocate a root node.")

        device = target_size.device
        num_graphs = int(target_size.numel())

        # ---- Initial root nodes ----

        # Explicitly initialize all roots at origin (0,0,0) 
        root_pos = th.zeros((num_graphs, 3), device=device)

        # Adjacency: start with no edges (one isolated root per graph)
        # SparseTensor requires row/col/value lists (empty) with proper shape.
        adj = SparseTensor(row=th.tensor([], dtype=th.long, device=device),
                           col=th.tensor([], dtype=th.long, device=device),
                           value=th.tensor([], dtype=th.float, device=device),
                           sparse_sizes=(num_graphs, num_graphs))

        batch = th.arange(num_graphs, device=device, dtype=th.long)  # root per graph
        parent_idx_1b = th.zeros(num_graphs, device=device, dtype=th.long)  # roots have parent 0
        leaf_idx = th.arange(num_graphs, device=device, dtype=th.long)      # all roots are leaves initially

        # Bootstrap expansion labels: set to branching (2) where capacity allows at least 3 nodes
        # If capacity <3, label as terminal (1) to avoid futile branching attempts.
        leaf_expansion = th.where(target_size >= 3, th.full_like(leaf_idx, 2), th.full_like(leaf_idx, 1))

        pos = root_pos
        # Persistent sibling order: -1 for nodes with no sibling assignment yet
        sibling_order = th.full((num_graphs,), -1, device=device, dtype=th.long)

        # Safety max steps: enough to reach capacity even if only one leaf expands each time.
        max_steps = int(target_size.max().item() * 2)  # generous upper bound
        step = 0
        terminated = False

        while not terminated and step < max_steps:
            adj, pos, leaf_idx, leaf_expansion, parent_idx_1b, batch, sibling_order, terminated = self.expand(
                adj,
                batch,
                target_size,
                model,
                pos=pos,
                leaf_idx=leaf_idx,
                leaf_expansion=leaf_expansion,
                parent_idx_1b=parent_idx_1b,
                sibling_order=sibling_order,
                step=step,
                ensure_progress=False,
                map_threshold=0.4,
            )
            step += 1

        # ---- Unbatch into per-graph geometric NetworkX graphs ----
        # Extract COO for adjacency and segment by batch
        row, col, val = adj.coo()
        graphs = []
        for g in range(num_graphs):
            g_mask = (batch == g)
            node_ids = th.nonzero(g_mask, as_tuple=False).flatten()
            # Map global node index -> local index
            local_map = {int(n.item()): i for i, n in enumerate(node_ids)}
            G = nx.Graph()
            # Add nodes with geometric position only
            for i_local, n_global in enumerate(node_ids.tolist()):
                G.add_node(
                    i_local,
                    pos=pos[n_global].detach().cpu().numpy(),
                )
            # Add edges where both endpoints belong to this graph (undirected, avoid duplicates)
            for r, c in zip(row.tolist(), col.tolist()):
                if r in local_map and c in local_map:
                    # Undirected; ensure single edge by adding only if r<=c
                    if local_map[r] <= local_map[c]:
                        G.add_edge(local_map[r], local_map[c])
            graphs.append(G)
        return graphs
    
    @th.no_grad()
    def expand(
        self,
        adj_reduced,
        batch_reduced,
        target_size,
        model: Module,
        *,
        pos: th.Tensor | None = None,
        leaf_idx: th.Tensor | None = None,
        leaf_expansion: th.Tensor | None = None,
        parent_idx_1b: th.Tensor | None = None,
        sibling_order: th.Tensor | None = None,
        step: int = 0,
        ensure_progress: bool = False,
        map_threshold: float = 0.5,
    ):
        """Expand graphs by one generation step using binary leaf branching.

        High-level overview:
          1. Validate required tensors are present (leaf-mode only).
          2. Compute remaining per-graph capacity relative to target_size.
          3. Map leaf expansion labels {1,2} -> spawn counts {0,2}.
          4. Enforce capacity (each branching costs 2 slots; all-or-nothing per leaf).
          5. Optionally force progress if capacity >=2 yet no expansions scheduled.
          6. Materialize new child nodes with Gaussian (optionally clipped) noise.
          7. Rebuild adjacency adding undirected parent-child edges.
          8. Define next leaf set (new children only).
          9. Forward pass to refine child positions & predict next expansion states.
         10. Threshold probabilities -> next labels {1,2}; compute termination.
        """
        if not all(x is not None for x in (pos, leaf_idx, leaf_expansion, parent_idx_1b)):
            raise ValueError("expand requires pos, leaf_idx, leaf_expansion, parent_idx_1b")

        # 1) Basic tensor / device prep
        device = pos.device
        parent_idx = parent_idx_1b - 1  # 0-based parent indices
        
        # -----ENFORCING DETERMINISTIC CONDITIONS-----

        # 2) Per-graph current size & remaining slots
        size_per_graph = scatter(th.ones_like(batch_reduced), batch_reduced)
        remaining_capacity = target_size.to(device) - size_per_graph
        if sibling_order is None:
            sibling_order = th.full((pos.size(0),), -1, device=device, dtype=th.long)

        if (remaining_capacity <= 0).all() or leaf_idx.numel() == 0:
            return adj_reduced, pos, leaf_idx.clone(), leaf_expansion.clone(), parent_idx_1b, batch_reduced, sibling_order.clone(), True

        # 3) Map labels -> spawn counts (binary branching)
        spawn_counts = (leaf_expansion == 2).long() * 2  # {0,2}
        leaf_batch = batch_reduced[leaf_idx]

        # 4) Capacity enforcement (no partial expansion of a leaf)
        spawn_counts_final = spawn_counts.clone()
        for g in target_size.nonzero().flatten():
            g_int = int(g.item())
            cap = int(remaining_capacity[g_int].item())
            if cap < 2:
                spawn_counts_final[leaf_batch == g_int] = 0
                continue
            mask_g = (leaf_batch == g_int)
            expanders = th.nonzero((spawn_counts_final == 2) & mask_g, as_tuple=False).flatten()
            needed = expanders.numel() * 2
            if needed <= cap:
                continue
            max_leaves = cap // 2
            # Fair selection: shuffle candidates before trimming to avoid index bias
            if self.deterministic_expansion:
                # Use deterministic shuffle for reproducibility (seed by graph + step)
                generator = th.Generator(device=expanders.device)
                generator.manual_seed(g_int * 10007 + step)  # combine graph id + step for unique seed
                perm = th.randperm(expanders.numel(), generator=generator, device=expanders.device)
            else:
                # Random shuffle for diverse selection
                perm = th.randperm(expanders.numel(), device=expanders.device)
            expanders_shuffled = expanders[perm]
            disable = expanders_shuffled[max_leaves:]  # disable excess after fair selection
            spawn_counts_final[disable] = 0

        # 5) Ensure progress if capacity still available
        if ensure_progress and (remaining_capacity >= 2).any():
            # Guarantee per-graph progress: for every graph that
            # (a) still has capacity for at least one 2-branch AND
            # (b) has at least one current leaf, AND
            # (c) has no leaf already scheduled to expand this step,
            # force exactly one leaf to expand (add 2 children).
            # This prevents graphs from stalling at size=3 (root + 2 children)
            # when the model prematurely predicts all leaves as terminal.
            num_graphs = int(target_size.numel())
            for g in range(num_graphs):
                if remaining_capacity[g] < 2:
                    continue  # not enough room for a binary expansion
                mask_g = (leaf_batch == g)
                if not mask_g.any():
                    continue  # no leaves to expand in this graph
                if (spawn_counts_final[mask_g] == 2).any():
                    continue  # already at least one expansion scheduled for this graph
                # Force the first leaf (deterministically or random) to expand
                leaf_indices_g = th.nonzero(mask_g, as_tuple=False).flatten()
                if self.deterministic_expansion:
                    # Deterministic ordering: pick smallest index (first in tensor)
                    forced_leaf = leaf_indices_g[0]
                else:
                    # Random selection among available leaves for diversity
                    rand_idx = th.randint(low=0, high=leaf_indices_g.numel(), size=(1,), device=leaf_indices_g.device)
                    forced_leaf = leaf_indices_g[rand_idx]
                spawn_counts_final[forced_leaf] = 2

        # 6) Count new children & early exit
        total_new_children = int(spawn_counts_final.sum().item())
        if total_new_children == 0:
            return adj_reduced, pos, leaf_idx.clone(), leaf_expansion.clone(), parent_idx_1b, batch_reduced, sibling_order.clone(), True

        # ----- EXPANSION -----

        # 7) Materialize children (positions, parents, batch ids)
        base_N = adj_reduced.size(0)
        new_child_positions = []
        new_child_parents = []
        new_child_batches = []
        parent_child_edges = []
        sibling_order_new = []  # track sibling order for newly created children (0=left,1=right)
        running_child_index = 0
        for li, sc in zip(leaf_idx.tolist(), spawn_counts_final.tolist()):
            if sc == 0:
                continue
            parent_pos = pos[li]
            # Use shared sampler for consistency with training masking
            noise = self._sample_noise((sc, parent_pos.shape[0]), device, sigma=self.leaf_noise_sigma, clip=self.leaf_noise_clip)
            child_pos = parent_pos.unsqueeze(0) + noise
            new_child_positions.append(child_pos)
            # record sibling order (0 for first, 1 for second) for binary branching
            for local_child in range(sc):
                global_child_idx = base_N + running_child_index
                parent_child_edges.append((li, global_child_idx))
                new_child_parents.append(li)
                new_child_batches.append(int(batch_reduced[li].item()))
                # spawn_counts are only 0 or 2; if ever >2, local_child gives order anyway
                running_child_index += 1
                # collect sibling order
                # build list lazily to avoid prealloc; will map after concatenation
                # list defined earlier
                sibling_order_new.append(0 if local_child == 0 else 1)
        if running_child_index != total_new_children:
            raise ValueError("Child accounting mismatch: expected %d got %d" % (total_new_children, running_child_index))

        new_child_positions_tensor = th.cat(new_child_positions, dim=0) if new_child_positions else pos.new_empty((0, 3))
        pos_new = th.cat([pos, new_child_positions_tensor], dim=0)
        parent_idx_new_0b = th.cat([parent_idx, th.tensor(new_child_parents, device=device, dtype=parent_idx.dtype)])
        parent_idx_1b_new = parent_idx_new_0b + 1
        batch_new = th.cat([batch_reduced, th.tensor(new_child_batches, device=device, dtype=batch_reduced.dtype)])

        # build sibling order tensor: -1 for existing nodes, {0,1} for new children (left/right)
        # Persistent sibling order: keep previous assignments, append new children
        sibling_order_next = th.cat([sibling_order, th.tensor(sibling_order_new, device=device, dtype=th.long)], dim=0)

        # 8) Rebuild adjacency (undirected parent-child edges)
        row_old, col_old, val_old = adj_reduced.coo()
        new_rows = []
        new_cols = []
        new_vals = []
        for p, c in parent_child_edges:
            new_rows.extend([p, c])
            new_cols.extend([c, p])
            new_vals.extend([1.0, 1.0])
        if new_rows:
            row_all = th.cat([row_old.to(device), th.tensor(new_rows, device=device)])
            col_all = th.cat([col_old.to(device), th.tensor(new_cols, device=device)])
            val_all = th.cat([val_old.to(device), th.tensor(new_vals, device=device)])
        else:
            row_all, col_all, val_all = row_old.to(device), col_old.to(device), val_old.to(device)
        adj_new = SparseTensor(row=row_all, col=col_all, value=val_all, sparse_sizes=(pos_new.size(0), pos_new.size(0)))

        # 9) Next leaf set (children only)
        leaf_idx_next = th.arange(base_N, base_N + total_new_children, device=device, dtype=leaf_idx.dtype)
        if leaf_idx_next.numel() == 0:
            return adj_new, pos_new, leaf_idx_next, leaf_idx_next.new_empty((0,), dtype=leaf_expansion.dtype), parent_idx_1b_new, batch_new, sibling_order_next, True

        # ----- POSITION REFINEMENT FOR LEAVES & NEXT EXPANSION PREDICTION -----

        # 10) Model forward to refine child positions & predict next expansion states
        feats_dim = getattr(model, 'feats_dim', 0)
        pos_dim = getattr(model, 'pos_dim', 3)
        if feats_dim > 0:
            # feature 0: is_leaf flag
            is_leaf_flag = pos_new.new_zeros((pos_new.size(0), 1))
            is_leaf_flag[leaf_idx_next] = 1.0
            features = [is_leaf_flag]
            feats_used = 1
            # feature 1: sibling left flag (persistent) if capacity permits
            if feats_used < feats_dim:                
                so = sibling_order_next
                sib_is_left = (so == 0).float().unsqueeze(-1)
                sib_is_left = th.where(so.unsqueeze(-1) >= 0, sib_is_left, sib_is_left.new_zeros(sib_is_left.shape))
                features.append(sib_is_left)
                feats_used += 1
            if feats_used < feats_dim:
                features.append(pos_new.new_zeros((pos_new.size(0), feats_dim - feats_used)))
            node_feats = th.cat(features, dim=-1)
            x_in = th.cat([pos_new[:, :pos_dim], node_feats], dim=-1)
        else:
            x_in = pos_new[:, :pos_dim]
        edge_index, edge_types = self._build_augmented_edge_index(parent_idx_new_0b)
        edge_attr = edge_types.unsqueeze(-1) if edge_types.numel() else edge_types.new_zeros((0, 1))
        out = model(x=x_in, edge_index=edge_index, batch=batch_new, edge_attr=edge_attr, parent_idx=parent_idx_new_0b)
        if not isinstance(out, dict) or 'rel_pred' not in out or 'expansion_pred' not in out:
            raise ValueError("Model must return dict with 'rel_pred' and 'expansion_pred'.")
        
        # relative position predictions from parents for all nodes
        rel_pred_all = out['rel_pred']
        # expansion predictions for all nodes
        expansion_pred_all = out['expansion_pred']

        # getting leaf predictions
        rel_pred_leaves = rel_pred_all[leaf_idx_next]
        expansion_pred_leaves = expansion_pred_all[leaf_idx_next]
        if expansion_pred_leaves.dim() == 1:
            expansion_pred_leaves = expansion_pred_leaves.unsqueeze(-1)
        parent_pos_for_children = pos_new[parent_idx_new_0b[leaf_idx_next]]

        # updating new leaf positions
        pos_new[leaf_idx_next] = parent_pos_for_children + rel_pred_leaves

        # calculating expansion labels for next step

        expansion_prob = expansion_pred_leaves.squeeze(-1).sigmoid() # training loss currently does not use logits
        leaf_expansion_next = (expansion_prob > map_threshold).long() + 1

        # 11) Termination condition for next step
        size_per_graph_new = scatter(th.ones_like(batch_new), batch_new)
        remaining_capacity_new = target_size.to(device) - size_per_graph_new
        terminated = (remaining_capacity_new < 2).all() or leaf_idx_next.numel() == 0

        return adj_new, pos_new, leaf_idx_next, leaf_expansion_next, parent_idx_1b_new, batch_new, sibling_order_next, terminated

    # ---------------------------------------------------------
    # 1) Build masked positions: replace leaf coords by parent + noise
    # ---------------------------------------------------------
    @th.no_grad()
    def _make_masked_positions(
        self,
        pos: th.Tensor,                 # [N,3] absolute coords (GT)
        leaf_idx: th.Tensor,            # [L]
        leaf_parent_idx: th.Tensor,     # [L]
        *,
        sigma: float,
        clip: float | None = None,
    ) -> th.Tensor:
        """
        Returns pos_in: [N,3] where leaves are replaced by parent_pos + Gaussian noise.
        Parents (and non-leaves) stay anchored.
        """
        device = pos.device
        pos_in = pos.clone()

        if leaf_idx.numel() == 0:
            return pos_in

        parent_pos = pos[leaf_parent_idx]                    # [L,3]
        # Use identical sampler as expansion for distributional match
        noise = self._sample_noise(parent_pos.shape, device, sigma=sigma, clip=clip)

        pos_in[leaf_idx] = parent_pos + noise
        return pos_in

    # ---------------------------------------------------------
    # 2) Ground-truth relative targets: leaf - parent (from GT)
    # ---------------------------------------------------------
    def _leaf_rel_targets(
        self,
        pos_gt: th.Tensor,              # [N,3] absolute GT coords
        leaf_idx: th.Tensor,            # [L]
        leaf_parent_idx: th.Tensor,     # [L]
    ) -> th.Tensor:
        if leaf_idx.numel() == 0:
            return pos_gt.new_zeros((0, 3))
        parent_pos = pos_gt[leaf_parent_idx]                 # [L,3]
        return pos_gt[leaf_idx] - parent_pos                 # [L,3]

    # ---------------------------------------------------------
    # 3) Optional permutation-aware positional loss via matching
    # ---------------------------------------------------------
    @staticmethod
    def _greedy_min_cost_matching(cost: th.Tensor) -> th.Tensor:
        """
        Simple greedy min-cost one-to-one matching.

        Args:
            cost: [K, K] cost matrix (assumed detached from autograd).

        Returns:
            assignment: LongTensor[K], where assignment[i] is the index of the
                        matched column (target) for row i (prediction).
        """
        K = int(cost.size(0))
        if K == 0:
            return cost.new_empty((0,), dtype=th.long)

        work = cost.clone()
        inf = (work.max().item() if work.numel() > 0 else 0.0) + 1.0
        assignment = work.new_full((K,), -1, dtype=th.long)

        for _ in range(K):
            flat = work.view(-1)
            _, idx = flat.min(dim=0)
            row = int(idx // K)
            col = int(idx % K)
            assignment[row] = col
            work[row, :] = inf
            work[:, col] = inf

        return assignment

    def _compute_leaf_pos_loss_with_matching(
        self,
        pred_rel: th.Tensor,          # [L,3] predicted parent-relative offsets for leaves
        tgt_rel: th.Tensor,           # [L,3] GT parent-relative offsets for leaves
        leaf_parent_idx: th.Tensor,   # [L] parent index per leaf (0-based)
    ) -> th.Tensor:
        """
        Per-parent permutation-invariant positional loss on leaves.

        For each parent:
          1) Collect its leaf children (pred_rel_group, tgt_rel_group),
          2) Build a K×K squared-distance cost matrix in relative space,
          3) Greedy min-cost matching,
          4) Accumulate MSE over matched pairs normalised by total #leaves.
        """
        if pred_rel.numel() == 0:
            return pred_rel.sum() * 0.0

        unique_parents, inverse, counts = leaf_parent_idx.unique(
            return_inverse=True, return_counts=True
        )
        total_loss = pred_rel.new_tensor(0.0)
        total_count = 0

        for parent, count in zip(unique_parents, counts):
            mask = (leaf_parent_idx == parent)
            idx = mask.nonzero(as_tuple=False).flatten()
            k = int(idx.numel())
            pred_group = pred_rel[idx]
            tgt_group = tgt_rel[idx]
            if k == 0:
                continue
            if k == 1:
                total_loss = total_loss + F.mse_loss(pred_group, tgt_group, reduction="sum")
                total_count += 1
                continue
            cost = (pred_group[:, None, :] - tgt_group[None, :, :]).pow(2).sum(dim=-1)
            assignment = self._greedy_min_cost_matching(cost.detach())
            matched_tgt = tgt_group[assignment]
            total_loss = total_loss + F.mse_loss(pred_group, matched_tgt, reduction="sum")
            total_count += k

        if total_count == 0:
            return pred_rel.sum() * 0.0
        return total_loss / float(total_count)

    # ---------------------------------------------------------
    # 4) Forward + loss (positional + expansion + optional sibling regularizer)
    # ---------------------------------------------------------
    def get_loss(self, batch, model: th.nn.Module):
        """
        Expected batch fields:
          - adj: SparseTensor [N×N] (undirected ok)
          - pos: Float [N,3] (absolute GT)
          - leaf_expansion: Long [L] in {1,2} (GT expansion states for leaves)
          - leaf_idx: Long [L]
          - parent_idx_1b: Long [N] (1-based; 0 denotes root; will be shifted in batching)
          - batch: Long [N]  (PyG batch vector)
        The network returns a dict with:
          - "node_state": [N, pos_dim + feats_dim]
          - "rel_pred"  : [N, 3]  (predicted parent-relative offsets for ALL nodes)
        """
        
        # --- DEBUG: Print detailed batch information ---
        if getattr(self, 'debug', False):
            print(f"\n[BatchDebug] step={self._debug_step} ============================")
            print(f"[BatchDebug] Batch attributes: {[attr for attr in dir(batch) if not attr.startswith('_')]}")
            
            if hasattr(batch, 'batch'):
                unique_graphs = batch.batch.unique().tolist()
                print(f"[BatchDebug] Graphs in batch: {unique_graphs} (total: {len(unique_graphs)})")
                for g_id in unique_graphs:
                    mask = (batch.batch == g_id)
                    print(f"[BatchDebug] Graph {g_id}: {mask.sum().item()} nodes")
            
            if hasattr(batch, 'pos'):
                print(f"[BatchDebug] pos.shape: {batch.pos.shape}, dtype: {batch.pos.dtype}")
                print(f"[BatchDebug] pos range: [{batch.pos.min().item():.4f}, {batch.pos.max().item():.4f}]")
            
            if hasattr(batch, 'adj'):
                print(f"[BatchDebug] adj.shape: {batch.adj.sizes()}, nnz: {batch.adj.nnz()}")
            
            if hasattr(batch, 'leaf_idx'):
                print(f"[BatchDebug] leaf_idx: {batch.leaf_idx.tolist()} (count: {len(batch.leaf_idx)})")
            
            if hasattr(batch, 'leaf_expansion'):
                print(f"[BatchDebug] leaf_expansion: {batch.leaf_expansion.tolist()}")
                expansion_counts = {}
                for exp in batch.leaf_expansion.tolist():
                    expansion_counts[exp] = expansion_counts.get(exp, 0) + 1
                print(f"[BatchDebug] expansion counts: {expansion_counts}")
            
            if hasattr(batch, 'parent_idx_1b'):
                print(f"[BatchDebug] parent_idx_1b: {batch.parent_idx_1b.tolist()}")
                root_mask = (batch.parent_idx_1b == 0)
                print(f"[BatchDebug] Root nodes (parent_idx_1b==0): {th.nonzero(root_mask, as_tuple=False).flatten().tolist()}")
            
            print(f"[BatchDebug] ==========================================\n")

        # --- parent indices (1-based in Data for safe batching) -> shift back to 0-based with -1 for roots
        if not hasattr(batch, "parent_idx_1b"):
            raise ValueError("Expected batch.parent_idx_1b (1-based parent indices). Please update dataloader.")
        parent_idx = batch.parent_idx_1b - 1                      # [N], -1 for roots

        # --- graph and positions
        pos_gt = batch.pos                             # [N,3] (absolute, untouched)
        edge_index, edge_types = self._build_augmented_edge_index(parent_idx)

        # --- tracking of leaves
        if not hasattr(batch, "leaf_idx"):
            raise ValueError("Expected batch.leaf_idx (leaf node indices). Please update dataloader.")
        
        # --- expansion state for leaves
        if not hasattr(batch, "leaf_expansion"):
            raise ValueError("Expected batch.leaf_expansion (leaf expansion states). Please update dataloader.")
        leaf_expansion = batch.leaf_expansion - 1       # [L] in {0,1}
        # leaf_expansion = leaf_expansion.float() * 2 - 1 # map to {-1,1} for regression

        # derive leaf -> parent mapping from global parent_idx
        leaf_parent_idx = parent_idx[batch.leaf_idx]
        assert (leaf_parent_idx >= 0).all(), "Leaf with no valid parent encountered."

        # --- prepare masked input positions for leaves
        pos_in = self._make_masked_positions(
            pos=pos_gt,
            leaf_idx=batch.leaf_idx,
            leaf_parent_idx=leaf_parent_idx,
            sigma=self.leaf_noise_sigma,
            clip=self.leaf_noise_clip,
        )                                              # [N,3]

        # --- Enhanced debug plotting (GT vs masked) with root/leaf coloring ---
        if getattr(self, 'debug', False):
            from utils.debug_helpers import plot_gt_and_masked_enhanced  # enhanced plotting function
            batch_vec = batch.batch if hasattr(batch, 'batch') else None
            if batch_vec is not None:
                unique_graphs = batch_vec.unique().tolist()
                plotted = 0
                for b_id in unique_graphs:
                    if plotted >= self.debug_max_batches:
                        break
                    node_mask = (batch_vec == b_id)
                    if node_mask.sum() == 0:
                        continue
                    try:
                        sub_indices = node_mask.nonzero(as_tuple=False).flatten()
                        pos_gt_sub = pos_gt[sub_indices]
                        pos_in_sub = pos_in[sub_indices]
                        
                        # Enhanced debug info for this graph
                        print(f"[PlotDebug] Graph {b_id}: processing {sub_indices.numel()} nodes")
                        print(f"[PlotDebug] Graph {b_id} node indices: {sub_indices.tolist()}")
                        
                        row, col, val = batch.adj.coo()
                        keep = node_mask[row] & node_mask[col]
                        row_sub = row[keep]
                        col_sub = col[keep]
                        val_sub = val[keep]
                        
                        print(f"[PlotDebug] Graph {b_id}: {len(row_sub)} edges after filtering")
                        
                        import torch as _t
                        from torch_sparse import SparseTensor as _ST
                        mapping = {int(gidx.item()): i for i, gidx in enumerate(sub_indices)}
                        
                        # Reindex edges
                        try:
                            if row_sub.numel() == 0:
                                row_mapped = _t.empty((0,), dtype=_t.long, device=row_sub.device)
                                col_mapped = _t.empty((0,), dtype=_t.long, device=row_sub.device)
                                val_mapped = _t.empty((0,), dtype=val_sub.dtype, device=val_sub.device)
                            else:
                                row_mapped = _t.tensor([mapping[int(r.item())] for r in row_sub], device=row_sub.device, dtype=_t.long)
                                col_mapped = _t.tensor([mapping[int(c.item())] for c in col_sub], device=col_sub.device, dtype=_t.long)
                                val_mapped = val_sub
                        except KeyError as ke:
                            logger.warning(f"Reindex KeyError graph={b_id}: {ke}; skipping plot")
                            continue
                            
                        # Build sparse tensor safely
                        adj_sub = _ST(row=row_mapped, col=col_mapped, value=val_mapped, sparse_sizes=(sub_indices.numel(), sub_indices.numel()))
                        
                        # Enhanced node type identification
                        leaf_global = set(batch.leaf_idx.tolist())
                        leaf_local_idx = [mapping[gidx] for gidx in sub_indices.tolist() if gidx in leaf_global]
                        
                        # Find root nodes (parent_idx_1b == 0)
                        root_global = set(th.nonzero(batch.parent_idx_1b == 0, as_tuple=False).flatten().tolist())
                        root_local_idx = [mapping[gidx] for gidx in sub_indices.tolist() if gidx in root_global]
                        
                        # Build expansion mapping
                        leaf_expansion_map = {}
                        for g_leaf, lab in zip(batch.leaf_idx.tolist(), batch.leaf_expansion.tolist()):
                            if g_leaf in mapping:
                                leaf_expansion_map[mapping[g_leaf]] = lab
                        leaf_expansion_local = [leaf_expansion_map[i] for i in leaf_local_idx]
                        
                        print(f"[PlotDebug] Graph {b_id}: roots={root_local_idx}, leaves={leaf_local_idx}")
                        print(f"[PlotDebug] Graph {b_id}: leaf_expansion={leaf_expansion_local}")
                        
                        gt_file, masked_file = plot_gt_and_masked_enhanced(
                            adj_sub,
                            pos_gt_sub,
                            pos_in_sub,
                            self.debug_dir,
                            prefix=f"trainstep{self._debug_step}",
                            step=self._debug_step,
                            batch_id=b_id,
                            leaf_local_idx=leaf_local_idx,
                            leaf_expansion=leaf_expansion_local,
                            root_local_idx=root_local_idx,
                        )
                        logger.info(
                            f"[ExpansionDebug] step={self._debug_step} graph={b_id} nodes={sub_indices.numel()} roots={len(root_local_idx)} leaves={len(leaf_local_idx)} saved: {gt_file.name}, {masked_file.name}"
                        )
                        plotted += 1
                    except Exception as e_plot:
                        import traceback, sys
                        tb = ''.join(traceback.format_exception(type(e_plot), e_plot, e_plot.__traceback__))
                        logger.error(f"Plot failure step={self._debug_step} graph={b_id}: {e_plot}\n{tb}")
                self._debug_step += 1

        # --- prepare EGNN input (positions + minimal node features)
        feats_dim = getattr(model, 'feats_dim', 0)
        pos_dim   = getattr(model, 'pos_dim', 3)

        if feats_dim > 0:
            # seed with simple is_leaf flag - could be extended later TODO
            is_leaf = pos_in.new_zeros((pos_in.size(0), 1))
            is_leaf[batch.leaf_idx] = 1.0
            features = [is_leaf]
            feats_used = 1

            # Add sibling order as binary feature for left child in binary trees
            if hasattr(batch, "sibling_order") and feats_used < feats_dim:
                so = batch.sibling_order.to(pos_in.device)
                sib_is_left = (so == 0).float().unsqueeze(-1)
                # Clamp roots (-1) to 0 for network stability as specified
                sib_is_left = th.where(so.unsqueeze(-1) >= 0, sib_is_left, th.zeros_like(sib_is_left))

                # check that number of left siblings == number of right siblings
                if getattr(self, 'debug', False):
                    if (so == 0).sum().item() != (so == 1).sum().item():
                        logger.warning("Unequal number of left/right siblings in batch; check data integrity.")
                features.append(sib_is_left)
                feats_used += 1
            
            # Fill remaining dimensions with zeros if needed
            if feats_used < feats_dim:
                extra = pos_in.new_zeros((pos_in.size(0), feats_dim - feats_used))
                features.append(extra)
            
            node_feats = th.cat(features, dim=-1)
            x_in = th.cat([pos_in, node_feats], dim=-1)
        else:
            x_in = pos_in[:, :pos_dim]

        edge_attr = edge_types.unsqueeze(-1) if edge_types.numel() else edge_types.new_zeros((0, 1))

        out = model(
            x=x_in,
            edge_index=edge_index,
            batch=batch.batch,
            edge_attr=edge_attr,
            parent_idx=parent_idx,
        )
        if isinstance(out, dict):
            pred_rel_all = out["rel_pred"]                # [N,3]
            pred_expansion_all = out["expansion_pred"]    # [N,1] or [N]
        else:
            raise ValueError("Network must return a dict with 'rel_pred' and 'expansion_pred'.")

        leaf_idx = batch.leaf_idx
        pred_rel = pred_rel_all[leaf_idx]                           # [L,3]
        pred_expansion = pred_expansion_all[leaf_idx]               # [L,1] or [L]

        # -- target relative offsets from parents for leaves
        tgt_rel  = self._leaf_rel_targets(pos_gt, leaf_idx, leaf_parent_idx)  # [L,3]

        # step_norm_gt   = tgt_rel.norm(dim=-1)      # [L]
        # step_norm_pred = pred_rel.norm(dim=-1)     # [L]
        # # print statistics
        # print(f"[LossDebug] step_norm_gt: mean={step_norm_gt.mean().item():.4f} std={step_norm_gt.std().item():.4f} min={step_norm_gt.min().item():.4f} max={step_norm_gt.max().item():.4f}")
        # print(f"[LossDebug] step_norm_pred: mean={step_norm_pred.mean().item():.4f} std={step_norm_pred.std().item():.4f} min={step_norm_pred.min().item():.4f} max={step_norm_pred.max().item():.4f}")
        # print(f"[LossDebug] step_norm_diff: mean={(step_norm_pred - step_norm_gt).mean().item():.4f} std={(step_norm_pred - step_norm_gt).std().item():.4f} min={(step_norm_pred - step_norm_gt).min().item():.4f} max={(step_norm_pred - step_norm_gt).max().item():.4f}")
        # print(f"Number of leaves: {leaf_idx.numel()}")

        # --- loss
        if pred_rel.numel() == 0:
            loss = pred_rel_all.sum() * 0.0
            metrics = {"leaf_pos_loss": 0.0, "leaf_expansion_loss": 0.0, "cumulative_loss": 0.0, "num_leaves": 0}
            return loss, metrics

        # Positional loss on leaves (matching optional)
        if getattr(self, "use_sibling_matching", False):
            leaf_pos_loss = self._compute_leaf_pos_loss_with_matching(
                pred_rel=pred_rel,
                tgt_rel=tgt_rel,
                leaf_parent_idx=leaf_parent_idx,
            )
        else:
            leaf_pos_loss = F.mse_loss(pred_rel, tgt_rel)
        # Ensure dimension compatibility for expansion loss
        if pred_expansion.dim() == 1:
            pred_expansion = pred_expansion.unsqueeze(-1)  # [L] -> [L,1]
        if leaf_expansion.dim() == 1:
            leaf_expansion = leaf_expansion.unsqueeze(-1)  # [L] -> [L,1]
        # leaf_expansion_loss = F.mse_loss(pred_expansion, leaf_expansion)
        # lets use a Logit Loss for better stability
        leaf_expansion_loss = F.binary_cross_entropy_with_logits(
            pred_expansion.float(),
            leaf_expansion.float(),
        )
        # --- sibling distance regularizer (prevents siblings collapsing) ---
        sibling_dist_loss = pred_rel.sum() * 0.0  # zero default
        if self.sibling_loss_weight > 0.0 and leaf_idx.numel() > 1:
            abs_gt_all = pos_gt[leaf_idx]
            parent_pos_in_all = pos_in[leaf_parent_idx]
            abs_pred_all = parent_pos_in_all + pred_rel
            unique_parents, inverse, counts = leaf_parent_idx.unique(
                return_inverse=True, return_counts=True
            )
            mask_multi = counts >= 2
            if mask_multi.any():
                pair_indices = []
                for parent in unique_parents[mask_multi]:
                    leaf_pos_for_parent = (leaf_parent_idx == parent).nonzero(as_tuple=False).flatten()
                    if leaf_pos_for_parent.numel() < 2:
                        continue
                    pair_indices.append(leaf_pos_for_parent[:2])
                if len(pair_indices) > 0:
                    pair_indices = th.stack(pair_indices, dim=0)
                    idx1 = pair_indices[:, 0]
                    idx2 = pair_indices[:, 1]
                    v_gt = abs_gt_all[idx2] - abs_gt_all[idx1]
                    v_pred = abs_pred_all[idx2] - abs_pred_all[idx1]
                    d_gt = v_gt.norm(dim=-1)
                    d_pred = v_pred.norm(dim=-1)
                    sibling_dist_loss = F.mse_loss(d_pred, d_gt)

        # combine losses with sibling regularizer
        loss = leaf_pos_loss + leaf_expansion_loss + self.sibling_loss_weight * sibling_dist_loss

        with th.no_grad():
            parent_pos_in = pos_in[leaf_parent_idx]                # [L,3]
            abs_pred = parent_pos_in + pred_rel                    # [L,3]

        metrics = {
            "leaf_pos_loss": float(leaf_pos_loss.item()),
            "leaf_expansion_loss": float(leaf_expansion_loss.item()),
            "sibling_dist_loss": float(sibling_dist_loss.item()),
            "cumulative_loss": float(loss.item()),
            "num_leaves": int(leaf_idx.numel()),
            # "abs_pred_mean_norm": float(abs_pred.norm(dim=-1).mean().item()),
        }
        return loss, metrics
