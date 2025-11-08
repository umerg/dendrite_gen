import networkx as nx
import torch as th
from torch.nn import Module
from torch_geometric.utils import to_edge_index
import torch.nn.functional as F
from torch_scatter import scatter
from torch_sparse import SparseTensor

from .method import Method

class Expansion_OneShot(Method):
    """Graph generation method generating graphs by local expansion."""

    def __init__(
        self,
        deterministic_expansion=False, # just sets seeding for reproducibility
        red_threshold=0,
        leaf_noise_sigma=0.05,           # <-- stddev of Gaussian around parent (same units as pos)
        leaf_noise_clip=None,            # <-- optional radius clamp (float) or None
    ):
        super().__init__(diffusion=None)
        self.deterministic_expansion = deterministic_expansion
        self.red_threshold = red_threshold
        self.leaf_noise_sigma = float(leaf_noise_sigma)
        self.leaf_noise_clip = leaf_noise_clip
    
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
          graphs: list[networkx.Graph] of length G.
          pos: final positions tensor [N,3].
          batch: batch vector [N].

        Note: For simplicity we return NetworkX graphs without node features; positions are
              returned separately so downstream code can attach them as needed.
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

        # Safety max steps: enough to reach capacity even if only one leaf expands each time.
        max_steps = int(target_size.max().item() * 2)  # generous upper bound
        step = 0
        terminated = False

        while not terminated and step < max_steps:
            adj, pos, leaf_idx, leaf_expansion, parent_idx_1b, batch, terminated = self.expand(
                adj,
                batch,
                target_size,
                model,
                pos=pos,
                leaf_idx=leaf_idx,
                leaf_expansion=leaf_expansion,
                parent_idx_1b=parent_idx_1b,
                step=step,
                ensure_progress=True,
                map_threshold=0.5,
            )
            step += 1

        # ---- Unbatch into per-graph NetworkX graphs ----
        # Extract COO for adjacency and segment by batch
        row, col, val = adj.coo()
        graphs = []
        for g in range(num_graphs):
            g_mask = (batch == g)
            node_ids = th.nonzero(g_mask, as_tuple=False).flatten()
            # Map global node index -> local index
            local_map = {int(n.item()): i for i, n in enumerate(node_ids)}
            G = nx.Graph()
            # Add nodes (store position metadata separately if desired)
            for i_local, n_global in enumerate(node_ids.tolist()):
                G.add_node(i_local)
            # Add edges where both endpoints belong to this graph
            for r, c in zip(row.tolist(), col.tolist()):
                if r in local_map and c in local_map:
                    # Undirected; ensure single edge by adding only if r<=c
                    if local_map[r] <= local_map[c]:
                        G.add_edge(local_map[r], local_map[c])
            graphs.append(G)

        return graphs, pos, batch
    
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
        step: int = 0,
        ensure_progress: bool = True,
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
        if (remaining_capacity <= 0).all() or leaf_idx.numel() == 0:
            return adj_reduced, pos, leaf_idx.clone(), leaf_expansion.clone(), parent_idx_1b, batch_reduced, True

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
        if ensure_progress and spawn_counts_final.sum().item() == 0 and (remaining_capacity >= 2).any():
            for g in range(target_size.numel()):
                if remaining_capacity[g] >= 2:
                    first_leaf_g = th.nonzero(leaf_batch == g, as_tuple=False)
                    if first_leaf_g.numel() > 0:
                        spawn_counts_final[first_leaf_g[0]] = 2
                        break

        # 6) Count new children & early exit
        total_new_children = int(spawn_counts_final.sum().item())
        if total_new_children == 0:
            return adj_reduced, pos, leaf_idx.clone(), leaf_expansion.clone(), parent_idx_1b, batch_reduced, True

        # ----- EXPANSION -----

        # 7) Materialize children (positions, parents, batch ids)
        base_N = adj_reduced.size(0)
        new_child_positions = []
        new_child_parents = []
        new_child_batches = []
        parent_child_edges = []
        running_child_index = 0
        for li, sc in zip(leaf_idx.tolist(), spawn_counts_final.tolist()):
            if sc == 0:
                continue
            parent_pos = pos[li]
            # Use shared sampler for consistency with training masking
            noise = self._sample_noise((sc, parent_pos.shape[0]), device, sigma=self.leaf_noise_sigma, clip=self.leaf_noise_clip)
            child_pos = parent_pos.unsqueeze(0) + noise
            new_child_positions.append(child_pos)
            for _ in range(sc):
                global_child_idx = base_N + running_child_index
                parent_child_edges.append((li, global_child_idx))
                new_child_parents.append(li)
                new_child_batches.append(int(batch_reduced[li].item()))
                running_child_index += 1
        if running_child_index != total_new_children:
            raise ValueError("Child accounting mismatch: expected %d got %d" % (total_new_children, running_child_index))

        new_child_positions_tensor = th.cat(new_child_positions, dim=0) if new_child_positions else pos.new_empty((0, 3))
        pos_new = th.cat([pos, new_child_positions_tensor], dim=0)
        parent_idx_new_0b = th.cat([parent_idx, th.tensor(new_child_parents, device=device, dtype=parent_idx.dtype)])
        parent_idx_1b_new = parent_idx_new_0b + 1
        batch_new = th.cat([batch_reduced, th.tensor(new_child_batches, device=device, dtype=batch_reduced.dtype)])

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
            return adj_new, pos_new, leaf_idx_next, leaf_idx_next.new_empty((0,), dtype=leaf_expansion.dtype), parent_idx_1b_new, batch_new, True

        # ----- POSITION REFINEMENT FOR LEAVES & NEXT EXPANSION PREDICTION -----

        # 10) Model forward to refine child positions & predict next expansion states
        feats_dim = getattr(model, 'feats_dim', 0)
        pos_dim = getattr(model, 'pos_dim', 3)
        if feats_dim > 0:
            is_leaf_flag = pos_new.new_zeros((pos_new.size(0), 1))
            is_leaf_flag[leaf_idx_next] = 1.0
            extra = pos_new.new_zeros((pos_new.size(0), feats_dim - 1)) if feats_dim > 1 else None
            node_feats = th.cat([is_leaf_flag, extra], dim=-1) if extra is not None else is_leaf_flag
            x_in = th.cat([pos_new[:, :pos_dim], node_feats], dim=-1)
        else:
            x_in = pos_new[:, :pos_dim]
        edge_index, _ = to_edge_index(adj_new)
        
        out = model(x=x_in, edge_index=edge_index, batch=batch_new, edge_attr=None, parent_idx=parent_idx_new_0b)
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

        # expansion_prob = expansion_pred_leaves.squeeze(-1).sigmoid() # training loss currently does not use logits
        leaf_expansion_next = (expansion_pred_leaves.squeeze(-1) > map_threshold).long() + 1

        # 11) Termination condition for next step
        size_per_graph_new = scatter(th.ones_like(batch_new), batch_new)
        remaining_capacity_new = target_size.to(device) - size_per_graph_new
        terminated = (remaining_capacity_new < 2).all() or leaf_idx_next.numel() == 0

        return adj_new, pos_new, leaf_idx_next, leaf_expansion_next, parent_idx_1b_new, batch_new, terminated

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
    # 3) Forward + loss (MSE on leaves only) assuming model → [N,3]
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

        # --- parent indices (1-based in Data for safe batching) -> shift back to 0-based with -1 for roots
        if not hasattr(batch, "parent_idx_1b"):
            raise ValueError("Expected batch.parent_idx_1b (1-based parent indices). Please update dataloader.")
        parent_idx = batch.parent_idx_1b - 1                      # [N], -1 for roots

        # --- graph and positions
        pos_gt = batch.pos                             # [N,3] (absolute, untouched)
        edge_index, _ = to_edge_index(batch.adj)       # (2,E)

        # --- tracking of leaves
        if not hasattr(batch, "leaf_idx"):
            raise ValueError("Expected batch.leaf_idx (leaf node indices). Please update dataloader.")
        
        # --- expansion state for leaves
        if not hasattr(batch, "leaf_expansion"):
            raise ValueError("Expected batch.leaf_expansion (leaf expansion states). Please update dataloader.")
        leaf_expansion = batch.leaf_expansion - 1       # [L] in {0,1}
        leaf_expansion = leaf_expansion.float() * 2 - 1 # map to {-1,1} for regression

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

        # --- prepare EGNN input (positions + minimal node features)
        feats_dim = getattr(model, 'feats_dim', 0)
        pos_dim   = getattr(model, 'pos_dim', 3)

        if feats_dim > 0:
            # seed with simple is_leaf flag - could be extended later TODO
            is_leaf = pos_in.new_zeros((pos_in.size(0), 1))
            is_leaf[batch.leaf_idx] = 1.0
            extra = pos_in.new_zeros((pos_in.size(0), feats_dim - 1)) if feats_dim > 1 else None
            node_feats = th.cat([is_leaf, extra], dim=-1) if extra is not None else is_leaf
            x_in = th.cat([pos_in, node_feats], dim=-1)
        else:
            x_in = pos_in[:, :pos_dim]

        out = model(
            x=x_in,
            edge_index=edge_index,
            batch=batch.batch,
            edge_attr=None,
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

        # --- loss
        if pred_rel.numel() == 0:
            loss = pred_rel_all.sum() * 0.0
            metrics = {"leaf_pos_loss": 0.0, "leaf_expansion_loss": 0.0, "cumulative_loss": 0.0, "num_leaves": 0}
            return loss, metrics

        leaf_pos_loss = F.mse_loss(pred_rel, tgt_rel)
        # Ensure dimension compatibility for expansion loss
        if pred_expansion.dim() == 1:
            pred_expansion = pred_expansion.unsqueeze(-1)  # [L] -> [L,1]
        if leaf_expansion.dim() == 1:
            leaf_expansion = leaf_expansion.unsqueeze(-1)  # [L] -> [L,1]
        leaf_expansion_loss = F.mse_loss(pred_expansion, leaf_expansion)
        loss = leaf_pos_loss + leaf_expansion_loss

        with th.no_grad():
            parent_pos_in = pos_in[leaf_parent_idx]                # [L,3]
            abs_pred = parent_pos_in + pred_rel                    # [L,3]

        metrics = {
            "leaf_pos_loss": float(leaf_pos_loss.item()),
            "leaf_expansion_loss": float(leaf_expansion_loss.item()),
            "cumulative_loss": float(loss.item()),
            "num_leaves": int(leaf_idx.numel()),
            # "abs_pred_mean_norm": float(abs_pred.norm(dim=-1).mean().item()),
        }
        return loss, metrics
