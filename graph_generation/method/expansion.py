import time
import networkx as nx
import torch as th
from torch.nn import Module
from torch_scatter import scatter
from torch_sparse import SparseTensor
import logging

logger = logging.getLogger(__name__)


def _t(device: th.device) -> float:
    """Return current wall time, syncing CUDA if needed for accurate GPU timing."""
    if device.type == 'cuda':
        th.cuda.synchronize(device)
    return time.perf_counter()

from .helpers import (
    build_directed_edge_index,
    compute_local_bases_for_leaves,
    decode_parent_indices,
    global_to_local,
    leaf_rel_targets,
    local_to_global,
    plot_diffusion_debug_trees,
    precompute_full_geometry,
    select_training_leaf_indices,
    size_ratio_feature_from_batch,
)
from .method import Method

class Expansion(Method):
    """Graph generation method generating graphs by local expansion."""

    EDGE_PARENT_TO_CHILD = 0
    EDGE_CHILD_TO_PARENT = 1

    def __init__(
        self,
        diffusion: Module | None = None,
        red_threshold: int = 0,
        expansion_loss_weight: float = 1.0,
        use_size_ratio: bool = True,
        max_tree_size: int = 500,
    ):
        super().__init__(diffusion=diffusion)
        self.red_threshold = red_threshold
        self.expansion_loss_weight = float(expansion_loss_weight)
        self.use_size_ratio = use_size_ratio
        self.max_tree_size = max_tree_size
    
    def sample_graphs(self, target_size: th.Tensor, model: Module, tmd: th.Tensor | None = None,
                      num_root_children: th.Tensor | int | None = None):
        """Generate graphs via iterative diffusion-based leaf expansion."""
        if self.diffusion is None:
            raise ValueError("Diffusion module is required for sampling.")
        if target_size.dim() != 1:
            raise ValueError("target_size must be a 1D tensor.")
        if (target_size < 1).any():
            raise ValueError("target_size entries must all be >= 1.")

        device = target_size.device
        num_graphs = int(target_size.numel())
        if tmd is not None:
            tmd = tmd.to(device=device)

        # Normalize num_root_children to a per-graph tensor
        if num_root_children is not None:
            if isinstance(num_root_children, int):
                nrc = th.full((num_graphs,), num_root_children, device=device, dtype=th.long)
            else:
                nrc = num_root_children.to(device=device, dtype=th.long)
        else:
            nrc = None

        pos = th.zeros((num_graphs, 3), device=device)
        adj = SparseTensor(
            row=th.tensor([], dtype=th.long, device=device),
            col=th.tensor([], dtype=th.long, device=device),
            value=th.tensor([], dtype=th.float, device=device),
            sparse_sizes=(num_graphs, num_graphs),
        )
        batch = th.arange(num_graphs, device=device, dtype=th.long)
        parent_idx_1b = th.zeros_like(batch)
        leaf_idx = batch.clone()
        leaf_expansion = th.ones_like(leaf_idx)  # root's spawn count is overridden in expand
        leaf_mask = th.ones((num_graphs,), device=device, dtype=th.bool)

        effective_max = min(int(target_size.max().item()), self.max_tree_size)
        max_steps = effective_max * 2
        step = 0
        terminated = False

        while not terminated and step < max_steps:
            (
                adj,
                pos,
                leaf_idx,
                leaf_expansion,
                parent_idx_1b,
                batch,
                leaf_mask,
                terminated,
            ) = self.expand(
                adj,
                batch,
                target_size,
                model,
                pos=pos,
                leaf_idx=leaf_idx,
                leaf_expansion=leaf_expansion,
                parent_idx_1b=parent_idx_1b,
                leaf_mask=leaf_mask,
                tmd=tmd,
                step=step,
                num_root_children=nrc,
            )
            step += 1

        row, col, _ = adj.coo()
        graphs = []
        for g in range(num_graphs):
            mask = batch == g
            node_ids = mask.nonzero(as_tuple=False).flatten()
            local_map = {int(n.item()): i for i, n in enumerate(node_ids)}
            G = nx.Graph()
            for i_local, n_global in enumerate(node_ids.tolist()):
                G.add_node(i_local, pos=pos[n_global].detach().cpu().numpy())
            for r, c in zip(row.tolist(), col.tolist()):
                if r in local_map and c in local_map:
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
        leaf_mask: th.Tensor | None = None,
        tmd: th.Tensor | None = None,
        step: int = 0,
        map_threshold: float = 0.0,
        num_root_children: th.Tensor | None = None,
    ):
        """Expand graphs by one generation step.

        Supports k-ary root expansion when num_root_children is provided.
        Non-root leaves use binary branching (0 or 2 children).
        """

        if not all(x is not None for x in (pos, leaf_idx, leaf_expansion, parent_idx_1b)):
            raise ValueError("expand requires pos, leaf_idx, leaf_expansion, parent_idx_1b tensors.")
        if self.diffusion is None:
            raise ValueError("Diffusion module must be provided for sampling.")

        device = pos.device
        _t_start = _t(device)
        _t_leaf_loop = _t_cat_growth = _t_sparse_rebuild = _t_diffusion_sample = 0.0
        parent_idx = parent_idx_1b - 1
        num_graphs = int(target_size.numel())

        if leaf_mask is None:
            leaf_mask = th.ones((pos.size(0),), device=device, dtype=th.bool)
        else:
            leaf_mask = leaf_mask.to(device=device)
            if leaf_mask.dtype != th.bool:
                leaf_mask = leaf_mask.bool()

        size_per_graph = scatter(
            th.ones_like(batch_reduced, dtype=target_size.dtype),
            batch_reduced,
            dim=0,
            dim_size=num_graphs,
        )
        remaining_capacity = target_size.to(device) - size_per_graph

        if leaf_idx.numel() == 0:
            return (
                adj_reduced,
                pos,
                leaf_idx.clone(),
                leaf_expansion.clone(),
                parent_idx_1b,
                batch_reduced,
                leaf_mask.clone(),
                True,
            )

        spawn_counts = (leaf_expansion == 2).long() * 2
        leaf_batch = batch_reduced[leaf_idx]
        spawn_counts_final = spawn_counts.clone()

        # Root nodes: spawn k children (from num_root_children) or 1 (legacy)
        is_root_leaf = parent_idx[leaf_idx] < 0
        if is_root_leaf.any():
            if num_root_children is not None:
                # Spawn k children for each root, if capacity allows
                root_k = num_root_children[leaf_batch[is_root_leaf]]
                has_capacity = (target_size[leaf_batch[is_root_leaf]] > 1).long()
                root_spawn = root_k * has_capacity
                spawn_counts_final[is_root_leaf] = root_spawn
            else:
                # Legacy: spawn exactly 1 child
                root_should_spawn = (target_size[leaf_batch] > 1).long()
                spawn_counts_final = th.where(is_root_leaf, root_should_spawn, spawn_counts_final)

        # Cap spawning so no graph exceeds max_tree_size
        if self.max_tree_size is not None and self.max_tree_size > 0:
            size_per_leaf_graph = size_per_graph[leaf_batch]
            over_cap = size_per_leaf_graph >= self.max_tree_size
            spawn_counts_final[over_cap] = 0

        total_new_children = int(spawn_counts_final.sum().item())
        if total_new_children == 0:
            return (
                adj_reduced,
                pos,
                leaf_idx.clone(),
                leaf_expansion.clone(),
                parent_idx_1b,
                batch_reduced,
                leaf_mask.clone(),
                True,
            )

        base_N = adj_reduced.size(0)
        leaf_mask_updated = leaf_mask.clone()
        expanded_mask = spawn_counts_final > 0
        if expanded_mask.any():
            leaf_mask_updated[leaf_idx[expanded_mask]] = False
        new_positions = []
        new_parents = []
        new_batches = []
        parent_child_edges = []
        # Ordinal tracking for spawn-order feature
        ordinal_new = []
        sibling_count_new = []
        running_child_index = 0

        _t_loop_0 = _t(device)
        for leaf_global, sc in zip(leaf_idx.tolist(), spawn_counts_final.tolist()):
            if sc == 0:
                continue
            parent_pos = pos[leaf_global].unsqueeze(0)
            placeholder = parent_pos.expand(sc, -1).clone()
            new_positions.append(placeholder)
            for local_child in range(sc):
                global_child = base_N + running_child_index
                parent_child_edges.append((leaf_global, global_child))
                new_parents.append(leaf_global)
                new_batches.append(int(batch_reduced[leaf_global].item()))
                ordinal_new.append(local_child)
                sibling_count_new.append(sc)
                running_child_index += 1

        if running_child_index != total_new_children:
            raise RuntimeError("Mismatch when creating child nodes.")

        _t_leaf_loop = _t(device) - _t_loop_0
        _t_cat_0 = _t(device)
        new_pos_tensor = th.cat(new_positions, dim=0) if new_positions else pos.new_empty((0, pos.size(1)))
        pos_new = th.cat([pos, new_pos_tensor], dim=0)
        parent_idx_new_0b = th.cat(
            [parent_idx, th.tensor(new_parents, device=device, dtype=parent_idx.dtype)]
        )
        parent_idx_1b_new = parent_idx_new_0b + 1
        batch_new = th.cat(
            [batch_reduced, th.tensor(new_batches, device=device, dtype=batch_reduced.dtype)]
        )

        # Compute child ordinal index for new children (raw integer, not normalized)
        ordinal_t = th.tensor(ordinal_new, device=device, dtype=th.float)
        geo_angle_new = ordinal_t  # integer child index: 0, 1, ..., k-1

        # Also track ordinals and sibling counts for frame computation
        child_ordinal_t = th.tensor(ordinal_new, device=device, dtype=th.long)
        sib_count_long = th.tensor(sibling_count_new, device=device, dtype=th.long)

        _t_cat_growth = _t(device) - _t_cat_0
        _t_sparse_0 = _t(device)
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
        adj_new = SparseTensor(
            row=row_all,
            col=col_all,
            value=val_all,
            sparse_sizes=(pos_new.size(0), pos_new.size(0)),
        )

        _t_sparse_rebuild = _t(device) - _t_sparse_0
        leaf_idx_next = th.arange(base_N, base_N + total_new_children, device=device, dtype=leaf_idx.dtype)
        new_leaf_flags = th.ones((leaf_idx_next.numel(),), device=device, dtype=th.bool)
        leaf_mask_next = th.cat([leaf_mask_updated, new_leaf_flags], dim=0)
        if leaf_idx_next.numel() == 0:
            return (
                adj_new,
                pos_new,
                leaf_idx_next,
                leaf_idx_next.new_empty((0,), dtype=leaf_expansion.dtype),
                parent_idx_1b_new,
                batch_new,
                leaf_mask_next,
                True,
            )

        node_counts_per_graph = scatter(
            th.ones_like(batch_new, dtype=target_size.dtype),
            batch_new,
            dim=0,
            dim_size=num_graphs,
        )

        # --- Build edge index, local bases, and precompute geometry BEFORE feature assembly ---
        edge_index, edge_types = build_directed_edge_index(
            parent_idx_new_0b,
            edge_parent_to_child=self.EDGE_PARENT_TO_CHILD,
            edge_child_to_parent=self.EDGE_CHILD_TO_PARENT,
        )
        if edge_types.numel():
            edge_attr = edge_types.unsqueeze(-1).to(pos_new.dtype)
        else:
            edge_attr = pos_new.new_zeros((0, 1))

        leaf_parent_idx_next = parent_idx_new_0b[leaf_idx_next]

        # Compute local bases for new leaf nodes (with ordinal info for shared root frames)
        leaf_fwd, leaf_side = compute_local_bases_for_leaves(
            pos_new, parent_idx_new_0b, leaf_parent_idx_next, model.uhat,
            child_ordinal=child_ordinal_t,
            sibling_count=sib_count_long,
        )

        # Precompute full geometry on P_0 — patched cheaply per diffusion step
        with th.no_grad():
            pre_geom_p0 = precompute_full_geometry(
                pos_new, parent_idx_new_0b, edge_index, model.uhat,
            )

        # Override leaf local bases in pre_geom_p0 with the frames from
        # compute_local_bases_for_leaves — these handle the degenerate root
        # case (step 0, placeholder positions) with random shared frames,
        # whereas precompute_full_geometry uses a deterministic fallback.
        # This ensures patch_geometry_for_noised_leaves uses the same
        # reference frame as the local↔global conversion in diffusion.sample().
        pre_geom_p0['local_forward'] = pre_geom_p0['local_forward'].clone()
        pre_geom_p0['local_sideways'] = pre_geom_p0['local_sideways'].clone()
        pre_geom_p0['local_forward'][leaf_idx_next] = leaf_fwd
        pre_geom_p0['local_sideways'][leaf_idx_next] = leaf_side

        # Build geo_feat: use precomputed geo_ordinal for internal nodes,
        # spawn-order ordinals for new leaves (whose positions are placeholders)
        geo_feat_all = pre_geom_p0['geo_ordinal'].clamp(min=0.0).clone()
        geo_feat_all[leaf_idx_next] = geo_angle_new  # raw child index (0, 1, ..., k-1)

        # --- Assemble node features ---
        MAX_CHILDREN = 10  # one-hot ordinal dimension
        feats_total = getattr(model, "feats_dim", 0)
        tmd_hidden_dim = getattr(model, "tmd_hidden_dim", 0)
        cond_dim = getattr(self.diffusion, "cond_dim", 0)
        avail_feats_dim = feats_total - cond_dim - tmd_hidden_dim
        if tmd_hidden_dim > 0 and avail_feats_dim < (MAX_CHILDREN + 4):
            raise ValueError(f"feats_dim - tmd_hidden_dim - cond_dim must be >= {MAX_CHILDREN + 4} when using TMD.")
        if avail_feats_dim > 0:
            N = pos_new.size(0)
            features = []
            feats_used = 0
            is_leaf = leaf_mask_next.to(dtype=pos_new.dtype).unsqueeze(-1)
            features.append(is_leaf)
            feats_used += 1

            if feats_used + MAX_CHILDREN <= avail_feats_dim:
                # One-hot ordinal encoding (10D): child i lights up bit i
                geo_idx = geo_feat_all.long().clamp(0, MAX_CHILDREN - 1)
                geo_onehot = pos_new.new_zeros((N, MAX_CHILDREN))
                # Set one-hot for actual children (geo_ordinal >= 0; sentinel is -1)
                child_mask = pre_geom_p0['geo_ordinal'] >= 0
                # Override for new leaves (whose geo_ordinal in pre_geom_p0 is stale)
                child_mask[leaf_idx_next] = True
                if child_mask.any():
                    geo_onehot[child_mask] = geo_onehot[child_mask].scatter_(
                        1, geo_idx[child_mask].unsqueeze(-1), 1.0
                    )
                features.append(geo_onehot)
                feats_used += MAX_CHILDREN

            if feats_used < avail_feats_dim:
                new_flag = pos_new.new_zeros((N, 1))
                new_flag[leaf_idx_next] = 1.0
                features.append(new_flag)
                feats_used += 1

            if self.use_size_ratio and feats_used < avail_feats_dim:
                ratio_graph = node_counts_per_graph.to(pos_new.dtype) / target_size.to(pos_new.dtype).clamp_min(1.0)
                ratio_nodes = ratio_graph[batch_new].unsqueeze(-1)
                features.append(ratio_nodes)
                feats_used += 1

            if feats_used < avail_feats_dim:
                pad = pos_new.new_zeros((N, avail_feats_dim - feats_used))
                features.append(pad)

            node_feats = th.cat(features, dim=-1)
        else:
            node_feats = pos_new.new_zeros((pos_new.size(0), 0))

        # --- Diffusion sampling with precomputed geometry ---
        model_kwargs = {"tmd": tmd} if tmd is not None else None
        _t_diff_0 = _t(device)
        rel_pred, exp_pred = self.diffusion.sample(
            node_feats=node_feats,
            edge_index=edge_index,
            batch=batch_new,
            edge_attr=edge_attr,
            P_0=pos_new,
            parent_idx=parent_idx_new_0b,
            leaf_idx=leaf_idx_next,
            leaf_parent_idx=leaf_parent_idx_next,
            model=model,
            model_kwargs=model_kwargs,
            local_forward=leaf_fwd,
            local_sideways=leaf_side,
            uhat=model.uhat,
            pre_geom_p0=pre_geom_p0,
        )

        _t_diffusion_sample = _t(device) - _t_diff_0
        # Convert local-frame predictions to global for position reconstruction
        rel_pred_global = local_to_global(rel_pred, leaf_fwd, leaf_side, model.uhat)
        parent_pos_for_children = pos_new[leaf_parent_idx_next]
        pos_new[leaf_idx_next] = parent_pos_for_children + rel_pred_global

        if exp_pred.dim() == 1:
            exp_pred = exp_pred.unsqueeze(-1)
        expansion_score = exp_pred.squeeze(-1)
        leaf_expansion_next = (expansion_score > map_threshold).long() + 1

        # Force-stop expansion for graphs at max_tree_size
        if self.max_tree_size is not None and self.max_tree_size > 0:
            leaf_batch_next = batch_new[leaf_idx_next]
            at_cap = node_counts_per_graph[leaf_batch_next] >= self.max_tree_size
            leaf_expansion_next[at_cap] = 1  # 1 = no expansion

        remaining_capacity_new = target_size.to(device) - node_counts_per_graph
        terminated = leaf_idx_next.numel() == 0

        return (
            adj_new,
            pos_new,
            leaf_idx_next,
            leaf_expansion_next,
            parent_idx_1b_new,
            batch_new,
            leaf_mask_next,
            terminated,
        )

    # ---------------------------------------------------------
    # 4) Forward + loss (positional + expansion)
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
        parent_idx = decode_parent_indices(batch).to(device=batch.pos.device)           # [N], -1 for roots

        # --- graph and positions
        pos_gt = batch.pos                             # [N,3] (absolute, untouched)
        edge_index, edge_types = build_directed_edge_index(
            parent_idx,
            edge_parent_to_child=self.EDGE_PARENT_TO_CHILD,
            edge_child_to_parent=self.EDGE_CHILD_TO_PARENT,
        )

        # --- tracking of leaves
        if not hasattr(batch, "leaf_idx"):
            raise ValueError("Expected batch.leaf_idx (leaf node indices). Please update dataloader.")
        leaf_idx_all = batch.leaf_idx

        # leaf_graphs_all = batch.batch[leaf_idx_all]
        # logger.info(
        # "[LeafAllDebug] leaf_idx_all.max()=%s, unique_leaf_graphs_all=%s, leaf_idx_all[:20]=%s",
        # int(leaf_idx_all.max().item()),
        # int(leaf_graphs_all.unique().numel()),
        # leaf_idx_all[:20].tolist()
        # )

        # --- expansion state for leaves
        if not hasattr(batch, "leaf_expansion"):
            raise ValueError("Expected batch.leaf_expansion (leaf expansion states). Please update dataloader.")
        leaf_expansion_all = batch.leaf_expansion - 1       # [L_total] in {0,1}

        leaf_idx_train = select_training_leaf_indices(batch)
        if leaf_idx_train.numel() == 0:
            leaf_parent_idx = parent_idx.new_empty((0,), dtype=parent_idx.dtype)
        else:
            leaf_parent_idx = parent_idx[leaf_idx_train]
            assert (leaf_parent_idx >= 0).all(), "Leaf with no valid parent encountered."

        # map per-node expansion labels so new leaves can be indexed directly
        # more an indexing step? because we map to allnodes and then back to only new leaves instead of all leaves
        leaf_targets_per_node = leaf_expansion_all.new_full((pos_gt.size(0),), -1)
        if leaf_idx_all.numel() > 0:
            leaf_targets_per_node[leaf_idx_all] = leaf_expansion_all.view(-1)
        leaf_expansion = leaf_targets_per_node[leaf_idx_train]
        if leaf_expansion.numel() > 0:
            valid_mask = leaf_expansion >= 0
            if not valid_mask.all(): # filter out any invalid leaves - extra safety? Shouldn't get triggered?
                leaf_idx_train = leaf_idx_train[valid_mask]
                leaf_parent_idx = leaf_parent_idx[valid_mask]
                leaf_expansion = leaf_expansion[valid_mask]

        # if leaf_idx_train.numel() > 0:
        #     N = int(batch.pos.size(0))
        #     ptr = getattr(batch, "ptr", None)
        #     num_graphs = int(ptr.numel() - 1) if ptr is not None else None
        #     leaf_sample = leaf_idx_train[:20].tolist()
        #     leaf_max = int(leaf_idx_train.max().item())
        #     ptr_sample = ptr[:5].tolist() if ptr is not None else None
        #     unshifted = leaf_max < N
        #     logger.info(
        #         "[IndexDebug] N=%s, num_graphs=%s, leaf_idx_train[:20]=%s, leaf_idx_train.max()=%s, batch.ptr[:5]=%s, unshifted=%s",
        #         N,
        #         num_graphs,
        #         leaf_sample,
        #         leaf_max,
        #         ptr_sample,
        #         unshifted,
        #     )
        
        # --- relative position conformation matrix for new/train leaves
        leaf_rel_pos_global = leaf_rel_targets(pos_gt, leaf_idx_train, leaf_parent_idx)  # [L,3]

        # --- compute full geometry on P_0 (geo_lr + SO(2) angles + edge decomposition)
        _t_glr_0 = _t(pos_gt.device)
        uhat = model.uhat
        with th.no_grad():
            pre_geom_p0 = precompute_full_geometry(
                pos_gt, parent_idx, edge_index, uhat,
                debug=getattr(self, "debug", False),
            )
        # --- Convert targets to local frame for SO(2)-equivariant loss
        local_fwd = pre_geom_p0['local_forward']
        local_side = pre_geom_p0['local_sideways']
        if leaf_idx_train.numel() > 0:
            leaf_fwd = local_fwd[leaf_idx_train]     # [L, 3]
            leaf_side = local_side[leaf_idx_train]    # [L, 3]
            leaf_rel_pos = global_to_local(leaf_rel_pos_global, leaf_fwd, leaf_side, uhat)
        else:
            leaf_fwd = leaf_rel_pos_global.new_zeros((0, 3))
            leaf_side = leaf_rel_pos_global.new_zeros((0, 3))
            leaf_rel_pos = leaf_rel_pos_global
        _t_geo_lr_loss = _t(pos_gt.device) - _t_glr_0

        # --- prepare EGNN input (positions + minimal node features)
        feats_total = getattr(model, 'feats_dim', 0)
        tmd_hidden_dim = getattr(model, "tmd_hidden_dim", 0)
        cond_dim = getattr(self.diffusion, "cond_dim", 0) if self.diffusion is not None else 0
        avail_feats_dim = feats_total - cond_dim - tmd_hidden_dim
        if tmd_hidden_dim > 0 and avail_feats_dim < 5:
            raise ValueError("feats_dim - tmd_hidden_dim - cond_dim must be >= 5 when using TMD.")
        tmd = getattr(batch, "tmd", None)
        if tmd_hidden_dim > 0 and tmd is None:
            raise ValueError("Expected batch.tmd when tmd_hidden_dim > 0.")
        
        # if tmd is not None:
        #     tmd_cpu = tmd.detach().cpu()
        #     tmd_round = (tmd_cpu * 1000).round() / 1000
        #     uniq, counts = th.unique(tmd_round, dim=0, return_counts=True)
        #     logger.info(
        #         "[TMD Debug] unique=%d counts=%s",
        #         int(uniq.size(0)),
        #         counts.tolist(),
        #     )
        #     logger.info(
        #         "[Unique TMDs] %s",
        #         uniq.tolist(),
        #     )

        MAX_CHILDREN = 10  # one-hot ordinal dimension
        if avail_feats_dim > 0:
            N_nodes = pos_gt.size(0)
            is_leaf = pos_gt.new_zeros((N_nodes, 1))
            is_leaf[batch.leaf_idx] = 1.0
            features = [is_leaf]
            feats_used = 1

            # One-hot ordinal encoding (10D): child i lights up bit i
            if feats_used + MAX_CHILDREN <= avail_feats_dim:
                geo_ordinal = pre_geom_p0['geo_ordinal'].to(device=pos_gt.device, dtype=pos_gt.dtype)
                geo_idx = geo_ordinal.long().clamp(0, MAX_CHILDREN - 1)
                geo_onehot = pos_gt.new_zeros((N_nodes, MAX_CHILDREN))
                child_mask = geo_ordinal >= 0  # sentinel -1 → all-zeros
                if child_mask.any():
                    geo_onehot[child_mask] = geo_onehot[child_mask].scatter_(
                        1, geo_idx[child_mask].unsqueeze(-1), 1.0
                    )
                features.append(geo_onehot)
                feats_used += MAX_CHILDREN

            # Add indicator for nodes flagged as newly expanded leaves
            if feats_used < avail_feats_dim:
                new_mask = batch.new_leaf_mask_from_next
                if isinstance(new_mask, th.Tensor):
                    new_mask_tensor = new_mask.to(pos_gt.device, dtype=pos_gt.dtype)
                else:
                    new_mask_tensor = pos_gt.new_tensor(new_mask, dtype=pos_gt.dtype)
                new_mask_tensor = new_mask_tensor.view(-1)
                if new_mask_tensor.numel() != N_nodes:
                    aligned = pos_gt.new_zeros(N_nodes)
                    count = min(new_mask_tensor.numel(), N_nodes)
                    if count > 0:
                        aligned[:count] = new_mask_tensor[:count]
                    new_mask_tensor = aligned
                features.append(new_mask_tensor.unsqueeze(-1))
                feats_used += 1

            # Graph size ratio feature (current nodes / total_tree_size), broadcast per node
            if self.use_size_ratio and feats_used < avail_feats_dim:
                size_ratio = size_ratio_feature_from_batch(
                    batch=batch,
                    device=pos_gt.device,
                    dtype=pos_gt.dtype,
                )
                if size_ratio is not None:
                    features.append(size_ratio)
                    feats_used += 1

            # Fill remaining dimensions with zeros if needed
            if feats_used < avail_feats_dim:
                extra = pos_gt.new_zeros((N_nodes, avail_feats_dim - feats_used))
                features.append(extra)

            node_feats = th.cat(features, dim=-1)
        else:
            node_feats = pos_gt.new_zeros((pos_gt.size(0), 0))

        if edge_types.numel():
            edge_attr = edge_types.unsqueeze(-1).to(pos_gt.dtype)
        else:
            edge_attr = pos_gt.new_zeros((0, 1))

        if self.diffusion is None:
            raise ValueError("Diffusion module must be provided for Expansion training.")

        # plot_diffusion_debug_trees(
        #     pos=pos_gt,
        #     parent_idx=parent_idx,
        #     batch_vec=batch.batch,
        #     leaf_idx_all=leaf_idx_all,
        #     leaf_idx_train=leaf_idx_train,
        #     geo_lr_mask=geo_lr_mask,
        #     leaf_targets_per_node=leaf_targets_per_node,
        # )

        _t_diff_loss_0 = _t(pos_gt.device)
        expansion_loss, position_loss = self.diffusion(
            node_feats=node_feats,
            edge_index=edge_index,
            batch=batch.batch,
            edge_attr=edge_attr,
            P_0=pos_gt,
            C_0=leaf_rel_pos,
            parent_idx=parent_idx,
            leaf_idx_train=leaf_idx_train,
            leaf_expansion=leaf_expansion,
            leaf_parent_idx=leaf_parent_idx,
            model=model,
            tmd=tmd,
            pre_geom_p0=pre_geom_p0,
            local_forward=leaf_fwd,
            local_sideways=leaf_side,
            uhat=uhat,
        )
        _t_diff_loss = _t(pos_gt.device) - _t_diff_loss_0
        # logger.info(
        #     "[get_loss N=%d L=%d] geo_lr_mask=%.4fs diffusion_forward=%.4fs",
        #     int(pos_gt.size(0)), int(leaf_idx_train.numel()),
        #     _t_geo_lr_loss, _t_diff_loss,
        # )

        loss = position_loss + self.expansion_loss_weight * expansion_loss
        metrics = {
            "leaf_pos_loss": float(position_loss.item()),
            "leaf_expansion_loss": float(expansion_loss.item()),
            "cumulative_loss": float(loss.item()),
            "num_leaves": int(leaf_idx_train.numel()),
            "num_total_leaves": int(leaf_idx_all.numel()),
        }
        return loss, metrics
