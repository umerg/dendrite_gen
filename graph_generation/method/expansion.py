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
    compute_geo_lr_mask,
    decode_parent_indices,
    leaf_rel_targets,
    plot_diffusion_debug_trees,
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
        deterministic_expansion: bool = False,      # just sets seeding for reproducibility
        red_threshold: int = 0,
        expansion_loss_weight: float = 1.0,
        use_size_ratio: bool = True,
    ):
        super().__init__(diffusion=diffusion)
        self.deterministic_expansion = deterministic_expansion
        self.red_threshold = red_threshold
        self.expansion_loss_weight = float(expansion_loss_weight)
        self.use_size_ratio = use_size_ratio
    
    def sample_graphs(self, target_size: th.Tensor, model: Module, tmd: th.Tensor | None = None):
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
        geo_lr_assign = th.full((num_graphs,), -1, device=device, dtype=th.long)
        leaf_mask = th.ones((num_graphs,), device=device, dtype=th.bool)

        max_steps = int(target_size.max().item() * 2)
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
                geo_lr_assign,
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
                geo_lr_assign=geo_lr_assign,
                leaf_mask=leaf_mask,
                tmd=tmd,
                step=step,
                ensure_progress=False,
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
        geo_lr_assign: th.Tensor | None = None,
        tmd: th.Tensor | None = None,
        step: int = 0,
        ensure_progress: bool = False,
        map_threshold: float = 0.0,
    ):
        """Expand graphs by one generation step using binary leaf branching.
        """

        if not all(x is not None for x in (pos, leaf_idx, leaf_expansion, parent_idx_1b)):
            raise ValueError("expand requires pos, leaf_idx, leaf_expansion, parent_idx_1b tensors.")
        if self.diffusion is None:
            raise ValueError("Diffusion module must be provided for sampling.")

        device = pos.device
        _t_start = _t(device)
        _t_leaf_loop = _t_cat_growth = _t_sparse_rebuild = _t_diffusion_sample = _t_geo_lr_mask = 0.0
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

        if geo_lr_assign is None:
            geo_lr_assign = th.full((pos.size(0),), -1, device=device, dtype=th.long)

        if leaf_idx.numel() == 0:  # (remaining_capacity <= 0).all() or 
            return (
                adj_reduced,
                pos,
                leaf_idx.clone(),
                leaf_expansion.clone(),
                parent_idx_1b,
                batch_reduced,
                geo_lr_assign.clone(),
                leaf_mask.clone(),
                True,
            )

        spawn_counts = (leaf_expansion == 2).long() * 2
        leaf_batch = batch_reduced[leaf_idx]
        spawn_counts_final = spawn_counts.clone()

        # Root nodes are non-binary: always spawn exactly 1 child
        is_root_leaf = parent_idx[leaf_idx] < 0
        if is_root_leaf.any():
            root_should_spawn = (target_size[leaf_batch] > 1).long()
            spawn_counts_final = th.where(is_root_leaf, root_should_spawn, spawn_counts_final)

        # Deterministic capacity cut-off (DISABLED)
        
        # for g in range(num_graphs):
        #     cap = int(remaining_capacity[g].item())
        #     if cap < 2:
        #         spawn_counts_final[leaf_batch == g] = 0
        #         continue
        #     mask_g = leaf_batch == g
        #     expanders = th.nonzero((spawn_counts_final == 2) & mask_g, as_tuple=False).flatten()
        #     needed = expanders.numel() * 2
        #     if needed <= cap:
        #         continue
        #     max_leaves = cap // 2
        #     if max_leaves <= 0:
        #         spawn_counts_final[expanders] = 0
        #         continue
        #     if self.deterministic_expansion:
        #         generator = th.Generator(device=expanders.device)
        #         generator.manual_seed(g * 10007 + step)
        #         perm = th.randperm(expanders.numel(), generator=generator, device=expanders.device)
        #     else:
        #         perm = th.randperm(expanders.numel(), device=expanders.device)
        #     disable = expanders[perm[max_leaves:]]
        #     spawn_counts_final[disable] = 0

        if ensure_progress and (remaining_capacity >= 2).any():
            for g in range(num_graphs):
                if remaining_capacity[g] < 2:
                    continue
                mask_g = leaf_batch == g
                if not mask_g.any():
                    continue
                if (spawn_counts_final[mask_g] == 2).any():
                    continue
                leaf_indices_g = th.nonzero(mask_g, as_tuple=False).flatten()
                if self.deterministic_expansion:
                    forced = leaf_indices_g[0]
                else:
                    rand_idx = th.randint(0, leaf_indices_g.numel(), (1,), device=leaf_indices_g.device)
                    forced = leaf_indices_g[rand_idx]
                spawn_counts_final[forced] = 2

        total_new_children = int(spawn_counts_final.sum().item())
        if total_new_children == 0:
            return (
                adj_reduced,
                pos,
                leaf_idx.clone(),
                leaf_expansion.clone(),
                parent_idx_1b,
                batch_reduced,
                geo_lr_assign.clone(),
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
        lr_assign_new = []
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
                lr_assign_new.append(0 if local_child == 0 else 1)
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
        geo_lr_assign_next = th.cat(
            [geo_lr_assign, th.tensor(lr_assign_new, device=device, dtype=th.long)], dim=0
        )

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
                geo_lr_assign_next,
                leaf_mask_next,
                True,
            )

        node_counts_per_graph = scatter(
            th.ones_like(batch_new, dtype=target_size.dtype),
            batch_new,
            dim=0,
            dim_size=num_graphs,
        )

        feats_total = getattr(model, "feats_dim", 0)
        tmd_hidden_dim = getattr(model, "tmd_hidden_dim", 0)
        cond_dim = getattr(self.diffusion, "cond_dim", 0)
        avail_feats_dim = feats_total - cond_dim - tmd_hidden_dim
        if tmd_hidden_dim > 0 and avail_feats_dim < 5:
            raise ValueError("feats_dim - tmd_hidden_dim - cond_dim must be >= 5 when using TMD.")
        if avail_feats_dim > 0:
            features = []
            feats_used = 0
            is_leaf = leaf_mask_next.to(dtype=pos_new.dtype).unsqueeze(-1)
            features.append(is_leaf)
            feats_used += 1

            if feats_used < avail_feats_dim:
                geo_feat = pos_new.new_zeros((pos_new.size(0), 1))
                mask = geo_lr_assign_next >= 0
                if mask.any():
                    geo_feat[mask] = (geo_lr_assign_next[mask] == 0).to(pos_new.dtype).unsqueeze(-1)
                features.append(geo_feat)
                feats_used += 1

            if feats_used < avail_feats_dim:
                new_flag = pos_new.new_zeros((pos_new.size(0), 1))
                new_flag[leaf_idx_next] = 1.0
                features.append(new_flag)
                feats_used += 1

            if self.use_size_ratio and feats_used < avail_feats_dim:
                ratio_graph = node_counts_per_graph.to(pos_new.dtype) / target_size.to(pos_new.dtype).clamp_min(1.0)
                ratio_nodes = ratio_graph[batch_new].unsqueeze(-1)
                features.append(ratio_nodes)
                feats_used += 1

            if feats_used < avail_feats_dim:
                pad = pos_new.new_zeros((pos_new.size(0), avail_feats_dim - feats_used))
                features.append(pad)

            node_feats = th.cat(features, dim=-1)
        else:
            node_feats = pos_new.new_zeros((pos_new.size(0), 0))

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
        )

        _t_diffusion_sample = _t(device) - _t_diff_0
        parent_pos_for_children = pos_new[leaf_parent_idx_next]
        pos_new[leaf_idx_next] = parent_pos_for_children + rel_pred

        _t_geo_0 = _t(device)
        if leaf_idx_next.numel() > 0:
            geo_lr_mask = compute_geo_lr_mask(pos_new, parent_idx_new_0b, debug=getattr(self, "debug", False))
            parent_new = parent_idx_new_0b[leaf_idx_next]
            counts = scatter(
                th.ones_like(parent_new),
                parent_new,
                dim=0,
                dim_size=pos_new.size(0),
            )
            valid = counts[parent_new] == 2
            if valid.any():
                geo_left = geo_lr_mask[leaf_idx_next][valid]
                geo_lr_assign_next = geo_lr_assign_next.clone()
                geo_lr_assign_next[leaf_idx_next[valid]] = (~geo_left).to(
                    dtype=geo_lr_assign_next.dtype
                )

        _t_geo_lr_mask = _t(device) - _t_geo_0
        logger.info(
            "[expand step=%d N=%d L=%d] leaf_loop=%.4fs cat_growth=%.4fs "
            "sparse_rebuild=%.4fs diffusion_sample=%.4fs geo_lr_mask=%.4fs total=%.4fs",
            step, pos_new.size(0), int(leaf_idx_next.numel()),
            _t_leaf_loop, _t_cat_growth, _t_sparse_rebuild,
            _t_diffusion_sample, _t_geo_lr_mask, _t(device) - _t_start,
        )

        if exp_pred.dim() == 1:
            exp_pred = exp_pred.unsqueeze(-1)
        expansion_score = exp_pred.squeeze(-1)
        leaf_expansion_next = (expansion_score > map_threshold).long() + 1

        remaining_capacity_new = target_size.to(device) - node_counts_per_graph
        terminated = leaf_idx_next.numel() == 0 # (remaining_capacity_new < 2).all() or

        return (
            adj_new,
            pos_new,
            leaf_idx_next,
            leaf_expansion_next,
            parent_idx_1b_new,
            batch_new,
            geo_lr_assign_next,
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

        leaf_graphs_all = batch.batch[leaf_idx_all]
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
        leaf_rel_pos = leaf_rel_targets(pos_gt, leaf_idx_train, leaf_parent_idx)  # [L,3]

        # --- compute geometric left/right mask for siblings
        _t_glr_0 = _t(pos_gt.device)
        geo_lr_mask = compute_geo_lr_mask(pos_gt, parent_idx, debug=getattr(self, "debug", False))
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

        if avail_feats_dim > 0:
            is_leaf = pos_gt.new_zeros((pos_gt.size(0), 1))
            is_leaf[batch.leaf_idx] = 1.0
            features = [is_leaf]
            feats_used = 1

            # Geometry-derived left/right bit for siblings
            if feats_used < avail_feats_dim:
                geo_left = geo_lr_mask.to(device=pos_gt.device, dtype=pos_gt.dtype).unsqueeze(-1)
                features.append(geo_left)
                feats_used += 1

            # Add indicator for nodes flagged as newly expanded leaves (when provided)
            if hasattr(batch, "new_leaf_mask_from_next") and feats_used < avail_feats_dim:
                new_mask = batch.new_leaf_mask_from_next
                if isinstance(new_mask, th.Tensor):
                    new_mask_tensor = new_mask.to(pos_gt.device, dtype=pos_gt.dtype)
                else:
                    new_mask_tensor = pos_gt.new_tensor(new_mask, dtype=pos_gt.dtype)
                new_mask_tensor = new_mask_tensor.view(-1)
                if new_mask_tensor.numel() != pos_gt.size(0):
                    aligned = pos_gt.new_zeros(pos_gt.size(0))
                    count = min(new_mask_tensor.numel(), pos_gt.size(0))
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
                extra = pos_gt.new_zeros((pos_gt.size(0), avail_feats_dim - feats_used))
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
        )
        _t_diff_loss = _t(pos_gt.device) - _t_diff_loss_0
        logger.info(
            "[get_loss N=%d L=%d] geo_lr_mask=%.4fs diffusion_forward=%.4fs",
            int(pos_gt.size(0)), int(leaf_idx_train.numel()),
            _t_geo_lr_loss, _t_diff_loss,
        )

        loss = position_loss + self.expansion_loss_weight * expansion_loss
        metrics = {
            "leaf_pos_loss": float(position_loss.item()),
            "leaf_expansion_loss": float(expansion_loss.item()),
            "cumulative_loss": float(loss.item()),
            "num_leaves": int(leaf_idx_train.numel()),
            "num_total_leaves": int(leaf_idx_all.numel()),
        }
        return loss, metrics
