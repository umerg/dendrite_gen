import networkx as nx
import torch as th
from torch.nn import Module
import torch.nn.functional as F
from torch_scatter import scatter
from torch_sparse import SparseTensor
import logging

logger = logging.getLogger(__name__)

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
        leaf_noise_sigma: float = 0.05,             # <-- stddev of Gaussian around parent (same units as pos)
        leaf_noise_clip: float | None = None,       # <-- optional radius clamp (float) or None
    ):
        super().__init__(diffusion=diffusion)
        self.deterministic_expansion = deterministic_expansion
        self.red_threshold = red_threshold
        self.leaf_noise_sigma = float(leaf_noise_sigma)
        self.leaf_noise_clip = leaf_noise_clip
    
    def sample_graphs(self, target_size: th.Tensor, model: Module):
        """Generate a batch of graphs starting from one root node per graph.
        """
        pass

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
        """

        pass

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

        # --- parent indices (1-based in Data for safe batching) -> shift back to 0-based with -1 for roots
        if not hasattr(batch, "parent_idx_1b"):
            raise ValueError("Expected batch.parent_idx_1b (1-based parent indices). Please update dataloader.")
        parent_idx = self._decode_parent_indices(batch)           # [N], -1 for roots

        # --- graph and positions
        pos_gt = batch.pos                             # [N,3] (absolute, untouched)
        edge_index, edge_types = self._build_directed_edge_index(parent_idx)

        # --- tracking of leaves
        if not hasattr(batch, "leaf_idx"):
            raise ValueError("Expected batch.leaf_idx (leaf node indices). Please update dataloader.")
        leaf_idx_all = batch.leaf_idx

        # --- expansion state for leaves
        if not hasattr(batch, "leaf_expansion"):
            raise ValueError("Expected batch.leaf_expansion (leaf expansion states). Please update dataloader.")
        leaf_expansion_all = batch.leaf_expansion - 1       # [L_total] in {0,1}

        leaf_idx_train = self._select_training_leaf_indices(batch)
        if leaf_idx_train.numel() == 0:
            leaf_parent_idx = parent_idx.new_empty((0,), dtype=parent_idx.dtype)
        else:
            leaf_parent_idx = parent_idx[leaf_idx_train]
            assert (leaf_parent_idx >= 0).all(), "Leaf with no valid parent encountered."

        # map per-node expansion labels so new leaves can be indexed directly
        leaf_targets_per_node = leaf_expansion_all.new_full((pos_gt.size(0),), -1)
        if leaf_idx_all.numel() > 0:
            leaf_targets_per_node[leaf_idx_all] = leaf_expansion_all.view(-1)
        leaf_expansion = leaf_targets_per_node[leaf_idx_train]
        if leaf_expansion.numel() > 0:
            valid_mask = leaf_expansion >= 0
            if not valid_mask.all():
                leaf_idx_train = leaf_idx_train[valid_mask]
                leaf_parent_idx = leaf_parent_idx[valid_mask]
                leaf_expansion = leaf_expansion[valid_mask]
        
        # --- relative position conformation matrix for new/train leaves 
        leaf_rel_pos = self._leaf_rel_targets(pos_gt, leaf_idx_train, leaf_parent_idx)  # [L,3]

        # --- compute geometric left/right mask for siblings
        geo_lr_mask = self._compute_geo_lr_mask(pos_gt, parent_idx)
         
        # --- prepare EGNN input (positions + minimal node features)
        feats_dim = getattr(model, 'feats_dim', 0)
        pos_dim   = getattr(model, 'pos_dim', 3)

        if feats_dim > 0:
            is_leaf = pos_gt.new_zeros((pos_gt.size(0), 1))
            is_leaf[batch.leaf_idx] = 1.0
            features = [is_leaf]
            feats_used = 1

            # Geometry-derived left/right bit for siblings
            if feats_used < feats_dim:
                geo_left = geo_lr_mask.to(device=pos_gt.device, dtype=pos_gt.dtype).unsqueeze(-1)
                features.append(geo_left)
                feats_used += 1

            # Add indicator for nodes flagged as newly expanded leaves (when provided)
            if hasattr(batch, "new_leaf_mask_from_next") and feats_used < feats_dim:
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

            # Graph size ratio feature (current nodes / target nodes), broadcast per node
            if feats_used < feats_dim:
                size_ratio = self._size_ratio_feature_from_batch(
                    batch=batch,
                    device=pos_gt.device,
                    dtype=pos_gt.dtype,
                )
                if size_ratio is not None:
                    features.append(size_ratio)
                    feats_used += 1

            # Fill remaining dimensions with zeros if needed
            if feats_used < feats_dim:
                extra = pos_gt.new_zeros((pos_gt.size(0), feats_dim - feats_used))
                features.append(extra)
            
            node_feats = th.cat(features, dim=-1)
        else:
            node_feats = None

        if edge_types.numel():
            edge_attr = edge_types.unsqueeze(-1).to(node_feats.dtype)
        else:
            edge_attr = node_feats.new_zeros((0, 1))

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
        )

        metrics = {
            "leaf_pos_loss": float(leaf_pos_loss.item()),
            "leaf_expansion_loss": float(leaf_expansion_loss.item()),
            "cumulative_loss": float(loss.item()),
        }
        return loss, metrics
