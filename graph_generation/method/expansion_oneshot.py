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
        deterministic_expansion=False,
        min_red_frac=0.0,
        max_red_frac=0.5,
        red_threshold=0,
        leaf_noise_sigma=0.05,           # <-- stddev of Gaussian around parent (same units as pos)
        leaf_noise_clip=None,            # <-- optional radius clamp (float) or None
    ):
        super().__init__(diffusion=None)
        self.deterministic_expansion = deterministic_expansion
        self.min_red_frac = min_red_frac
        self.max_red_frac = max_red_frac
        self.red_threshold = red_threshold

        self.leaf_noise_sigma = float(leaf_noise_sigma)
        self.leaf_noise_clip = float(leaf_noise_clip) if leaf_noise_clip is not None else None

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
        noise = th.randn_like(parent_pos) * sigma            # isotropic
        if clip is not None and clip > 0:
            # project noise to a ball of radius=clip (keeps density reasonable)
            nrm = noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            scale = th.minimum(th.ones_like(nrm), clip / nrm)
            noise = noise * scale

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
    def sample_graphs(self, target_size: th.Tensor, model: Module):
        raise NotImplementedError("Expansion_OneShot.sample_graphs is not supported yet.")

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
            raise ValueError("Network must return a dict with 'rel_pred' and 'expansion_red'.")

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
        leaf_expansion_loss = F.mse_loss(pred_expansion, leaf_expansion.unsqueeze(-1))
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
