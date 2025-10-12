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
    def get_loss(self, batch, model: th.nn.Module):
        """
        Expected batch fields:
          - adj: SparseTensor [N×N] (undirected ok)
          - pos: Float [N,3] (absolute GT)
          - leaf_idx: Long [L]
          - leaf_parent_idx: Long [L]  (>=0, already batched-shifted)
          - batch: Long [N]  (PyG batch vector)
          - reduction_level, target_size (not used here)
        Assumes `model(x, edge_index, batch, edge_attr=None, ...) -> [N,3]`
        producing RELATIVE offsets per node.
        """

        # --- sanity
        assert (batch.leaf_parent_idx >= 0).all(), "Found -1 in leaf_parent_idx (unexpected root-parent leaf)."

        # --- graph and positions
        pos_gt = batch.pos                             # [N,3] (absolute, untouched)
        edge_index, _ = to_edge_index(batch.adj)       # (2,E)

        # --- build noisy-masked input positions (parents anchored)
        pos_in = self._make_masked_positions(
            pos=pos_gt,
            leaf_idx=batch.leaf_idx,
            leaf_parent_idx=batch.leaf_parent_idx,
            sigma=self.leaf_noise_sigma,
            clip=self.leaf_noise_clip,
        )                                              # [N,3]

        # --- prepare EGNN input (positions + optional trivial features)
        x_in = pos_in
        # Try to detect expected feats_dim from model (EGNN_Sparse_Network has attr feats_dim & pos_dim)
        feats_dim = getattr(model, 'feats_dim', None)
        pos_dim   = getattr(model, 'pos_dim', 3)
        if feats_dim is not None:
            if feats_dim > 0:
                # Append zero feature matrix (optionally could encode is_leaf)
                # shape: [N, feats_dim]
                zero_feats = pos_in.new_zeros((pos_in.size(0), feats_dim))
                # Example optional feature (commented): is_leaf flag
                # is_leaf = pos_in.new_zeros((pos_in.size(0), 1))
                # is_leaf[batch.leaf_idx] = 1.0
                # zero_feats[:, :1] = is_leaf.squeeze(-1)
                x_in = th.cat([pos_in, zero_feats], dim=-1)  # [N, pos_dim + feats_dim]
            else:
                # Ensure we only pass positions if feats_dim == 0
                x_in = pos_in[:, :pos_dim]

        # --- run the EGNN (expects x with positions first and optional features after)
        pred_rel_all = model(
            x=x_in,
            edge_index=edge_index,
            batch=batch.batch,
            edge_attr=None,
        )  # [N, pos_dim]

        # --- gather predictions and targets for leaves only
        leaf_idx = batch.leaf_idx
        pred_rel = pred_rel_all[leaf_idx]                           # [L,3]
        tgt_rel  = self._leaf_rel_targets(pos_gt, leaf_idx, batch.leaf_parent_idx)  # [L,3]

        # --- loss
        if pred_rel.numel() == 0:
            loss = pred_rel_all.sum() * 0.0
            metrics = {"leaf_pos_loss": 0.0, "num_leaves": 0}
            return loss, metrics

        loss = F.mse_loss(pred_rel, tgt_rel)

        # (optional) monitor absolute predictions for leaves
        with th.no_grad():
            parent_pos_in = pos_in[batch.leaf_parent_idx]          # [L,3]
            abs_pred = parent_pos_in + pred_rel                    # [L,3] (for debugging)

        metrics = {
            "leaf_pos_loss": float(loss.item()),
            "num_leaves": int(leaf_idx.numel()),
            # "abs_pred_mean_norm": float(abs_pred.norm(dim=-1).mean().item()),
        }
        return loss, metrics
