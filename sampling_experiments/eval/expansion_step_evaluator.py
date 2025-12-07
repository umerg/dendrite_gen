"""Step-by-step evaluator mirroring Expansion_OneShot masking logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.nn import Module

from graph_generation.method.expansion_oneshot import Expansion_OneShot


@dataclass
class StepEvalRecord:
    """Prediction + metadata bundle for one reduced graph."""

    sequence_idx: int | None
    graph_batch_idx: int
    step_idx: int | None
    reduction_level: int | None
    total_tree_size: int | None
    num_nodes: int
    num_leaves: int
    leaf_local_indices: List[int]
    parent_local_indices: List[int]
    leaf_global_indices: List[int]
    parent_global_indices: List[int]
    gt_leaf_positions: np.ndarray
    masked_leaf_positions: np.ndarray
    gt_parent_positions: np.ndarray
    predicted_rel: np.ndarray
    predicted_abs: np.ndarray
    gt_rel: np.ndarray
    expansion_logits: np.ndarray
    expansion_probs: np.ndarray
    gt_expansion_labels: np.ndarray
    sibling_order_leaf: np.ndarray | None
    metadata: Dict[str, Any] = field(default_factory=dict)
    losses: Dict[str, float] = field(default_factory=dict)


class ExpansionStepEvaluator(Expansion_OneShot):
    """Subclass of Expansion_OneShot that exposes per-step predictions."""

    def _construct_model_inputs(self, batch, pos_in: th.Tensor, edge_types: th.Tensor, model: Module):
        feats_dim = getattr(model, "feats_dim", 0)
        pos_dim = getattr(model, "pos_dim", 3)

        if feats_dim > 0:
            is_leaf = pos_in.new_zeros((pos_in.size(0), 1))
            is_leaf[batch.leaf_idx] = 1.0
            features = [is_leaf]
            feats_used = 1

            if hasattr(batch, "sibling_order") and feats_used < feats_dim:
                so = batch.sibling_order.to(pos_in.device)
                sib_is_left = (so == 0).float().unsqueeze(-1)
                sib_is_left = th.where(so.unsqueeze(-1) >= 0, sib_is_left, th.zeros_like(sib_is_left))
                features.append(sib_is_left)
                feats_used += 1

            if hasattr(batch, "new_leaf_mask_from_next") and feats_used < feats_dim:
                new_mask = batch.new_leaf_mask_from_next
                if isinstance(new_mask, th.Tensor):
                    new_mask_tensor = new_mask.to(pos_in.device, dtype=pos_in.dtype)
                else:
                    new_mask_tensor = pos_in.new_tensor(new_mask, dtype=pos_in.dtype)
                new_mask_tensor = new_mask_tensor.view(-1)
                if new_mask_tensor.numel() != pos_in.size(0):
                    aligned = pos_in.new_zeros(pos_in.size(0))
                    count = min(new_mask_tensor.numel(), pos_in.size(0))
                    if count > 0:
                        aligned[:count] = new_mask_tensor[:count]
                    new_mask_tensor = aligned
                features.append(new_mask_tensor.unsqueeze(-1))
                feats_used += 1

            # if feats_used < feats_dim:
            #     size_ratio = self._size_ratio_feature_from_batch(
            #         batch=batch,
            #         device=pos_in.device,
            #         dtype=pos_in.dtype,
            #     )
            #     if size_ratio is not None:
            #         features.append(size_ratio)
            #         feats_used += 1

            if feats_used < feats_dim:
                extra = pos_in.new_zeros((pos_in.size(0), feats_dim - feats_used))
                features.append(extra)

            node_feats = th.cat(features, dim=-1)
            x_in = th.cat([pos_in, node_feats], dim=-1)
        else:
            x_in = pos_in[:, :pos_dim]

        if edge_types.numel():
            edge_attr = edge_types.unsqueeze(-1).to(x_in.dtype)
        else:
            edge_attr = x_in.new_zeros((0, 1))
        return x_in, edge_attr

    def _compute_graph_losses(
        self,
        *,
        pred_rel: th.Tensor,
        tgt_rel: th.Tensor,
        pred_exp: th.Tensor,
        leaf_exp: th.Tensor,
        leaf_parent_idx: th.Tensor,
        pos_gt_leaves: th.Tensor,
        pos_in: th.Tensor,
    ) -> Dict[str, float]:
        if pred_rel.numel() == 0:
            return {
                "leaf_pos_loss": 0.0,
                "leaf_expansion_loss": 0.0,
                "sibling_dist_loss": 0.0,
                "total_loss": 0.0,
            }

        if self.use_sibling_matching:
            leaf_pos_loss = self._compute_leaf_pos_loss_with_matching(
                pred_rel=pred_rel,
                tgt_rel=tgt_rel,
                leaf_parent_idx=leaf_parent_idx,
            )
        else:
            leaf_pos_loss = F.mse_loss(pred_rel, tgt_rel)

        if pred_exp.dim() == 1:
            pred_exp = pred_exp.unsqueeze(-1)
        if leaf_exp.dim() == 1:
            leaf_exp = leaf_exp.unsqueeze(-1)
        leaf_expansion_loss = F.binary_cross_entropy_with_logits(
            pred_exp.float(),
            leaf_exp.float(),
        )

        sibling_dist_loss = pred_rel.sum() * 0.0
        if self.sibling_loss_weight > 0.0 and pred_rel.size(0) > 1:
            parent_pos_in_all = pos_in[leaf_parent_idx]
            abs_pred_all = parent_pos_in_all + pred_rel
            unique_parents, counts = th.unique(leaf_parent_idx, return_counts=True)
            mask_multi = counts >= 2
            if mask_multi.any():
                pair_indices = []
                for parent in unique_parents[mask_multi]:
                    leaf_pos_for_parent = (leaf_parent_idx == parent).nonzero(as_tuple=False).flatten()
                    if leaf_pos_for_parent.numel() < 2:
                        continue
                    pair_indices.append(leaf_pos_for_parent[:2])
                if pair_indices:
                    pair_indices = th.stack(pair_indices, dim=0)
                    idx1 = pair_indices[:, 0]
                    idx2 = pair_indices[:, 1]
                    v_gt = pos_gt_leaves[idx2] - pos_gt_leaves[idx1]
                    v_pred = abs_pred_all[idx2] - abs_pred_all[idx1]
                    d_gt = v_gt.norm(dim=-1)
                    d_pred = v_pred.norm(dim=-1)
                    sibling_dist_loss = F.mse_loss(d_pred, d_gt)

        total = leaf_pos_loss + leaf_expansion_loss + self.sibling_loss_weight * sibling_dist_loss
        return {
            "leaf_pos_loss": float(leaf_pos_loss.item()),
            "leaf_expansion_loss": float(leaf_expansion_loss.item()),
            "sibling_dist_loss": float(sibling_dist_loss.item()),
            "total_loss": float(total.item()),
        }

    def collect_step_predictions(self, batch, model: Module) -> List[StepEvalRecord]:
        """Run a forward pass and return per-graph prediction bundles."""
        if not hasattr(batch, "parent_idx_1b"):
            raise ValueError("Batch is missing parent_idx_1b required for evaluation.")
        if not hasattr(batch, "leaf_idx"):
            raise ValueError("Batch is missing leaf_idx required for evaluation.")
        if not hasattr(batch, "leaf_expansion"):
            raise ValueError("Batch is missing leaf_expansion required for evaluation.")

        parent_idx = batch.parent_idx_1b - 1
        pos_gt = batch.pos
        edge_index, edge_types = self._build_directed_edge_index(parent_idx)
        leaf_idx = batch.leaf_idx
        leaf_expansion = batch.leaf_expansion - 1
        leaf_parent_idx = parent_idx[leaf_idx]

        pos_in = self._make_masked_positions(
            pos=pos_gt,
            leaf_idx=leaf_idx,
            leaf_parent_idx=leaf_parent_idx,
            sigma=self.leaf_noise_sigma,
            clip=self.leaf_noise_clip,
        )

        x_in, edge_attr = self._construct_model_inputs(batch, pos_in, edge_types, model)
        out = model(
            x=x_in,
            edge_index=edge_index,
            batch=batch.batch,
            edge_attr=edge_attr,
            parent_idx=parent_idx,
        )
        if not isinstance(out, dict):
            raise ValueError("Model must return a dict with 'rel_pred' and 'expansion_pred'.")

        pred_rel_all = out["rel_pred"]
        pred_expansion_all = out["expansion_pred"]

        leaf_pred_rel = pred_rel_all[leaf_idx]
        leaf_pred_expansion = pred_expansion_all[leaf_idx]
        tgt_rel = self._leaf_rel_targets(pos_gt, leaf_idx, leaf_parent_idx)
        parent_pos_in = pos_in[leaf_parent_idx]
        abs_pred = parent_pos_in + leaf_pred_rel

        num_graphs = int(batch.num_graphs) if hasattr(batch, "num_graphs") else int(batch.batch.max().item()) + 1
        batch_vec = batch.batch
        leaf_batch = batch_vec[leaf_idx]

        step_idx_attr = getattr(batch, "step_idx", None)
        sequence_idx_attr = getattr(batch, "sequence_id", None)
        reduction_level_attr = getattr(batch, "reduction_level", None)
        total_tree_attr = getattr(batch, "total_tree_size", None)
        target_size_attr = getattr(batch, "target_size", None)

        sibling_order_attr = getattr(batch, "sibling_order", None)
        new_leaf_idx_attr = getattr(batch, "new_leaf_idx_from_next", None)
        new_leaf_mask_attr = getattr(batch, "new_leaf_mask_from_next", None)

        records: List[StepEvalRecord] = []

        for graph_idx in range(num_graphs):
            node_mask = batch_vec == graph_idx
            node_indices = th.nonzero(node_mask, as_tuple=False).flatten()
            if node_indices.numel() == 0:
                continue
            node_ids = node_indices.tolist()
            local_map = {int(g_idx): i for i, g_idx in enumerate(node_ids)}

            leaf_mask = leaf_batch == graph_idx
            leaf_indices = leaf_idx[leaf_mask]
            parent_indices = leaf_parent_idx[leaf_mask]

            def _to_numpy(tensor: th.Tensor) -> np.ndarray:
                if tensor.numel() == 0:
                    shape = (0,) + tuple(tensor.shape[1:])
                    return np.zeros(shape, dtype=np.float32)
                return tensor.detach().cpu().numpy()

            metadata: Dict[str, Any] = {
                "node_global_ids": [int(i) for i in node_ids],
                "target_size": int(target_size_attr[graph_idx].item()) if target_size_attr is not None else None,
            }

            if new_leaf_idx_attr is not None and new_leaf_idx_attr.numel() > 0:
                new_leaf_ids = []
                for idx in new_leaf_idx_attr.tolist():
                    if int(batch_vec[idx].item()) == graph_idx:
                        new_leaf_ids.append(int(idx))
                metadata["new_leaf_global_indices"] = new_leaf_ids
                metadata["new_leaf_local_indices"] = [local_map[i] for i in new_leaf_ids] if new_leaf_ids else []

            if new_leaf_mask_attr is not None:
                mask_local = new_leaf_mask_attr[node_mask]
                metadata["new_leaf_mask_local"] = mask_local.detach().cpu().numpy()

            sibling_order_leaf = None
            if sibling_order_attr is not None:
                sibling_order_leaf = sibling_order_attr[leaf_indices].detach().cpu().numpy()

            leaf_local = [local_map[int(idx)] for idx in leaf_indices.tolist()]
            parent_local = [local_map[int(idx)] for idx in parent_indices.tolist()]

            gt_leaf_pos = pos_gt[leaf_indices]
            masked_leaf_pos = pos_in[leaf_indices]
            gt_parent_pos = pos_gt[parent_indices] if parent_indices.numel() > 0 else gt_leaf_pos.new_zeros((0, 3))

            pred_rel = leaf_pred_rel[leaf_mask]
            pred_abs = abs_pred[leaf_mask]
            gt_rel_graph = tgt_rel[leaf_mask]
            pred_exp_logits = leaf_pred_expansion[leaf_mask]
            pred_probs = th.sigmoid(pred_exp_logits)
            gt_expansion_targets = leaf_expansion[leaf_mask]
            raw_leaf_labels = batch.leaf_expansion[leaf_mask]
            metadata["raw_leaf_expansion"] = raw_leaf_labels.detach().cpu().view(-1).numpy() if raw_leaf_labels.numel() else []

            losses = self._compute_graph_losses(
                pred_rel=pred_rel,
                tgt_rel=gt_rel_graph,
                pred_exp=pred_exp_logits,
                leaf_exp=leaf_expansion[leaf_mask],
                leaf_parent_idx=parent_indices,
                pos_gt_leaves=gt_leaf_pos,
                pos_in=pos_in,
            )

            record = StepEvalRecord(
                sequence_idx=int(sequence_idx_attr[graph_idx].item()) if sequence_idx_attr is not None else None,
                graph_batch_idx=graph_idx,
                step_idx=int(step_idx_attr[graph_idx].item()) if step_idx_attr is not None else None,
                reduction_level=int(reduction_level_attr[graph_idx].item()) if reduction_level_attr is not None else None,
                total_tree_size=int(total_tree_attr[graph_idx].item()) if total_tree_attr is not None else None,
                num_nodes=int(node_indices.numel()),
                num_leaves=int(leaf_indices.numel()),
                leaf_local_indices=leaf_local,
                parent_local_indices=parent_local,
                leaf_global_indices=[int(i) for i in leaf_indices.tolist()],
                parent_global_indices=[int(i) for i in parent_indices.tolist()],
                gt_leaf_positions=_to_numpy(gt_leaf_pos),
                masked_leaf_positions=_to_numpy(masked_leaf_pos),
                gt_parent_positions=_to_numpy(gt_parent_pos),
                predicted_rel=_to_numpy(pred_rel),
                predicted_abs=_to_numpy(pred_abs),
                gt_rel=_to_numpy(gt_rel_graph),
                expansion_logits=_to_numpy(pred_exp_logits.view(-1)),
                expansion_probs=_to_numpy(pred_probs.view(-1)),
                gt_expansion_labels=_to_numpy(gt_expansion_targets.view(-1)),
                sibling_order_leaf=sibling_order_leaf,
                metadata=metadata,
                losses=losses,
            )
            records.append(record)

        return records
