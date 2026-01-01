import networkx as nx
import torch as th
from torch.nn import Module
import torch.nn.functional as F
from torch_scatter import scatter
from torch_sparse import SparseTensor
import logging

def _build_directed_edge_index(
    self, parent_idx: th.Tensor
) -> tuple[th.Tensor, th.Tensor]:
    """Return (edge_index, edge_types) for explicit parent<->child directions."""
    device = parent_idx.device
    dtype = parent_idx.dtype
    src_list: list[int] = []
    dst_list: list[int] = []
    type_list: list[int] = []

    for child, parent in enumerate(parent_idx.tolist()):
        if parent < 0:
            continue
        # parent -> child
        src_list.append(parent)
        dst_list.append(child)
        type_list.append(self.EDGE_PARENT_TO_CHILD)
        # child -> parent
        src_list.append(child)
        dst_list.append(parent)
        type_list.append(self.EDGE_CHILD_TO_PARENT)

    if src_list:
        edge_index = th.tensor([src_list, dst_list], device=device, dtype=dtype)
        edge_types = th.tensor(type_list, device=device, dtype=dtype)
    else:
        edge_index = parent_idx.new_zeros((2, 0))
        edge_types = parent_idx.new_zeros((0,))

    return edge_index, edge_types

def _graph_target_sizes_from_batch(self, batch, device: th.device) -> th.Tensor | None:
    """Extract per-graph target sizes from a batched PyG Data object."""
    target_attr = getattr(batch, "target_size", None)
    if target_attr is None:
        return None
    if not isinstance(target_attr, th.Tensor):
        target_tensor = th.as_tensor(target_attr)
    else:
        target_tensor = target_attr
    target_tensor = target_tensor.to(device=device, dtype=th.float32).view(-1)
    if target_tensor.numel() == 0:
        return None
    batch_vec = getattr(batch, "batch", None)
    if batch_vec is None or batch_vec.numel() == 0:
        return target_tensor
    batch_vec = batch_vec.to(device)
    num_graphs = int(batch_vec.max().item()) + 1
    if target_tensor.numel() == num_graphs:
        return target_tensor
    if target_tensor.numel() == batch_vec.numel():
        ones = target_tensor.new_ones(batch_vec.size(0))
        sum_per = scatter(target_tensor, batch_vec, dim=0, dim_size=num_graphs)
        counts = scatter(ones, batch_vec, dim=0, dim_size=num_graphs).clamp_min(1.0)
        return sum_per / counts
    if target_tensor.numel() == 1:
        return target_tensor.repeat(num_graphs)
    if target_tensor.numel() > num_graphs:
        return target_tensor[:num_graphs]
    pad = target_tensor.new_full((num_graphs - target_tensor.numel(),), target_tensor[-1])
    return th.cat([target_tensor, pad], dim=0)

def _size_ratio_feature_from_batch(
    self,
    batch,
    device: th.device,
    dtype: th.dtype,
) -> th.Tensor | None:
    """Compute per-node (current_size / target_size) feature for a batch."""
    batch_vec = getattr(batch, "batch", None)
    if batch_vec is None or batch_vec.numel() == 0:
        return None
    batch_vec = batch_vec.to(device)
    target_sizes = self._graph_target_sizes_from_batch(batch, device)
    if target_sizes is None:
        return None
    num_graphs = int(target_sizes.numel())
    ones = target_sizes.new_ones(batch_vec.size(0))
    graph_counts = scatter(ones, batch_vec, dim=0, dim_size=num_graphs)
    ratio_graph = graph_counts / target_sizes.clamp_min(1.0)
    ratio_nodes = ratio_graph[batch_vec].to(dtype).unsqueeze(-1)
    return ratio_nodes

@staticmethod
def _global_inplane_basis(uhat: th.Tensor, eps: float = 1e-8) -> tuple[th.Tensor, th.Tensor]:
    ref = th.tensor([1.0, 0.0, 0.0], dtype=uhat.dtype, device=uhat.device)
    ref_proj = ref - (ref @ uhat) * uhat
    if ref_proj.norm() <= eps:
        ref = th.tensor([0.0, 1.0, 0.0], dtype=uhat.dtype, device=uhat.device)
        ref_proj = ref - (ref @ uhat) * uhat
    e1 = ref_proj / (ref_proj.norm() + eps)
    e2 = th.cross(uhat, e1)
    e2 = e2 / (e2.norm() + eps)
    return e1, e2

def _compute_geo_lr_mask(
    self,
    pos: th.Tensor,
    parent_idx: th.Tensor,
    eps: float = 1e-8,
    tol: float = 1e-6,
) -> th.Tensor:
    """Return boolean mask marking the geometrically-defined left child per parent."""
    if pos.numel() == 0:
        return pos.new_zeros((0,), dtype=th.bool)
    device = pos.device
    dtype = pos.dtype
    N = pos.size(0)
    parent = parent_idx.to(device=device)
    has_parent = parent >= 0

    gp = parent.new_full((N,), -1)
    if has_parent.any():
        parents = parent[has_parent]
        gp_values = parent.new_full((parents.numel(),), -1)
        positive_mask = parents >= 0
        if positive_mask.any():
            gp_values[positive_mask] = parent[parents[positive_mask]].clamp(min=-1)
        gp[has_parent] = gp_values

    uhat = pos.new_zeros((pos.size(1),), dtype=dtype)
    uhat[-1] = 1.0
    global_e1, _ = self._global_inplane_basis(uhat, eps=eps)
    uhat_vec = uhat.view(1, -1)

    v_in = th.zeros((N, pos.size(1)), device=device, dtype=dtype)
    has_gp_mask = gp >= 0
    if has_gp_mask.any():
        sel = has_gp_mask.nonzero(as_tuple=False).flatten()
        v_in[sel] = pos[parent[sel]] - pos[gp[sel]]
    fallback_mask = has_parent & ~has_gp_mask
    if fallback_mask.any():
        v_in[fallback_mask] = global_e1.view(1, -1)

    v_out = th.zeros((N, pos.size(1)), device=device, dtype=dtype)
    if has_parent.any():
        sel = has_parent.nonzero(as_tuple=False).flatten()
        v_out[sel] = pos[sel] - pos[parent[sel]]

    du_in = (v_in * uhat_vec).sum(dim=-1, keepdim=True)
    du_out = (v_out * uhat_vec).sum(dim=-1, keepdim=True)
    v_in_perp = v_in - du_in * uhat_vec
    v_out_perp = v_out - du_out * uhat_vec

    nin = v_in_perp.norm(dim=-1, keepdim=True)
    nout = v_out_perp.norm(dim=-1, keepdim=True)
    v_in_unit = v_in_perp / (nin + eps)
    degenerate = (nin <= eps) | (~has_parent).view(-1, 1)
    if degenerate.any():
        v_in_unit = v_in_unit.clone()
        v_in_unit[degenerate.squeeze(-1)] = global_e1
    v_out_unit = v_out_perp / (nout + eps)

    cospsi = (v_in_unit * v_out_unit).sum(dim=-1, keepdim=True)
    cross = th.cross(v_in_unit, v_out_unit, dim=-1)
    sinpsi = (cross * uhat_vec).sum(dim=-1, keepdim=True)
    mask = has_parent.view(-1, 1)
    cospsi = th.where(mask, cospsi, cospsi.new_ones(cospsi.shape))
    sinpsi = th.where(mask, sinpsi, sinpsi.new_zeros(sinpsi.shape))

    lr_mask = th.zeros((N,), dtype=th.bool, device=device)

    # Special-case: children whose parent is a root (parent idx == -1) lack a
    # grandparent, so assign left/right purely from +/- z relative to the root.
    handled_parents = th.zeros((N,), dtype=th.bool, device=device)
    root_nodes = (parent == -1).nonzero(as_tuple=False).flatten()
    if not root_nodes.numel():
        logger.warning("[GeoLR] No root with parent==-1 found; override skipped for this graph.")
    else:
        for r in root_nodes.tolist():
            child_idx = (parent == r).nonzero(as_tuple=False).flatten()
            if child_idx.numel() == 0:
                continue
            parent_z = pos[r, -1]
            child_z = pos[child_idx, -1]
            lr_mask[child_idx] = child_z >= parent_z
            handled_parents[r] = True

    unique_parents = parent.unique()
    for p in unique_parents.tolist():
        if p < 0:
            continue
        if handled_parents[p]:
            continue
        child_idx = (parent == p).nonzero(as_tuple=False).flatten()
        if child_idx.numel() != 2:
            continue
        s = sinpsi[child_idx, 0]
        c = cospsi[child_idx, 0]
        if (s[0] * s[1] < -tol):
            lr_mask[child_idx[0]] = bool(s[0] > 0)
            lr_mask[child_idx[1]] = bool(s[1] > 0)
        else:
            theta = th.atan2(s, c)
            idx_left = child_idx[int(th.argmax(theta))]
            lr_mask[idx_left] = True

    if getattr(self, "debug", False):
        for p in unique_parents.tolist():
            if p < 0:
                continue
            child_idx = (parent == p).nonzero(as_tuple=False).flatten()
            if child_idx.numel() != 2:
                continue
            left_count = int(lr_mask[child_idx].sum().item())
            if left_count != 1:
                logger.warning(f"[GeoLR] Parent {p} has {left_count} left assignments (expected 1).")

    return lr_mask

@staticmethod
def _decode_parent_indices(batch) -> th.Tensor:
    """Convert batched parent_idx_1b -> 0-based with -1 for roots, even after PyG offsets."""
    parent_idx_1b = batch.parent_idx_1b
    if not isinstance(parent_idx_1b, th.Tensor):
        parent_idx_1b = th.as_tensor(parent_idx_1b)
    parent_idx = parent_idx_1b - 1
    batch_vec = getattr(batch, "batch", None)
    ptr = getattr(batch, "ptr", None)
    if batch_vec is not None and ptr is not None:
        offsets = ptr[batch_vec]
        root_mask = parent_idx_1b == offsets
    else:
        root_mask = parent_idx_1b == 0
    if root_mask.any():
        parent_idx = parent_idx.clone()
        parent_idx[root_mask] = -1
    return parent_idx

def _select_training_leaf_indices(self, batch) -> th.Tensor:
    """Return indices of leaves that should contribute to masking/loss."""
    base = getattr(batch, "leaf_idx", None)
    if base is None:
        raise ValueError("Expected batch.leaf_idx to select leaves for training.")
    candidate = getattr(batch, "new_leaf_idx_from_next", None)
    if candidate is None:
        return base
    if isinstance(candidate, th.Tensor):
        new_idx = candidate.to(device=base.device, dtype=base.dtype)
    else:
        new_idx = th.as_tensor(candidate, device=base.device, dtype=base.dtype)
    if new_idx.numel() == 0:
        return new_idx
    leaf_mask = getattr(batch, "leaf_mask", None)
    if leaf_mask is not None:
        if isinstance(leaf_mask, th.Tensor):
            lm = leaf_mask.to(device=new_idx.device)
            if lm.dtype != th.bool:
                lm = lm.bool()
            valid = lm[new_idx]
            if not valid.all():
                new_idx = new_idx[valid]
    return new_idx

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