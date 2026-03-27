import itertools
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch as th
from torch_scatter import scatter

logger = logging.getLogger(__name__)
_DIFFUSION_PLOT_COUNTER = itertools.count()


def build_directed_edge_index(
    parent_idx: th.Tensor,
    edge_parent_to_child: int = 0,
    edge_child_to_parent: int = 1,
) -> tuple[th.Tensor, th.Tensor]:
    """Return (edge_index, edge_types) for explicit parent/child directions."""
    device = parent_idx.device
    dtype = parent_idx.dtype
    src_list: list[int] = []
    dst_list: list[int] = []
    type_list: list[int] = []

    for child, parent in enumerate(parent_idx.tolist()):
        if parent < 0:
            continue
        src_list.append(parent)
        dst_list.append(child)
        type_list.append(edge_parent_to_child)
        src_list.append(child)
        dst_list.append(parent)
        type_list.append(edge_child_to_parent)

    if src_list:
        edge_index = th.tensor([src_list, dst_list], device=device, dtype=dtype)
        edge_types = th.tensor(type_list, device=device, dtype=dtype)
    else:
        edge_index = parent_idx.new_zeros((2, 0))
        edge_types = parent_idx.new_zeros((0,))
    return edge_index, edge_types


def graph_target_sizes_from_batch(batch, device: th.device) -> Optional[th.Tensor]:
    """Extract per-graph total tree sizes from a batched PyG Data object."""
    target_attr = getattr(batch, "total_tree_size", None)
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


def size_ratio_feature_from_batch(
    batch,
    device: th.device,
    dtype: th.dtype,
) -> Optional[th.Tensor]:
    """Compute per-node (current_size / total_tree_size) feature for a batch."""
    batch_vec = getattr(batch, "batch", None)
    if batch_vec is None or batch_vec.numel() == 0:
        return None
    batch_vec = batch_vec.to(device)
    target_sizes = graph_target_sizes_from_batch(batch, device)
    if target_sizes is None:
        return None
    num_graphs = int(target_sizes.numel())
    ones = target_sizes.new_ones(batch_vec.size(0))
    graph_counts = scatter(ones, batch_vec, dim=0, dim_size=num_graphs)
    ratio_graph = graph_counts / target_sizes.clamp_min(1.0)
    ratio_nodes = ratio_graph[batch_vec].to(dtype).unsqueeze(-1)
    return ratio_nodes


def global_inplane_basis(uhat: th.Tensor, eps: float = 1e-8) -> Tuple[th.Tensor, th.Tensor]:
    """Return an orthogonal basis spanning the plane orthogonal to `uhat`."""
    ref = th.tensor([1.0, 0.0, 0.0], dtype=uhat.dtype, device=uhat.device)
    ref_proj = ref - (ref @ uhat) * uhat
    if ref_proj.norm() <= eps:
        ref = th.tensor([0.0, 1.0, 0.0], dtype=uhat.dtype, device=uhat.device)
        ref_proj = ref - (ref @ uhat) * uhat
    e1 = ref_proj / (ref_proj.norm() + eps)
    e2 = th.cross(uhat, e1)
    e2 = e2 / (e2.norm() + eps)
    return e1, e2


def _order_root_children_by_uhat(
    offsets: th.Tensor,
    uhat: th.Tensor,
    eps: float = 1e-8,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """SO(2)-invariant ordering of root children.

    child_0 = lowest uhat component (tiebreak: largest perp-plane distance).
    Remaining children ordered clockwise relative to child_0's perp direction.

    Args:
        offsets: [k, 3] offsets from root to each child
        uhat:    [3] SO(2) axis
        eps:     numerical tolerance

    Returns:
        sorted_idx:   [k] indices sorted by ordinal (child_0 first)
        fwd0_unit:    [3] forward direction (root→child_0 in perp plane), zero if degenerate
        delta_angles: [k] angle of each child relative to child_0 (child_0 gets 0.0)
    """
    k = offsets.size(0)
    device = offsets.device
    dtype = offsets.dtype
    uhat_vec = uhat.view(1, -1)

    # Project onto uhat and perp plane
    uhat_components = (offsets * uhat_vec).sum(dim=-1)          # [k]
    offsets_perp = offsets - uhat_components.unsqueeze(-1) * uhat_vec  # [k, 3]
    perp_dist = offsets_perp.norm(dim=-1)                       # [k]

    # Select child_0: lowest uhat component, tiebreak largest perp distance
    min_uhat = uhat_components.min()
    is_min = uhat_components <= min_uhat + eps
    tied_perp = th.where(is_min, perp_dist, perp_dist.new_tensor(-1.0))
    child0_local = int(tied_perp.argmax().item())

    fwd0 = offsets_perp[child0_local]
    fwd0_norm = fwd0.norm()

    delta_angles = th.zeros(k, device=device, dtype=dtype)

    if fwd0_norm <= eps:
        # Degenerate: all children on uhat axis — sort by uhat ascending
        _, sorted_idx = uhat_components.sort()
        return sorted_idx, th.zeros(3, device=device, dtype=dtype), delta_angles

    fwd0_unit = fwd0 / fwd0_norm
    side0 = th.cross(uhat, fwd0_unit)
    side0 = side0 / (side0.norm() + eps)

    # Compute angle of each child relative to fwd0
    proj_fwd = (offsets_perp * fwd0_unit.unsqueeze(0)).sum(dim=-1)   # [k]
    proj_side = (offsets_perp * side0.unsqueeze(0)).sum(dim=-1)      # [k]
    angles_from_fwd = th.atan2(proj_side, proj_fwd)                  # [k]

    # Sort clockwise (descending angle); child_0 has angle ~0 and should be first
    _, sorted_idx = (-angles_from_fwd).sort()

    # Ensure child_0 is first (it should be, but handle float precision)
    c0_rank = (sorted_idx == child0_local).nonzero(as_tuple=False).flatten()
    if c0_rank.numel() > 0 and c0_rank[0].item() != 0:
        # Swap child_0 to position 0
        r = c0_rank[0].item()
        sorted_idx = sorted_idx.clone()
        sorted_idx[0], sorted_idx[r] = sorted_idx[r].clone(), sorted_idx[0].clone()

    # delta_angles = angle of each child relative to fwd0 (child_0 gets 0.0)
    delta_angles = angles_from_fwd.clone()
    delta_angles[child0_local] = 0.0

    return sorted_idx, fwd0_unit, delta_angles


def global_to_local(
    offset_global: th.Tensor,
    forward: th.Tensor,
    sideways: th.Tensor,
    uhat: th.Tensor,
) -> th.Tensor:
    """Project 3D global-frame offsets into a per-node local basis.

    Args:
        offset_global: [L, 3] offsets in global frame
        forward:       [L, 3] per-node forward basis vector
        sideways:      [L, 3] per-node sideways basis vector
        uhat:          [3] SO(2) axis (broadcast)

    Returns:
        [L, 3] offsets in local frame (forward, sideways, axial)
    """
    uhat_b = uhat.unsqueeze(0).expand_as(offset_global)
    comp_fwd = (offset_global * forward).sum(-1, keepdim=True)
    comp_side = (offset_global * sideways).sum(-1, keepdim=True)
    comp_axial = (offset_global * uhat_b).sum(-1, keepdim=True)
    return th.cat([comp_fwd, comp_side, comp_axial], dim=-1)


def local_to_global(
    offset_local: th.Tensor,
    forward: th.Tensor,
    sideways: th.Tensor,
    uhat: th.Tensor,
) -> th.Tensor:
    """Reconstruct 3D global-frame offsets from a per-node local basis.

    Args:
        offset_local: [L, 3] offsets in local frame (forward, sideways, axial)
        forward:      [L, 3] per-node forward basis vector
        sideways:     [L, 3] per-node sideways basis vector
        uhat:         [3] SO(2) axis (broadcast)

    Returns:
        [L, 3] offsets in global frame
    """
    uhat_b = uhat.unsqueeze(0).expand_as(forward)
    return (
        offset_local[:, 0:1] * forward
        + offset_local[:, 1:2] * sideways
        + offset_local[:, 2:3] * uhat_b
    )


def compute_local_bases(
    pos: th.Tensor,
    parent_idx: th.Tensor,
    uhat: th.Tensor,
    geo_lr_mask: th.Tensor,
    eps: float = 1e-8,
    geo_delta_theta: th.Tensor | None = None,
) -> dict:
    """Compute per-node local coordinate frames in the plane perpendicular to uhat.

    For each node i with parent p:
      - forward_p = normalize(v_in_perp) where v_in = pos[p] - pos[grandparent(p)]
        projected onto the plane perpendicular to uhat.
      - sideways_p = uhat × forward_p

    For root's children: each child gets its own frame.
      - Child 0 (leftmost): forward = root→child_0 direction (perp-projected)
      - Child i: forward = child_0's forward rotated by Δθ_i around uhat
    If geo_delta_theta is not provided, falls back to shared left-child frame.

    Args:
        pos:             [N, 3] node positions
        parent_idx:      [N] 0-based parent indices (-1 for roots)
        uhat:            [3] SO(2) axis
        geo_lr_mask:     [N] bool, True = geometrically-defined LEFT child
        eps:             numerical tolerance
        geo_delta_theta: [N] float, relative angle from child 0 for root children (from compute_root_child_angles)

    Returns:
        dict with 'local_forward': [N, 3], 'local_sideways': [N, 3]
    """
    N = pos.size(0)
    device = pos.device
    dtype = pos.dtype
    parent = parent_idx.clone()
    has_parent = parent >= 0

    # Grandparent lookup
    gp = parent.new_full((N,), -1)
    if has_parent.any():
        gp[has_parent] = parent[parent[has_parent].clamp(min=0)].clamp(min=-1)
        # Fix: nodes whose parent is root have gp = parent[root] which is -1 already
        # but clamp might have turned it to 0 — re-mask
        gp[has_parent] = torch.where(
            parent[parent[has_parent].clamp(min=0)] >= 0,
            parent[parent[has_parent].clamp(min=0)],
            parent.new_full((has_parent.sum(),), -1),
        )

    has_gp = gp >= 0

    # --- v_in for non-root nodes with grandparent: pos[parent] - pos[grandparent]
    v_in = torch.zeros_like(pos)
    if has_gp.any():
        sel = has_gp.nonzero(as_tuple=False).flatten()
        v_in[sel] = pos[parent[sel]] - pos[gp[sel]]

    # --- Children of root: per-child frames using geo_delta_theta
    fallback_mask = has_parent & ~has_gp
    if fallback_mask.any():
        is_root = parent == -1
        uhat_vec = uhat.view(1, -1)
        for r in is_root.nonzero(as_tuple=False).flatten().tolist():
            children = (parent == r).nonzero(as_tuple=False).flatten()
            if children.numel() == 0:
                continue
            # Find child 0 (leftmost): the one with geo_delta_theta == 0 or geo_lr_mask == True
            if geo_delta_theta is not None:
                child0_local_idx = geo_delta_theta[children].abs().argmin()
            else:
                left_children = children[geo_lr_mask[children]]
                child0_local_idx = 0 if left_children.numel() == 0 else (children == left_children[0]).nonzero(as_tuple=False).flatten()[0]
            child0 = children[child0_local_idx]
            ref_dir = pos[child0] - pos[r]

            if geo_delta_theta is not None and children.numel() > 1:
                # Per-child frame: rotate ref_dir by each child's delta_theta
                du = (ref_dir * uhat.view(-1)).sum()
                ref_perp = ref_dir - du * uhat
                ref_norm = ref_perp.norm()
                if ref_norm > eps:
                    fwd0 = ref_perp / ref_norm
                    side0 = torch.cross(uhat, fwd0)
                    side0 = side0 / (side0.norm() + eps)
                    for c in children.tolist():
                        dt = geo_delta_theta[c].item()
                        v_in[c] = torch.cos(torch.tensor(dt)) * fwd0 + torch.sin(torch.tensor(dt)) * side0
                else:
                    # Degenerate: all children get same ref_dir, will hit fallback
                    for c in children.tolist():
                        v_in[c] = ref_dir
            else:
                # Legacy fallback: all children share left-child direction
                for c in children.tolist():
                    v_in[c] = ref_dir

    # --- Project onto plane perpendicular to uhat and normalize
    uhat_vec = uhat.view(1, -1)
    du_in = (v_in * uhat_vec).sum(dim=-1, keepdim=True)
    v_in_perp = v_in - du_in * uhat_vec
    nin = v_in_perp.norm(dim=-1, keepdim=True)

    forward = v_in_perp / (nin + eps)

    # Degenerate cases: v_in parallel to uhat or zero
    degenerate = nin.squeeze(-1) <= eps
    if degenerate.any():
        e1, _ = global_inplane_basis(uhat, eps=eps)
        forward = forward.clone()
        forward[degenerate] = e1.unsqueeze(0)

    sideways = torch.cross(uhat_vec.expand_as(forward), forward, dim=-1)
    sideways = sideways / (sideways.norm(dim=-1, keepdim=True) + eps)

    return {
        'local_forward': forward,
        'local_sideways': sideways,
    }


def compute_local_bases_for_leaves(
    pos: th.Tensor,
    parent_idx: th.Tensor,
    leaf_parent_idx: th.Tensor,
    uhat: th.Tensor,
    eps: float = 1e-8,
    child_ordinal: th.Tensor | None = None,
    sibling_count: th.Tensor | None = None,
) -> tuple[th.Tensor, th.Tensor]:
    """Compute local bases for new leaf nodes during inference.

    Used during expand() when full precompute_full_geometry is not available.

    When child_ordinal and sibling_count are provided and root children are
    spawned all at once (degenerate: all at parent position), child 0 gets a
    random forward direction and children 1..k-1 get frames rotated by
    2π*i/k from child 0's frame.

    Args:
        pos:              [N, 3] node positions
        parent_idx:       [N] 0-based parent indices (-1 for roots)
        leaf_parent_idx:  [L] indices of parent nodes for new leaves
        uhat:             [3] SO(2) axis
        eps:              numerical tolerance
        child_ordinal:    [L] int, 0-based ordinal of each leaf among its siblings (optional)
        sibling_count:    [L] int, total siblings for each leaf's parent (optional)

    Returns:
        (leaf_fwd [L, 3], leaf_side [L, 3])
    """
    L = leaf_parent_idx.numel()
    device = pos.device

    if L == 0:
        return pos.new_zeros((0, 3)), pos.new_zeros((0, 3))

    # Grandparent of each parent
    gp = parent_idx[leaf_parent_idx]  # [L], grandparent indices

    has_gp = gp >= 0
    v_in = torch.zeros((L, 3), device=device, dtype=pos.dtype)

    if has_gp.any():
        sel = has_gp.nonzero(as_tuple=False).flatten()
        v_in[sel] = pos[leaf_parent_idx[sel]] - pos[gp[sel]]

    # For root parents (no grandparent): use direction to existing child if available
    no_gp = ~has_gp
    if no_gp.any():
        sel = no_gp.nonzero(as_tuple=False).flatten()
        for s in sel.tolist():
            p = leaf_parent_idx[s].item()
            children = (parent_idx == p).nonzero(as_tuple=False).flatten()
            # Exclude placeholder children (those with the same position as parent)
            real_children = []
            for c in children.tolist():
                if (pos[c] - pos[p]).norm() > eps:
                    real_children.append(c)
            if real_children:
                v_in[s] = pos[real_children[0]] - pos[p]
            # else: v_in stays zero, will hit degenerate fallback

    # Project onto plane perpendicular to uhat
    uhat_vec = uhat.view(1, -1)
    du_in = (v_in * uhat_vec).sum(dim=-1, keepdim=True)
    v_in_perp = v_in - du_in * uhat_vec
    nin = v_in_perp.norm(dim=-1, keepdim=True)

    forward = v_in_perp / (nin + eps)

    # Degenerate fallback
    degenerate = nin.squeeze(-1) <= eps
    if degenerate.any():
        forward = forward.clone()

        if child_ordinal is not None and sibling_count is not None:
            # Structured rotation for root children spawned at once:
            # Group degenerate leaves by parent, pick one random forward per parent,
            # then rotate each child by 2π*ordinal/k
            degen_sel = degenerate.nonzero(as_tuple=False).flatten()
            degen_parents = leaf_parent_idx[degen_sel]
            unique_parents, inv = torch.unique(degen_parents, return_inverse=True)

            e1_v, e2_v = global_inplane_basis(uhat, eps=eps)
            # One random angle per unique degenerate parent
            base_theta = torch.rand(unique_parents.numel(), device=device, dtype=pos.dtype) * (2 * torch.pi)

            for j, up in enumerate(unique_parents.tolist()):
                group_mask = inv == j
                group_indices = degen_sel[group_mask]
                k = int(sibling_count[group_indices[0]].item())
                theta0 = base_theta[j]
                ordinals = child_ordinal[group_indices]
                angles = theta0 + 2 * torch.pi * ordinals.to(dtype=pos.dtype) / max(k, 1)
                forward[group_indices] = (
                    angles.cos().unsqueeze(-1) * e1_v.unsqueeze(0)
                    + angles.sin().unsqueeze(-1) * e2_v.unsqueeze(0)
                )
        else:
            # Legacy fallback: random angle per degenerate leaf
            degen_sel = degenerate.nonzero(as_tuple=False).flatten()
            theta = torch.rand(degen_sel.numel(), device=device, dtype=pos.dtype) * (2 * torch.pi)
            e1_v, e2_v = global_inplane_basis(uhat, eps=eps)
            forward[degen_sel] = (
                theta.cos().unsqueeze(-1) * e1_v.unsqueeze(0)
                + theta.sin().unsqueeze(-1) * e2_v.unsqueeze(0)
            )

    sideways = torch.cross(uhat_vec.expand_as(forward), forward, dim=-1)
    sideways = sideways / (sideways.norm(dim=-1, keepdim=True) + eps)

    return forward, sideways


def compute_branch_angles_parent_centric(
    coors: torch.Tensor,
    parent_idx: torch.Tensor,
    uhat: torch.Tensor,
    eps: float = 1e-8,
    return_intermediates: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Returns (cosψ, sinψ, cosθ) per node i (> root), describing the angle of the branch
    parent(i) -> i relative to the incoming direction at parent(i), plus the angle
    between parent(i)->i and uhat.

    If return_intermediates=True, additionally returns a dict with v_in, v_out, has_gp
    needed for leaf patching.
    """
    N = coors.size(0)

    parent = parent_idx.clone()
    has_parent = parent >= 0

    # grandparent of each node
    gp = parent.new_full((N,), -1)
    gp[has_parent] = parent_idx[parent[has_parent]].clamp(min=-1)

    # incoming direction at parent
    v_in = torch.zeros_like(coors)
    has_gp = gp >= 0
    if has_gp.any():
        sel = has_gp.nonzero(as_tuple=False).flatten()
        v_in[sel] = coors[parent[sel]] - coors[gp[sel]]
    fallback_mask = has_parent & ~has_gp
    if fallback_mask.any():
        sel = fallback_mask.nonzero(as_tuple=False).flatten()
        v_in[sel] = coors[sel] - coors[parent[sel]]

    # outgoing direction parent -> child
    v_out = torch.zeros_like(coors)
    if has_parent.any():
        sel = has_parent.nonzero(as_tuple=False).flatten()
        v_out[sel] = coors[sel] - coors[parent[sel]]

    # project onto plane orthogonal to axis
    du_in = (v_in @ uhat).unsqueeze(-1)
    du_out = (v_out @ uhat).unsqueeze(-1)
    v_in_perp = v_in - du_in * uhat
    v_out_perp = v_out - du_out * uhat

    nin = v_in_perp.norm(dim=-1, keepdim=True)
    nout = v_out_perp.norm(dim=-1, keepdim=True)
    v_in_unit = v_in_perp / (nin + eps)
    v_out_unit = v_out_perp / (nout + eps)

    cospsi = (v_in_unit * v_out_unit).sum(dim=-1, keepdim=True)
    cross = torch.cross(v_in_unit, v_out_unit, dim=-1)
    sinpsi = (cross * uhat).sum(dim=-1, keepdim=True)

    v_out_norm = (nout.pow(2) + du_out.pow(2)).sqrt()
    cos_theta = du_out / (v_out_norm + eps)

    cospsi = torch.where(has_parent.view(-1, 1), cospsi, torch.ones_like(cospsi))
    sinpsi = torch.where(has_parent.view(-1, 1), sinpsi, torch.zeros_like(sinpsi))
    cos_theta = torch.where(has_parent.view(-1, 1), cos_theta, torch.ones_like(cos_theta))

    if return_intermediates:
        intermediates = {
            'v_in': v_in,
            'v_out': v_out,
            'has_gp': has_gp,
        }
        return cospsi, sinpsi, cos_theta, intermediates
    return cospsi, sinpsi, cos_theta


def assign_branch_angles_to_edges(
    edge_index: torch.Tensor,
    parent_idx: torch.Tensor,
    cospsi_node: torch.Tensor,
    sinpsi_node: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign node-level branch angles (cosψ/sinψ) to directed edges."""
    src, dst = edge_index
    device = cospsi_node.device
    dtype = cospsi_node.dtype
    num_edges = src.size(0)
    cos_edge = torch.zeros((num_edges, 1), device=device, dtype=dtype)
    sin_edge = torch.zeros_like(cos_edge)

    mask_parent_to_child = (parent_idx[dst] == src)
    if mask_parent_to_child.any():
        child_idx = dst[mask_parent_to_child]
        cos_edge[mask_parent_to_child] = cospsi_node[child_idx]
        sin_edge[mask_parent_to_child] = sinpsi_node[child_idx]

    mask_child_to_parent = (parent_idx[src] == dst)
    if mask_child_to_parent.any():
        child_idx = src[mask_child_to_parent]
        cos_edge[mask_child_to_parent] = cospsi_node[child_idx]
        sin_edge[mask_child_to_parent] = sinpsi_node[child_idx]

    return cos_edge, sin_edge


def assign_parent_scalar_to_edges(
    edge_index: torch.Tensor,
    parent_idx: torch.Tensor,
    scalar_node: torch.Tensor,
) -> torch.Tensor:
    """Assign parent-centric per-node scalar to directed edges (both directions share child value)."""
    src, dst = edge_index
    device = scalar_node.device
    dtype = scalar_node.dtype
    num_edges = src.size(0)
    scalar_edge = torch.zeros((num_edges, 1), device=device, dtype=dtype)

    mask_parent_to_child = (parent_idx[dst] == src)
    if mask_parent_to_child.any():
        child_idx = dst[mask_parent_to_child]
        scalar_edge[mask_parent_to_child] = scalar_node[child_idx]

    mask_child_to_parent = (parent_idx[src] == dst)
    if mask_child_to_parent.any():
        child_idx = src[mask_child_to_parent]
        scalar_edge[mask_child_to_parent] = scalar_node[child_idx]

    return scalar_edge


def compute_geo_lr_mask(
    pos: th.Tensor,
    parent_idx: th.Tensor,
    *,
    uhat: th.Tensor | None = None,
    debug: bool = False,
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

    if uhat is None:
        uhat = pos.new_zeros((pos.size(1),), dtype=dtype)
        uhat[-1] = 1.0
    global_e1, _ = global_inplane_basis(uhat, eps=eps)
    uhat_vec = uhat.view(1, -1)

    v_in = th.zeros((N, pos.size(1)), device=device, dtype=dtype)
    has_gp_mask = gp >= 0
    if has_gp_mask.any():
        sel = has_gp_mask.nonzero(as_tuple=False).flatten()
        v_in[sel] = pos[parent[sel]] - pos[gp[sel]]
    fallback_mask = has_parent & ~has_gp_mask

    v_out = th.zeros((N, pos.size(1)), device=device, dtype=dtype)
    if has_parent.any():
        sel = has_parent.nonzero(as_tuple=False).flatten()
        v_out[sel] = pos[sel] - pos[parent[sel]]

    if fallback_mask.any():
        # Root children: use v_out as v_in (geometric, no global axis).
        # Unused for their L/R decision (Z-comparison path); yields sinψ=0 for sinψ codepath.
        v_in[fallback_mask] = v_out[fallback_mask]

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

    # --- vectorized root-children handling (replaces loop over root_nodes) ---
    is_root = (parent == -1)                                      # [N]
    parent_clamped = parent.clamp(min=0)                          # [N], safe for indexing
    parent_is_root_node = has_parent & is_root[parent_clamped]    # [N] children of roots
    if not is_root.any():
        logger.warning("[GeoLR] No root with parent==-1 found; override skipped for this graph.")
    else:
        counts = scatter(
            th.ones(N, device=device, dtype=pos.dtype),
            parent_clamped, dim=0, dim_size=N, reduce='sum',
        )                                                          # [N] child-count per node
        sibling_count = counts[parent_clamped]                    # [N]
        single_root_ch = parent_is_root_node & (sibling_count == 1)
        multi_root_ch  = parent_is_root_node & (sibling_count > 1)
        lr_mask = lr_mask.masked_fill(single_root_ch, True)
        if multi_root_ch.any():
            # Compare root children with each other (not with root).
            # Convention: lower z = left (True), higher z = right (False).
            z_rc = pos[multi_root_ch, -1]
            p_rc = parent_clamped[multi_root_ch]
            max_z = scatter(z_rc, p_rc, dim=0, dim_size=N, reduce='max')
            lr_mask[multi_root_ch] = z_rc < max_z[p_rc] - 1e-7
    handled_parents = is_root                                      # [N] bool

    # --- vectorized binary-parent handling (replaces loop over unique_parents) ---
    if not is_root.any():
        counts = scatter(
            th.ones(N, device=device, dtype=pos.dtype),
            parent_clamped, dim=0, dim_size=N, reduce='sum',
        )
    binary_parents = (counts == 2) & ~handled_parents             # [N]
    in_binary = has_parent & binary_parents[parent_clamped] & ~parent_is_root_node  # [N]

    if in_binary.any():
        s = sinpsi[:, 0]                                           # [N]
        c = cospsi[:, 0]                                           # [N]
        s_bin = s[in_binary]
        p_bin = parent_clamped[in_binary]

        # Product s0*s1 per parent via identity: (s0+s1)^2 - s0^2 - s1^2) / 2
        sum_s  = scatter(s_bin,      p_bin, dim=0, dim_size=N, reduce='sum')
        sum_s2 = scatter(s_bin ** 2, p_bin, dim=0, dim_size=N, reduce='sum')
        product = (sum_s ** 2 - sum_s2) * 0.5                     # [N]

        # Case 1: opposite-sign sines → left child is the one with sin > 0
        case1_parent = (product < -tol) & binary_parents
        case1_nodes  = in_binary & case1_parent[parent_clamped]
        if case1_nodes.any():
            lr_mask[case1_nodes] = s[case1_nodes] > 0

        # Case 2: same-sign or near-zero → left child is the one with larger atan2 angle
        case2_parent = binary_parents & ~case1_parent
        case2_nodes  = in_binary & case2_parent[parent_clamped]
        if case2_nodes.any():
            theta = th.atan2(s, c)                                 # [N]
            max_theta = scatter(
                theta[case2_nodes], parent_clamped[case2_nodes],
                dim=0, dim_size=N, reduce='max',
            )                                                      # [N]
            lr_mask[case2_nodes] = (
                theta[case2_nodes] >= max_theta[parent_clamped[case2_nodes]] - 1e-7
            )

    if debug:
        unique_parents = parent.unique()
        for p in unique_parents.tolist():
            if p < 0:
                continue
            child_idx = (parent == p).nonzero(as_tuple=False).flatten()
            if child_idx.numel() != 2:
                continue
            left_count = int(lr_mask[child_idx].sum().item())
            if left_count != 1:
                logger.warning(
                    f"[GeoLR] Parent {p} has {left_count} left assignments (expected 1)."
                )

    return lr_mask


def compute_root_child_angles(
    pos: th.Tensor,
    parent_idx: th.Tensor,
    uhat: th.Tensor,
    geo_lr_mask: th.Tensor,
    *,
    eps: float = 1e-8,
) -> Tuple[th.Tensor, th.Tensor]:
    """Compute per-node ordinal feature and relative angles for root children.

    For root children (k ≥ 1):
      1. Select child_0 by lowest uhat component (tiebreak: largest perp distance).
      2. Order remaining children clockwise relative to child_0's perp direction.
      3. Compute Δθ_i = angle of child i relative to child 0's direction.
      4. Ordinal feature = i / max(k-1, 1).

    This ordering is SO(2)-invariant (no global axes used).

    For binary interior children: left (geo_lr_mask=True) → 0.0, right → 1.0.
    For root/parentless nodes: sentinel -1.0.

    Args:
        pos:          [N, 3] node positions
        parent_idx:   [N] 0-based parent indices (-1 for roots)
        uhat:         [3] SO(2) axis
        geo_lr_mask:  [N] bool, True = left child (from compute_geo_lr_mask)

    Returns:
        (geo_ordinal [N] float, geo_delta_theta [N] float)
        - geo_ordinal: i/(k-1) for root children, 0.0/1.0 for binary L/R, -1.0 sentinel
        - geo_delta_theta: relative angle from child 0 for root children (radians),
          0.0 for others (used for frame construction in compute_local_bases)
    """
    N = pos.size(0)
    device = pos.device
    dtype = pos.dtype
    parent = parent_idx.to(device=device)

    geo_ordinal = th.full((N,), -1.0, device=device, dtype=dtype)
    geo_delta_theta = th.zeros((N,), device=device, dtype=dtype)

    if N == 0:
        return geo_ordinal, geo_delta_theta

    has_parent = parent >= 0
    is_root = parent == -1
    parent_clamped = parent.clamp(min=0)

    # --- Binary interior children: left=0.0, right=1.0 ---
    # (children whose parent is NOT root and has exactly 2 children)
    counts = scatter(
        th.ones(N, device=device, dtype=dtype),
        parent_clamped, dim=0, dim_size=N, reduce='sum',
    )
    parent_is_root = has_parent & is_root[parent_clamped]
    binary_interior = has_parent & ~parent_is_root & (counts[parent_clamped] == 2)
    if binary_interior.any():
        geo_ordinal[binary_interior] = (~geo_lr_mask[binary_interior]).to(dtype)
        # left (True) → 0.0, right (False) → 1.0

    # --- Root children: SO(2)-invariant ordering by uhat component ---
    if not is_root.any():
        return geo_ordinal, geo_delta_theta

    root_children_mask = parent_is_root  # [N] bool
    if not root_children_mask.any():
        return geo_ordinal, geo_delta_theta

    # For each root, process its children
    root_indices = is_root.nonzero(as_tuple=False).flatten()
    for r in root_indices.tolist():
        children = (parent == r).nonzero(as_tuple=False).flatten()
        k = children.numel()
        if k == 0:
            continue

        if k == 1:
            geo_ordinal[children[0]] = 0.0
            geo_delta_theta[children[0]] = 0.0
            continue

        offsets = pos[children] - pos[r].unsqueeze(0)  # [k, 3]
        sorted_idx, _fwd0, delta_angles = _order_root_children_by_uhat(
            offsets, uhat, eps=eps,
        )

        for rank, si in enumerate(sorted_idx.tolist()):
            c = children[si]
            geo_ordinal[c] = rank / max(k - 1, 1)
            geo_delta_theta[c] = delta_angles[si]

    return geo_ordinal, geo_delta_theta


def compute_geo_lr_for_new_leaves(
    pos: th.Tensor,            # [N, D] all node positions
    parent_idx: th.Tensor,     # [N] parent indices (0-based, root=-1)
    new_leaf_idx: th.Tensor,   # [K] indices of new leaf nodes
    *,
    uhat: th.Tensor | None = None,
    debug: bool = False,
    eps: float = 1e-8,
    tol: float = 1e-6,
) -> Tuple[th.Tensor, th.Tensor]:
    """Compute geo L/R mask for only the specified new leaves (O(K) not O(N)).

    Returns (lr_mask[K], valid[K]) where:
      - lr_mask[i] = True means new_leaf_idx[i] is the LEFT child
      - valid[i] = True means new_leaf_idx[i] has a binary parent
        (only lr_mask[valid] entries are meaningful)
    """
    K = new_leaf_idx.numel()
    device = pos.device
    dtype = pos.dtype

    lr_mask = th.zeros(K, dtype=th.bool, device=device)
    valid = th.zeros(K, dtype=th.bool, device=device)

    if K == 0:
        return lr_mask, valid

    # --- 1. Group by parent, identify binary pairs ---
    parent_of_leaf = parent_idx[new_leaf_idx]                       # [K]
    unique_parents, inv = th.unique(parent_of_leaf, return_inverse=True)  # [M], [K]
    M = unique_parents.numel()
    counts = scatter(th.ones(K, device=device), inv, dim=0,
                     dim_size=M, reduce='sum')                      # [M]
    sibling_count = counts[inv]                                     # [K]
    valid = sibling_count == 2                                      # [K]

    if not valid.any():
        return lr_mask, valid

    # --- 2. Gather only valid leaves ---
    valid_leaf_global = new_leaf_idx[valid]                          # [V]
    valid_parent = parent_idx[valid_leaf_global]                     # [V]
    V = valid_leaf_global.numel()

    # --- 3. Grandparent lookup ---
    gp = parent_idx[valid_parent]                                   # [V]

    # --- 4. uhat and basis ---
    D = pos.size(1)
    if uhat is None:
        uhat = pos.new_zeros((D,), dtype=dtype)
        uhat[-1] = 1.0
    global_e1, _ = global_inplane_basis(uhat, eps=eps)
    uhat_vec = uhat.view(1, -1)                                    # [1, D]

    # --- 5. Direction vectors (size [V, D]) ---
    has_gp = gp >= 0                                                # [V]
    v_in = global_e1.view(1, -1).expand(V, -1).clone()             # [V, D] init to fallback
    if has_gp.any():
        v_in[has_gp] = pos[valid_parent[has_gp]] - pos[gp[has_gp]]

    v_out = pos[valid_leaf_global] - pos[valid_parent]              # [V, D]

    # --- 6. Project onto plane ⊥ uhat, normalize, cross product ---
    du_in = (v_in * uhat_vec).sum(dim=-1, keepdim=True)
    du_out = (v_out * uhat_vec).sum(dim=-1, keepdim=True)
    v_in_perp = v_in - du_in * uhat_vec
    v_out_perp = v_out - du_out * uhat_vec

    nin = v_in_perp.norm(dim=-1, keepdim=True)
    nout = v_out_perp.norm(dim=-1, keepdim=True)
    v_in_unit = v_in_perp / (nin + eps)
    degenerate = nin <= eps
    if degenerate.any():
        v_in_unit = v_in_unit.clone()
        v_in_unit[degenerate.squeeze(-1)] = global_e1
    v_out_unit = v_out_perp / (nout + eps)

    cospsi = (v_in_unit * v_out_unit).sum(dim=-1)                   # [V]
    cross = th.cross(v_in_unit, v_out_unit, dim=-1)
    sinpsi = (cross * uhat_vec).sum(dim=-1)                         # [V]

    # --- 7. L/R decision per parent (compact space) ---
    # Map valid parents to compact indices [0..M_valid)
    valid_parent_unique, valid_inv = th.unique(valid_parent, return_inverse=True)  # [M_v], [V]
    M_v = valid_parent_unique.numel()

    # Product s0*s1 per parent: (sum_s)^2 - sum_s2) / 2
    sum_s = scatter(sinpsi, valid_inv, dim=0, dim_size=M_v, reduce='sum')
    sum_s2 = scatter(sinpsi ** 2, valid_inv, dim=0, dim_size=M_v, reduce='sum')
    product = (sum_s ** 2 - sum_s2) * 0.5                          # [M_v]

    lr_valid = th.zeros(V, dtype=th.bool, device=device)

    # Case 1: opposite-sign sines
    case1_parent = product < -tol                                   # [M_v]
    case1_nodes = case1_parent[valid_inv]                           # [V]
    if case1_nodes.any():
        lr_valid[case1_nodes] = sinpsi[case1_nodes] > 0

    # Case 2: same-sign or near-zero
    case2_nodes = ~case1_nodes                                      # [V]
    if case2_nodes.any():
        theta = th.atan2(sinpsi, cospsi)                            # [V]
        max_theta = scatter(
            theta[case2_nodes], valid_inv[case2_nodes],
            dim=0, dim_size=M_v, reduce='max',
        )
        lr_valid[case2_nodes] = (
            theta[case2_nodes] >= max_theta[valid_inv[case2_nodes]] - 1e-7
        )

    # --- 8. Write back into K-sized output ---
    lr_mask[valid] = lr_valid

    if debug:
        for j in range(M_v):
            sib_mask = valid_inv == j
            left_count = int(lr_valid[sib_mask].sum().item())
            if left_count != 1:
                parent_id = valid_parent_unique[j].item()
                logger.warning(
                    f"[GeoLR-new] Parent {parent_id} has {left_count} left "
                    f"assignments (expected 1)."
                )

    return lr_mask, valid


def compute_geo_angle_for_new_leaves(
    pos: th.Tensor,
    parent_idx: th.Tensor,
    new_leaf_idx: th.Tensor,
    *,
    uhat: th.Tensor | None = None,
    eps: float = 1e-8,
) -> Tuple[th.Tensor, th.Tensor]:
    """Compute ordinal angular feature for new leaves (generalizes compute_geo_lr_for_new_leaves).

    Handles any sibling count ≥ 1 (not just binary pairs).
    For root children: sorts clockwise from x-axis, assigns i/(k-1).
    For binary children: uses parity logic, left=0.0, right=1.0.
    For single children: assigns 0.0.

    Returns:
        (geo_angle [K] float, valid [K] bool)
        - geo_angle: ordinal feature i/(k-1) for each new leaf
        - valid: True for leaves where the computation succeeded
    """
    K = new_leaf_idx.numel()
    device = pos.device
    dtype = pos.dtype

    geo_angle = th.zeros(K, dtype=dtype, device=device)
    valid = th.zeros(K, dtype=th.bool, device=device)

    if K == 0:
        return geo_angle, valid

    D = pos.size(1)
    if uhat is None:
        uhat = pos.new_zeros((D,), dtype=dtype)
        uhat[-1] = 1.0

    parent_of_leaf = parent_idx[new_leaf_idx]
    unique_parents, inv = th.unique(parent_of_leaf, return_inverse=True)
    M = unique_parents.numel()
    counts = scatter(th.ones(K, device=device), inv, dim=0, dim_size=M, reduce='sum')
    sibling_count = counts[inv]

    # All leaves with at least 1 sibling are valid
    valid = sibling_count >= 1

    uhat_vec = uhat.view(1, -1)

    for j in range(M):
        group_mask = inv == j
        group_indices = group_mask.nonzero(as_tuple=False).flatten()
        k = int(counts[j].item())
        parent_node = unique_parents[j].item()

        if k == 1:
            geo_angle[group_indices[0]] = 0.0
            valid[group_indices[0]] = True
            continue

        # Get offsets from parent, project onto perp plane
        leaf_globals = new_leaf_idx[group_indices]
        offsets = pos[leaf_globals] - pos[parent_node].unsqueeze(0)

        # Check if parent is root
        is_root_parent = parent_idx[parent_node] < 0

        if is_root_parent:
            # SO(2)-invariant ordering by uhat component
            sorted_idx, _fwd0, _delta = _order_root_children_by_uhat(
                offsets, uhat, eps=eps,
            )
            for rank, si in enumerate(sorted_idx.tolist()):
                geo_angle[group_indices[si]] = rank / max(k - 1, 1)
        elif k == 2:
            # Binary: use the existing parity approach (simplified)
            # Compute v_in for the parent
            du = (offsets * uhat_vec).sum(dim=-1, keepdim=True)
            offsets_perp = offsets - du * uhat_vec

            gp = parent_idx[parent_node]
            if gp >= 0:
                v_in = pos[parent_node] - pos[gp]
            else:
                # Non-root degenerate fallback: global axis is acceptable here
                e1, _ = global_inplane_basis(uhat, eps=eps)
                v_in = e1
            du_in = (v_in * uhat.view(-1)).sum()
            v_in_perp = v_in - du_in * uhat
            nin = v_in_perp.norm()
            if nin <= eps:
                # Degenerate: global axis fallback (non-root, rare)
                e1, _ = global_inplane_basis(uhat, eps=eps)
                v_in_unit = e1
            else:
                v_in_unit = v_in_perp / nin

            # sin of angle between v_in and each child's v_out
            for idx_local in range(2):
                gi = group_indices[idx_local]
                v_out = offsets_perp[idx_local]
                nout = v_out.norm()
                if nout <= eps:
                    continue
                v_out_unit = v_out / nout
                cross = th.cross(v_in_unit, v_out_unit)
                sin_val = (cross * uhat).sum()
                # left (sin > 0) = 0.0, right = 1.0
                geo_angle[gi] = 0.0 if sin_val > 0 else 1.0

            # Ensure exactly one is 0.0 and one is 1.0
            vals = geo_angle[group_indices]
            if vals[0] == vals[1]:
                geo_angle[group_indices[1]] = 1.0 - geo_angle[group_indices[0]]

    return geo_angle, valid


def compute_geo_lr_mask_f2(
    pos: th.Tensor,
    parent_idx: th.Tensor,
    *,
    debug: bool = False,
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
    global_e1, _ = global_inplane_basis(uhat, eps=eps)
    uhat_vec = uhat.view(1, -1)

    v_in = th.zeros((N, pos.size(1)), device=device, dtype=dtype)
    has_gp_mask = gp >= 0
    if has_gp_mask.any():
        sel = has_gp_mask.nonzero(as_tuple=False).flatten()
        v_in[sel] = pos[parent[sel]] - pos[gp[sel]]
    fallback_mask = has_parent & ~has_gp_mask

    v_out = th.zeros((N, pos.size(1)), device=device, dtype=dtype)
    if has_parent.any():
        sel = has_parent.nonzero(as_tuple=False).flatten()
        v_out[sel] = pos[sel] - pos[parent[sel]]

    if fallback_mask.any():
        # Root children: use v_out as v_in (geometric, no global axis).
        # Unused for their L/R decision (Z-comparison path); yields sinψ=0 for sinψ codepath.
        v_in[fallback_mask] = v_out[fallback_mask]

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

    handled_parents = th.zeros((N,), dtype=th.bool, device=device)
    root_nodes = (parent == -1).nonzero(as_tuple=False).flatten()
    if not root_nodes.numel():
        logger.warning("[GeoLR] No root with parent==-1 found; override skipped for this graph.")
    else:
        for r in root_nodes.tolist():
            child_idx = (parent == r).nonzero(as_tuple=False).flatten()
            if child_idx.numel() == 0:
                continue
            if child_idx.numel() == 2:
                child_z = pos[child_idx, -1]
                if float(child_z[0].item()) <= float(child_z[1].item()):
                    left_idx = int(child_idx[0].item())
                    right_idx = int(child_idx[1].item())
                else:
                    left_idx = int(child_idx[1].item())
                    right_idx = int(child_idx[0].item())
                lr_mask[left_idx] = True
                lr_mask[right_idx] = False
            else:
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

    if debug:
        for p in unique_parents.tolist():
            if p < 0:
                continue
            child_idx = (parent == p).nonzero(as_tuple=False).flatten()
            if child_idx.numel() != 2:
                continue
            left_count = int(lr_mask[child_idx].sum().item())
            if left_count != 1:
                logger.warning(
                    f"[GeoLR] Parent {p} has {left_count} left assignments (expected 1)."
                )

    return lr_mask


def precompute_full_geometry(
    pos: th.Tensor,
    parent_idx: th.Tensor,
    edge_index: th.Tensor,
    uhat: th.Tensor,
    *,
    eps: float = 1e-8,
    tol: float = 1e-6,
    debug: bool = False,
) -> dict:
    """Compute all geometry (geo_lr_mask + SO(2) edge/node features) once on P_0.

    Returns a dict compatible with the ``pre_geom`` format expected by
    ``SO2_EGNN.forward()`` plus extras needed for leaf-patching.
    """
    # 1. geo_lr_mask (node feature for left/right sibling assignment)
    geo_lr_mask = compute_geo_lr_mask(pos, parent_idx, uhat=uhat, debug=debug, eps=eps, tol=tol)

    # 1b. Root child angular ordering and per-child relative angles
    geo_ordinal, geo_delta_theta = compute_root_child_angles(
        pos, parent_idx, uhat, geo_lr_mask, eps=eps,
    )

    # 2. Node-level branch angles with intermediates for later patching
    cospsi_node, sinpsi_node, cos_theta_node, intermediates = compute_branch_angles_parent_centric(
        pos, parent_idx, uhat, eps=eps, return_intermediates=True,
    )

    # 3. Edge-level SO(2) decomposition
    src, dst = edge_index
    rel_coors = pos[dst] - pos[src]                         # (E, 3)
    du = (rel_coors @ uhat)                                  # (E,)
    r_par = du[:, None] * uhat                               # (E, 3)
    r_perp = rel_coors - r_par                               # (E, 3)
    rho = r_perp.norm(dim=-1, keepdim=True).clamp_min(eps)   # (E, 1)
    du = du[:, None]                                         # (E, 1)

    # 4. Assign node angles to edges
    cospsi_edge, sinpsi_edge = assign_branch_angles_to_edges(
        edge_index, parent_idx, cospsi_node, sinpsi_node,
    )
    cos_theta_edge = assign_parent_scalar_to_edges(
        edge_index, parent_idx, cos_theta_node,
    )

    # 5. Local coordinate bases for SO(2)-equivariant loss (with per-child root frames)
    local_bases = compute_local_bases(
        pos, parent_idx, uhat, geo_lr_mask, eps=eps,
        geo_delta_theta=geo_delta_theta,
    )

    return {
        # edge-level (used by SO2_EGNN layers)
        'rel_coors': rel_coors,
        'r_perp': r_perp,
        'rho': rho,
        'du': du,
        'cospsi_edge': cospsi_edge,
        'sinpsi_edge': sinpsi_edge,
        'cos_theta_edge': cos_theta_edge,
        # node-level
        'cospsi_node': cospsi_node,
        'sinpsi_node': sinpsi_node,
        'cos_theta_node': cos_theta_node,
        # for geo_lr feature (kept for binary interior children)
        'geo_lr_mask': geo_lr_mask,
        # angular ordering for root children
        'geo_ordinal': geo_ordinal,
        'geo_delta_theta': geo_delta_theta,
        # intermediates for leaf patching
        'v_in': intermediates['v_in'],
        'v_out': intermediates['v_out'],
        'has_gp': intermediates['has_gp'],
        # local bases for SO(2)-equivariant loss
        'local_forward': local_bases['local_forward'],
        'local_sideways': local_bases['local_sideways'],
    }


def patch_geometry_for_noised_leaves(
    pre_geom_p0: dict,
    P_t: th.Tensor,
    leaf_idx_train: th.Tensor,
    parent_idx: th.Tensor,
    edge_index: th.Tensor,
    uhat: th.Tensor,
    *,
    eps: float = 1e-8,
) -> dict:
    """Patch P_0 geometry for noised leaf positions in P_t.

    Only leaf-related node angles and affected edges are recomputed;
    everything else is reused from *pre_geom_p0*.
    """
    if leaf_idx_train.numel() == 0:
        return pre_geom_p0

    device = P_t.device
    N = P_t.size(0)
    src, dst = edge_index

    # --- 1. Identify affected edges (any edge touching a noised leaf) ---
    leaf_set = th.zeros(N, dtype=th.bool, device=device)
    leaf_set[leaf_idx_train] = True
    affected = leaf_set[src] | leaf_set[dst]  # (E,) bool

    # --- 2. Patch node-level angles for leaf nodes ---
    parent = parent_idx
    parent_clamped = parent.clamp(min=0)

    # v_out_new for leaves: P_t[leaf] - P_t[parent[leaf]]
    # (parent pos is unchanged in P_t since parents are internal)
    v_out_new = P_t[leaf_idx_train] - P_t[parent_clamped[leaf_idx_train]]  # (L, 3)

    # v_in reused from P_0 for most leaves (parent/grandparent are internal, unchanged)
    v_in_p0 = pre_geom_p0['v_in']                           # (N, 3)
    has_gp_p0 = pre_geom_p0['has_gp']                       # (N,) bool
    v_in_leaf = v_in_p0[leaf_idx_train]                      # (L, 3)

    # Root-child leaves: v_in is locked to the P_0-based per-child frame
    # (computed in compute_local_bases with geo_delta_theta). Do NOT update
    # with noised v_out — the frame must remain stable through diffusion.

    # Perpendicular projections
    du_in = (v_in_leaf @ uhat).unsqueeze(-1)                 # (L, 1)
    du_out = (v_out_new @ uhat).unsqueeze(-1)                # (L, 1)
    v_in_perp = v_in_leaf - du_in * uhat                     # (L, 3)
    v_out_perp = v_out_new - du_out * uhat                   # (L, 3)

    nin = v_in_perp.norm(dim=-1, keepdim=True)
    nout = v_out_perp.norm(dim=-1, keepdim=True)
    v_in_unit = v_in_perp / (nin + eps)
    v_out_unit = v_out_perp / (nout + eps)

    cospsi_leaf = (v_in_unit * v_out_unit).sum(dim=-1, keepdim=True)
    cross = th.cross(v_in_unit, v_out_unit, dim=-1)
    sinpsi_leaf = (cross * uhat).sum(dim=-1, keepdim=True)

    v_out_norm = (nout.pow(2) + du_out.pow(2)).sqrt()
    cos_theta_leaf = du_out / (v_out_norm + eps)

    # For nodes without a valid parent, angles are trivial (shouldn't happen for leaves, but safety)
    has_parent_leaf = parent[leaf_idx_train] >= 0
    hp = has_parent_leaf.view(-1, 1)
    cospsi_leaf = th.where(hp, cospsi_leaf, th.ones_like(cospsi_leaf))
    sinpsi_leaf = th.where(hp, sinpsi_leaf, th.zeros_like(sinpsi_leaf))
    cos_theta_leaf = th.where(hp, cos_theta_leaf, th.ones_like(cos_theta_leaf))

    # Clone node-level tensors and scatter patched values at leaf indices
    cospsi_node = pre_geom_p0['cospsi_node'].clone()
    sinpsi_node = pre_geom_p0['sinpsi_node'].clone()
    cos_theta_node = pre_geom_p0['cos_theta_node'].clone()
    cospsi_node[leaf_idx_train] = cospsi_leaf
    sinpsi_node[leaf_idx_train] = sinpsi_leaf
    cos_theta_node[leaf_idx_train] = cos_theta_leaf

    # --- 3. Patch edge-level quantities for affected edges ---
    rel_coors = pre_geom_p0['rel_coors'].clone()
    rel_coors_new = P_t[dst[affected]] - P_t[src[affected]]
    rel_coors[affected] = rel_coors_new

    du_edge = pre_geom_p0['du'].clone()                      # (E, 1)
    rho_edge = pre_geom_p0['rho'].clone()                    # (E, 1)
    r_perp_edge = pre_geom_p0['r_perp'].clone()              # (E, 3)

    du_new = (rel_coors_new @ uhat).unsqueeze(-1)            # (A, 1)
    r_par_new = du_new * uhat                                # (A, 3)
    r_perp_new = rel_coors_new - r_par_new                   # (A, 3)
    rho_new = r_perp_new.norm(dim=-1, keepdim=True).clamp_min(eps)  # (A, 1)

    du_edge[affected] = du_new
    rho_edge[affected] = rho_new
    r_perp_edge[affected] = r_perp_new

    # Reassign edge angle features from patched node angles
    cospsi_edge, sinpsi_edge = assign_branch_angles_to_edges(
        edge_index, parent_idx, cospsi_node, sinpsi_node,
    )
    cos_theta_edge = assign_parent_scalar_to_edges(
        edge_index, parent_idx, cos_theta_node,
    )

    return {
        'rel_coors': rel_coors,
        'r_perp': r_perp_edge,
        'rho': rho_edge,
        'du': du_edge,
        'cospsi_edge': cospsi_edge,
        'sinpsi_edge': sinpsi_edge,
        'cos_theta_edge': cos_theta_edge,
        'cospsi_node': cospsi_node,
        'sinpsi_node': sinpsi_node,
        'cos_theta_node': cos_theta_node,
    }


def decode_parent_indices(batch) -> th.Tensor:
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


def select_training_leaf_indices(batch, candidate_attr: str = "new_leaf_idx_from_next") -> th.Tensor:
    """Return indices of leaves that should contribute to masking/loss."""
    base = getattr(batch, "leaf_idx", None)
    if base is None:
        raise ValueError("Expected batch.leaf_idx to select leaves for training.")
    candidate = getattr(batch, candidate_attr, None)
    if candidate is None:
        raise ValueError(
            f"Expected batch.{candidate_attr} to be set (even if empty), "
            f"but got None. Check your dataset/collation."
        )
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


def leaf_rel_targets(
    pos_gt: th.Tensor,
    leaf_idx: th.Tensor,
    leaf_parent_idx: th.Tensor,
) -> th.Tensor:
    """Compute parent-relative targets for leaves."""
    if leaf_idx.numel() == 0:
        return pos_gt.new_zeros((0, 3))
    parent_pos = pos_gt[leaf_parent_idx]
    return pos_gt[leaf_idx] - parent_pos


def plot_diffusion_debug_trees(
    *,
    pos: th.Tensor,
    parent_idx: th.Tensor,
    batch_vec: th.Tensor,
    leaf_idx_all: th.Tensor,
    leaf_idx_train: th.Tensor,
    geo_lr_mask: th.Tensor,
    leaf_targets_per_node: Optional[th.Tensor] = None,
    out_dir: Optional[Path] = None,
) -> list[Path]:
    """Plot per-graph tree layouts with node coloring and a metadata table."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("[DebugPlot] matplotlib unavailable: %s", exc)
        return []

    if pos.numel() == 0:
        return []
    if out_dir is None:
        root_dir = Path(__file__).resolve().parents[2]
        out_dir = root_dir / "debug_plots" / "diffusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_cpu = pos.detach().cpu()
    parent_cpu = parent_idx.detach().cpu()
    batch_cpu = batch_vec.detach().cpu()
    leaf_all_cpu = leaf_idx_all.detach().cpu()
    leaf_train_cpu = leaf_idx_train.detach().cpu()
    geo_lr_cpu = geo_lr_mask.detach().cpu()
    leaf_targets_cpu = None
    if leaf_targets_per_node is not None:
        leaf_targets_cpu = leaf_targets_per_node.detach().cpu()

    num_graphs = int(batch_cpu.max().item()) + 1 if batch_cpu.numel() else 0
    leaf_all_set = set(leaf_all_cpu.tolist())
    leaf_train_set = set(leaf_train_cpu.tolist())
    saved: list[Path] = []

    for graph_id in range(num_graphs):
        node_mask = batch_cpu == graph_id
        node_idx = node_mask.nonzero(as_tuple=False).flatten()
        if node_idx.numel() == 0:
            continue
        node_list = node_idx.tolist()
        idx_to_local = {idx: i for i, idx in enumerate(node_list)}
        node_set = set(node_list)
        num_nodes = len(node_list)

        pos_xyz = pos_cpu[node_idx, :3] if pos_cpu.size(1) >= 3 else pos_cpu[node_idx]
        if pos_xyz.size(1) == 1:
            pos_xyz = th.cat([pos_xyz, th.zeros_like(pos_xyz), th.zeros_like(pos_xyz)], dim=1)
        elif pos_xyz.size(1) == 2:
            pos_xyz = th.cat([pos_xyz, th.zeros_like(pos_xyz[:, :1])], dim=1)
        xs = pos_xyz[:, 0].numpy()
        ys = pos_xyz[:, 1].numpy()
        zs = pos_xyz[:, 2].numpy()

        colors: list[str] = []
        for n in node_list:
            if parent_cpu[n].item() < 0:
                colors.append("#ffd700")
            elif n in leaf_train_set:
                colors.append("#d62728")
            elif n in leaf_all_set:
                colors.append("#2ca02c")
            else:
                colors.append("#1f77b4")

        width = max(12.0, min(0.5 * num_nodes, 24.0))
        height = max(10.0, min(0.4 * num_nodes, 20.0))
        fig = plt.figure(figsize=(width, height))
        ax_graph = fig.add_subplot(2, 1, 1, projection="3d")
        ax_table = fig.add_subplot(2, 1, 2)

        for child in node_list:
            parent = int(parent_cpu[child].item())
            if parent < 0 or parent not in node_set:
                continue
            child_idx = idx_to_local[child]
            parent_idx_local = idx_to_local[parent]
            ax_graph.plot(
                [xs[parent_idx_local], xs[child_idx]],
                [ys[parent_idx_local], ys[child_idx]],
                [zs[parent_idx_local], zs[child_idx]],
                color="#b0b0b0",
                linewidth=1.0,
                zorder=1,
            )

        ax_graph.scatter(xs, ys, zs, s=80, c=colors, edgecolors="black", linewidths=0.5, zorder=2)
        for i, n in enumerate(node_list):
            ax_graph.text(
                xs[i],
                ys[i],
                zs[i],
                str(n),
                fontsize=8,
                ha="center",
                va="center",
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                zorder=3,
            )
        ax_graph.set_title(f"Graph {graph_id} (nodes={num_nodes})")
        ax_graph.set_axis_off()
        ax_graph.view_init(elev=20, azim=45)
        try:
            ax_graph.set_box_aspect((1, 1, 1))
        except Exception:
            pass

        table_rows = []
        for n in node_list:
            pos_vals = pos_cpu[n]
            if parent_cpu[n].item() < 0:
                geo_lr = "root"
            else:
                geo_lr = "L" if bool(geo_lr_cpu[n].item()) else "R"
            parent_val = int(parent_cpu[n].item())
            if leaf_targets_cpu is None:
                exp_state = "N/A"
            else:
                val = int(leaf_targets_cpu[n].item())
                exp_state = str(val) if val >= 0 else "N/A"
            x_val = f"{float(pos_vals[0].item()):.3f}"
            y_val = f"{float(pos_vals[1].item()):.3f}" if pos_vals.numel() > 1 else "0.000"
            z_val = f"{float(pos_vals[2].item()):.3f}" if pos_vals.numel() > 2 else "0.000"
            table_rows.append([str(n), x_val, y_val, z_val, str(parent_val), geo_lr, exp_state])

        ax_table.axis("off")
        table = ax_table.table(
            cellText=table_rows,
            colLabels=["node", "x", "y", "z", "parent_idx", "geo_lr", "expansion"],
            loc="center",
        )
        table.auto_set_font_size(False)
        table_font = max(6, 10 - num_nodes // 20)
        table.set_fontsize(table_font)
        table.scale(1.0, 1.2)

        fig.tight_layout()
        plot_id = next(_DIFFUSION_PLOT_COUNTER)
        out_file = out_dir / f"tree_n{num_nodes}_id{plot_id}.png"
        fig.savefig(out_file, dpi=150)
        plt.close(fig)
        saved.append(out_file)

    return saved
