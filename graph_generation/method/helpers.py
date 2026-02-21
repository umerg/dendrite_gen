import itertools
import logging
from pathlib import Path
from typing import Optional, Tuple

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


def compute_geo_lr_mask(
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

    handled_parents = th.zeros((N,), dtype=th.bool, device=device)
    root_nodes = (parent == -1).nonzero(as_tuple=False).flatten()
    if not root_nodes.numel():
        logger.warning("[GeoLR] No root with parent==-1 found; override skipped for this graph.")
    else:
        for r in root_nodes.tolist():
            child_idx = (parent == r).nonzero(as_tuple=False).flatten()
            if child_idx.numel() == 0:
                continue
            if child_idx.numel() == 1:
                lr_mask[child_idx[0]] = True  # single child: assign as left (index 0)
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
