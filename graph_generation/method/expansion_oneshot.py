import networkx as nx
import torch as th
from torch.nn import Module
import torch.nn.functional as F
from torch_scatter import scatter
from torch_sparse import SparseTensor
import logging

logger = logging.getLogger(__name__)

from .method import Method

class Expansion_OneShot(Method):
    """Graph generation method generating graphs by local expansion."""

    EDGE_PARENT_TO_CHILD = 0
    EDGE_CHILD_TO_PARENT = 1

    def __init__(
        self,
        deterministic_expansion: bool = False,      # just sets seeding for reproducibility
        red_threshold: int = 0,
        leaf_noise_sigma: float = 0.05,             # <-- stddev of Gaussian around parent (same units as pos)
        leaf_noise_clip: float | None = None,       # <-- optional radius clamp (float) or None
        sibling_loss_weight: float = 0.8,           # weight for sibling distance regularizer
        use_sibling_matching: bool = False,         # if True, use per-parent matching for positional loss
        debug: bool = False,
        debug_max_batches: int = 2,
        debug_dir: str | None = None,
    ):
        super().__init__(diffusion=None)
        self.deterministic_expansion = deterministic_expansion
        self.red_threshold = red_threshold
        self.leaf_noise_sigma = float(leaf_noise_sigma)
        self.leaf_noise_clip = leaf_noise_clip
        self.sibling_loss_weight = float(sibling_loss_weight)
        self.use_sibling_matching = bool(use_sibling_matching)
        self.debug = debug
        self._debug_step = 0
        from pathlib import Path as _P
        self.debug_dir = _P(debug_dir) if debug_dir is not None else _P.cwd() / "debug_graphs"
        self.debug_max_batches = debug_max_batches
        if self.debug:
            logger.info(f"Expansion_OneShot debug enabled; plots will be saved under: {self.debug_dir}")
    
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

    def _maybe_debug_root_children(
        self,
        batch,
        parent_idx: th.Tensor,
        geo_lr_mask: th.Tensor,
        leaf_idx_train: th.Tensor,
        pos_gt: th.Tensor,
        pos_masked: th.Tensor,
    ) -> None:
        """Capture debug artifacts for graphs reduced to root + two children."""
        if not getattr(self, "debug", False):
            return
        batch_vec = getattr(batch, "batch", None)
        if batch_vec is None:
            return
        try:
            from utils.debug_helpers import log_root_children_debug
        except Exception:
            return

        sibling_order = getattr(batch, "sibling_order", None)
        sibling_order = sibling_order.to(parent_idx.device) if sibling_order is not None else None
        leaf_idx_all = getattr(batch, "leaf_idx", None)
        leaf_all_set = set(leaf_idx_all.tolist()) if leaf_idx_all is not None else set()
        leaf_expansion_all = getattr(batch, "leaf_expansion", None)
        leaf_expansion_per_node = None
        if leaf_idx_all is not None and leaf_expansion_all is not None:
            leaf_idx_tensor = leaf_idx_all.to(device=parent_idx.device, dtype=th.long)
            if isinstance(leaf_expansion_all, th.Tensor):
                leaf_expansion_tensor = leaf_expansion_all.to(device=parent_idx.device)
            else:
                leaf_expansion_tensor = th.as_tensor(leaf_expansion_all).to(device=parent_idx.device)
            leaf_expansion_tensor = leaf_expansion_tensor.view(-1)
            if leaf_idx_tensor.numel() == leaf_expansion_tensor.numel():
                leaf_expansion_per_node = leaf_expansion_tensor.new_full(
                    (parent_idx.size(0),), -1
                )
                leaf_expansion_per_node[leaf_idx_tensor] = leaf_expansion_tensor
        leaf_train_set = set(leaf_idx_train.tolist()) if leaf_idx_train.numel() > 0 else set()
        new_leaf_idx = getattr(batch, "new_leaf_idx_from_next", None)
        new_leaf_set = set(new_leaf_idx.tolist()) if new_leaf_idx is not None else set()

        limit = getattr(self, "debug_max_batches", None)
        if limit is not None and limit < 0:
            limit = None

        matches_by_size = {}
        matches = []
        unique_graphs = batch_vec.unique(sorted=True)
        for graph_id in unique_graphs.tolist():
            graph_mask = (batch_vec == graph_id)
            node_idx = graph_mask.nonzero(as_tuple=False).flatten()
            graph_size = int(node_idx.numel())
            if graph_size < 9 or graph_size > 13: # change debug graph size range here
                continue
            node_ids = node_idx.tolist()
            global_to_local = {int(g): i for i, g in enumerate(node_ids)}
            parent_global = parent_idx[node_idx].tolist()
            parent_local = [-1 if p < 0 else global_to_local.get(int(p), -1) for p in parent_global]
            parent_local_tensor = th.tensor(parent_local, dtype=th.long, device=parent_idx.device)
            root_local_mask = parent_local_tensor < 0
            if root_local_mask.sum().item() != 1:
                continue
            root_local = int(root_local_mask.nonzero(as_tuple=False)[0].item())
            child_count = int((parent_local_tensor == root_local).sum().item())
            if graph_size == 3 and child_count != 2:
                continue

            geo_local = geo_lr_mask[node_idx]
            pos_gt_local = pos_gt[node_idx]
            pos_mask_local = pos_masked[node_idx]
            leaf_expansion_local = (
                leaf_expansion_per_node[node_idx].detach().cpu()
                if leaf_expansion_per_node is not None
                else None
            )

            def _build_mask(id_set):
                return th.tensor(
                    [gid in id_set for gid in node_ids],
                    dtype=th.bool,
                    device=parent_idx.device,
                )

            leaf_mask_local = _build_mask(leaf_all_set) if leaf_idx_all is not None else th.zeros(
                len(node_ids), dtype=th.bool, device=parent_idx.device
            )
            leaf_train_mask_local = _build_mask(leaf_train_set) if leaf_train_set else th.zeros(
                len(node_ids), dtype=th.bool, device=parent_idx.device
            )
            new_leaf_mask_local = _build_mask(new_leaf_set) if new_leaf_set else None

            edges = []
            for child_local, parent_local_idx in enumerate(parent_local):
                if parent_local_idx >= 0:
                    edges.append((child_local, parent_local_idx))
                    edges.append((parent_local_idx, child_local))
            if edges:
                edge_index = th.tensor(edges, dtype=th.long, device=parent_idx.device).t()
            else:
                edge_index = th.empty((2, 0), dtype=th.long, device=parent_idx.device)
            adj_local = SparseTensor(
                row=edge_index[0],
                col=edge_index[1],
                sparse_sizes=(len(node_ids), len(node_ids)),
            )

            sibling_local = (
                sibling_order[node_idx].detach().cpu()
                if sibling_order is not None
                else th.full((graph_size,), -1, dtype=th.long)
            )

            matches_by_size.setdefault(graph_size, 0)
            matches_by_size[graph_size] += 1
            matches.append(
                {
                    "graph_id": graph_id,
                    "node_ids": node_ids,
                    "parent_local": parent_local_tensor.detach().cpu(),
                    "pos_gt": pos_gt_local.detach().cpu(),
                    "pos_masked": pos_mask_local.detach().cpu(),
                    "geo_lr_mask": geo_local.detach().cpu(),
                    "leaf_mask": leaf_mask_local.detach().cpu(),
                    "leaf_train_mask": leaf_train_mask_local.detach().cpu(),
                    "new_leaf_mask": new_leaf_mask_local.detach().cpu()
                    if new_leaf_mask_local is not None
                    else None,
                    "geo_lr_mask": geo_local.detach().cpu(),
                    "sibling_order": sibling_local,
                    "leaf_expansion_state": leaf_expansion_local,
                    "adj": adj_local.cpu(),
                    "graph_size": graph_size,
                }
            )

        total_matches = len(matches)
        if total_matches == 0:
            return

        logged_this_call = 0
        for item in matches:
            if limit is not None and self._debug_step >= limit:
                break
            log_root_children_debug(
                out_dir=self.debug_dir / "root_children",
                step=self._debug_step,
                batch_index=int(self._debug_step),
                graph_index=item["graph_id"],
                node_ids=item["node_ids"],
                parent_local=item["parent_local"],
                pos_gt=item["pos_gt"],
                pos_masked=item["pos_masked"],
                geo_lr_mask=item["geo_lr_mask"],
                sibling_order=item["sibling_order"],
                leaf_mask=item["leaf_mask"],
                leaf_train_mask=item["leaf_train_mask"],
                new_leaf_mask=item["new_leaf_mask"],
                leaf_expansion_state=item["leaf_expansion_state"],
                adj=item["adj"],
                graph_size=item["graph_size"],
            )
            self._debug_step += 1
            logged_this_call += 1

        logger.info(
            "[RootChildrenDebug] small-graph counts this batch: %s; logged: %d (limit=%s, total_logged=%d)",
            ", ".join(f"{size}:{count}" for size, count in sorted(matches_by_size.items())),
            logged_this_call,
            "∞" if limit is None else str(limit),
            self._debug_step,
        )

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
          graphs: list[networkx.Graph] of length G. Each node carries:
              - 'pos': np.ndarray (3,) geometric position
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
        # Persistent sibling order: -1 for nodes with no sibling assignment yet
        sibling_order = th.full((num_graphs,), -1, device=device, dtype=th.long)
        leaf_mask = th.ones((num_graphs,), device=device, dtype=th.bool)

        # Safety max steps: enough to reach capacity even if only one leaf expands each time.
        max_steps = int(target_size.max().item() * 2)  # generous upper bound
        step = 0
        terminated = False

        while not terminated and step < max_steps:
            adj, pos, leaf_idx, leaf_expansion, parent_idx_1b, batch, sibling_order, leaf_mask, terminated = self.expand(
                adj,
                batch,
                target_size,
                model,
                pos=pos,
                leaf_idx=leaf_idx,
                leaf_expansion=leaf_expansion,
                parent_idx_1b=parent_idx_1b,
                sibling_order=sibling_order,
                leaf_mask=leaf_mask,
                step=step,
                ensure_progress=False,
                map_threshold=0.5,
            )
            step += 1

        # ---- Unbatch into per-graph geometric NetworkX graphs ----
        # Extract COO for adjacency and segment by batch
        row, col, val = adj.coo()
        graphs = []
        for g in range(num_graphs):
            g_mask = (batch == g)
            node_ids = th.nonzero(g_mask, as_tuple=False).flatten()
            # Map global node index -> local index
            local_map = {int(n.item()): i for i, n in enumerate(node_ids)}
            G = nx.Graph()
            # Add nodes with geometric position only
            for i_local, n_global in enumerate(node_ids.tolist()):
                G.add_node(
                    i_local,
                    pos=pos[n_global].detach().cpu().numpy(),
                )
            # Add edges where both endpoints belong to this graph (undirected, avoid duplicates)
            for r, c in zip(row.tolist(), col.tolist()):
                if r in local_map and c in local_map:
                    # Undirected; ensure single edge by adding only if r<=c
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
        sibling_order: th.Tensor | None = None,
        leaf_mask: th.Tensor | None = None,
        step: int = 0,
        ensure_progress: bool = False,
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
        if leaf_mask is None:
            leaf_mask = th.ones((pos.size(0),), device=device, dtype=th.bool)
        else:
            leaf_mask = leaf_mask.to(device=device)
            if leaf_mask.dtype != th.bool:
                leaf_mask = leaf_mask.bool()
        
        # -----ENFORCING DETERMINISTIC CONDITIONS-----

        # 2) Per-graph current size & remaining slots
        size_per_graph = scatter(th.ones_like(batch_reduced), batch_reduced)
        remaining_capacity = target_size.to(device) - size_per_graph
        if sibling_order is None:
            sibling_order = th.full((pos.size(0),), -1, device=device, dtype=th.long)

        if (remaining_capacity <= 0).all() or leaf_idx.numel() == 0:
            return (
                adj_reduced,
                pos,
                leaf_idx.clone(),
                leaf_expansion.clone(),
                parent_idx_1b,
                batch_reduced,
                sibling_order.clone(),
                leaf_mask.clone(),
                True,
            )

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
        if ensure_progress and (remaining_capacity >= 2).any():
            # Guarantee per-graph progress: for every graph that
            # (a) still has capacity for at least one 2-branch AND
            # (b) has at least one current leaf, AND
            # (c) has no leaf already scheduled to expand this step,
            # force exactly one leaf to expand (add 2 children).
            # This prevents graphs from stalling at size=3 (root + 2 children)
            # when the model prematurely predicts all leaves as terminal.
            num_graphs = int(target_size.numel())
            for g in range(num_graphs):
                if remaining_capacity[g] < 2:
                    continue  # not enough room for a binary expansion
                mask_g = (leaf_batch == g)
                if not mask_g.any():
                    continue  # no leaves to expand in this graph
                if (spawn_counts_final[mask_g] == 2).any():
                    continue  # already at least one expansion scheduled for this graph
                # Force the first leaf (deterministically or random) to expand
                leaf_indices_g = th.nonzero(mask_g, as_tuple=False).flatten()
                if self.deterministic_expansion:
                    # Deterministic ordering: pick smallest index (first in tensor)
                    forced_leaf = leaf_indices_g[0]
                else:
                    # Random selection among available leaves for diversity
                    rand_idx = th.randint(low=0, high=leaf_indices_g.numel(), size=(1,), device=leaf_indices_g.device)
                    forced_leaf = leaf_indices_g[rand_idx]
                spawn_counts_final[forced_leaf] = 2

        # 6) Count new children & early exit
        total_new_children = int(spawn_counts_final.sum().item())
        if total_new_children == 0:
            return (
                adj_reduced,
                pos,
                leaf_idx.clone(),
                leaf_expansion.clone(),
                parent_idx_1b,
                batch_reduced,
                sibling_order.clone(),
                leaf_mask.clone(),
                True,
            )

        # ----- EXPANSION -----

        # 7) Materialize children (positions, parents, batch ids)
        base_N = adj_reduced.size(0)
        new_child_positions = []
        leaf_mask_updated = leaf_mask.clone()
        expanded_mask = spawn_counts_final == 2
        if expanded_mask.any():
            leaf_mask_updated[leaf_idx[expanded_mask]] = False
        new_child_parents = []
        new_child_batches = []
        parent_child_edges = []
        sibling_order_new = []  # track sibling order for newly created children (0=left,1=right)
        running_child_index = 0
        for li, sc in zip(leaf_idx.tolist(), spawn_counts_final.tolist()):
            if sc == 0:
                continue
            parent_pos = pos[li]
            # Use shared sampler for consistency with training masking
            noise = self._sample_noise((sc, parent_pos.shape[0]), device, sigma=self.leaf_noise_sigma, clip=self.leaf_noise_clip)
            child_pos = parent_pos.unsqueeze(0) + noise
            new_child_positions.append(child_pos)
            # record sibling order (0 for first, 1 for second) for binary branching
            for local_child in range(sc):
                global_child_idx = base_N + running_child_index
                parent_child_edges.append((li, global_child_idx))
                new_child_parents.append(li)
                new_child_batches.append(int(batch_reduced[li].item()))
                # spawn_counts are only 0 or 2; if ever >2, local_child gives order anyway
                running_child_index += 1
                # collect sibling order
                # build list lazily to avoid prealloc; will map after concatenation
                # list defined earlier
                sibling_order_new.append(0 if local_child == 0 else 1)
        if running_child_index != total_new_children:
            raise ValueError("Child accounting mismatch: expected %d got %d" % (total_new_children, running_child_index))

        new_child_positions_tensor = th.cat(new_child_positions, dim=0) if new_child_positions else pos.new_empty((0, 3))
        pos_new = th.cat([pos, new_child_positions_tensor], dim=0)
        parent_idx_new_0b = th.cat([parent_idx, th.tensor(new_child_parents, device=device, dtype=parent_idx.dtype)])
        parent_idx_1b_new = parent_idx_new_0b + 1
        batch_new = th.cat([batch_reduced, th.tensor(new_child_batches, device=device, dtype=batch_reduced.dtype)])

        # build sibling order tensor: -1 for existing nodes, {0,1} for new children (left/right)
        # Persistent sibling order: keep previous assignments, append new children
        sibling_order_next = th.cat([sibling_order, th.tensor(sibling_order_new, device=device, dtype=th.long)], dim=0)

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
                sibling_order_next,
                leaf_mask_next,
                True,
            )

        num_graphs_total = int(target_size.numel())
        node_counts_per_graph = scatter(
            th.ones_like(batch_new),
            batch_new,
            dim=0,
            dim_size=num_graphs_total,
        )

        # ----- POSITION REFINEMENT FOR LEAVES & NEXT EXPANSION PREDICTION -----

        # 10) Model forward to refine child positions & predict next expansion states
        feats_dim = getattr(model, 'feats_dim', 0)
        pos_dim = getattr(model, 'pos_dim', 3)
        if feats_dim > 0:
            # feature 0: is_leaf flag
            is_leaf_flag = leaf_mask_next.to(dtype=pos_new.dtype).unsqueeze(-1)
            features = [is_leaf_flag]
            feats_used = 1
            # feature 1: sibling left flag (persistent) if capacity permits
            if feats_used < feats_dim:                
                so = sibling_order_next
                sib_is_left = (so == 0).float().unsqueeze(-1)
                sib_is_left = th.where(so.unsqueeze(-1) >= 0, sib_is_left, sib_is_left.new_zeros(sib_is_left.shape))
                features.append(sib_is_left)
                feats_used += 1
            # feature 2: indicator for freshly expanded leaves
            if feats_used < feats_dim:
                new_leaf_flag = pos_new.new_zeros((pos_new.size(0), 1))
                new_leaf_flag[leaf_idx_next] = 1.0
                features.append(new_leaf_flag)
                feats_used += 1
            # feature 3: current size / target size ratio broadcast per node
            if feats_used < feats_dim:
                target_size_float = target_size.to(pos_new.dtype)
                ratio_graph = node_counts_per_graph.to(pos_new.dtype) / target_size_float.clamp_min(1.0)
                ratio_nodes = ratio_graph[batch_new].unsqueeze(-1)
                features.append(ratio_nodes)
                feats_used += 1
            if feats_used < feats_dim:
                features.append(pos_new.new_zeros((pos_new.size(0), feats_dim - feats_used)))
            node_feats = th.cat(features, dim=-1)
            x_in = th.cat([pos_new[:, :pos_dim], node_feats], dim=-1)
        else:
            x_in = pos_new[:, :pos_dim]
        edge_index, edge_types = self._build_directed_edge_index(parent_idx_new_0b)
        if edge_types.numel():
            edge_attr = edge_types.unsqueeze(-1).to(x_in.dtype)
        else:
            edge_attr = x_in.new_zeros((0, 1))
        out = model(x=x_in, edge_index=edge_index, batch=batch_new, edge_attr=edge_attr, parent_idx=parent_idx_new_0b)
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

        expansion_prob = expansion_pred_leaves.squeeze(-1).sigmoid() # training loss currently does not use logits
        leaf_expansion_next = (expansion_prob > map_threshold).long() + 1

        # 11) Termination condition for next step
        remaining_capacity_new = target_size.to(device) - node_counts_per_graph
        terminated = (remaining_capacity_new < 2).all() or leaf_idx_next.numel() == 0

        return (
            adj_new,
            pos_new,
            leaf_idx_next,
            leaf_expansion_next,
            parent_idx_1b_new,
            batch_new,
            sibling_order_next,
            leaf_mask_next,
            terminated,
        )

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
    # 3) Optional permutation-aware positional loss via matching
    # ---------------------------------------------------------
    @staticmethod
    def _greedy_min_cost_matching(cost: th.Tensor) -> th.Tensor:
        """
        Simple greedy min-cost one-to-one matching.

        Args:
            cost: [K, K] cost matrix (assumed detached from autograd).

        Returns:
            assignment: LongTensor[K], where assignment[i] is the index of the
                        matched column (target) for row i (prediction).
        """
        K = int(cost.size(0))
        if K == 0:
            return cost.new_empty((0,), dtype=th.long)

        work = cost.clone()
        inf = (work.max().item() if work.numel() > 0 else 0.0) + 1.0
        assignment = work.new_full((K,), -1, dtype=th.long)

        for _ in range(K):
            flat = work.view(-1)
            _, idx = flat.min(dim=0)
            row = int(idx // K)
            col = int(idx % K)
            assignment[row] = col
            work[row, :] = inf
            work[:, col] = inf

        return assignment

    def _compute_leaf_pos_loss_with_matching(
        self,
        pred_rel: th.Tensor,          # [L,3] predicted parent-relative offsets for leaves
        tgt_rel: th.Tensor,           # [L,3] GT parent-relative offsets for leaves
        leaf_parent_idx: th.Tensor,   # [L] parent index per leaf (0-based)
    ) -> th.Tensor:
        """
        Per-parent permutation-invariant positional loss on leaves.

        For each parent:
          1) Collect its leaf children (pred_rel_group, tgt_rel_group),
          2) Build a K×K squared-distance cost matrix in relative space,
          3) Greedy min-cost matching,
          4) Accumulate MSE over matched pairs normalised by total #leaves.
        """
        if pred_rel.numel() == 0:
            return pred_rel.sum() * 0.0

        unique_parents, inverse, counts = leaf_parent_idx.unique(
            return_inverse=True, return_counts=True
        )
        total_loss = pred_rel.new_tensor(0.0)
        total_count = 0

        for parent, count in zip(unique_parents, counts):
            mask = (leaf_parent_idx == parent)
            idx = mask.nonzero(as_tuple=False).flatten()
            k = int(idx.numel())
            pred_group = pred_rel[idx]
            tgt_group = tgt_rel[idx]
            if k == 0:
                continue
            if k == 1:
                total_loss = total_loss + F.mse_loss(pred_group, tgt_group, reduction="sum")
                total_count += 1
                continue
            cost = (pred_group[:, None, :] - tgt_group[None, :, :]).pow(2).sum(dim=-1)
            assignment = self._greedy_min_cost_matching(cost.detach())
            matched_tgt = tgt_group[assignment]
            total_loss = total_loss + F.mse_loss(pred_group, matched_tgt, reduction="sum")
            total_count += k

        if total_count == 0:
            return pred_rel.sum() * 0.0
        return total_loss / float(total_count)

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

        # --- prepare masked input positions for leaves
        pos_in = self._make_masked_positions(
            pos=pos_gt,
            leaf_idx=leaf_idx_train,
            leaf_parent_idx=leaf_parent_idx,
            sigma=self.leaf_noise_sigma,
            clip=self.leaf_noise_clip,
        )                                              # [N,3]
        geo_lr_mask = self._compute_geo_lr_mask(pos_gt, parent_idx)
        self._maybe_debug_root_children(
            batch=batch,
            parent_idx=parent_idx,
            geo_lr_mask=geo_lr_mask,
            leaf_idx_train=leaf_idx_train,
            pos_gt=pos_gt,
            pos_masked=pos_in,
        )
         
        # --- prepare EGNN input (positions + minimal node features)
        feats_dim = getattr(model, 'feats_dim', 0)
        pos_dim   = getattr(model, 'pos_dim', 3)

        if feats_dim > 0:
            # seed with simple is_leaf flag - could be extended later TODO
            is_leaf = pos_in.new_zeros((pos_in.size(0), 1))
            is_leaf[batch.leaf_idx] = 1.0
            features = [is_leaf]
            feats_used = 1

            # Geometry-derived left/right bit for siblings
            if feats_used < feats_dim:
                geo_left = geo_lr_mask.to(device=pos_in.device, dtype=pos_in.dtype).unsqueeze(-1)
                features.append(geo_left)
                feats_used += 1

            # Add indicator for nodes flagged as newly expanded leaves (when provided)
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

            # Graph size ratio feature (current nodes / target nodes), broadcast per node
            if feats_used < feats_dim:
                size_ratio = self._size_ratio_feature_from_batch(
                    batch=batch,
                    device=pos_in.device,
                    dtype=pos_in.dtype,
                )
                if size_ratio is not None:
                    features.append(size_ratio)
                    feats_used += 1

            # Fill remaining dimensions with zeros if needed
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

        out = model(
            x=x_in,
            edge_index=edge_index,
            batch=batch.batch,
            edge_attr=edge_attr,
            parent_idx=parent_idx,
        )
        if isinstance(out, dict):
            pred_rel_all = out["rel_pred"]                # [N,3]
            pred_expansion_all = out["expansion_pred"]    # [N,1] or [N]
        else:
            raise ValueError("Network must return a dict with 'rel_pred' and 'expansion_pred'.")

        pred_rel = pred_rel_all[leaf_idx_train]                           # [L,3]
        pred_expansion = pred_expansion_all[leaf_idx_train]               # [L,1] or [L]

        # -- target relative offsets from parents for leaves
        tgt_rel  = self._leaf_rel_targets(pos_gt, leaf_idx_train, leaf_parent_idx)  # [L,3]

        # --- loss
        if pred_rel.numel() == 0:
            loss = pred_rel_all.sum() * 0.0
            metrics = {
                "leaf_pos_loss": 0.0,
                "leaf_expansion_loss": 0.0,
                "cumulative_loss": 0.0,
                "num_leaves": int(leaf_idx_train.numel()),
                "num_total_leaves": int(leaf_idx_all.numel()),
            }
            return loss, metrics

        # Positional loss on leaves (matching optional)
        if getattr(self, "use_sibling_matching", False):
            leaf_pos_loss = self._compute_leaf_pos_loss_with_matching(
                pred_rel=pred_rel,
                tgt_rel=tgt_rel,
                leaf_parent_idx=leaf_parent_idx,
            )
        else:
            leaf_pos_loss = F.mse_loss(pred_rel, tgt_rel)
        # Ensure dimension compatibility for expansion loss
        if pred_expansion.dim() == 1:
            pred_expansion = pred_expansion.unsqueeze(-1)  # [L] -> [L,1]
        if leaf_expansion.dim() == 1:
            leaf_expansion = leaf_expansion.unsqueeze(-1)  # [L] -> [L,1]
        # leaf_expansion_loss = F.mse_loss(pred_expansion, leaf_expansion)
        # lets use a Logit Loss for better stability
        leaf_expansion_loss = F.binary_cross_entropy_with_logits(
            pred_expansion.float(),
            leaf_expansion.float(),
        )
        # --- sibling distance regularizer (prevents siblings collapsing) ---
        sibling_dist_loss = pred_rel.sum() * 0.0  # zero default
        if self.sibling_loss_weight > 0.0 and leaf_idx_train.numel() > 1:
            abs_gt_all = pos_gt[leaf_idx_train]
            parent_pos_in_all = pos_in[leaf_parent_idx]
            abs_pred_all = parent_pos_in_all + pred_rel
            unique_parents, inverse, counts = leaf_parent_idx.unique(
                return_inverse=True, return_counts=True
            )
            mask_multi = counts >= 2
            if mask_multi.any():
                pair_indices = []
                for parent in unique_parents[mask_multi]:
                    leaf_pos_for_parent = (leaf_parent_idx == parent).nonzero(as_tuple=False).flatten()
                    if leaf_pos_for_parent.numel() < 2:
                        continue
                    pair_indices.append(leaf_pos_for_parent[:2])
                if len(pair_indices) > 0:
                    pair_indices = th.stack(pair_indices, dim=0)
                    idx1 = pair_indices[:, 0]
                    idx2 = pair_indices[:, 1]
                    v_gt = abs_gt_all[idx2] - abs_gt_all[idx1]
                    v_pred = abs_pred_all[idx2] - abs_pred_all[idx1]
                    d_gt = v_gt.norm(dim=-1)
                    d_pred = v_pred.norm(dim=-1)
                    sibling_dist_loss = F.mse_loss(d_pred, d_gt)

        # combine losses with sibling regularizer
        loss = leaf_pos_loss + leaf_expansion_loss + self.sibling_loss_weight * sibling_dist_loss

        with th.no_grad():
            parent_pos_in = pos_in[leaf_parent_idx]                # [L,3]
            abs_pred = parent_pos_in + pred_rel                    # [L,3]

        metrics = {
            "leaf_pos_loss": float(leaf_pos_loss.item()),
            "leaf_expansion_loss": float(leaf_expansion_loss.item()),
            "sibling_dist_loss": float(sibling_dist_loss.item()),
            "cumulative_loss": float(loss.item()),
            "num_leaves": int(leaf_idx_train.numel()),
            "num_total_leaves": int(leaf_idx_all.numel()),
            # "abs_pred_mean_norm": float(abs_pred.norm(dim=-1).mean().item()),
        }
        return loss, metrics
