"""Interactive wrapper around Expansion_OneShot for per-step tracing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import networkx as nx
import torch as th
from torch.nn import Module
from torch_scatter import scatter
from torch_sparse import SparseTensor

from graph_generation.method.expansion_oneshot import Expansion_OneShot


@dataclass
class GraphStepTrace:
    """Metadata captured for a single expansion step of one graph."""

    step_idx: int
    graph_id: int
    node_positions: th.Tensor
    edges: List[tuple[int, int]]
    leaf_ids: List[int]
    expansion_logits: List[float] | None = None
    expansion_probs: List[float] | None = None
    rel_pred: th.Tensor | None = None
    noise_sample: th.Tensor | None = None
    sibling_order: th.Tensor | None = None
    remaining_capacity: int | None = None
    enforced_progress: bool = False
    extras: Dict[str, Any] | None = None


class InteractiveExpansionOneShot(Expansion_OneShot):
    """Extension of Expansion_OneShot that emits traces for visualization."""

    DEFAULT_MAP_THRESHOLD = 0.3

    def sample_graphs_with_trace(
        self,
        target_size: th.Tensor,
        model,
        *,
        map_threshold: float | None = None,
        ensure_progress: bool = False,
    ) -> tuple[list[nx.Graph], list[list[GraphStepTrace]]]:
        """Run sampling while collecting per-graph traces."""
        if target_size.dim() != 1:
            raise ValueError("target_size must be a 1D tensor of per-graph capacities.")
        if (target_size < 1).any():
            raise ValueError("target_size entries must be >=1.")

        device = target_size.device
        num_graphs = int(target_size.numel())
        threshold = map_threshold if map_threshold is not None else self.DEFAULT_MAP_THRESHOLD

        # ---- Replicate base sampler initialization ----
        root_pos = th.zeros((num_graphs, 3), device=device)
        adj = SparseTensor(
            row=th.tensor([], dtype=th.long, device=device),
            col=th.tensor([], dtype=th.long, device=device),
            value=th.tensor([], dtype=th.float, device=device),
            sparse_sizes=(num_graphs, num_graphs),
        )

        batch = th.arange(num_graphs, device=device, dtype=th.long)
        parent_idx_1b = th.zeros(num_graphs, device=device, dtype=th.long)
        leaf_idx = th.arange(num_graphs, device=device, dtype=th.long)
        leaf_expansion = th.where(target_size >= 3, th.full_like(leaf_idx, 2), th.full_like(leaf_idx, 1))
        pos = root_pos
        sibling_order = th.full((num_graphs,), -1, device=device, dtype=th.long)
        leaf_mask = th.ones((num_graphs,), device=device, dtype=th.bool)
        max_steps = int(target_size.max().item() * 2)

        traces: list[list[GraphStepTrace]] = [[] for _ in range(num_graphs)]

        # capture the initial state as step 0
        initial_traces = self._capture_step_traces(
            step_idx=0,
            adj=adj,
            pos=pos,
            batch=batch,
            leaf_idx=leaf_idx,
            prev_leaf_idx=None,
            sibling_order=sibling_order,
            target_size=target_size,
        )
        for g in range(num_graphs):
            traces[g].append(initial_traces[g])

        step = 0
        terminated = False
        while not terminated and step < max_steps:
            prev_leaf_idx = leaf_idx.clone()
            (
                adj,
                pos,
                leaf_idx,
                leaf_expansion,
                parent_idx_1b,
                batch,
                sibling_order,
                leaf_mask,
                terminated,
                debug_payload,
            ) = self.expand(
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
                ensure_progress=ensure_progress,
                map_threshold=threshold,
                return_debug_payload=True,
            )
            step += 1
            step_traces = self._capture_step_traces(
                step_idx=step,
                adj=adj,
                pos=pos,
                batch=batch,
                leaf_idx=leaf_idx,
                prev_leaf_idx=prev_leaf_idx,
                sibling_order=sibling_order,
                target_size=target_size,
                debug_payload=debug_payload,
            )
            for g in range(num_graphs):
                traces[g].append(step_traces[g])

        graphs = self._build_final_graphs(adj, batch, pos, num_graphs)
        return graphs, traces

    def _capture_step_traces(
        self,
        *,
        step_idx: int,
        adj: SparseTensor,
        pos: th.Tensor,
        batch: th.Tensor,
        leaf_idx: th.Tensor,
        prev_leaf_idx: Optional[th.Tensor],
        sibling_order: th.Tensor,
        target_size: th.Tensor,
        debug_payload: dict | None = None,
    ) -> list[GraphStepTrace]:
        """Snapshot the per-graph state after an expansion step."""
        num_graphs = int(target_size.numel())
        traces: list[GraphStepTrace] = []

        target_cpu = target_size.detach().cpu()
        pos_cpu = pos.detach().cpu()
        batch_cpu = batch.detach().cpu()
        sibling_cpu = sibling_order.detach().cpu()
        leaf_cpu = leaf_idx.detach().cpu()
        prev_leaf_cpu = prev_leaf_idx.detach().cpu() if prev_leaf_idx is not None else None

        row, col, _ = adj.coo()
        row_list = row.detach().cpu().tolist()
        col_list = col.detach().cpu().tolist()

        for g in range(num_graphs):
            node_mask = (batch_cpu == g)
            node_ids = node_mask.nonzero(as_tuple=False).flatten()
            node_ids_list = node_ids.tolist()
            local_map = {int(global_id): idx for idx, global_id in enumerate(node_ids_list)}

            node_positions = pos_cpu[node_ids]
            sibling_local = sibling_cpu[node_ids]
            remaining_capacity = int(target_cpu[g].item()) - len(node_ids_list)

            leaf_local: list[int] = []
            for idx in leaf_cpu.tolist():
                if idx in local_map:
                    leaf_local.append(local_map[idx])

            expanded_local: list[int] = []
            if prev_leaf_cpu is not None:
                for idx in prev_leaf_cpu.tolist():
                    if idx in local_map:
                        expanded_local.append(local_map[idx])

            edges: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            for r, c in zip(row_list, col_list):
                if r in local_map and c in local_map:
                    u = local_map[r]
                    v = local_map[c]
                    key = (u, v) if u <= v else (v, u)
                    if key in seen:
                        continue
                    seen.add(key)
                    edges.append(key)

            info = debug_payload.get(g) if debug_payload else None
            expansion_logits = info.get("expansion_logits") if info else None
            expansion_probs = info.get("expansion_probs") if info else None
            rel_pred_tensor = (
                th.tensor(info["rel_pred"], dtype=pos_cpu.dtype) if info and info.get("rel_pred") else None
            )
            noise_tensor = (
                th.tensor(info["noise"], dtype=pos_cpu.dtype) if info and info.get("noise") else None
            )
            extras: Dict[str, Any] | None = {"expanded_leaf_ids": expanded_local} if expanded_local else None
            if info and info.get("leaf_global_ids"):
                extras = extras or {}
                extras["leaf_global_ids"] = info["leaf_global_ids"]

            trace = GraphStepTrace(
                step_idx=step_idx,
                graph_id=g,
                node_positions=node_positions.clone(),
                edges=edges,
                leaf_ids=leaf_local,
                expansion_logits=expansion_logits,
                expansion_probs=expansion_probs,
                rel_pred=rel_pred_tensor,
                noise_sample=noise_tensor,
                sibling_order=sibling_local.clone(),
                remaining_capacity=remaining_capacity,
                enforced_progress=False,
                extras=extras,
            )
            traces.append(trace)

        return traces

    def _build_final_graphs(
        self,
        adj: SparseTensor,
        batch: th.Tensor,
        pos: th.Tensor,
        num_graphs: int,
    ) -> list[nx.Graph]:
        """Deterministically rebuild final NetworkX graphs identical to base implementation."""
        row, col, _ = adj.coo()
        graphs: list[nx.Graph] = []
        row_list = row.tolist()
        col_list = col.tolist()

        for g in range(num_graphs):
            g_mask = (batch == g)
            node_ids = th.nonzero(g_mask, as_tuple=False).flatten()
            node_ids_list = node_ids.tolist()
            local_map = {int(n): i for i, n in enumerate(node_ids_list)}
            G = nx.Graph()
            for i_local, n_global in enumerate(node_ids_list):
                G.add_node(
                    i_local,
                    pos=pos[n_global].detach().cpu().numpy(),
                )
            seen: set[tuple[int, int]] = set()
            for r, c in zip(row_list, col_list):
                if r in local_map and c in local_map:
                    u = local_map[r]
                    v = local_map[c]
                    key = (u, v) if u <= v else (v, u)
                    if key in seen:
                        continue
                    seen.add(key)
                    G.add_edge(u, v)
            graphs.append(G)
        return graphs

    def _build_debug_payload(
        self,
        *,
        leaf_idx_next: th.Tensor,
        batch_new: th.Tensor,
        expansion_pred_leaves: th.Tensor,
        expansion_prob: th.Tensor,
        rel_pred_leaves: th.Tensor,
        noise_samples: th.Tensor | None,
    ) -> dict[int, dict]:
        payload: dict[int, dict] = {}
        if leaf_idx_next.numel() == 0:
            return payload
        leaf_ids = leaf_idx_next.detach().cpu().tolist()
        graph_ids = batch_new[leaf_idx_next].detach().cpu().tolist()
        logits = expansion_pred_leaves.detach().cpu().view(-1).tolist()
        probs = expansion_prob.detach().cpu().view(-1).tolist()
        rel_entries = rel_pred_leaves.detach().cpu().tolist()
        noise_entries = noise_samples.detach().cpu().tolist() if noise_samples is not None else None

        for idx, graph_id in enumerate(graph_ids):
            entry = payload.setdefault(
                int(graph_id),
                {
                    "leaf_global_ids": [],
                    "expansion_logits": [],
                    "expansion_probs": [],
                    "rel_pred": [],
                    "noise": [],
                },
            )
            entry["leaf_global_ids"].append(int(leaf_ids[idx]))
            entry["expansion_logits"].append(float(logits[idx]))
            entry["expansion_probs"].append(float(probs[idx]))
            entry["rel_pred"].append(rel_entries[idx])
            if noise_entries is not None:
                entry["noise"].append(noise_entries[idx])
        if noise_entries is None:
            for entry in payload.values():
                entry["noise"] = []
        return payload

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
        return_debug_payload: bool = False,
    ):
        """Override of base expand adding optional debug payload."""
        if not all(x is not None for x in (pos, leaf_idx, leaf_expansion, parent_idx_1b)):
            raise ValueError("expand requires pos, leaf_idx, leaf_expansion, parent_idx_1b")

        device = pos.device
        parent_idx = parent_idx_1b - 1
        if leaf_mask is None:
            leaf_mask = th.ones((pos.size(0),), device=device, dtype=th.bool)
        else:
            leaf_mask = leaf_mask.to(device=device)
            if leaf_mask.dtype != th.bool:
                leaf_mask = leaf_mask.bool()

        size_per_graph = scatter(th.ones_like(batch_reduced), batch_reduced)
        remaining_capacity = target_size.to(device) - size_per_graph
        if sibling_order is None:
            sibling_order = th.full((pos.size(0),), -1, device=device, dtype=th.long)

        if (remaining_capacity <= 0).all() or leaf_idx.numel() == 0:
            result = (
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
            if return_debug_payload:
                return (*result, {})
            return result

        spawn_counts = (leaf_expansion == 2).long() * 2
        leaf_batch = batch_reduced[leaf_idx]

        spawn_counts_final = spawn_counts.clone()
        for g in target_size.nonzero().flatten():
            g_int = int(g.item())
            cap = int(remaining_capacity[g_int].item())
            if cap < 2:
                spawn_counts_final[leaf_batch == g_int] = 0
                continue
            mask_g = leaf_batch == g_int
            expanders = th.nonzero((spawn_counts_final == 2) & mask_g, as_tuple=False).flatten()
            needed = expanders.numel() * 2
            if needed <= cap:
                continue
            max_leaves = cap // 2
            if self.deterministic_expansion:
                generator = th.Generator(device=expanders.device)
                generator.manual_seed(g_int * 10007 + step)
                perm = th.randperm(expanders.numel(), generator=generator, device=expanders.device)
            else:
                perm = th.randperm(expanders.numel(), device=expanders.device)
            expanders_shuffled = expanders[perm]
            disable = expanders_shuffled[max_leaves:]
            spawn_counts_final[disable] = 0

        if ensure_progress and (remaining_capacity >= 2).any():
            num_graphs = int(target_size.numel())
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
                    forced_leaf = leaf_indices_g[0]
                else:
                    rand_idx = th.randint(low=0, high=leaf_indices_g.numel(), size=(1,), device=leaf_indices_g.device)
                    forced_leaf = leaf_indices_g[rand_idx]
                spawn_counts_final[forced_leaf] = 2

        leaf_mask_updated = leaf_mask.clone()
        expanded_mask = spawn_counts_final == 2
        if expanded_mask.any():
            leaf_mask_updated[leaf_idx[expanded_mask]] = False

        total_new_children = int(spawn_counts_final.sum().item())
        if total_new_children == 0:
            result = (
                adj_reduced,
                pos,
                leaf_idx.clone(),
                leaf_expansion.clone(),
                parent_idx_1b,
                batch_reduced,
                sibling_order.clone(),
                leaf_mask_updated,
                True,
            )
            if return_debug_payload:
                return (*result, {})
            return result

        base_N = adj_reduced.size(0)
        new_child_positions = []
        new_child_parents = []
        new_child_batches = []
        parent_child_edges = []
        sibling_order_new = []
        running_child_index = 0
        noise_records = [] if return_debug_payload else None
        for li, sc in zip(leaf_idx.tolist(), spawn_counts_final.tolist()):
            if sc == 0:
                continue
            parent_pos = pos[li]
            noise = self._sample_noise(
                (sc, parent_pos.shape[0]),
                device,
                sigma=self.leaf_noise_sigma,
                clip=self.leaf_noise_clip,
            )
            if noise_records is not None:
                noise_records.append(noise.clone())
            child_pos = parent_pos.unsqueeze(0) + noise
            new_child_positions.append(child_pos)
            for local_child in range(sc):
                global_child_idx = base_N + running_child_index
                parent_child_edges.append((li, global_child_idx))
                new_child_parents.append(li)
                new_child_batches.append(int(batch_reduced[li].item()))
                running_child_index += 1
                sibling_order_new.append(0 if local_child == 0 else 1)
        if running_child_index != total_new_children:
            raise ValueError(
                "Child accounting mismatch: expected %d got %d" % (total_new_children, running_child_index)
            )

        new_child_positions_tensor = (
            th.cat(new_child_positions, dim=0) if new_child_positions else pos.new_empty((0, 3))
        )
        pos_new = th.cat([pos, new_child_positions_tensor], dim=0)
        parent_idx_new_0b = th.cat(
            [parent_idx, th.tensor(new_child_parents, device=device, dtype=parent_idx.dtype)]
        )
        parent_idx_1b_new = parent_idx_new_0b + 1
        batch_new = th.cat([batch_reduced, th.tensor(new_child_batches, device=device, dtype=batch_reduced.dtype)])

        sibling_order_next = th.cat(
            [sibling_order, th.tensor(sibling_order_new, device=device, dtype=th.long)],
            dim=0,
        )

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

        leaf_idx_next = th.arange(base_N, base_N + total_new_children, device=device, dtype=leaf_idx.dtype)
        new_leaf_flags = th.ones((leaf_idx_next.numel(),), device=device, dtype=th.bool)
        leaf_mask_next = th.cat([leaf_mask_updated, new_leaf_flags], dim=0)
        if leaf_idx_next.numel() == 0:
            result = (
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
            if return_debug_payload:
                return (*result, {})
            return result

        num_graphs_total = int(target_size.numel())
        node_counts_per_graph = scatter(
            th.ones_like(batch_new),
            batch_new,
            dim=0,
            dim_size=num_graphs_total,
        )

        feats_dim = getattr(model, "feats_dim", 0)
        pos_dim = getattr(model, "pos_dim", 3)
        if feats_dim > 0:
            is_leaf_flag = leaf_mask_next.to(dtype=pos_new.dtype).unsqueeze(-1)
            features = [is_leaf_flag]
            feats_used = 1
            if feats_used < feats_dim:
                so = sibling_order_next
                sib_is_left = (so == 0).float().unsqueeze(-1)
                sib_is_left = th.where(so.unsqueeze(-1) >= 0, sib_is_left, sib_is_left.new_zeros(sib_is_left.shape))
                features.append(sib_is_left)
                feats_used += 1
            if feats_used < feats_dim:
                new_leaf_flag = pos_new.new_zeros((pos_new.size(0), 1))
                new_leaf_flag[leaf_idx_next] = 1.0
                features.append(new_leaf_flag)
                feats_used += 1
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
        if not isinstance(out, dict) or "rel_pred" not in out or "expansion_pred" not in out:
            raise ValueError("Model must return dict with 'rel_pred' and 'expansion_pred'.")

        rel_pred_all = out["rel_pred"]
        expansion_pred_all = out["expansion_pred"]

        rel_pred_leaves = rel_pred_all[leaf_idx_next]
        expansion_pred_leaves = expansion_pred_all[leaf_idx_next]
        if expansion_pred_leaves.dim() == 1:
            expansion_pred_leaves = expansion_pred_leaves.unsqueeze(-1)
        parent_pos_for_children = pos_new[parent_idx_new_0b[leaf_idx_next]]
        pos_new[leaf_idx_next] = parent_pos_for_children + rel_pred_leaves

        if leaf_idx_next.numel() > 0:
            geo_lr_mask = self._compute_geo_lr_mask(pos_new, parent_idx_new_0b)
            parent_new = parent_idx_new_0b[leaf_idx_next]
            counts = scatter(
                th.ones_like(parent_new),
                parent_new,
                dim=0,
                dim_size=pos_new.size(0),
            )
            valid = counts[parent_new] == 2
            if valid.any():
                sib_left = geo_lr_mask[leaf_idx_next][valid]
                sibling_order_next = sibling_order_next.clone()
                sibling_order_next[leaf_idx_next[valid]] = (~sib_left).to(
                    dtype=sibling_order_next.dtype
                )

        expansion_prob = expansion_pred_leaves.squeeze(-1).sigmoid()
        leaf_expansion_next = (expansion_prob > map_threshold).long() + 1

        remaining_capacity_new = target_size.to(device) - node_counts_per_graph
        terminated = (remaining_capacity_new < 2).all() or leaf_idx_next.numel() == 0

        result = (
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

        if return_debug_payload:
            if noise_records:
                noise_payload = th.cat(noise_records, dim=0)
            else:
                noise_payload = None
            payload = self._build_debug_payload(
                leaf_idx_next=leaf_idx_next,
                batch_new=batch_new,
                expansion_pred_leaves=expansion_pred_leaves,
                expansion_prob=expansion_prob,
                rel_pred_leaves=rel_pred_leaves,
                noise_samples=noise_payload,
            )
            return (*result, payload)
        return result
