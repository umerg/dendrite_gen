"""Interactive wrapper around the diffusion-based Expansion sampler."""

from __future__ import annotations

from typing import Any, Dict, Optional

import networkx as nx
import torch as th
from torch.nn import Module
from torch_scatter import scatter
from torch_sparse import SparseTensor

from graph_generation.method.expansion import Expansion
from graph_generation.method.helpers import build_directed_edge_index

from .expansion_interactive import GraphStepTrace


class InteractiveDiffusionExpansion(Expansion):
    """Diffusion-driven Expansion method that captures per-step traces."""

    DEFAULT_MAP_THRESHOLD = 0.0

    def sample_graphs_with_trace(
        self,
        target_size: th.Tensor,
        model,
        *,
        map_threshold: float | None = None,
        ensure_progress: bool = False,
    ) -> tuple[list[nx.Graph], list[list[GraphStepTrace]]]:
        """Run sampling while collecting graph+trace artifacts."""
        if self.diffusion is None:
            raise ValueError("InteractiveDiffusionExpansion requires a diffusion module.")
        if target_size.dim() != 1:
            raise ValueError("target_size must be a 1D tensor.")
        if (target_size < 1).any():
            raise ValueError("target_size entries must all be >= 1.")

        device = target_size.device
        num_graphs = int(target_size.numel())
        threshold = map_threshold if map_threshold is not None else self.DEFAULT_MAP_THRESHOLD

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
        expandable = (target_size >= 3).long()
        leaf_expansion = th.where(expandable.bool(), th.full_like(leaf_idx, 2), th.full_like(leaf_idx, 1))
        geo_lr_assign = th.full((num_graphs,), -1, device=device, dtype=th.long)
        leaf_mask = th.ones((num_graphs,), device=device, dtype=th.bool)
        max_steps = int(target_size.max().item() * 2)

        traces: list[list[GraphStepTrace]] = [[] for _ in range(num_graphs)]
        initial_traces = self._capture_step_traces(
            step_idx=0,
            adj=adj,
            pos=pos,
            batch=batch,
            leaf_idx=leaf_idx,
            prev_leaf_idx=None,
            geo_lr_assign=geo_lr_assign,
            target_size=target_size,
            debug_payload=None,
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
                geo_lr_assign,
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
                geo_lr_assign=geo_lr_assign,
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
                geo_lr_assign=geo_lr_assign,
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
        geo_lr_assign: th.Tensor,
        target_size: th.Tensor,
        debug_payload: dict | None,
    ) -> list[GraphStepTrace]:
        """Snapshot per-graph tensors into GraphStepTrace entries."""
        num_graphs = int(target_size.numel())
        traces: list[GraphStepTrace] = []

        target_cpu = target_size.detach().cpu()
        pos_cpu = pos.detach().cpu()
        batch_cpu = batch.detach().cpu()
        geo_cpu = geo_lr_assign.detach().cpu()
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
            geo_local = geo_cpu[node_ids]
            remaining_capacity = int(target_cpu[g].item()) - len(node_ids_list)

            leaf_local: list[int] = [local_map[idx] for idx in leaf_cpu.tolist() if idx in local_map]
            expanded_local: list[int] = []
            if prev_leaf_cpu is not None:
                expanded_local = [local_map[idx] for idx in prev_leaf_cpu.tolist() if idx in local_map]

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
                noise_sample=None,
                sibling_order=geo_local.clone(),
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
        """Rebuild NetworkX graphs identical to base Expansion.sample_graphs."""
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
        expansion_logits: th.Tensor,
        expansion_probs: th.Tensor,
        rel_pred_leaves: th.Tensor,
    ) -> dict[int, dict]:
        payload: dict[int, dict] = {}
        if leaf_idx_next.numel() == 0:
            return payload

        leaf_ids = leaf_idx_next.detach().cpu().tolist()
        graph_ids = batch_new[leaf_idx_next].detach().cpu().tolist()
        logits = expansion_logits.detach().cpu().view(-1).tolist()
        probs = expansion_probs.detach().cpu().view(-1).tolist()
        rel_entries = rel_pred_leaves.detach().cpu().tolist()

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
        return payload

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
        step: int = 0,
        ensure_progress: bool = False,
        map_threshold: float = 0.0,
        return_debug_payload: bool = False,
    ):
        """Copy of Expansion.expand with optional debug payload output."""
        if self.diffusion is None:
            raise ValueError("Diffusion module must be provided for sampling.")
        if not all(x is not None for x in (pos, leaf_idx, leaf_expansion, parent_idx_1b)):
            raise ValueError("expand requires pos, leaf_idx, leaf_expansion, parent_idx_1b tensors.")

        device = pos.device
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

        if (remaining_capacity <= 0).all() or leaf_idx.numel() == 0:
            result = (
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
            if return_debug_payload:
                return (*result, {})
            return result

        spawn_counts = (leaf_expansion == 2).long() * 2
        leaf_batch = batch_reduced[leaf_idx]
        spawn_counts_final = spawn_counts.clone()

        for g in range(num_graphs):
            cap = int(remaining_capacity[g].item())
            if cap < 2:
                spawn_counts_final[leaf_batch == g] = 0
                continue
            mask_g = leaf_batch == g
            expanders = th.nonzero((spawn_counts_final == 2) & mask_g, as_tuple=False).flatten()
            needed = expanders.numel() * 2
            if needed <= cap:
                continue
            max_leaves = cap // 2
            if max_leaves <= 0:
                spawn_counts_final[expanders] = 0
                continue
            if self.deterministic_expansion:
                generator = th.Generator(device=expanders.device)
                generator.manual_seed(g * 10007 + step)
                perm = th.randperm(expanders.numel(), generator=generator, device=expanders.device)
            else:
                perm = th.randperm(expanders.numel(), device=expanders.device)
            disable = expanders[perm[max_leaves:]]
            spawn_counts_final[disable] = 0

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
            result = (
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
            if return_debug_payload:
                return (*result, {})
            return result

        base_N = adj_reduced.size(0)
        leaf_mask_updated = leaf_mask.clone()
        expanded_mask = spawn_counts_final == 2
        if expanded_mask.any():
            leaf_mask_updated[leaf_idx[expanded_mask]] = False
        new_positions = []
        new_parents = []
        new_batches = []
        parent_child_edges = []
        lr_assign_new = []
        running_child_index = 0

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
                geo_lr_assign_next,
                leaf_mask_next,
                True,
            )
            if return_debug_payload:
                return (*result, {})
            return result

        node_counts_per_graph = scatter(
            th.ones_like(batch_new, dtype=target_size.dtype),
            batch_new,
            dim=0,
            dim_size=num_graphs,
        )

        feats_total = getattr(model, "feats_dim", 0)
        cond_dim = getattr(self.diffusion, "cond_dim", 0)
        feats_dim = max(feats_total - cond_dim, 0)
        if feats_dim > 0:
            features = []
            feats_used = 0
            is_leaf = leaf_mask_next.to(dtype=pos_new.dtype).unsqueeze(-1)
            features.append(is_leaf)
            feats_used += 1

            if feats_used < feats_dim:
                geo_feat = pos_new.new_zeros((pos_new.size(0), 1))
                mask = geo_lr_assign_next >= 0
                if mask.any():
                    geo_feat[mask] = (geo_lr_assign_next[mask] == 0).to(pos_new.dtype).unsqueeze(-1)
                features.append(geo_feat)
                feats_used += 1

            if feats_used < feats_dim:
                new_flag = pos_new.new_zeros((pos_new.size(0), 1))
                new_flag[leaf_idx_next] = 1.0
                features.append(new_flag)
                feats_used += 1

            if feats_used < feats_dim:
                ratio_graph = node_counts_per_graph.to(pos_new.dtype) / target_size.to(pos_new.dtype).clamp_min(1.0)
                ratio_nodes = ratio_graph[batch_new].unsqueeze(-1)
                features.append(ratio_nodes)
                feats_used += 1

            if feats_used < feats_dim:
                pad = pos_new.new_zeros((pos_new.size(0), feats_dim - feats_used))
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
            model_kwargs=None,
        )

        parent_pos_for_children = pos_new[leaf_parent_idx_next]
        pos_new[leaf_idx_next] = parent_pos_for_children + rel_pred

        if exp_pred.dim() == 1:
            exp_pred = exp_pred.unsqueeze(-1)
        expansion_score = exp_pred.squeeze(-1)
        expansion_prob = expansion_score.sigmoid()
        leaf_expansion_next = (expansion_score > map_threshold).long() + 1

        remaining_capacity_new = target_size.to(device) - node_counts_per_graph
        terminated = (remaining_capacity_new < 2).all() or leaf_idx_next.numel() == 0

        result = (
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

        if return_debug_payload:
            payload = self._build_debug_payload(
                leaf_idx_next=leaf_idx_next,
                batch_new=batch_new,
                expansion_logits=expansion_score,
                expansion_probs=expansion_prob,
                rel_pred_leaves=rel_pred,
            )
            return (*result, payload)
        return result
