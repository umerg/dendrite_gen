import logging
import math
import time

import torch as th
import torch.nn.functional as F
from torch.nn import Module

from graph_generation.method.helpers import (
    local_to_global,
    patch_geometry_for_noised_leaves,
)

logger = logging.getLogger(__name__)


class DenoisingDiffusionModel(Module):
    """Noise-conditioned diffusion loss for training-time denoising."""

    P_mean = -1.2
    P_std = 1.2
    sigma_min = 0.002
    sigma_max = 4.0
    cond_dim = 2  # e_t feature + log_sigma feature per node

    def __init__(self, num_steps: int = 1):
        super().__init__()
        self.num_steps = num_steps

    def forward(
        self,
        *,
        node_feats: th.Tensor | None,
        edge_index: th.Tensor,
        batch: th.Tensor,
        edge_attr: th.Tensor,
        P_0: th.Tensor,
        C_0: th.Tensor,
        parent_idx: th.Tensor,
        leaf_idx_train: th.Tensor,
        leaf_expansion: th.Tensor,
        leaf_parent_idx: th.Tensor,
        model: Module,
        tmd: th.Tensor | None = None,
        pre_geom_p0: dict | None = None,
        local_forward: th.Tensor | None = None,
        local_sideways: th.Tensor | None = None,
        uhat: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Compute σ-conditioned denoising losses for positional + expansion targets."""
        device = P_0.device
        num_leaves = leaf_idx_train.numel()
        if node_feats is None:
            node_feats = P_0.new_zeros((P_0.size(0), 0))

        if num_leaves == 0:
            zero = P_0.new_zeros(())
            return zero, zero

        leaf_expansion = leaf_expansion.to(dtype=P_0.dtype).view(-1, 1)
        e_0 = 2.0 * leaf_expansion - 1.0 # map [0,1] to [-1,1]

        if batch.numel() == 0:
            raise ValueError("Batch vector is empty; cannot sample σ.")
        num_graphs = int(batch.max().item()) + 1
        sigma_graph = (
            th.randn((num_graphs,), device=device) * self.P_std + self.P_mean
        ).exp()
        sigma_graph = sigma_graph.clamp(self.sigma_min, self.sigma_max)
        log_sigma_graph = sigma_graph.log()

        leaf_batch = batch[leaf_idx_train]
        sigma_leaf = sigma_graph[leaf_batch].view(-1, 1)

        eps_pos = th.randn_like(C_0)
        eps_exp = th.randn_like(e_0)
        C_t = C_0 + sigma_leaf * eps_pos  # noising in local frame (isotropic)
        e_t = e_0 + sigma_leaf * eps_exp

        P_t = P_0.clone()
        if local_forward is not None and local_sideways is not None and uhat is not None:
            # Convert local-frame C_t to global for position placement
            C_t_global = local_to_global(C_t, local_forward, local_sideways, uhat)
            P_t[leaf_idx_train] = P_0[leaf_parent_idx] + C_t_global
        else:
            P_t[leaf_idx_train] = P_0[leaf_parent_idx] + C_t

        # Patch precomputed P_0 geometry for noised leaf positions
        pre_geom = None
        if pre_geom_p0 is not None:
            with th.no_grad():
                pre_geom = patch_geometry_for_noised_leaves(
                    pre_geom_p0, P_t, leaf_idx_train, parent_idx,
                    edge_index, model.uhat,
                )

        N = P_0.size(0)
        e_feat = P_0.new_zeros((N, 1))
        e_feat[leaf_idx_train] = e_t
        log_sigma_node = log_sigma_graph[batch].view(N, 1)
        node_feats_t = th.cat([node_feats, e_feat, log_sigma_node], dim=-1)

        x_in = th.cat([P_t, node_feats_t], dim=-1)
        if device.type == 'cuda':
            th.cuda.synchronize(device)
        _t0_model = time.perf_counter()
        out = model(
            x=x_in,
            edge_index=edge_index,
            batch=batch,
            edge_attr=edge_attr,
            parent_idx=parent_idx,
            tmd=tmd,
            pre_geom=pre_geom,
        )
        if device.type == 'cuda':
            th.cuda.synchronize(device)
        # logger.info(
        #     "[diffusion.forward N=%d L=%d] model_call=%.4fs",
        #     P_0.size(0), num_leaves, time.perf_counter() - _t0_model,
        # )
        if not isinstance(out, dict):
            raise ValueError("Model must return dict with 'rel_pred' and 'expansion_pred'.")
        rel_pred_all = out["rel_pred"]
        exp_pred_all = out["expansion_pred"]

        C_pred = rel_pred_all[leaf_idx_train]
        e_pred = exp_pred_all[leaf_idx_train]
        if e_pred.dim() == 1:
            e_pred = e_pred.unsqueeze(-1)

        pos_loss = F.mse_loss(C_pred, C_0)
        exp_loss = F.mse_loss(e_pred, e_0)
        return exp_loss, pos_loss

    @staticmethod
    def make_sigma_schedule(num_steps: int, sigma_max: float, sigma_min: float, device: th.device) -> th.Tensor:
        """Return monotonically decreasing sigma schedule ending with 0."""
        steps = max(int(num_steps), 1)
        ramp = th.linspace(0.0, 1.0, steps=steps, device=device)
        sigmas = sigma_max * (sigma_min / sigma_max) ** ramp
        sigmas = th.cat([sigmas, sigmas.new_zeros(1)], dim=0)
        return sigmas

    @th.no_grad()
    def sample(
        self,
        *,
        node_feats: th.Tensor | None,
        edge_index: th.Tensor,
        batch: th.Tensor,
        edge_attr: th.Tensor,
        P_0: th.Tensor,
        parent_idx: th.Tensor,
        leaf_idx: th.Tensor,
        leaf_parent_idx: th.Tensor,
        model: Module,
        model_kwargs: dict | None = None,
        tmd: th.Tensor | None = None,
        local_forward: th.Tensor | None = None,
        local_sideways: th.Tensor | None = None,
        uhat: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Deterministically denoise leaves via σ-schedule (ancestral-free)."""
        device = P_0.device
        model_kwargs = model_kwargs or {}
        if tmd is not None:
            model_kwargs = {**model_kwargs, "tmd": tmd}
        if node_feats is None:
            node_feats = P_0.new_zeros((P_0.size(0), 0))

        if leaf_idx.numel() == 0:
            zero_pos = P_0.new_zeros((0, 3))
            zero_exp = P_0.new_zeros((0, 1))
            return zero_pos, zero_exp

        node_feats = node_feats.to(device=device)
        parent_idx = parent_idx.to(device=device)
        leaf_idx = leaf_idx.to(device=device, dtype=th.long)
        leaf_parent_idx = leaf_parent_idx.to(device=device, dtype=th.long)

        sigmas = self.make_sigma_schedule(self.num_steps, self.sigma_max, self.sigma_min, device)
        sigma_init = float(sigmas[0].item())

        L = leaf_idx.numel()
        N = P_0.size(0)
        parent_pos = P_0[leaf_parent_idx]
        C = th.randn((L, 3), device=device) * sigma_init
        e = th.randn((L, 1), device=device) * sigma_init

        C0_pred = th.zeros_like(C)
        e0_pred = th.zeros_like(e)

        def _st() -> float:
            if device.type == 'cuda':
                th.cuda.synchronize(device)
            return time.perf_counter()

        _acc_clone, _acc_alloc, _acc_model = 0.0, 0.0, 0.0

        for step in range(self.num_steps):
            sigma_cur = float(sigmas[step].item())
            sigma_next = float(sigmas[step + 1].item())
            sigma_cur_clamped = max(sigma_cur, 1e-12)
            log_sigma = math.log(sigma_cur_clamped)

            _t0 = _st()
            P_cur = P_0.clone()
            _acc_clone += _st() - _t0

            if local_forward is not None and local_sideways is not None and uhat is not None:
                C_global = local_to_global(C, local_forward, local_sideways, uhat)
                P_cur[leaf_idx] = parent_pos + C_global
            else:
                P_cur[leaf_idx] = parent_pos + C

            _t0 = _st()
            e_feat = P_0.new_zeros((N, 1))
            e_feat[leaf_idx] = e
            log_sigma_feat = P_0.new_full((N, 1), log_sigma)
            node_feats_t = th.cat([node_feats, e_feat, log_sigma_feat], dim=-1)
            x_in = th.cat([P_cur, node_feats_t], dim=-1)
            _acc_alloc += _st() - _t0

            _t0 = _st()
            out = model(
                x=x_in,
                edge_index=edge_index,
                batch=batch,
                edge_attr=edge_attr,
                parent_idx=parent_idx,
                **model_kwargs,
            )
            _acc_model += _st() - _t0

            if not isinstance(out, dict):
                raise ValueError("Model must return dict with 'rel_pred' and 'expansion_pred'.")
            rel_pred_all = out["rel_pred"]
            exp_pred_all = out["expansion_pred"]

            C0_pred = rel_pred_all[leaf_idx]
            e0_pred = exp_pred_all[leaf_idx]
            if e0_pred.dim() == 1:
                e0_pred = e0_pred.unsqueeze(-1)

            inv_sigma = 1.0 / sigma_cur_clamped
            eps_C = (C - C0_pred) * inv_sigma
            eps_e = (e - e0_pred) * inv_sigma

            C = C0_pred + sigma_next * eps_C
            e = e0_pred + sigma_next * eps_e

        # logger.info(
        #     "[diffusion.sample num_steps=%d N=%d L=%d] "
        #     "clone_total=%.4fs alloc+cat_total=%.4fs model_total=%.4fs",
        #     self.num_steps, N, L, _acc_clone, _acc_alloc, _acc_model,
        # )
        return C0_pred, e0_pred
