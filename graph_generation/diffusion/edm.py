
"""
EDM-style diffusion for *leaf-only* geometric binary tree expansion.

This module adapts the "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM)
preconditioning + loss weighting + Karras/Heun sampler to the tree-growth paradigm where the
diffused variables live only on the *new leaves*:

  - C: leaf-relative offset vectors (L x 3), where P_leaf = P_parent + C
  - e: leaf expansion scalar (L x 1) in [-1, 1] (typically mapped from {0,1} via 2y-1)

The rest of the graph (existing nodes) is treated as conditioning context.

Key design choice:
- We apply EDM preconditioning in the *leaf variable space* (C,e) while keeping the rest of the
  graph unscaled. The model still consumes full-graph coordinates and features, but the noised
  leaf offsets and expansion scalars are fed in preconditioned form.

"""

import math
from dataclasses import dataclass
from typing import Any

import torch as th
import torch.nn.functional as F
from torch.nn import Module


@dataclass
class EDMSamplerConfig:
    """Sampler hyperparameters (Karras schedule + optional churn)."""
    rho: float = 6.0
    S_churn: float = 0.0   # set >0 to enable stochastic churn; 0 keeps deterministic sampling
    S_min: float = 0.05
    S_max: float = 50.0
    S_noise: float = 1.003


class EDMDiffusionModel(Module):
    """
    Leaf-only EDM diffusion.

    Training:
      - sample σ per graph from log-normal: σ = exp(N(P_mean, P_std))
      - add noise: x = x0 + σ ε
      - EDM preconditioning:
          c_in   = 1 / sqrt(σ_data^2 + σ^2)
          c_skip = σ_data^2 / (σ_data^2 + σ^2)
          c_out  = σ σ_data / sqrt(σ_data^2 + σ^2)
        network sees x_in = c_in * x (in leaf space; injected via leaf coords/features)
        prediction: x0_hat = c_skip * x + c_out * F(x_in, σ)
      - EDM weighting:
          w(σ) = (σ^2 + σ_data^2) / (σ σ_data)^2

    Sampling:
      - Karras σ schedule with Heun (2nd order) correction.
    """

    # Log-normal sampling hyperparams.
    P_mean: float = -0.8
    P_std: float = 1.5

    # Noise range used both for sampling.
    sigma_min: float = 0.01
    sigma_max: float = 5.0

    # EDM parameter: typical std of clean data in the diffused variable space.
    #   sigma_data ≈ std(C_0) (and optionally include e_0, but excluded for now).
    sigma_data: float = 0.45

    # Model conditioning: node feature channels include leaf expansion feature + log_sigma per node.
    cond_dim: int = 2  # e feature + log_sigma feature per node

    def __init__(
        self,
        num_steps: int = 64,
        *,
        sampler_cfg: EDMSamplerConfig | None = None,
    ):
        super().__init__()
        self.num_steps = int(num_steps)
        self.sampler_cfg = sampler_cfg or EDMSamplerConfig()

    # ---------------------------
    # Helpers: schedules, coeffs
    # ---------------------------
    @staticmethod
    def _edm_coeffs(sigma: th.Tensor, sigma_data: float) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Return (c_in, c_skip, c_out) for EDM preconditioning.
        sigma: (...,) tensor
        """
        sd2 = float(sigma_data) ** 2
        s2 = sigma ** 2
        denom = (s2 + sd2).sqrt()
        c_in = 1.0 / denom
        c_skip = sd2 / (s2 + sd2)
        c_out = sigma * float(sigma_data) / denom
        return c_in, c_skip, c_out

    @staticmethod
    def _edm_weight(sigma: th.Tensor, sigma_data: float) -> th.Tensor:
        """EDM loss weight w(σ) = (σ^2 + σ_data^2) / (σ σ_data)^2."""
        sd = float(sigma_data)
        return (sigma ** 2 + sd ** 2) / (sigma * sd) ** 2

    @staticmethod
    def make_karras_schedule(
        num_steps: int,
        sigma_max: float,
        sigma_min: float,
        rho: float,
        device: th.device,
        dtype: th.dtype = th.float64,
    ) -> th.Tensor:
        """
        Karras schedule as in EDM. Returns tensor of length (num_steps + 1) with final 0 appended.
        """
        steps = int(num_steps)
        if steps <= 1:
            sig = th.tensor([float(sigma_max), 0.0], device=device, dtype=dtype)
            return sig
        i = th.arange(steps, device=device, dtype=dtype)
        inv_rho = 1.0 / float(rho)
        smax = float(sigma_max) ** inv_rho
        smin = float(sigma_min) ** inv_rho
        t_steps = (smax + (i / (steps - 1)) * (smin - smax)) ** float(rho)
        t_steps = th.cat([t_steps, th.zeros_like(t_steps[:1])], dim=0)  # append 0
        return t_steps

    # ---------------------------
    # Training loss
    # ---------------------------
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
        cell_class: th.Tensor | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        """
        Compute EDM-weighted denoising loss for leaf offsets C and leaf expansion e.

        Returns: (exp_loss, pos_loss)
        """
        device = P_0.device
        dtype = P_0.dtype
        model_kwargs = model_kwargs or {}
        if tmd is not None:
            model_kwargs = {**model_kwargs, "tmd": tmd}
        if cell_class is not None:
            model_kwargs = {**model_kwargs, "cell_class": cell_class}

        if node_feats is None:
            node_feats = P_0.new_zeros((P_0.size(0), 0))
        node_feats = node_feats.to(device=device)

        leaf_idx_train = leaf_idx_train.to(device=device, dtype=th.long)
        leaf_parent_idx = leaf_parent_idx.to(device=device, dtype=th.long)

        num_leaves = leaf_idx_train.numel()
        if num_leaves == 0:
            zero = P_0.new_zeros(())
            return zero, zero

        # Map expansion label {0,1} -> e0 in [-1,1].
        e_0 = (2.0 * leaf_expansion.to(device=device, dtype=dtype).view(-1, 1)) - 1.0

        # Sample σ per graph (log-normal).
        if batch.numel() == 0:
            raise ValueError("Batch vector is empty; cannot sample σ.")
        num_graphs = int(batch.max().item()) + 1
        sigma_graph = (th.randn((num_graphs,), device=device) * self.P_std + self.P_mean).exp()

        leaf_batch = batch[leaf_idx_train]
        sigma_leaf = sigma_graph[leaf_batch].view(-1, 1)  # (L,1)

        # Noise.
        eps_C = th.randn_like(C_0)
        eps_e = th.randn_like(e_0)
        C_t = C_0 + sigma_leaf * eps_C
        e_t = e_0 + sigma_leaf * eps_e

        # EDM preconditioning in leaf space.
        c_in, c_skip, c_out = self._edm_coeffs(sigma_leaf, self.sigma_data)  # (L,1)
        C_in = c_in * C_t
        e_in = c_in * e_t

        # Build full-graph inputs.
        P_t = P_0.clone()
        P_t[leaf_idx_train] = P_0[leaf_parent_idx] + C_in  # inject preconditioned leaf offsets

        N = P_0.size(0)
        e_feat = P_0.new_zeros((N, 1))
        e_feat[leaf_idx_train] = e_in

        log_sigma_graph = sigma_graph.log() / 4.0  # scale down for numerical stability
        log_sigma_node = log_sigma_graph[batch].view(N, 1)
        node_feats_t = th.cat([node_feats, e_feat, log_sigma_node], dim=-1)

        x_in = th.cat([P_t, node_feats_t], dim=-1)

        out = model(
            x=x_in,
            edge_index=edge_index,
            batch=batch,
            edge_attr=edge_attr,
            parent_idx=parent_idx,
            **model_kwargs,
        )
        if not isinstance(out, dict):
            raise ValueError("Model must return dict with 'rel_pred' and 'expansion_pred'.")
        rel_out = out["rel_pred"][leaf_idx_train]          # (L,3) => interpreted as EDM net output
        exp_out = out["expansion_pred"][leaf_idx_train]    # (L,1) or (L,)
        if exp_out.dim() == 1:
            exp_out = exp_out.unsqueeze(-1)

        # Compute x0 prediction via EDM combination.
        # NOTE: combine with the *unpreconditioned* noised variables (C_t, e_t).
        C0_pred = (c_skip * C_t) + (c_out * rel_out)
        e0_pred = (c_skip * e_t) + (c_out * exp_out)

        # Weighted losses.
        w = self._edm_weight(sigma_leaf, self.sigma_data)  # (L,1)
        pos_loss = (w * (C0_pred - C_0) ** 2).mean()
        exp_loss = (w * (e0_pred - e_0) ** 2).mean()

        return exp_loss, pos_loss

    # ---------------------------
    # Sampling
    # ---------------------------
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
        model_kwargs: dict[str, Any] | None = None,
        tmd: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        """
        EDM sampler for leaf offsets and expansion scalar.

        Returns:
          C_sample: (L,3) leaf-relative offsets (in the original variable space)
          e_sample: (L,1) expansion scalar in [-1,1]
        """
        device = P_0.device
        dtype = P_0.dtype
        model_kwargs = model_kwargs or {}
        if tmd is not None:
            model_kwargs = {**model_kwargs, "tmd": tmd}

        if node_feats is None:
            node_feats = P_0.new_zeros((P_0.size(0), 0))
        node_feats = node_feats.to(device=device)

        leaf_idx = leaf_idx.to(device=device, dtype=th.long)
        leaf_parent_idx = leaf_parent_idx.to(device=device, dtype=th.long)

        L = leaf_idx.numel()
        if L == 0:
            return P_0.new_zeros((0, 3)), P_0.new_zeros((0, 1))

        N = P_0.size(0)
        parent_pos = P_0[leaf_parent_idx]

        # Karras schedule.
        t_steps = self.make_karras_schedule(
            num_steps=self.num_steps,
            sigma_max=self.sigma_max,
            sigma_min=self.sigma_min,
            rho=self.sampler_cfg.rho,
            device=device,
            dtype=th.float64,
        )

        # Initialize latents x ~ N(0, σ0^2).
        C_next = th.randn((L, 3), device=device, dtype=th.float64) * t_steps[0]
        e_next = th.randn((L, 1), device=device, dtype=th.float64) * t_steps[0]

        # For compatibility with EDM's self-conditioning interface, we keep last x0_hat around.
        C0_hat = th.zeros_like(C_next)
        e0_hat = th.zeros_like(e_next)

        # Main loop.
        for i in range(self.num_steps):
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]

            C_cur = C_next
            e_cur = e_next

            # Optional churn (stochasticity). Set S_churn=0 for deterministic sampling.
            gamma = 0.0
            if (self.sampler_cfg.S_churn > 0.0) and (self.sampler_cfg.S_min <= float(t_cur) <= self.sampler_cfg.S_max):
                gamma = min(self.sampler_cfg.S_churn / self.num_steps, math.sqrt(2.0) - 1.0)
            t_hat = t_cur + gamma * t_cur

            if gamma > 0.0:
                # add extra noise
                noise_scale = (t_hat ** 2 - t_cur ** 2).sqrt() * self.sampler_cfg.S_noise
                C_hat = C_cur + noise_scale * th.randn_like(C_cur)
                e_hat = e_cur + noise_scale * th.randn_like(e_cur)
            else:
                C_hat, e_hat = C_cur, e_cur

            # Denoise at t_hat.
            C0_hat, e0_hat = self._denoise_leaf_vars(
                node_feats=node_feats,
                edge_index=edge_index,
                batch=batch,
                edge_attr=edge_attr,
                P_0=P_0,
                parent_idx=parent_idx,
                leaf_idx=leaf_idx,
                parent_pos=parent_pos,
                C=C_hat,
                e=e_hat,
                sigma=float(t_hat),
                model=model,
                model_kwargs=model_kwargs,
            )

            # Euler step.
            # d = (x - x0_hat)/sigma
            sigma_hat = float(max(float(t_hat), 1e-12))
            dC = (C_hat - C0_hat) / sigma_hat
            de = (e_hat - e0_hat) / sigma_hat
            C_next = C_hat + (float(t_next) - float(t_hat)) * dC
            e_next = e_hat + (float(t_next) - float(t_hat)) * de

            # Heun (2nd-order) correction, except at last step.
            if i < self.num_steps - 1 and float(t_next) > 0.0:
                C0_hat_2, e0_hat_2 = self._denoise_leaf_vars(
                    node_feats=node_feats,
                    edge_index=edge_index,
                    batch=batch,
                    edge_attr=edge_attr,
                    P_0=P_0,
                    parent_idx=parent_idx,
                    leaf_idx=leaf_idx,
                    parent_pos=parent_pos,
                    C=C_next,
                    e=e_next,
                    sigma=float(t_next),
                    model=model,
                    model_kwargs=model_kwargs,
                )
                sigma_next = float(max(float(t_next), 1e-12))
                dC_prime = (C_next - C0_hat_2) / sigma_next
                de_prime = (e_next - e0_hat_2) / sigma_next

                C_next = C_hat + (float(t_next) - float(t_hat)) * (0.5 * dC + 0.5 * dC_prime)
                e_next = e_hat + (float(t_next) - float(t_hat)) * (0.5 * de + 0.5 * de_prime)

        return C_next.to(dtype=dtype), e_next.to(dtype=dtype)

    # ---------------------------
    # Internal: one denoise call
    # ---------------------------
    def _denoise_leaf_vars(
        self,
        *,
        node_feats: th.Tensor,
        edge_index: th.Tensor,
        batch: th.Tensor,
        edge_attr: th.Tensor,
        P_0: th.Tensor,
        parent_idx: th.Tensor,
        leaf_idx: th.Tensor,
        parent_pos: th.Tensor,
        C: th.Tensor,           # (L,3) float64
        e: th.Tensor,           # (L,1) float64
        sigma: float,
        model: Module,
        model_kwargs: dict[str, Any],
    ) -> tuple[th.Tensor, th.Tensor]:
        """
        Run one EDM denoising evaluation at noise level sigma (scalar).

        Returns:
          C0_hat, e0_hat in float64 (leaf variable space).
        """
        device = P_0.device
        # scalar coeffs (same for all leaves because sigma is scalar during sampling)
        sigma_t = th.full((leaf_idx.numel(), 1), float(sigma), device=device, dtype=th.float64)
        c_in, c_skip, c_out = self._edm_coeffs(sigma_t, self.sigma_data)

        # preconditioned inputs (leaf space)
        C_in = (c_in * C).to(dtype=P_0.dtype)
        e_in = (c_in * e).to(dtype=P_0.dtype)

        # build full-graph inputs
        P_t = P_0.clone()
        P_t[leaf_idx] = parent_pos + C_in

        N = P_0.size(0)
        e_feat = P_0.new_zeros((N, 1))
        e_feat[leaf_idx] = e_in

        log_sigma_feat = P_0.new_full((N, 1), math.log(max(float(sigma), 1e-12)) / 4.0)  # scale down for numerical stability
        node_feats_t = th.cat([node_feats, e_feat, log_sigma_feat], dim=-1)

        x_in = th.cat([P_t, node_feats_t], dim=-1)

        out = model(
            x=x_in,
            edge_index=edge_index,
            batch=batch,
            edge_attr=edge_attr,
            parent_idx=parent_idx,
            **model_kwargs,
        )
        if not isinstance(out, dict):
            raise ValueError("Model must return dict with 'rel_pred' and 'expansion_pred'.")
        rel_out = out["rel_pred"][leaf_idx].to(dtype=th.float64)       # (L,3)
        exp_out = out["expansion_pred"][leaf_idx].to(dtype=th.float64) # (L,1) or (L,)
        if exp_out.dim() == 1:
            exp_out = exp_out.unsqueeze(-1)

        # x0_hat = c_skip * x + c_out * F(c_in*x, sigma)
        C0_hat = (c_skip * C) + (c_out * rel_out)
        e0_hat = (c_skip * e) + (c_out * exp_out)
        return C0_hat, e0_hat