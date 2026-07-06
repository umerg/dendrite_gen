import logging
import time

import torch as th
import torch.nn.functional as F
from torch.nn import Module

from graph_generation.method.helpers import (
    local_to_global,
    patch_geometry_for_noised_leaves,
)
from graph_generation.diffusion.diagnostics import compute_flow_diagnostics

logger = logging.getLogger(__name__)


class FlowMatchingModel(Module):
    """Conditional flow-matching alternative to DenoisingDiffusionModel.

    Transports a Gaussian prior (t=0) to data (t=1) along the linear / optimal-transport
    path ``x_t = (1 - t) * x_noise + t * x_data``. The network keeps the same
    data (x_1) prediction parameterization as the diffusion model — it predicts the clean
    targets ``C_0`` (local-frame child offset) and ``e_0`` (expansion decision in [-1, 1]) —
    so this is a drop-in replacement: model architecture, the ``model(...)`` call, the
    ``rel_pred``/``expansion_pred`` outputs, and all of ``expansion.py`` are unchanged. The
    only model-facing difference is that the second conditioning feature carries the flow
    time ``t`` instead of ``log sigma``.

    Sampling integrates the probability-flow ODE with explicit Euler steps, backing out the
    velocity from the data prediction: ``v = (x_hat_1 - x_t) / (1 - t)``.
    """

    cond_dim = 2  # e_t feature + time feature per node

    def __init__(
        self,
        num_steps: int = 1,
        prior_std: float = 1.0,
        time_dist: str = "uniform",
        beta_a: float = 2.0,
        beta_b: float = 1.0,
        sigma_min: float = 0.0,
        prior_std_pos: list | tuple | None = None,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.prior_std = float(prior_std)
        if time_dist not in ("uniform", "beta"):
            raise ValueError(f"time_dist must be 'uniform' or 'beta', got '{time_dist}'.")
        self.time_dist = time_dist
        self.beta_a = float(beta_a)
        self.beta_b = float(beta_b)
        self.sigma_min = float(sigma_min)
        # Optional anisotropic (per-axis) prior std for the local-frame position offset
        # C = (forward, sideways, axial). None -> isotropic scalar prior_std (unchanged
        # behavior). When set, this normalizes the prior to the data's per-axis C_0 scale.
        if prior_std_pos is not None:
            prior_std_pos = tuple(float(s) for s in prior_std_pos)
            if len(prior_std_pos) != 3:
                raise ValueError(f"prior_std_pos must have length 3, got {len(prior_std_pos)}.")
        self.prior_std_pos = prior_std_pos

    def _pos_scale(self, device: th.device, dtype: th.dtype):
        """Per-axis prior std for the position offset C [.,3]; scalar if isotropic."""
        if self.prior_std_pos is None:
            return self.prior_std
        return th.tensor(self.prior_std_pos, device=device, dtype=dtype).view(1, 3)

    def _sample_time(self, num_graphs: int, device: th.device) -> th.Tensor:
        """Sample one flow time t in [0, 1] per graph, shape [num_graphs]."""
        if self.time_dist == "uniform":
            return th.rand((num_graphs,), device=device)
        # beta
        beta = th.distributions.Beta(self.beta_a, self.beta_b)
        return beta.sample((num_graphs,)).to(device=device)

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
        pre_geom_p0: dict | None = None,
        local_forward: th.Tensor | None = None,
        local_sideways: th.Tensor | None = None,
        uhat: th.Tensor | None = None,
        return_diag_arrays: bool = False,
    ) -> tuple[th.Tensor, th.Tensor, dict]:
        """Compute flow-matching (data-prediction) losses for positional + expansion targets.

        Returns ``(exp_loss, pos_loss, diag)`` where ``diag`` is a flat dict of stratified
        training diagnostics (see ``compute_flow_diagnostics``); empty when there are no leaves.
        When ``return_diag_arrays`` is set, ``diag['_arrays']`` additionally carries the raw
        per-leaf tensors (implied clean prediction ``C_pred``, target ``C_0``, flow time
        ``t_leaf``, root-child mask) for teacher-forced diagnostics — off by default so the
        training path and ``Trainer.log`` (which only flattens float leaves) are unchanged.
        """
        device = P_0.device
        num_leaves = leaf_idx_train.numel()
        if node_feats is None:
            node_feats = P_0.new_zeros((P_0.size(0), 0))

        if num_leaves == 0:
            zero = P_0.new_zeros(())
            return zero, zero, {}

        leaf_expansion = leaf_expansion.to(dtype=P_0.dtype).view(-1, 1)
        e_0 = 2.0 * leaf_expansion - 1.0  # map [0,1] to [-1,1]

        if batch.numel() == 0:
            raise ValueError("Batch vector is empty; cannot sample flow time.")
        num_graphs = int(batch.max().item()) + 1
        t_graph = self._sample_time(num_graphs, device)

        leaf_batch = batch[leaf_idx_train]
        t_leaf = t_graph[leaf_batch].view(-1, 1)

        # Linear / OT path: noise at t=0, data at t=1.
        C_noise = th.randn_like(C_0) * self._pos_scale(C_0.device, C_0.dtype)
        e_noise = th.randn_like(e_0) * self.prior_std
        C_t = (1.0 - t_leaf) * C_noise + t_leaf * C_0
        e_t = (1.0 - t_leaf) * e_noise + t_leaf * e_0
        if self.sigma_min > 0.0:
            C_t = C_t + self.sigma_min * th.randn_like(C_t)
            e_t = e_t + self.sigma_min * th.randn_like(e_t)

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
        t_node = t_graph[batch].view(N, 1)
        node_feats_t = th.cat([node_feats, e_feat, t_node], dim=-1)

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
            cell_class=cell_class,
            pre_geom=pre_geom,
        )
        if device.type == 'cuda':
            th.cuda.synchronize(device)
        if not isinstance(out, dict):
            raise ValueError("Model must return dict with 'rel_pred' and 'expansion_pred'.")
        rel_pred_all = out["rel_pred"]
        exp_pred_all = out["expansion_pred"]

        C_pred = rel_pred_all[leaf_idx_train]
        e_pred = exp_pred_all[leaf_idx_train]
        if e_pred.dim() == 1:
            e_pred = e_pred.unsqueeze(-1)

        # Data-prediction loss: regress the clean targets directly.
        pos_loss = F.mse_loss(C_pred, C_0)
        exp_loss = F.mse_loss(e_pred, e_0)

        # Stratified, teacher-forced training diagnostics (cheap, no-grad). These share
        # the exact targets/frames the loss is computed against and are returned as a
        # third element; `expansion.get_loss` unpacks them tolerantly so the other
        # diffusion variants (2-tuple return) are unaffected.
        with th.no_grad():
            if self.prior_std_pos is not None:
                prior_var = tuple(s * s for s in self.prior_std_pos)
            else:
                prior_var = (self.prior_std ** 2,) * 3
            # A leaf is a root-child iff its parent is the root (parent_idx == -1).
            is_root_child = parent_idx[leaf_parent_idx] < 0
            diag = compute_flow_diagnostics(
                C_pred=C_pred, C_0=C_0, e_pred=e_pred, e_0=e_0,
                t_leaf=t_leaf, is_root_child=is_root_child, prior_var=prior_var,
            )
        if return_diag_arrays:
            diag = {**diag, "_arrays": {
                "C_pred": C_pred.detach(), "C_0": C_0.detach(),
                "t_leaf": t_leaf.detach(), "is_root_child": is_root_child.detach(),
            }}
        return exp_loss, pos_loss, diag

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
        cell_class: th.Tensor | None = None,
        local_forward: th.Tensor | None = None,
        local_sideways: th.Tensor | None = None,
        uhat: th.Tensor | None = None,
        pre_geom_p0: dict | None = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Integrate the probability-flow ODE from noise (t=0) to data (t=1) via Euler steps."""
        device = P_0.device
        model_kwargs = model_kwargs or {}
        if tmd is not None:
            model_kwargs = {**model_kwargs, "tmd": tmd}
        if cell_class is not None:
            model_kwargs = {**model_kwargs, "cell_class": cell_class}
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

        L = leaf_idx.numel()
        N = P_0.size(0)
        parent_pos = P_0[leaf_parent_idx]

        # Time grid t = 0 .. 1 (the loop never evaluates t=1 itself).
        steps = max(int(self.num_steps), 1)
        grid = th.linspace(0.0, 1.0, steps=steps + 1, device=device)

        # Initialise from the Gaussian prior at t=0.
        C = th.randn((L, 3), device=device) * self._pos_scale(device, P_0.dtype)
        e = th.randn((L, 1), device=device) * self.prior_std

        C1_pred = th.zeros_like(C)
        e1_pred = th.zeros_like(e)

        for step in range(steps):
            t_cur = float(grid[step].item())
            t_next = float(grid[step + 1].item())
            dt = t_next - t_cur

            P_cur = P_0.clone()
            if local_forward is not None and local_sideways is not None and uhat is not None:
                C_global = local_to_global(C, local_forward, local_sideways, uhat)
                P_cur[leaf_idx] = parent_pos + C_global
            else:
                P_cur[leaf_idx] = parent_pos + C

            e_feat = P_0.new_zeros((N, 1))
            e_feat[leaf_idx] = e
            t_feat = P_0.new_full((N, 1), t_cur)
            node_feats_t = th.cat([node_feats, e_feat, t_feat], dim=-1)
            x_in = th.cat([P_cur, node_feats_t], dim=-1)

            # Patch precomputed P_0 geometry for noised leaf positions
            pre_geom_t = None
            if pre_geom_p0 is not None:
                pre_geom_t = patch_geometry_for_noised_leaves(
                    pre_geom_p0, P_cur, leaf_idx, parent_idx,
                    edge_index, uhat,
                )

            out = model(
                x=x_in,
                edge_index=edge_index,
                batch=batch,
                edge_attr=edge_attr,
                parent_idx=parent_idx,
                pre_geom=pre_geom_t,
                **model_kwargs,
            )

            if not isinstance(out, dict):
                raise ValueError("Model must return dict with 'rel_pred' and 'expansion_pred'.")
            rel_pred_all = out["rel_pred"]
            exp_pred_all = out["expansion_pred"]

            C1_pred = rel_pred_all[leaf_idx]
            e1_pred = exp_pred_all[leaf_idx]
            if e1_pred.dim() == 1:
                e1_pred = e1_pred.unsqueeze(-1)

            # Data-prediction velocity: v = (x_hat_1 - x_t) / (1 - t). The final step has
            # (1 - t_cur) == dt, so the Euler update lands exactly on the clean prediction.
            inv_one_minus_t = 1.0 / max(1.0 - t_cur, 1e-12)
            vel_C = (C1_pred - C) * inv_one_minus_t
            vel_e = (e1_pred - e) * inv_one_minus_t
            C = C + dt * vel_C
            e = e + dt * vel_e

        # After integration C ≈ C1_pred (and e ≈ e1_pred); return the integrated state.
        return C, e
