import numpy as np
import torch as th
from torch.nn import Module

class DenoisingDiffusionModel(Module):
    P_mean = -1.2
    P_std = 1.2
    sigma_data = 0.5
    sigma_min = 0.002
    sigma_max = 80
    rho = 7
    S_min = 0.05
    S_max = 50
    S_noise = 1.003
    S_churn = 40

    def __init__(self, num_steps):
        self.num_steps = num_steps

    @property
    def device(self):
        assert hasattr(self, "_device")
        return self._device

    def to(self, device):
        self._device = device
        self.model_wrapper.to(device)
        return self

    def get_loss(self, node_feats, edge_index, batch, edge_attr, P_0, C_0, parent_idx, leaf_idx_train, leaf_expansion, leaf_parent_idx, model):
        # rescale expansion from {1,2} to {-1, 1}
        leaf_expansion = (leaf_expansion - 1) * 2 - 1

        # sample noise level
        num_graphs = batch.max().item() + 1
        rnd_normal = th.randn((num_graphs,), device=self.device)
        t = (rnd_normal * self.P_std + self.P_mean).exp() # change to matrix level TODO

        # sample noise
        expansion_noise = th.randn_like(leaf_expansion) * t[batch]
        position_noise = th.randn_like(C_0) * t[batch]

        # add noise
        e_t = leaf_expansion + expansion_noise  # update e_0 with noisy leaf expansion to make e_t
        C_t = C_0 + position_noise

        # update P_0 with noisy leaf position to make P_t

        # update node_feats to include e_t

        # create input features as x_in as concat of P_t and node_feats

        # make prediction - SKIPPING EDM FOR NOW AND USING BASIC DIFFUSION MODEL
        expansion_pred, pos_pred = self.model(
            edge_index=edge_index,
            batch=batch,
            x_in=x_in,
            edge_attr=edge_attr,
            parent_idx=parent_idx,
        )

        # compute loss
        weight = (t**2 + self.sigma_data**2) / (t * self.sigma_data) ** 2
        expansion_loss = weight[batch] * (expansion_pred - leaf_expansion) ** 2
        pos_loss = weight[batch] * (pos_pred - C_0) ** 2

        return expansion_loss, pos_loss

    @th.no_grad()
    def sample(self, edge_index, batch, model, model_kwargs):
        pass

