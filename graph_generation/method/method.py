from abc import ABC, abstractmethod

import torch as th
from torch.nn import Module


class Method(ABC):
    """Interface for graph generation methods."""

    def __init__(self, diffusion=None):
        self.diffusion = diffusion

    @abstractmethod
    def sample_graphs(self, target_size: th.Tensor, model: Module):
        pass

    @abstractmethod
    def get_loss(self, batch, model: Module):
        pass

    @property
    def device(self):
        return self._device

    def to(self, device):
        self._device = device
        if self.diffusion is not None:
            self.diffusion.to(device)
        return self
