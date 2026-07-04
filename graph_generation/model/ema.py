import copy

import numpy as np
from torch.nn import Module


class EMA(Module):
    """Exponential Moving Average for model parameters.

    If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).
    """

    def __init__(self, model, beta=0.9999, gamma=1, power=1):
        assert 0 < beta < 1

        super().__init__()
        self._model = [model]  # hack to not register model as submodule
        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()
        self.beta = beta
        self.gamma = gamma
        self.power = power

        self.train_param_names = [
            name for name, param in self.model.named_parameters() if param.requires_grad
        ]

    @property
    def model(self):
        return self._model[0]

    def update(self, step):
        decay = 1 - (1 + step * self.gamma) ** (-self.power)
        decay = np.clip(decay, 0.0, self.beta)

        for (name, param), (ema_name, ema_param) in zip(
            self.model.named_parameters(), self.ema_model.named_parameters()
        ):
            assert name == ema_name
            if name not in self.train_param_names:
                continue

            new_ema_param = decay * ema_param.data + (1 - decay) * param.data
            ema_param.data.copy_(new_ema_param)
            assert ema_param.requires_grad == False

    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)

    # Proxy attribute access (e.g., feats_dim) to underlying model or ema_model.
    def __getattr__(self, name):
        # Prefer ema_model attributes (updated weights) then base model.
        if name in {"_model", "ema_model", "beta", "gamma", "power", "train_param_names"}:
            return super().__getattr__(name)
        if hasattr(self.ema_model, name):
            return getattr(self.ema_model, name)
        if hasattr(self.model, name):
            return getattr(self.model, name)
        # Fall back to default behavior (may raise AttributeError)
        return super().__getattr__(name)


class EMA1(Module):
    def __init__(self, model):
        super().__init__()
        self._model = [model]  # hack to not register model as submodule

    @property
    def model(self):
        return self._model[0]

    def update(self, step):
        pass

    def forward(self, *args, **kwargs):
        training_mode = self.model.training
        self.model.eval()
        res = self.model(*args, **kwargs)
        self.model.train(training_mode)
        return res

    def __getattr__(self, name):
        if name in {"_model"}:
            return super().__getattr__(name)
        if hasattr(self.model, name):
            return getattr(self.model, name)
        return super().__getattr__(name)
