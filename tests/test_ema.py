"""Unit tests for the EMA (exponential moving average) of model parameters.

The key regression test is ``test_ema_smooths_slowly``: it pins down that a high
decay factor produces *slow* smoothing (the EMA barely moves toward a changed live
param). The previous implementation had the update operands swapped
(``decay * param + (1 - decay) * ema`` instead of ``decay * ema + (1 - decay) * param``),
which made the "EMA" track the live weights almost exactly — no averaging at all.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch as th

from graph_generation.model import EMA, EMA1


def _fill_params(model, value):
    with th.no_grad():
        for p in model.parameters():
            p.fill_(value)


def test_ema_smooths_slowly():
    """With decay saturated at 0.99, one update should move the EMA only ~1%
    toward a changed live param. The pre-fix (swapped) formula gives ~0.99 instead."""
    model = th.nn.Linear(4, 4)
    _fill_params(model, 0.0)

    ema = EMA(model, beta=0.99, gamma=1, power=1)  # ema_model deepcopy -> also 0.0

    # Move the live weights far away, then take a single high-decay step.
    _fill_params(model, 1.0)
    # step large -> decay = 1 - (1 + step)**-1 ~ 1.0, clipped to beta = 0.99.
    ema.update(step=10_000)

    # Correct EMA: 0.99 * 0.0 + 0.01 * 1.0 = 0.01 (moves 1% toward the new value).
    # Swapped bug:  0.99 * 1.0 + 0.01 * 0.0 = 0.99 (basically copies the new value).
    for p in ema.ema_model.parameters():
        assert th.allclose(p, th.full_like(p, 0.01), atol=1e-6)
        assert p.requires_grad is False


def test_ema_converges_to_held_constant():
    """Holding the live params at a constant should drive the EMA to that constant."""
    model = th.nn.Linear(3, 3)
    _fill_params(model, 0.0)

    ema = EMA(model, beta=0.999, gamma=1, power=1)

    target = 5.0
    _fill_params(model, target)
    for step in range(1, 20_001):
        ema.update(step=step)

    for p in ema.ema_model.parameters():
        assert th.allclose(p, th.full_like(p, target), atol=1e-3)


def test_ema1_is_noop():
    """EMA1 is the beta=1 passthrough: update() changes nothing and forward() uses
    the live model."""
    model = th.nn.Linear(2, 2)
    _fill_params(model, 0.3)
    before = [p.detach().clone() for p in model.parameters()]

    ema1 = EMA1(model)
    ema1.update(step=5)  # no-op

    for p, b in zip(model.parameters(), before):
        assert th.equal(p, b)

    # forward proxies to the (eval-mode) live model
    x = th.randn(6, 2)
    model.eval()
    with th.no_grad():
        expected = model(x)
        got = ema1(x)
    assert th.allclose(got, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
