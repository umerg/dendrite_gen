"""Unit tests for build_optimizer (configurable Adam / AdamW).

Verifies the optimizer factory: default cfg reproduces plain Adam with zero
weight decay (bit-for-bit legacy behavior), AdamW is selectable with weight
decay, the name is case-insensitive, and unknown names raise. No model/data or
Trainer construction needed — a bare Linear + SimpleNamespace cfg suffices.
"""
from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch as th

from graph_generation.training import build_optimizer


def _params():
    return th.nn.Linear(4, 4).parameters()


def test_default_is_plain_adam_zero_weight_decay():
    cfg = SimpleNamespace(lr=1e-3)  # no optimizer / weight_decay fields
    opt = build_optimizer(_params(), cfg)
    assert isinstance(opt, th.optim.Adam)
    assert not isinstance(opt, th.optim.AdamW)  # Adam, not the AdamW subclass
    assert opt.param_groups[0]["lr"] == 1e-3
    assert opt.param_groups[0]["weight_decay"] == 0.0


def test_adamw_with_weight_decay():
    cfg = SimpleNamespace(lr=5e-4, optimizer="adamw", weight_decay=0.01)
    opt = build_optimizer(_params(), cfg)
    assert isinstance(opt, th.optim.AdamW)
    assert opt.param_groups[0]["lr"] == 5e-4
    assert opt.param_groups[0]["weight_decay"] == 0.01


def test_optimizer_name_is_case_insensitive():
    cfg = SimpleNamespace(lr=1e-3, optimizer="AdamW")
    opt = build_optimizer(_params(), cfg)
    assert isinstance(opt, th.optim.AdamW)


def test_explicit_adam_with_weight_decay():
    cfg = SimpleNamespace(lr=1e-3, optimizer="adam", weight_decay=0.05)
    opt = build_optimizer(_params(), cfg)
    assert isinstance(opt, th.optim.Adam)
    assert not isinstance(opt, th.optim.AdamW)
    assert opt.param_groups[0]["weight_decay"] == 0.05


def test_unknown_optimizer_raises():
    cfg = SimpleNamespace(lr=1e-3, optimizer="lion")
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer(_params(), cfg)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
