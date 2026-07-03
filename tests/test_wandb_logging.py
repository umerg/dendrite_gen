"""Unit tests for Trainer.log() wandb batching.

Verifies the logging refactor that fixes validation plots landing at "random"
wandb steps: each log() call must flush a SINGLE wandb.log(payload, step=...)
with all leaves flattened, tagged train_step, and Figures wrapped as captioned
wandb.Image. Batching keeps wandb's internal _step aligned with the training
step so image/media panels index by the true step.

These tests exercise log() in isolation (no model/data/Trainer construction) by
binding the unbound methods onto a lightweight stand-in object.
"""
from __future__ import annotations

from types import SimpleNamespace, MethodType
from unittest.mock import MagicMock
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import graph_generation.training as training_mod
from graph_generation.training import Trainer


def _make_logger_stub(step: int, wandb_run):
    """Minimal object with just the attributes log()/_collect_log() touch."""
    obj = SimpleNamespace(
        cfg=SimpleNamespace(wandb=SimpleNamespace(logging=True)),
        wandb_run=wandb_run,
        step=step,
    )
    obj.log = MethodType(Trainer.log, obj)
    obj._collect_log = MethodType(Trainer._collect_log, obj)
    return obj


def test_log_batches_into_single_call_with_step(monkeypatch):
    # Ensure wandb is treated as available regardless of the test env.
    monkeypatch.setattr(training_mod, "wandb", MagicMock())
    run = MagicMock()
    obj = _make_logger_stub(step=5000, wandb_run=run)

    obj.log({
        "validation": {"ema_0.999": {"dist": {"mmd": 0.1}}},
        "timing": {"validation_total_s": 2.0},
    })

    # Exactly one wandb.log call for the whole nested dict.
    assert run.log.call_count == 1
    args, kwargs = run.log.call_args
    payload = args[0]

    # Explicit step pins wandb's internal counter to the true training step.
    assert kwargs["step"] == 5000
    assert payload["train_step"] == 5000

    # Nested keys flattened with '/'.
    assert payload["validation/ema_0.999/dist/mmd"] == 0.1
    assert payload["timing/validation_total_s"] == 2.0


def test_log_wraps_figures_with_step_caption(monkeypatch):
    fake_wandb = MagicMock()
    monkeypatch.setattr(training_mod, "wandb", fake_wandb)
    run = MagicMock()
    obj = _make_logger_stub(step=3000, wandb_run=run)

    fig = plt.figure()
    obj.log({"validation": {"ema_1": {"examples": fig, "sampling_s": 1.5}}})
    plt.close(fig)

    # Figure routed through wandb.Image with the true-step caption.
    fake_wandb.Image.assert_called_once()
    _, img_kwargs = fake_wandb.Image.call_args
    assert img_kwargs["caption"] == "step 3000"

    # Still a single batched log call carrying both the image and the scalar.
    assert run.log.call_count == 1
    payload = run.log.call_args[0][0]
    assert "validation/ema_1/examples" in payload
    assert payload["validation/ema_1/sampling_s"] == 1.5
    assert run.log.call_args[1]["step"] == 3000


def test_run_metadata_pushed_to_wandb_config():
    import torch as th

    run = MagicMock()
    obj = SimpleNamespace(
        wandb_run=run,
        train_graphs=list(range(2000)),
        validation_graphs=list(range(200)),
        test_graphs=list(range(200)),
        device="cpu",
    )
    obj._log_run_metadata = MethodType(Trainer._log_run_metadata, obj)

    model = th.nn.Linear(256, 256)  # 256*256 + 256 = 65792 params (~0.26 MB fp32)
    num_parameters = sum(p.numel() for p in model.parameters())
    obj._log_run_metadata(model, num_parameters)

    run.config.update.assert_called_once()
    args, kwargs = run.config.update.call_args
    meta = args[0]
    assert kwargs["allow_val_change"] is True
    assert meta["model_num_parameters"] == 65792
    assert meta["num_train_graphs"] == 2000
    assert meta["num_val_graphs"] == 200
    assert meta["num_test_graphs"] == 200
    assert meta["device"] == "cpu"
    assert meta["model_size_mb"] > 0


def test_run_metadata_noop_without_wandb():
    obj = SimpleNamespace(wandb_run=None)
    obj._log_run_metadata = MethodType(Trainer._log_run_metadata, obj)
    # Should not raise even though no dataset/device attrs are present.
    obj._log_run_metadata(model=object(), num_parameters=0)


def test_log_noop_when_wandb_disabled(monkeypatch):
    monkeypatch.setattr(training_mod, "wandb", MagicMock())
    run = MagicMock()
    obj = _make_logger_stub(step=100, wandb_run=run)
    obj.cfg.wandb.logging = False  # disabled

    obj.log({"training": {"loss": 0.5}})

    run.log.assert_not_called()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
