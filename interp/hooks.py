"""Activation capture / editing on SO2_EGNN_Network.

Site names (all are the 256-d feature slice; coordinates are never included):
  ``in``  — input features entering block 0 (raw features + tmd/class embeddings)
  ``L{i}`` — output of EGNN block i (i = 0..n_layers-1)
  ``A{i}`` — node features after the global-attention sub-block of block i (global blocks only)
  ``G{i}`` — global-token states after block i's ISAB, shape [B*m, 256], graph-major

Global blocks are ``mpnn_layers[i] = ModuleList([GlobalLinearAttention_Sparse, SO2_EGNN])``.
The network discards the updated tokens, so ``G{i}`` is a fresh per-layer readout.
Pass the bare network (``ema.ema_model``), not the EMA wrapper.
"""
from __future__ import annotations

import contextlib
from typing import Callable, Iterable

import torch as th
from torch import nn

POS_DIM = 3


def global_block_ids(model) -> list[int]:
    return [i for i, L in enumerate(model.mpnn_layers) if isinstance(L, nn.ModuleList)]


def egnn_module(model, i: int) -> nn.Module:
    L = model.mpnn_layers[i]
    return L[1] if isinstance(L, nn.ModuleList) else L


def all_sites(model) -> list[str]:
    g = global_block_ids(model)
    return (["in"] + [f"L{i}" for i in range(len(model.mpnn_layers))]
            + [f"A{i}" for i in g] + [f"G{i}" for i in g])


def _parse(site: str) -> tuple[str, int]:
    return (site, 0) if site == "in" else (site[0], int(site[1:]))


class ActivationRecorder(contextlib.AbstractContextManager):
    """Record activations at ``sites`` for every forward call while active.

    ``store[site]`` is a list with one tensor per network forward (detached, cast to
    ``dtype``, moved to ``device``). ``rows`` optionally selects node rows per call:
    a callable ``() -> LongTensor`` evaluated at hook time (e.g. reading InputCapture).
    """

    def __init__(self, model, sites: Iterable[str], dtype=th.float16, device="cpu",
                 rows: Callable[[], th.Tensor] | None = None):
        self.model, self.sites = model, list(sites)
        self.dtype, self.device, self.rows = dtype, device, rows
        self.store: dict[str, list[th.Tensor]] = {s: [] for s in self.sites}
        self._handles = []

    def _keep(self, site, t: th.Tensor, node_level: bool = True):
        t = t.detach()
        if node_level and self.rows is not None:
            t = t[self.rows()]
        self.store[site].append(t.to(device=self.device, dtype=self.dtype).clone())

    def __enter__(self):
        for site in self.sites:
            kind, i = _parse(site)
            if kind == "in":
                h = self.model.mpnn_layers[0].register_forward_pre_hook(
                    lambda m, args, s=site: self._keep(s, args[0][:, POS_DIM:]))
            elif kind == "L":
                h = egnn_module(self.model, i).register_forward_hook(
                    lambda m, a, out, s=site: self._keep(s, out[:, POS_DIM:]))
            elif kind == "A":
                h = self.model.mpnn_layers[i][0].register_forward_hook(
                    lambda m, a, out, s=site: self._keep(s, out[0]))
            elif kind == "G":
                h = self.model.mpnn_layers[i][0].register_forward_hook(
                    lambda m, a, out, s=site: self._keep(s, out[1], node_level=False))
            else:
                raise ValueError(f"Unknown site {site!r}")
            self._handles.append(h)
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def pop(self) -> dict[str, list[th.Tensor]]:
        out, self.store = self.store, {s: [] for s in self.sites}
        return out


class InputCapture(contextlib.AbstractContextManager):
    """Capture the network's keyword inputs (x, batch, parent_idx, tmd, ...) per call."""

    KEYS = ("x", "batch", "parent_idx", "edge_index", "tmd", "cell_class")

    def __init__(self, model):
        self.model, self.calls, self._h = model, [], None

    def __enter__(self):
        def pre(m, args, kwargs):
            self.calls.append({k: kwargs.get(k) for k in self.KEYS})
        self._h = self.model.register_forward_pre_hook(pre, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        self._h.remove()
        return False

    @property
    def last(self) -> dict:
        return self.calls[-1]


def _graph_mean(t: th.Tensor, batch: th.Tensor) -> th.Tensor:
    B = int(batch.max().item()) + 1
    s = t.new_zeros((B, t.size(1))).index_add_(0, batch, t)
    c = t.new_zeros((B, 1)).index_add_(0, batch, t.new_ones((t.size(0), 1)))
    return (s / c.clamp_min(1))[batch]


class AttnAblation(contextlib.AbstractContextManager):
    """Ablate the tokens→nodes read (``attn2``) in global blocks.

    mode: "zero" (no global read — note this also drops attn2's output bias, so it removes a
    constant offset as well as routing; "mean" keeps it), "mean" (replace with a fixed per-block vector from
    ``means[i]``), "graphmean" (each node gets its graph's average read — keeps the
    broadcast, removes node-specific routing). ``ff_x`` and the token stream are untouched.
    Implemented as a per-instance override because hooks on attn2 never fire
    (``batched_forward`` calls ``Attention.forward`` directly).
    """

    def __init__(self, model, blocks: Iterable[int] | None = None, mode: str = "zero",
                 means: dict[int, th.Tensor] | None = None):
        if mode not in ("zero", "mean", "graphmean", "record"):
            raise ValueError(mode)
        self.model, self.mode, self.means = model, mode, means or {}
        # mode="record": pass-through that accumulates per-block sums of the attn2 output,
        # used to estimate the dataset-mean read for mode="mean" (see ``recorded_means``).
        self.sums: dict[int, th.Tensor] = {}
        self.counts: dict[int, int] = {}
        self.blocks = list(blocks) if blocks is not None else global_block_ids(model)
        self._patched = []

    def __enter__(self):
        for i in self.blocks:
            attn2 = self.model.mpnn_layers[i][0].attn2
            orig = attn2.sparse_forward_with_separate_batches  # bound method

            def patched(q, kv, *, q_batch=None, kv_batch=None, mask=None, _orig=orig, _i=i):
                if self.mode == "zero":
                    return th.zeros_like(q)
                if self.mode == "mean":
                    return self.means[_i].to(q).expand_as(q).clone()
                out = _orig(q, kv, q_batch=q_batch, kv_batch=kv_batch, mask=mask)
                if self.mode == "record":
                    s = out.detach().sum(0)
                    self.sums[_i] = self.sums.get(_i, 0) + s
                    self.counts[_i] = self.counts.get(_i, 0) + out.size(0)
                    return out
                return _graph_mean(out, q_batch) if q_batch is not None else out.mean(0, keepdim=True).expand_as(out)

            attn2.sparse_forward_with_separate_batches = patched
            self._patched.append(attn2)
        return self

    def recorded_means(self) -> dict[int, th.Tensor]:
        return {i: self.sums[i] / max(self.counts[i], 1) for i in self.sums}

    def __exit__(self, *exc):
        for attn2 in self._patched:
            del attn2.sparse_forward_with_separate_batches  # restore the class method
        self._patched.clear()
        return False


class SiteEditor(contextlib.AbstractContextManager):
    """Replace the features at an ``L{i}`` or ``A{i}`` site with ``fn(feats) -> feats``.

    Used for amnesic erasure (apply to all nodes) and steering (masked add). ``fn``
    returning ``None`` leaves the output untouched (exact no-op).
    """

    def __init__(self, model, site: str, fn: Callable[[th.Tensor], th.Tensor | None]):
        kind, i = _parse(site)
        if kind not in ("L", "A"):
            raise ValueError("SiteEditor supports L{i} and A{i} sites.")
        self.model, self.kind, self.i, self.fn, self._h = model, kind, i, fn, None

    def __enter__(self):
        if self.kind == "L":
            def hook(m, a, out):
                new = self.fn(out[:, POS_DIM:])
                if new is None:
                    return None
                return th.cat([out[:, :POS_DIM], new], dim=-1)
            self._h = egnn_module(self.model, self.i).register_forward_hook(hook)
        else:
            def hook(m, a, out):
                new = self.fn(out[0])
                return None if new is None else (new, out[1])
            self._h = self.model.mpnn_layers[self.i][0].register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        self._h.remove()
        return False


@contextlib.contextmanager
def fixed_flow_time(diffusion, t0: float):
    """Make the flow model's training forward use t = t0 for every graph."""
    def fixed(num_graphs, device):
        return th.full((num_graphs,), float(t0), device=device)
    diffusion._sample_time = fixed
    try:
        yield
    finally:
        del diffusion._sample_time


@contextlib.contextmanager
def keyed_flow_noise(keys: "np.ndarray"):
    """Make the flow-matching forward's prior noise a function of per-leaf keys only.

    ``VFlowMatchingModel.forward`` draws ``randn_like(C_0)`` ([L,3]) then ``randn_like(e_0)``
    ([L,1]) over all training leaves of the batch, so by default a node's noise depends on
    how items were packed into batches. ``keys`` is an int array [L, k] (e.g. seed, t,
    neuron, level, node) in training-leaf order; each leaf gets 4 standard normals from its
    own seeded stream (first 3 → C, 4th → e). Only those two calls are intercepted;
    sigma_min > 0 (an extra draw) raises.
    """
    import numpy as np

    keys = np.asarray(keys, dtype=np.int64)
    z = np.stack([np.random.default_rng(np.random.SeedSequence([int(v) & 0x7FFFFFFF for v in k]))
                  .standard_normal(4) for k in keys]) if len(keys) else np.zeros((0, 4))
    orig = th.randn_like
    calls = {"n": 0}

    def fake(t, *a, **kw):
        n = calls["n"]
        calls["n"] += 1
        if t.dim() == 2 and t.size(0) == len(keys) and n == 0 and t.size(1) == 3:
            return th.as_tensor(z[:, :3], dtype=t.dtype, device=t.device)
        if t.dim() == 2 and t.size(0) == len(keys) and n == 1 and t.size(1) == 1:
            return th.as_tensor(z[:, 3:], dtype=t.dtype, device=t.device)
        raise RuntimeError(f"keyed_flow_noise: unexpected randn_like call #{n} with shape {tuple(t.shape)}")

    th.randn_like = fake
    try:
        yield
    finally:
        th.randn_like = orig
