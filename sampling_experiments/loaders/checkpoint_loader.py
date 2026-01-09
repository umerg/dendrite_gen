"""Helpers for constructing models/methods and loading checkpoints for sampling runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch as th

import graph_generation as gg


@dataclass
class SamplingContext:
    """Container describing everything needed for an interactive sampling run."""

    cfg: Any
    model: th.nn.Module
    method: Any
    device: th.device
    checkpoint_path: Path
    ema_beta: float | None = None
    checkpoint_step: int | None = None


def _instantiate_model(cfg, *, method_name: str):
    """Mirror the model construction logic from main.py for eval-time loading."""
    edge_embedding_nums = [2]
    edge_embedding_dims = [4]
    edge_attr_dim = 1
    if method_name == "expansion_augmented":
        edge_embedding_nums = [3]
        edge_embedding_dims = [4]

    model_name = getattr(cfg.model, "name", None)
    if model_name is None:
        raise ValueError("cfg.model.name must be specified.")

    base_kwargs = dict(
        n_layers=cfg.model.num_layers,
        feats_dim=cfg.model.feats_dim,
        pos_dim=3,
        m_dim=cfg.model.m_dim,
        edge_embedding_nums=edge_embedding_nums,
        edge_embedding_dims=edge_embedding_dims,
        edge_attr_dim=edge_attr_dim,
        dropout=cfg.model.dropout,
        norm_feats=cfg.model.norm_feats,
        global_linear_attn_every=cfg.model.global_linear_attn_every,
        global_linear_attn_heads=cfg.model.global_linear_attn_heads,
        global_linear_attn_dim_head=cfg.model.global_linear_attn_dim_head,
        num_global_tokens=cfg.model.num_global_tokens,
        offset_head_hidden=cfg.model.offset_head_hidden,
    )

    if model_name == "egnn":
        return gg.model.SO2_EGNN_Sparse_Network(
            use_global_fallback_frames=cfg.model.use_global_fallback_frames,
            **base_kwargs,
        )
    if model_name == "egnn_multihead":
        return gg.model.SO2_EGNN_Sparse_Network_MultiHead(
            use_global_fallback_frames=cfg.model.use_global_fallback_frames,
            **base_kwargs,
        )
    if model_name == "egnn_simple":
        kwargs = base_kwargs.copy()
        kwargs.pop("use_global_fallback_frames", None)
        return gg.model.SO2_EGNN_Sparse_Network_Simple(**kwargs)
    if model_name == "egnn_geometry_aware":
        kwargs = base_kwargs.copy()
        kwargs.pop("use_global_fallback_frames", None)
        return gg.model.SO2_EGNN_Sparse_Network_Geometry_Aware(**kwargs)

    raise ValueError(f"Unknown model name: {model_name}")


def _instantiate_method(cfg, *, method_name: str, method_cls: type | None = None):
    """Return Expansion method matching cfg.method.name with overrides."""
    diffusion_cfg = getattr(cfg, "diffusion", None)
    diffusion = None
    if diffusion_cfg is not None:
        diffusion_name = getattr(diffusion_cfg, "name", None)
        if diffusion_name == "basic":
            diffusion = gg.diffusion.DenoisingDiffusionModel(
                num_steps=diffusion_cfg.num_steps,
            )
        elif diffusion_name is not None:
            raise ValueError(f"Unknown diffusion name: {diffusion_name}")

    cls = method_cls
    if cls is None:
        if method_name == "expansion":
            cls = gg.method.Expansion_OneShot
        elif method_name == "expansion_augmented":
            cls = gg.method.Expansion_OneShot_Augmented
        else:
            raise ValueError(f"Unknown method name: {method_name}")

    if issubclass(cls, gg.method.Expansion):
        if diffusion is None:
            raise ValueError("Diffusion config is required for Expansion-based samplers.")
        expansion_loss_weight = getattr(cfg.method, "expansion_loss_weight", 1.0)
        method_kwargs = dict(
            diffusion=diffusion,
            deterministic_expansion=cfg.method.deterministic_expansion,
            red_threshold=cfg.reduction.red_threshold,
            expansion_loss_weight=expansion_loss_weight,
        )
    elif issubclass(cls, gg.method.Expansion_OneShot):
        method_kwargs = dict(
            deterministic_expansion=cfg.method.deterministic_expansion,
            red_threshold=cfg.reduction.red_threshold,
            leaf_noise_sigma=cfg.method.leaf_noise_sigma,
            leaf_noise_clip=cfg.method.leaf_noise_clip,
            sibling_loss_weight=cfg.method.sibling_loss_weight,
            use_sibling_matching=cfg.method.use_sibling_matching,
            debug=cfg.debugging,
            debug_max_batches=getattr(cfg, "debugging_max_batches", 2),
            debug_dir=getattr(cfg, "debugging_dir", None),
        )
    elif issubclass(cls, gg.method.Expansion_OneShot_Augmented):
        method_kwargs = dict(
            deterministic_expansion=cfg.method.deterministic_expansion,
            red_threshold=cfg.reduction.red_threshold,
            leaf_noise_sigma=cfg.method.leaf_noise_sigma,
            leaf_noise_clip=cfg.method.leaf_noise_clip,
            sibling_loss_weight=cfg.method.sibling_loss_weight,
            use_sibling_matching=cfg.method.use_sibling_matching,
            debug=cfg.debugging,
            debug_max_batches=getattr(cfg, "debugging_max_batches", 2),
            debug_dir=getattr(cfg, "debugging_dir", None),
        )
    else:
        raise ValueError(f"Unsupported method class: {cls}")

    return cls(**method_kwargs)



def load_sampling_items(
    cfg,
    *,
    checkpoint: str | Path,
    ema_beta: float | None = None,
    device: str = "cpu",
    method_cls: type | None = None,
) -> SamplingContext:
    """Load model/method weights similar to Trainer but for interactive eval."""
    if cfg is None:
        raise ValueError("cfg must be provided to load sampling items.")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    method_name = getattr(cfg.method, "name", None)
    if method_name is None:
        raise ValueError("cfg.method.name must be set.")

    device_obj = th.device(device)

    model = _instantiate_model(cfg, method_name=method_name).to(device_obj)
    method = _instantiate_method(cfg, method_name=method_name, method_cls=method_cls).to(device_obj)

    checkpoint = th.load(checkpoint_path, map_location=device_obj)
    if ema_beta is None:
        state_key = "model"
    else:
        beta_str = str(ema_beta)
        if beta_str.endswith(".0"):
            beta_str = beta_str.rstrip("0").rstrip(".")
        state_key = f"model_ema_{beta_str}"
    if state_key not in checkpoint:
        available = ", ".join(checkpoint.keys())
        raise KeyError(f"Checkpoint missing key '{state_key}'. Available keys: {available}")

    state_dict = checkpoint[state_key]
    if not state_dict:
        fallback_key = "model"
        if state_key != fallback_key and fallback_key in checkpoint and checkpoint[fallback_key]:
            print(
                f"[loader] Requested checkpoint key '{state_key}' is empty; "
                f"falling back to '{fallback_key}'."
            )
            state_dict = checkpoint[fallback_key]
            state_key = fallback_key
        else:
            print(f"[loader] Warning: checkpoint key '{state_key}' is empty.")
    sample_keys = list(state_dict.keys())[:10]
    print(f"[loader] Loading checkpoint '{checkpoint_path.name}' -> key '{state_key}' "
          f"({len(state_dict)} tensors). Sample keys: {sample_keys}")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as err:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        msg = ["Strict checkpoint loading failed."]
        msg.append(str(err))
        if missing:
            preview = ", ".join(missing[:10])
            msg.append(f"Missing keys ({len(missing)} total): {preview}")
        if unexpected:
            preview = ", ".join(unexpected[:10])
            msg.append(f"Unexpected keys ({len(unexpected)} total): {preview}")
        raise RuntimeError("\n".join(msg)) from err

    return SamplingContext(
        cfg=cfg,
        model=model,
        method=method,
        device=device_obj,
        checkpoint_path=checkpoint_path,
        ema_beta=ema_beta,
        checkpoint_step=checkpoint.get("step"),
    )
