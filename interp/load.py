"""Config / model / data loading for interpretability runs.

Builds the model through the training path (``main.get_expansion_items``) so the
architecture is byte-identical to the checkpoint, then loads EMA, raw, or no weights.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np
import torch as th
from omegaconf import DictConfig

import graph_generation as gg
from utils.data_loading import load_swc_graph
from utils.tmd import compute_tmd_mixed

REPO_ROOT = Path(__file__).resolve().parent.parent
EMA_KEY = "model_ema_0.999"
EMA_PREFIX = "ema_model."


def compose_cfg(config_name: str, overrides: Sequence[str] = ()) -> DictConfig:
    """Compose a hydra config exactly as ``main.py -cn <config_name>`` would."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(REPO_ROOT / "config"), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


def build_diffusion(cfg: DictConfig):
    """Replicate main.py's diffusion branch (flow / flow_v only; the live stack)."""
    dc = cfg.diffusion
    name = getattr(dc, "name", None)
    if name not in ("flow", "flow_v"):
        raise ValueError(f"interp supports flow/flow_v diffusion only, got {name!r}.")
    cls = gg.diffusion.VFlowMatchingModel if name == "flow_v" else gg.diffusion.FlowMatchingModel
    ps = getattr(dc, "prior_std_pos", None)
    return cls(
        num_steps=dc.num_steps,
        prior_std=getattr(dc, "prior_std", 1.0),
        time_dist=getattr(dc, "time_dist", "uniform"),
        beta_a=getattr(dc, "beta_a", 2.0),
        beta_b=getattr(dc, "beta_b", 1.0),
        sigma_min=getattr(dc, "sigma_min", 0.0),
        prior_std_pos=(list(ps) if ps is not None else None),
    )


def _ensure_root(G: nx.Graph) -> nx.Graph:
    root = G.graph.get("root", None)
    if root is None or root not in G.nodes:
        G.graph["root"] = next(iter(G.nodes))
    return G


def load_split(cfg: DictConfig, split: str, max_n: int | None = None,
               data_dir: str | Path | None = None) -> list[nx.Graph]:
    """Load ``<data_dir>/<split>/*.swc`` in filename order (same order as main.py).

    Each graph gets ``G.graph['neuron_id']`` (the SWC filename stem) and ``['split']``.
    """
    root = Path(data_dir if data_dir is not None else cfg.dataset.data_dir) / split
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = sorted(f for f in root.iterdir()
                   if f.is_file() and f.name.endswith(".swc") and not f.name.startswith("._"))
    if max_n is not None:
        files = files[: int(max_n)]
    graphs = []
    for f in files:
        G = _ensure_root(load_swc_graph(f))
        G.graph["neuron_id"] = f.name[: -len(".swc")]
        G.graph["split"] = split
        graphs.append(G)
    return graphs


def uhat_of(cfg: DictConfig) -> np.ndarray:
    return np.asarray(cfg.model.so2_axis, dtype=float).reshape(3)


def tmd_vector(G: nx.Graph, cfg: DictConfig) -> np.ndarray:
    """Conditioning TMD vector, computed on the UNSCALED graph exactly as main.py does."""
    return compute_tmd_mixed(
        G,
        filtrations=list(cfg.model.tmd_filtrations),
        n_bins=int(cfg.model.tmd_bins),
        uhat=uhat_of(cfg),
    )


def ema_state_dict(ckpt: dict) -> dict:
    """Strip the EMA wrapper prefix so the dict loads into a bare SO2_EGNN_Network."""
    if EMA_KEY not in ckpt:
        raise KeyError(f"Checkpoint has no '{EMA_KEY}' (keys: {sorted(ckpt)}).")
    sd = ckpt[EMA_KEY]
    bad = [k for k in sd if not k.startswith(EMA_PREFIX)]
    if bad:
        raise ValueError(f"Unexpected non-EMA keys in {EMA_KEY}: {bad[:5]}")
    return {k[len(EMA_PREFIX):]: v for k, v in sd.items()}


def build_method_and_model(cfg: DictConfig, ckpt_path: str | Path | None = None,
                           which: str = "ema", seed: int = 0,
                           pool: list[nx.Graph] | None = None):
    """Return ``(method, model, meta)`` with ``model`` in eval mode.

    which: "ema" (EMA 0.999 weights, what eval/sampling uses), "raw" (``ckpt['model']``),
    or "random" (architecture only, seeded init — the random-init probe control).
    ``pool`` is a handful of graphs used only to satisfy ``get_expansion_items``'s
    dataset construction (defaults to the first 2 val graphs).
    """
    from main import get_expansion_items

    if str(cfg.model.root_child_order) != "axial_extent":
        raise ValueError("interp assumes root_child_order=axial_extent (all parity configs).")
    if pool is None:
        pool = load_split(cfg, "val", max_n=2)

    th.manual_seed(seed)  # makes "random" reproducible; harmless otherwise
    items = get_expansion_items(cfg, pool, diffusion=build_diffusion(cfg))
    model, method = items["model"], items["method"]

    meta = {"config": cfg.get("name"), "which": which, "seed": seed,
            "ckpt": str(ckpt_path) if ckpt_path else None, "git_sha": _git_sha()}
    if which == "random":
        pass
    else:
        if ckpt_path is None:
            raise ValueError(f"which={which!r} needs a ckpt_path.")
        ckpt = th.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ema_state_dict(ckpt) if which == "ema" else ckpt["model"]
        model.load_state_dict(sd, strict=True)
        meta.update(step=int(ckpt.get("step", -1)), run_id=ckpt.get("run_id"))

    if not np.allclose(model.uhat.detach().cpu().numpy().reshape(3), uhat_of(cfg)):
        raise ValueError("model.uhat disagrees with cfg.model.so2_axis.")
    model.eval()
    return method, model, meta


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return None
