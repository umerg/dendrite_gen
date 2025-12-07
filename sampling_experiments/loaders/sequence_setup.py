"""Helpers for building reduction sequences for a single graph."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
from omegaconf import DictConfig

import graph_generation as gg
from graph_generation.data.reduction_dataset import RandRedDataset
from graph_generation.data.data import ReducedGraphData

from sampling_experiments.loaders.checkpoint_loader import SamplingContext, load_sampling_items
from sampling_experiments.utils import load_hydra_config
from utils.data_loading import load_swc_graph, nx_graph_to_adj_pos


@dataclass
class ReductionSequenceBundle:
    """Container describing the per-step reduction sequence for one graph."""

    graph: nx.Graph
    graph_path: Path | None
    adjacency: sp.spmatrix
    positions: np.ndarray
    node_order: np.ndarray
    steps: list[ReducedGraphData]
    reduction_seed: int


@dataclass
class SequenceSetupResult:
    """Output of :func:`prepare_sequence_setup` for notebook consumption."""

    cfg: DictConfig
    context: SamplingContext
    reduction_bundle: ReductionSequenceBundle


def _ensure_graph_positions(graph: nx.Graph) -> nx.Graph:
    """Validate that each node has a 3D position and coerce dtype to float32."""
    for node in graph.nodes:
        pos = graph.nodes[node].get("pos", None)
        if pos is None:
            raise ValueError(f"Node {node} is missing 'pos' attribute; cannot run reduction.")
        arr = np.asarray(pos, dtype=np.float32)
        if arr.shape != (3,):
            raise ValueError(f"Node {node} position must have shape (3,), got {arr.shape}")
        graph.nodes[node]["pos"] = arr
    return graph


def load_graph_from_path(graph_path: str | Path) -> nx.Graph:
    """Load a single graph from disk (gpickle/pickle/SWC) with 3D positions."""
    path = Path(graph_path)
    if not path.exists():
        raise FileNotFoundError(f"Graph path not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".gpickle":
        graph = nx.read_gpickle(path)
    elif suffix in {".pkl", ".pickle"}:
        with path.open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, nx.Graph):
            raise TypeError(f"Pickle at {path} did not contain a NetworkX graph.")
        graph = obj
    elif suffix.endswith(".swc"):
        graph = load_swc_graph(path)
    elif suffix == ".graphml":
        graph = nx.read_graphml(path)
    else:
        raise ValueError(
            f"Unsupported graph format '{path.suffix}'. "
            "Provide a .gpickle, .pkl/.pickle containing a NetworkX graph, or an .swc file."
        )

    if isinstance(graph, nx.DiGraph):
        graph = nx.Graph(graph)

    if not nx.is_tree(graph):
        raise ValueError(f"Loaded graph {path} is not a tree; reduction expects trees.")
    graph = _ensure_graph_positions(graph)
    return graph


def _build_reduction_factory(cfg) -> gg.reduction.ReductionFactory:
    reduction_cfg = getattr(cfg, "reduction", None)
    if reduction_cfg is None:
        raise ValueError("Config missing 'reduction' section required for ReductionFactory.")
    return gg.reduction.ReductionFactory(
        mode=reduction_cfg.mode,
        cherry_p=reduction_cfg.cherry_p,
        ensure_progress=reduction_cfg.ensure_progress,
        root=reduction_cfg.root,
        contract_root=reduction_cfg.contract_root,
    )


def _annotate_step_indices(sequence: list[ReducedGraphData]) -> None:
    """Attach monotonically increasing step_idx attributes to ReducedGraphData."""
    for step_idx, data in enumerate(sequence):
        tensor_val = th.tensor(int(step_idx), dtype=th.long)
        data.step_idx = tensor_val
        data.sequence_id = tensor_val


def _build_reduction_sequence(
    adj,
    pos,
    *,
    factory: gg.reduction.ReductionFactory,
    seed: int = 0,
) -> list[ReducedGraphData]:
    dataset = _SingleGraphReductionDataset(adjs=[adj], poses=[pos], red_factory=factory)
    rng = np.random.default_rng(seed)
    reducer = factory(adj.copy(), rng=rng)
    sequence = dataset.get_random_reduction_sequence(reducer, pos.copy(), rng)
    if not sequence:
        raise RuntimeError("Reduction sequence is empty; check that the graph contains at least one node.")
    _annotate_step_indices(sequence)
    return sequence


def prepare_sequence_setup(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    graph_path: str | Path,
    overrides: Sequence[str] | None = None,
    ema_beta: float | None = None,
    device: str = "cpu",
    reduction_seed: int = 0,
    method_cls: type | None = None,
) -> SequenceSetupResult:
    """Load config/model and build the reduction sequence for a specific graph."""
    cfg = load_hydra_config(config_path, overrides or [])
    if method_cls is not None:
        cfg.method.name = getattr(cfg.method, "name", None) or "expansion"

    context = load_sampling_items(
        cfg=cfg,
        checkpoint=checkpoint_path,
        ema_beta=ema_beta,
        device=device,
        method_cls=method_cls,
    )

    graph = load_graph_from_path(graph_path)
    adjacency, positions, node_order = nx_graph_to_adj_pos(graph)

    factory = _build_reduction_factory(cfg)
    sequence = _build_reduction_sequence(adjacency, positions, factory=factory, seed=reduction_seed)

    bundle = ReductionSequenceBundle(
        graph=graph,
        graph_path=Path(graph_path),
        adjacency=adjacency,
        positions=positions,
        node_order=node_order,
        steps=sequence,
        reduction_seed=reduction_seed,
    )
    return SequenceSetupResult(cfg=cfg, context=context, reduction_bundle=bundle)
class _SingleGraphReductionDataset(RandRedDataset):
    """Minimal concrete subclass to reuse RandRedDataset helpers without streaming."""

    def __iter__(self):  # pragma: no cover - not used for iteration
        raise NotImplementedError("Iteration is not supported for the single-graph helper.")
