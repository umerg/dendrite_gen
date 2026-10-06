"""Per-neuron reduction sequences that remember which original node each row is.

``PrecomputedRedDataset`` drops the coarse→fine mapping and the neuron id. This module
rebuilds the identical sequences (same factory, same apical flag, same item fields) and
additionally attaches to every item:

  * ``fine_idx``  [n_ℓ] long — index of each row in the level-0 (original) node order
  * ``neuron``    scalar     — position of the neuron in the input list
  * ``level``     scalar     — list index in the sequence (0 = finest)
  * ``n_levels``  scalar     — sequence length (== max hop depth D for depth reduction)

Under ``type=depth, mode=deterministic, contract_root=False`` level ℓ contains exactly
the original nodes with hop depth ≤ D-ℓ (``closed_form_fine_idx``); the composed
``survivor_mask`` is the ground truth and the closed form is checked against it in tests.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import torch as th
from omegaconf import DictConfig

import graph_generation as gg
from graph_generation.data.reduction_dataset import RandRedDataset
from utils.data_loading import nx_graph_to_adj_pos

from .load import tmd_vector, uhat_of


def hop_depth(parent: np.ndarray) -> np.ndarray:
    """Hop depth from a parent array (root has parent -1). Parents need not precede children."""
    n = parent.shape[0]
    depth = np.full(n, -1, dtype=np.int64)
    depth[parent < 0] = 0
    for _ in range(n):
        todo = depth < 0
        if not todo.any():
            break
        ready = todo & (depth[np.maximum(parent, 0)] >= 0)
        depth[ready] = depth[parent[ready]] + 1
    return depth


def closed_form_fine_idx(depth0: np.ndarray, D: int, level: int) -> np.ndarray:
    return np.nonzero(depth0 <= D - level)[0]


class IdTrackingRedDataset(RandRedDataset):
    """Not iterable — use ``sequence_with_ids`` / ``build_items``."""

    def __iter__(self):
        raise TypeError("IdTrackingRedDataset is not iterable; use build_items().")

    def sequence_with_ids(self, graph, pos, neuron: int, tmd=None, cell_class=None):
        # Line-for-line mirror of RandRedDataset.get_random_reduction_sequence, plus
        # composition of survivor_mask into fine_idx.
        data = []
        pos = pos.astype(np.float32)
        is_apical = self._compute_apical_flag(graph, pos)
        fine_idx = np.arange(graph.n, dtype=np.int64)
        while True:
            reduced_graph = graph.get_reduced_graph()

            forced_new = None
            if not reduced_graph.did_contract:
                root = graph._state.root
                children = graph._state.children.get(root, []) if root is not None else []
                if children:
                    forced_new = np.array(children, dtype=np.int64)

            rgd = self._build_reduced_graph_data(
                graph, pos, forced_new_leaf_idx=forced_new, tmd=tmd, cell_class=cell_class,
                is_apical=is_apical,
            )
            rgd.fine_idx = th.from_numpy(fine_idx.copy())
            rgd.neuron = th.tensor(int(neuron), dtype=th.long)
            rgd.level = th.tensor(len(data), dtype=th.long)
            data.append(rgd)

            if not reduced_graph.did_contract:
                break

            pos = pos[reduced_graph.survivor_mask]
            if is_apical is not None:
                is_apical = is_apical[reduced_graph.survivor_mask]
            fine_idx = fine_idx[reduced_graph.survivor_mask]
            graph = reduced_graph

        for rgd in data:
            rgd.n_levels = th.tensor(len(data), dtype=th.long)
        return data


def make_factory(cfg: DictConfig):
    if getattr(cfg.reduction, "type", "cherry") != "depth":
        raise ValueError("interp assumes reduction.type=depth (all parity configs).")
    # Root children are identified downstream as "parent is local node 0" and the closed-form
    # level structure assumes the root is never contracted.
    if cfg.reduction.root != 0 or bool(cfg.reduction.contract_root):
        raise ValueError("interp assumes reduction.root=0 and contract_root=False (all parity configs).")
    kwargs = dict(
        mode=cfg.reduction.mode,
        cherry_p=cfg.reduction.cherry_p,
        ensure_progress=cfg.reduction.ensure_progress,
        root=cfg.reduction.root,
        contract_root=cfg.reduction.contract_root,
    )
    if hasattr(cfg.reduction, "weighted_reduction"):
        kwargs["weighted_reduction"] = cfg.reduction.weighted_reduction
    return gg.depth_reduction.DepthReductionFactory(**kwargs)


def build_items(graphs: list[nx.Graph], cfg: DictConfig, with_tmd: bool | None = None,
                with_class: bool | None = None):
    """Build every reduction item for every graph, tagged with ids.

    Returns ``(items, per_neuron)`` where ``per_neuron[i]`` holds the level-0 arrays
    (``adj``, ``pos`` scaled, ``parent``, ``depth``, ``node_order``) needed for targets.
    TMD / cell class are attached when the model is conditioned on them (``None`` → auto).
    """
    if with_tmd is None:
        with_tmd = int(getattr(cfg.model, "tmd_hidden_dim", 0)) > 0
    if with_class is None:
        with_class = int(getattr(cfg.model, "class_hidden_dim", 0)) > 0
    psf = getattr(cfg.dataset, "pos_scale_factor", None)
    factory = make_factory(cfg)
    ds = IdTrackingRedDataset(
        adjs=[], poses=[], red_factory=factory, uhat=uhat_of(cfg),
        root_child_order=str(cfg.model.root_child_order),
    )
    rng = np.random.default_rng(0)  # only feeds sibling_order, which the model never reads
    items, per_neuron = [], []
    for i, G in enumerate(graphs):
        A, P, node_order = nx_graph_to_adj_pos(G)
        if psf is not None:
            P = P / float(psf)
        reducer = factory(A, rng=rng)
        st = reducer._state
        parent = np.array([(-1 if st.parent[u] is None else st.parent[u]) for u in range(reducer.n)],
                          dtype=np.int64)
        tmd = tmd_vector(G, cfg) if with_tmd else None
        cc = G.graph.get("cell_class") if with_class else None
        seq = ds.sequence_with_ids(reducer, P.copy(), neuron=i, tmd=tmd, cell_class=cc)
        items.extend(seq)
        per_neuron.append(dict(
            neuron=i, neuron_id=G.graph.get("neuron_id"), adj=A, pos=P.astype(np.float64),
            parent=parent, depth=hop_depth(parent), node_order=node_order,
            n_levels=len(seq), cell_class=G.graph.get("cell_class"),
            is_apical=(seq[0].is_apical_root_child.numpy() if hasattr(seq[0], "is_apical_root_child")
                       else None),
        ))
    return items, per_neuron
