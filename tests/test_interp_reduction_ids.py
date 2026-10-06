"""Node-id mapping and probe targets for interp (needs the local neurons_conditional_full copy)."""
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch as th
from torch_geometric.data import Batch

DATA_DIR = Path(os.environ.get("INTERP_DATA_DIR", "/Users/umer/Documents/neurons_conditional_full"))
pytestmark = pytest.mark.skipif(not (DATA_DIR / "val").is_dir(), reason="neuron data not available")


@pytest.fixture(scope="module")
def built():
    from interp.load import compose_cfg, load_split
    from interp.reduction_ids import build_items

    cfg = compose_cfg("parity_neurons_uncond", [f"dataset.data_dir={DATA_DIR}"])
    graphs = load_split(cfg, "val", max_n=50)
    items, per = build_items(graphs, cfg, with_tmd=False)
    return cfg, graphs, items, per


def _by_neuron(items):
    out = {}
    for it in items:
        out.setdefault(int(it.neuron), []).append(it)
    return out


def test_fine_idx_matches_closed_form_and_positions(built):
    from interp.reduction_ids import closed_form_fine_idx

    _, _, items, per = built
    for i, seq in _by_neuron(items).items():
        info = per[i]
        D = info["n_levels"]
        assert D == int(info["depth"].max()), "depth reduction yields one level per hop depth"
        for it in seq:
            lvl = int(it.level)
            fine = it.fine_idx.numpy()
            np.testing.assert_array_equal(fine, closed_form_fine_idx(info["depth"], D, lvl))
            np.testing.assert_array_equal(it.pos.numpy(), info["pos"][fine].astype(np.float32))


def test_frontier_labels_and_apical(built):
    from interp.targets import final_node_targets

    cfg, _, items, per = built
    uhat = np.asarray(cfg.model.so2_axis, dtype=float)
    for i, seq in _by_neuron(items).items():
        info = per[i]
        final = final_node_targets(info["parent"], info["depth"], info["pos"], info["is_apical"], uhat)
        D = info["n_levels"]
        for it in seq:
            lvl, fine = int(it.level), it.fine_idx.numpy()
            frontier = fine[it.new_leaf_idx_from_next.numpy()]
            assert set(frontier) == set(np.nonzero(info["depth"] == D - lvl)[0])
            leaves = fine[it.leaf_idx.numpy()]
            lab2 = it.leaf_expansion.numpy() == 2
            np.testing.assert_array_equal(lab2, final.height_final.to_numpy()[leaves] >= 1)
            ap = fine[it.is_apical_root_child.numpy()]
            assert len(ap) == 1 and final.is_apical_subtree[ap[0]] and info["parent"][ap[0]] == 0


def test_batching_keeps_ids_unoffset(built):
    _, _, items, _ = built
    b = Batch.from_data_list(items[:5])
    np.testing.assert_array_equal(b.fine_idx.numpy(), np.concatenate([it.fine_idx.numpy() for it in items[:5]]))
    assert b.neuron.shape == (5,) and b.level.shape == (5,)


def test_targets_match_structural_metrics(built):
    from interp.targets import final_node_targets, level_targets
    from validation.structural_metrics import _postorder_subtree_stats

    cfg, graphs, _, per = built
    uhat = np.asarray(cfg.model.so2_axis, dtype=float)
    for G, info in list(zip(graphs, per))[:10]:
        final = final_node_targets(info["parent"], info["depth"], info["pos"], info["is_apical"], uhat)
        _, leaves, strahler = _postorder_subtree_stats(G, G.graph["root"])
        order = info["node_order"]
        np.testing.assert_array_equal(final.strahler_final, [strahler[n] for n in order])
        np.testing.assert_array_equal(final.leaves_final, [leaves[n] for n in order])
        sizes = [len(nx.descendants(nx.bfs_tree(G, G.graph["root"]), n)) for n in order]
        np.testing.assert_array_equal(final.desc_final, sizes)
        # visible + hidden partition the future at every level
        for lvl in range(info["n_levels"]):
            lt = level_targets(final, info["n_levels"], lvl)
            assert (lt.hidden_desc >= 0).all() and (lt.hidden_height >= 0).all()
            assert not lt[~lt.is_frontier & (lt.depth < lt.visible_depth)].hidden_desc.isna().any()
