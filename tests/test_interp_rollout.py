"""Generation-time intervention plumbing (methodology §11b): rollout driver, lineage keys,
masks, ablation options, metrics, classifier. Random-init model; small local data."""
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch as th

DATA_DIR = Path(os.environ.get("INTERP_DATA_DIR", "/Users/umer/Documents/neurons_conditional_full"))
needs_data = pytest.mark.skipif(not (DATA_DIR / "val").is_dir(), reason="neuron data not available")


# ----------------------------------------------------------------------------- pure units

def test_lineage_keys_depend_only_on_lineage():
    from interp.rollout import RolloutState

    # Two graphs (roots 0, 1). Run A: root0 → 2 children, root1 → 3 children; then node 2 → 2.
    a = RolloutState.new(0, [10, 11])
    a.grow(np.array([-1, -1, 0, 0, 1, 1, 1]))
    a.grow(np.array([-1, -1, 0, 0, 1, 1, 1, 2, 2]))
    # Run B: same trees, but graph 1's children come first and node 2 does not branch while an
    # extra pair hangs off node 4 — different global indices for the shared lineages.
    b = RolloutState.new(0, [10, 11])
    b.grow(np.array([-1, -1, 1, 1, 1, 0, 0]))
    b.grow(np.array([-1, -1, 1, 1, 1, 0, 0, 2, 2, 5, 5]))
    # graph-0 children: A nodes 2,3 == B nodes 5,6; node-2 children: A 7,8 == B 9,10
    assert (a.keys[[2, 3]] == b.keys[[5, 6]]).all() and (a.keys[[7, 8]] == b.keys[[9, 10]]).all()
    assert (a.keys[[4, 5, 6]] == b.keys[[2, 3, 4]]).all()
    assert len(set(a.keys.tolist())) == len(a.keys)
    assert a.owner.tolist() == [-1, -1, 0, 1, 0, 1, 2, 0, 0]
    assert a.depth.tolist() == [0, 0, 1, 1, 1, 1, 1, 2, 2]
    # different seed → different keys
    assert (RolloutState.new(1, [10, 11]).keys != a.keys[:2]).all()


def test_keyed_normals_are_standard_and_deterministic():
    from interp.rollout import normals, splitmix

    keys = splitmix(np.arange(20000, dtype=np.uint64))
    z = normals(keys, 4)
    assert np.allclose(z, normals(keys, 4))
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1) < 0.02
    assert abs(np.corrcoef(z[:, 0], z[:, 1])[0, 1]) < 0.03


def test_graph_layernorm_removes_only_the_constant_component():
    from torch_geometric.nn import LayerNorm

    th.manual_seed(0)
    ln = LayerNorm(16)          # default mode='graph', as in egnn_so2
    x = th.randn(30, 16)
    batch = th.tensor([0] * 12 + [1] * 18)
    ref = ln(x, batch)
    shift = th.zeros_like(x)
    shift[:12] = 3.7            # a scalar added to every node and channel of graph 0
    assert th.allclose(ln(x + shift, batch), ref, atol=1e-5)
    v = th.randn(16)            # a direction added to every node of graph 0
    out = ln(x + th.where((batch == 0).unsqueeze(-1), v, th.zeros(16)), batch)
    assert not th.allclose(out[:12], ref[:12], atol=1e-3)
    assert th.allclose(out[12:], ref[12:], atol=1e-6)      # other graphs untouched
    v0 = v - v.mean()           # the mean of v drops out
    out0 = ln(x + th.where((batch == 0).unsqueeze(-1), v0, th.zeros(16)), batch)
    assert th.allclose(out, out0, atol=1e-5)


def _tree(edges, pos):
    G = nx.Graph()
    for n, p in pos.items():
        G.add_node(n, pos=np.asarray(p, float))
    G.add_edges_from(edges)
    G.graph["root"] = 0
    return G


def test_tall_arm_stats_hand_tree():
    from interp.metrics import arm_table, size_stats, tall_arm_stats

    u = (0, 1, 0)
    # soma 0; arm A (node 1): reaches -10 along y; arm B (node 2): +7; arm C (node 3): 2
    G = _tree([(0, 1), (1, 4), (0, 2), (2, 5), (0, 3)],
              {0: (0, 0, 0), 1: (0, -5, 0), 4: (0, -10, 1), 2: (1, 3, 0), 5: (1, 7, 0), 3: (2, 2, 0)})
    a = arm_table(G, u)
    assert a.reach.tolist() == [10, 7, 2] and a.slot.tolist() == [0, 1, 2]
    s = tall_arm_stats(G, u)
    assert s["n_tall"] == 2 and s["ge2_tall"] and not s["one_tall"]       # 7 ≥ 0.6·10
    assert s["dominance"] == pytest.approx(10 / 7) and s["tallest_is_slot0"] and s["tallest_neg"]
    assert s["apical_reach"] == 10 and s["basal_reach_max"] == 7 and s["basal_nodes"] == 3
    G.nodes[5]["pos"] = np.array([1, 5, 0])
    assert tall_arm_stats(G, u)["one_tall"]                                  # 5 < 6
    # recorded owner overrides id order
    for n, o in {1: 2, 2: 0, 3: 1}.items():
        G.nodes[n]["owner"] = o
    assert tall_arm_stats(G, u)["tallest_slot"] == 2
    z = size_stats(G, u)
    assert z["n_nodes"] == 6 and z["n_tips"] == 3 and z["max_depth"] == 2


def test_class_clf_separates_toy_classes():
    from interp.class_clf import MorphoClassifier

    rng = np.random.default_rng(0)
    feats, y = [], []
    for c in (0, 1):
        for _ in range(60):
            feats.append(dict(axial_extent=100 + 80 * c + rng.normal(0, 10), total_extent=200.0,
                              node_count=50 + rng.normal(0, 5), strahler=3.0))
            y.append(c)
    clf = MorphoClassifier((0, 1, 0)).fit_features(feats, y)
    acc = (clf.classes_[clf.proba_features(feats).argmax(1)] == np.array(y)).mean()
    assert acc > 0.95
    sf = MorphoClassifier((0, 1, 0), size_free=True).fit_features(feats, y)
    assert "node_count" not in sf.keys and "axial_extent" in sf.keys


# ----------------------------------------------------------------------------- model + sampler

@pytest.fixture(scope="module")
def mm():
    from interp.load import build_method_and_model, compose_cfg, load_split

    cfg = compose_cfg("parity_neurons_uncond")
    graphs = sorted(load_split(cfg, "val", max_n=30, data_dir=DATA_DIR), key=len)[:4]
    method, model, _ = build_method_and_model(cfg, which="random", seed=0, pool=graphs[:2])
    method.max_tree_size = 40                     # keep CPU rollouts short
    return cfg, method, model, graphs


def _same(A, B, atol=1e-5):
    if len(A) != len(B):
        return False
    for g, h in zip(A, B):
        if g.number_of_nodes() != h.number_of_nodes() or set(g.edges) != set(h.edges):
            return False
        if not all(np.allclose(g.nodes[n]["pos"], h.nodes[n]["pos"], atol=atol) for n in g.nodes):
            return False
    return True


@needs_data
def test_unkeyed_generate_matches_sample_graphs(mm):
    from interp.rollout import generate, gt_inputs

    cfg, method, model, graphs = mm
    G = generate(method, model, graphs, cfg, seed=3, keyed=False, batch_items=64)
    inp = gt_inputs(graphs, cfg, model)
    order = np.argsort(inp["size"], kind="stable")
    th.manual_seed(3 * 100_003)
    ref = method.sample_graphs(target_size=th.as_tensor(inp["size"][order]), model=model,
                               num_root_children=th.as_tensor(inp["nrc"][order]))
    psf = float(cfg.dataset.pos_scale_factor)
    for g in ref:
        for n in g.nodes:
            g.nodes[n]["pos"] = g.nodes[n]["pos"] * psf
    assert _same([G[i] for i in order], ref)


@needs_data
def test_keyed_generate_is_deterministic_and_packing_independent(mm):
    from interp.rollout import generate

    cfg, method, model, graphs = mm
    a = generate(method, model, graphs, cfg, seed=0, batch_items=64)
    b = generate(method, model, graphs, cfg, seed=0, batch_items=64)
    assert _same(a, b, atol=0)
    c = generate(method, model, graphs, cfg, seed=0, batch_items=1)     # each neuron alone
    assert _same(a, c, atol=1e-3)
    d = generate(method, model, graphs, cfg, seed=1, batch_items=64)
    assert not _same(a, d)


@needs_data
def test_owner_matches_ancestry_and_slots(mm):
    from interp.rollout import generate

    cfg, method, model, graphs = mm
    for G, gt in zip(generate(method, model, graphs, cfg, seed=0), graphs):
        kids = sorted(G.neighbors(0))
        assert kids == list(range(1, len(kids) + 1))          # soma children = ids 1..k in spawn order
        assert len(kids) == gt.degree[gt.graph["root"]] or G.graph["cap_hit"]
        H = G.copy()
        H.remove_node(0)
        for slot, c in enumerate(kids):
            assert all(G.nodes[n]["owner"] == slot for n in nx.node_connected_component(H, c))
        assert G.nodes[0]["owner"] == -1


@needs_data
def test_zero_edit_and_empty_mask_ablation_are_identity(mm):
    from interp import hooks as H
    from interp.rollout import AddDirection, generate, make_ablation, make_editor

    cfg, method, model, graphs = mm
    ref = generate(method, model, graphs, cfg, seed=0)
    v = np.ones(256, np.float32)
    ed = make_editor(model, "L5", lambda st: AddDirection(st, v, 0.0, "all"))
    assert _same(generate(method, model, graphs, cfg, seed=0, interventions=[ed]), ref, atol=0)
    # an ablation restricted to no node at all is the identity
    abl = lambda st: H.AttnAblation(model, mode="zero", node_mask=lambda: th.zeros_like(st.frontier_t))
    assert _same(generate(method, model, graphs, cfg, seed=0, interventions=[abl]), ref, atol=1e-6)
    # a large push on basal nodes changes the rollout
    big = make_editor(model, "L5", lambda st: AddDirection(st, v, 50.0, "basal"))
    assert not _same(generate(method, model, graphs, cfg, seed=0, interventions=[big]), ref)
    # per-t record → per-t mean ablation runs
    rec = H.AttnAblation(model, mode="record")
    from interp.rollout import _rec_ctx
    generate(method, model, graphs[:2], cfg, seed=0, interventions=[lambda st: _rec_ctx(rec, st)])
    means = rec.recorded_means()
    assert {k[1] for k in means} == set(range(10)) and all(isinstance(k, tuple) for k in means)
    out = generate(method, model, graphs[:2], cfg, seed=0,
                   interventions=[make_ablation(model, "mean", None, means, target="basal")])
    assert len(out) == 2


@needs_data
def test_rollout_cli_and_analysis(mm, tmp_path):
    import pickle

    from interp.analyses import s_rollouts
    from interp.rollout import main

    cfg, method, model, graphs = mm
    # CLI on a 3-neuron subset of val with a random model; E0 conditions
    main(["--config", "parity_neurons_uncond", "--which", "random", "--exp", "e0", "--split", "val",
          "--data-dir", str(DATA_DIR), "--max-neurons", "3", "--seeds", "0", "--out", str(tmp_path),
          "--device", "cpu", "--max-tree-size", "40"])
    d = tmp_path / "e0"
    names = {p.name for p in d.glob("*.pkl")}
    assert {"intact_seed0.pkl", "abl_mean_seed0.pkl", "abl_graphmean_seed0.pkl", "abl_mean_b13_seed0.pkl"} <= names
    with open(d / "intact_seed0.pkl", "rb") as f:
        r = pickle.load(f)
    assert len(r["pred_graphs"]) == 3 and r["meta"]["keyed"]
    s_rollouts.main(["--exp-dir", str(d), "--config", "parity_neurons_uncond", "--split", "val",
                     "--data-dir", str(DATA_DIR), "--B", "50", "--no-dist"])
    import pandas as pd
    eff = pd.read_csv(d / "analysis" / "effects.csv")
    assert {"abl_mean", "abl_zero"} <= set(eff.condition) and {"ge2_tall", "n_nodes"} <= set(eff.outcome)
    assert (d / "analysis" / "report.txt").exists()


@needs_data
def test_class_and_reach_conditions_from_dirs(mm, tmp_path):
    from interp.analyses import s_rollouts
    from interp.rollout import main

    rng = np.random.default_rng(0)
    d = {"L5/sigma": np.array(1.0), "A5/sigma": np.array(1.0)}
    for k in ("v", "random", "onmanifold"):
        d[f"L5/class2/{k}"] = rng.standard_normal(256).astype(np.float32)
        d[f"A5/reach/{k}"] = rng.standard_normal(256).astype(np.float32)
    np.savez(tmp_path / "dirs.npz", **d)
    common = ["--config", "parity_neurons_uncond", "--which", "random", "--split", "val", "--data-dir", str(DATA_DIR),
              "--max-neurons", "2", "--seeds", "0", "--out", str(tmp_path), "--device", "cpu", "--max-tree-size", "30",
              "--dirs", str(tmp_path / "dirs.npz")]
    main(common + ["--exp", "class", "--sites", "L5", "--alphas", "1,-1", "--kinds", "v,random"])
    names = {p.stem for p in (tmp_path / "class").glob("*.pkl")}
    assert {"intact_seed0", "L5_class2_v_a1_all_seed0", "L5_class2_v_a-1_all_seed0", "L5_class2_random_a1_all_seed0"} <= names
    main(common + ["--exp", "reach", "--sites", "A5", "--alphas", "2", "--kinds", "v", "--targets", "basal,apical"])
    names = {p.stem for p in (tmp_path / "reach").glob("*.pkl")}
    assert {"abl_mean_basal_seed0", "A5_reach_v_a2_basal_seed0", "A5_reach_v_a2_apical_seed0"} <= names
    s_rollouts.main(["--exp-dir", str(tmp_path / "reach"), "--config", "parity_neurons_uncond", "--split", "val",
                     "--data-dir", str(DATA_DIR), "--B", "20"])
    assert (tmp_path / "reach" / "analysis" / "dist.csv").exists()
