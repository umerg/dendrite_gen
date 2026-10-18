"""P1 / P2 end-to-end on a small random-init extraction (pipeline check, not a result)."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA_DIR = Path(os.environ.get("INTERP_DATA_DIR", "/Users/umer/Documents/neurons_conditional_full"))
pytestmark = pytest.mark.skipif(not (DATA_DIR / "val").is_dir(), reason="neuron data not available")


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    from interp.extract import main

    out = tmp_path_factory.mktemp("store")
    main(["--config", "parity_neurons_uncond", "--which", "random:0", "--splits", "val",
          "--max-neurons", "40", "--t", "0", "--data-dir", str(DATA_DIR), "--out", str(out),
          "--device", "cpu", "--dtype", "float32"])
    return out


def test_row_table(store):
    from interp.baselines import KHOP_COLS, OBS_FRONTIER, row_table
    from interp.store import Store

    st = Store.open(store, 0.0, "val")
    rows = row_table(st)
    assert len(rows) == len(st.index)
    assert (rows.row_sparse.to_numpy() == np.arange(len(rows))).all()
    f = rows[rows.is_frontier]
    assert not f[OBS_FRONTIER[:5] + KHOP_COLS].isna().any().any()
    assert (f.depth == f.visible_depth).all()
    assert ((f.hidden_desc == f.desc_final) & (f.vis_desc == 0)).all()
    # the hop-1 neighbourhood of a frontier node is its parent (+ its own children: none visible)
    assert (f.hop1_n == 1).all()
    # parent-indexed path geometry on frontier rows = the parent's own values in the final tree
    from interp.targets import add_path_geometry
    tf = pd.concat([add_path_geometry(g) for _, g in st.targets_final().groupby("neuron")])
    par = f[["neuron", "parent", "parent_pathdist", "parent_radial", "parent_radial_cos"]].merge(
        tf[["neuron", "node", "pathdist_soma", "radial", "in_edge_radial_cos"]],
        left_on=["neuron", "parent"], right_on=["neuron", "node"], validate="many_to_one")
    assert len(par) == len(f)
    np.testing.assert_allclose(par.parent_pathdist, par.pathdist_soma)
    np.testing.assert_allclose(par.parent_radial, par.radial)
    np.testing.assert_allclose(par.parent_radial_cos, par.in_edge_radial_cos)
    assert (f.parent_pathdist >= f.parent_radial - 1e-9).all()      # path ≥ straight line
    assert (rows.vis_strahler >= 1).all()
    # row table is cached
    from interp.baselines import ROWS_VERSION
    assert (st.path / f"rows_v{ROWS_VERSION}.parquet").exists()


def test_p1_p2_run(store, tmp_path):
    from interp.analyses import p1_local, p2_lookahead

    recs, diag = p1_local.run(store, 0.0, "val", tmp_path, B=20)
    fr = recs[(recs.tag == "is_frontier/all") & (recs.site == "in")]
    assert fr.score.iloc[0] > 0.99          # new_leaf_flag is an input channel
    assert set(diag.tag) == {"depth/context", "depth/frontier"}
    for tag in ("ctx/parent_edge/frontier", "path/parent_pathdist/frontier", "dend/own_minus_apical/basal",
                "dend/mine_taller/basal", "morpho/strahler_ge3/internal"):
        g = recs[recs.tag == tag]
        assert len(g) and g.delta.notna().all() and g.score_obs.notna().all(), tag
    assert recs[recs.tag == "morpho/strahler_ge3/internal"].kind.iloc[0] == "binary"
    # tree-level targets are no longer probed on node rows (they live in P5's tree/ targets)
    assert not recs.tag.isin(["morpho/log_n_visible/frontier", "morpho/tallest_reach/frontier"]).any()
    assert "nonconv" in recs.columns
    bins = pd.read_csv(tmp_path / "p1_bins_val_t0.csv")
    assert {"tag", "site", "depth_lo", "depth_hi", "rho", "rho_obs", "rho_full",
            "rho_tree", "rho_dend", "share_tree", "rho_tree_obs", "rho_tree_full"} <= set(bins.columns)
    assert not bins.tag.str.startswith("depth/").any() and "own_edge/frontier" in set(bins.tag)
    assert bins.rho_tree.notna().any() and bins.share_tree.between(0, 1).any()
    from interp.analyses.report import report
    rep = report(tmp_path)
    assert "probe ρ within depth bin" in rep and "within snapshot" in rep

    p2 = p2_lookahead.run(store, 0.0, "val", tmp_path, B=20)
    prim = p2[p2.tag == "primary/y_h2"]
    assert {"delta", "delta_lo", "delta_hi", "score_obs"} <= set(prim.columns)
    assert len(prim) == prim.site.nunique() > 10
    assert {"apical/y_h2", "nonapical/y_h2"} <= set(p2.tag)
    assert p2[p2.tag == "apical/y_h2"].delta.notna().all()
    b2 = pd.read_csv(tmp_path / "p2_bins_val_t0.csv")
    assert {"primary/y_h2", "apical/y_h2", "nonapical/y_h2"} <= set(b2.tag) and "rho_tree" in b2
    d = np.load(tmp_path / "directions_p2_val_t0.npz")
    assert d["L7/v_grow"].shape == (256,)
    v = d["L7/v_orth"]
    assert abs(v @ d["L7/v_label"]) < 1e-6 * np.linalg.norm(v) * np.linalg.norm(d["L7/v_label"]) + 1e-9
    assert (tmp_path / "p1_val_t0.csv").exists() and (tmp_path / "p2_val_t0.csv").exists()


@pytest.fixture(scope="module")
def ablated_store(tmp_path_factory):
    from interp.extract import main

    out = tmp_path_factory.mktemp("ablated")
    main(["--config", "parity_neurons_uncond", "--which", "random:0", "--splits", "val",
          "--max-neurons", "25", "--t", "0", "--data-dir", str(DATA_DIR), "--out", str(out),
          "--device", "cpu", "--dtype", "float32", "--ablate", "zero"])
    return out


def test_ablation_only_changes_downstream_of_first_global_block(store, ablated_store):
    from interp.store import Store

    a, b = Store.open(store, 0.0, "val"), Store.open(ablated_store, 0.0, "val")
    key = ["neuron", "level", "node"]
    m = a.index[a.index.row_dense >= 0].merge(b.index[b.index.row_dense >= 0], on=key, suffixes=("", "_b"))
    ra, rb = m, m.assign(row_dense=m.row_dense_b, row_sparse=m.row_sparse_b)
    assert len(m) > 100 and b.meta["ablate"] == "zero"
    np.testing.assert_array_equal(a.acts("in", ra), b.acts("in", rb))
    np.testing.assert_array_equal(a.acts("L0", ra), b.acts("L0", rb))
    assert not np.allclose(a.acts("A1", ra), b.acts("A1", rb))
    # the per-neuron files are written alongside
    assert b.neuron_tmd().shape == (25, 512) and len(b.neuron_meta()) == 25


def test_p3_p5_p6_run(store, ablated_store, tmp_path):
    from interp.analyses import p3_latent, p5_stop, p6_apical

    p3 = p3_latent.run(store, 0.0, "val", tmp_path, B=20, min_neurons=10, tmd_input=True, n_pcs=2,
                       sites=["G1", "pool_L5"])
    assert {"tmd_pc1"} <= set(p3.target) and "baseline:partial_tree" in set(p3.site)
    assert {"delta", "delta_lo"} <= set(p3.columns)

    p5 = p5_stop.run(store, 0.0, "val", tmp_path, B=20, sites=["G11", "pool_L11"])
    lr = p5[(p5.target == "levels_remaining") & (p5.site == "baseline:observables")]
    assert lr.score.iloc[0] > 0.0      # observables beat predicting the mean
    tree = p5[p5.target.str.startswith("tree/")]
    assert {"tree/log_n_visible", "tree/tallest_reach"} <= set(tree.target)
    assert "baseline:tree_average" in set(tree.site) and tree[tree.site == "G11"].delta.notna().all()

    p6 = p6_apical.run(store, 0.0, "val", tmp_path, ablated={"zero": str(ablated_store)}, B=20)
    assert {"a_own_lineage", "b_cross", "c_pos_err", "c_ablation"} <= set(p6.part)
    modes = set(p6[p6.part == "c_ablation"]["mode"])
    assert modes == {"intact", "fixed_probe", "refit"}
    for f in ("p3_val_t0.csv", "p5_val_t0.csv", "p6_val_t0.csv"):
        assert (tmp_path / f).exists()


def test_within_tree(store, tmp_path):
    from interp.analyses import within_tree

    df = within_tree.run(store, 0.0, "val", tmp_path, max_neurons=30)
    assert {"rho_pooled", "rho_within", "share_between"} <= set(df.columns)
    orc = df[df.site == "oracle:tree_mean"]
    assert len(orc) and (orc.rho_within.dropna().abs() < 1e-9).all()   # tree-only knowledge scores 0 within trees
    assert (tmp_path / "within_tree_val_t0.csv").exists()
    assert "rho_within" in within_tree.table(df, "rho_within")
