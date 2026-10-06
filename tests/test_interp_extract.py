"""End-to-end extraction on a few val neurons with a random-init model."""
import os
from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(os.environ.get("INTERP_DATA_DIR", "/Users/umer/Documents/neurons_conditional_full"))
pytestmark = pytest.mark.skipif(not (DATA_DIR / "val").is_dir(), reason="neuron data not available")


@pytest.fixture(scope="module", params=["parity_neurons_uncond", "parity_neurons_tmd"])
def extracted(request, tmp_path_factory):
    from interp.extract import main

    out = tmp_path_factory.mktemp(request.param)
    args = ["--config", request.param, "--which", "random:0", "--splits", "val", "--max-neurons", "6",
            "--t", "0,0.9", "--batch-items", "7", "--data-dir", str(DATA_DIR), "--out", str(out),
            "--device", "cpu", "--dtype", "float32"]
    main(args)
    out2 = tmp_path_factory.mktemp(request.param + "_rerun")
    rerun = args[:-6] + ["--out", str(out2), "--device", "cpu", "--dtype", "float32"]
    rerun[rerun.index("--batch-items") + 1] = "13"   # different batch packing
    main(rerun)
    return out, out2


def test_store_shapes_and_index(extracted):
    from interp.store import Store

    out, _ = extracted
    for t in (0.0, 0.9):
        st = Store.open(out, t, "val")
        idx = st.index
        assert (idx.t == t).all()
        dense = idx[idx.row_dense >= 0]
        assert len(dense) == st.meta["n_dense"] and len(idx) == st.meta["n_sparse"]
        assert set(dense.tier) <= {"A", "B1"} and (idx.tier[idx.row_dense < 0] == "B2").all()
        # frontier rows: one per non-root node of each neuron
        tf = st.targets_final()
        assert idx.is_frontier.sum() == int((tf.depth > 0).sum())
        # root children carry an ordinal slot; slot 0 is the apical
        rc = idx[idx.is_root_child].merge(tf[["neuron", "node", "is_apical_subtree"]], on=["neuron", "node"])
        assert (rc.slot >= 0).all()
        assert ((rc.slot == 0) == rc.is_apical_subtree).all()
        for s in st.meta["sites_dense"]:
            assert st.acts(s, dense).shape == (len(dense), 256)
        for s in st.meta["sites_sparse"]:
            assert st.acts(s).shape == (len(idx), 256)
        assert st.graph_acts("G11").shape == (st.meta["n_items"], 8, 256)
        assert np.isfinite(st.acts("L11", dense)).all()


def test_frontier_flow_inputs(extracted):
    from interp.store import Store

    out, _ = extracted
    st0, st9 = Store.open(out, 0.0, "val"), Store.open(out, 0.9, "val")
    f9 = st9.index[st9.index.is_frontier]
    assert f9.pos_err.notna().all()
    # at t=0.9 the flowed expansion input is close to the GT label (±1)
    tf = st9.targets_final()
    lab = f9.merge(tf[["neuron", "node", "height_final"]], on=["neuron", "node"])
    agree = np.sign(lab.e_t) == np.where(lab.height_final >= 1, 1, -1)
    assert agree.mean() > 0.9
    # e_t / e1_implied are only defined on the frontier (training leaves)
    assert st0.index[~st0.index.is_frontier].e_t.isna().all() and st0.index[st0.index.is_frontier].e_t.notna().all()


def test_extraction_is_independent_of_batch_packing(extracted):
    """Flow noise is keyed per (seed, t, neuron, level, node), so re-packing batches must not
    change any input. Activations agree up to float rounding of padded batched attention; a
    random-init net amplifies rounding ~10x per block, so only early sites are compared tightly."""
    from interp.store import Store

    a, b = extracted
    for t in (0.0, 0.9):
        sa, sb = Store.open(a, t, "val"), Store.open(b, t, "val")
        assert sa.index[["neuron", "level", "node"]].equals(sb.index[["neuron", "level", "node"]])
        np.testing.assert_array_equal(sa.index.e_t, sb.index.e_t)
        np.testing.assert_array_equal(sa.acts("in"), sb.acts("in"))
        for s, tol in (("L0", 1e-5), ("A1", 1e-5), ("L3", 1e-3), ("G1", 1e-4)):
            x, y = (sa.graph_acts(s), sb.graph_acts(s)) if s.startswith("G") else (sa.acts(s), sb.acts(s))
            np.testing.assert_allclose(x, y, rtol=tol, atol=tol)


def test_overflow_guard(extracted):
    """A site with recorded fp16 overflow must be refused by the store, never returned as inf."""
    from interp.store import Store

    out, _ = extracted
    st = Store.open(out, 0.0, "val")
    assert st.meta["dtype"] == "float32" and st.meta["overflow"] == {}
    st.meta["overflow"] = {"L11": 3}
    with pytest.raises(ValueError, match="overflow"):
        st.acts("L11")
    assert np.isfinite(st.acts("L0")).all()
