"""Hook plumbing on a random-init parity model: capture is side-effect free, sites are right."""
import os
from pathlib import Path

import pytest
import torch as th
from torch_geometric.data import Batch

DATA_DIR = Path(os.environ.get("INTERP_DATA_DIR", "/Users/umer/Documents/neurons_conditional_full"))
pytestmark = pytest.mark.skipif(not (DATA_DIR / "val").is_dir(), reason="neuron data not available")


@pytest.fixture(scope="module", params=["parity_neurons_uncond", "parity_neurons_tmd"])
def setup(request):
    from interp.load import build_method_and_model, compose_cfg, load_split
    from interp.reduction_ids import build_items

    cfg = compose_cfg(request.param, [f"dataset.data_dir={DATA_DIR}"])
    graphs = load_split(cfg, "val", max_n=4)
    method, model, _ = build_method_and_model(cfg, which="random", pool=graphs[:2])
    items, _ = build_items(graphs, cfg)
    return method, model, Batch.from_data_list(items[:12])


def _run(method, model, batch, t0=0.3, seed=0):
    from interp.hooks import fixed_flow_time

    with fixed_flow_time(method.diffusion, t0):
        th.manual_seed(seed)
        return method.teacher_forced_diagnostics(batch, model)


def test_recorder_is_side_effect_free_and_sites_consistent(setup):
    from interp.hooks import ActivationRecorder, InputCapture, all_sites, global_block_ids

    method, model, batch = setup
    ref = _run(method, model, batch)
    outs = []
    h = model.register_forward_hook(lambda m, a, o: outs.append(o["node_state"]))
    with ActivationRecorder(model, all_sites(model), dtype=th.float32) as rec, InputCapture(model) as cap:
        got = _run(method, model, batch)
    h.remove()
    assert th.equal(ref["cp"], got["cp"])
    acts = rec.pop()
    last = len(model.mpnn_layers) - 1
    assert th.equal(acts[f"L{last}"][0], outs[0][:, 3:])
    N, B = batch.num_nodes, batch.num_graphs
    for s, v in acts.items():
        want = (B * model.global_tokens.shape[0], model.feats_dim) if s.startswith("G") else (N, model.feats_dim)
        assert v[0].shape == want, s
    assert global_block_ids(model) == [1, 3, 5, 7, 9, 11]
    # flow time is the last raw input channel and is fixed
    assert th.allclose(cap.last["x"][:, -1], th.full((N,), 0.3))
    assert th.allclose(got["t"], th.full_like(got["t"], 0.3))


def test_attn_ablation_changes_output_and_restores(setup):
    from interp.hooks import AttnAblation

    method, model, batch = setup
    ref = _run(method, model, batch)
    for mode in ("zero", "graphmean"):
        with AttnAblation(model, mode=mode):
            abl = _run(method, model, batch)
        assert not th.equal(ref["cp"], abl["cp"]), mode
    assert th.equal(ref["cp"], _run(method, model, batch)["cp"])


def test_site_editor_noop_and_effect(setup):
    from interp.hooks import SiteEditor

    method, model, batch = setup
    ref = _run(method, model, batch)
    with SiteEditor(model, "L5", lambda f: None):
        assert th.equal(ref["cp"], _run(method, model, batch)["cp"])
    with SiteEditor(model, "L5", lambda f: f + 0.0):
        assert th.equal(ref["cp"], _run(method, model, batch)["cp"])
    with SiteEditor(model, "L5", lambda f: f + 1.0):
        assert not th.equal(ref["cp"], _run(method, model, batch)["cp"])
