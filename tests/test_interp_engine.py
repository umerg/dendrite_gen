"""Checkpointing and parallel execution of analysis units."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from interp.analyses import common as C


def _unit(a, b):
    return [dict(site=f"s{a}", score=float(a * b), tag="t")]


def test_checkpoint_resume_and_signature(tmp_path):
    units = [(f"k{i}", _unit, dict(a=i, b=2)) for i in range(5)]
    ck = C.Checkpoint(tmp_path, "x", dict(a=1))
    full = C.run_units(units, ck, n_jobs=1)
    assert [r["score"] for r in full] == [0, 2, 4, 6, 8]
    # simulate a kill: drop the last two units and leave a truncated line
    lines = ck.path.read_text().splitlines()
    ck.path.write_text("\n".join(lines[:-2]) + '\n{"key": "k4", "re')
    ck2 = C.Checkpoint(tmp_path, "x", dict(a=1))
    assert set(ck2.done) == {"k0", "k1", "k2"}
    assert C.run_units(units, ck2, n_jobs=1) == full
    # changed settings: old checkpoint moved aside, start fresh
    ck3 = C.Checkpoint(tmp_path, "x", dict(a=2))
    assert ck3.done == {} and (tmp_path / "x.partial.stale0").exists()


def test_parallel_equals_serial(tmp_path):
    units = [(f"k{i}", _unit, dict(a=i, b=3)) for i in range(7)]
    serial = C.run_units(units, C.Checkpoint(tmp_path / "s", "x", {}), n_jobs=1)
    par = C.run_units(units, C.Checkpoint(tmp_path / "p", "x", {}), n_jobs=3)
    assert par == serial            # unit order is preserved regardless of completion order


def test_lazy_not_computed_when_all_done(tmp_path):
    calls = []

    def expensive():
        calls.append(1)
        return 1
    lz = C.Lazy(expensive)
    units = [("k0", _unit, dict(a=lz, b=1))]
    ck = C.Checkpoint(tmp_path, "x", {})
    C.run_units(units, ck, n_jobs=1)
    assert calls == [1]
    lz2 = C.Lazy(expensive)
    C.run_units([("k0", _unit, dict(a=lz2, b=1))], C.Checkpoint(tmp_path, "x", {}), n_jobs=1)
    assert calls == [1]             # resumed: the lazy value was never needed


DATA_DIR = Path(os.environ.get("INTERP_DATA_DIR", "/Users/umer/Documents/neurons_conditional_full"))


@pytest.mark.skipif(not (DATA_DIR / "val").is_dir(), reason="neuron data not available")
def test_p5_parallel_and_resume_match_serial(tmp_path, monkeypatch):
    from interp.analyses import p5_stop
    from interp.extract import main

    store = tmp_path / "store"
    main(["--config", "parity_neurons_uncond", "--which", "random:0", "--splits", "val", "--max-neurons", "30",
          "--t", "0", "--data-dir", str(DATA_DIR), "--out", str(store), "--device", "cpu", "--dtype", "float32"])
    sites = ["G1", "G11", "pool_L11"]
    monkeypatch.setenv("INTERP_JOBS", "1")
    a = p5_stop.run(store, 0.0, "val", tmp_path / "serial", B=20, sites=sites)
    monkeypatch.setenv("INTERP_JOBS", "3")
    b = p5_stop.run(store, 0.0, "val", tmp_path / "par", B=20, sites=sites)
    pd.testing.assert_frame_equal(a, b)
    # resume after losing half the units gives the same table
    part = tmp_path / "par" / "p5_val_t0.partial.jsonl"
    lines = part.read_text().splitlines()
    part.write_text("\n".join(lines[: 1 + (len(lines) - 1) // 2]) + "\n")
    c = p5_stop.run(store, 0.0, "val", tmp_path / "par", B=20, sites=sites)
    pd.testing.assert_frame_equal(a, c)


def test_fmt_tolerates_partial_records():
    for rec in (dict(site="L1", score=0.5), dict(site="L1", score=0.5, delta=0.1),
                dict(part="a", target="t", site="L1", bin_lo=2, bin_hi=99, score=float("nan"), delta=0.1),
                dict(site="x", score=0.5, lo=0.4, hi=0.6, delta=0.1, delta_lo=0.0, delta_hi=0.2, score_obs=0.3)):
        assert isinstance(C.fmt(rec), str)
