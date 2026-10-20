"""Report figures render from CSVs with the analyses' column layout (synthetic numbers)."""
import numpy as np
import pandas as pd

from interp.analyses.figures import make_all

SITES = ["in", "L0", "A1", "L1", "L3", "L5", "L7", "L9", "L11"]
GRAPH = ["pool_in", "pool_L0", "G1", "G5", "G11", "pool_L11"]


def _p2_bins(rng):
    rows = []
    for tag in ("primary/log_desc", "primary/log_pathlen", "nonapical/log_desc"):
        for lo, hi in [(3, 4), (5, 6), (7, 8), (9, 12), (13, 99)]:
            for s in SITES:
                rows.append(dict(tag=tag, site=s, depth_lo=lo, depth_hi=hi, rho=rng.random(),
                                 rho_tree=rng.random(), rho_dend=rng.random(), share_tree=0.4))
    return pd.DataFrame(rows)


def _graph(rng, targets, extra=None):
    rows = []
    for t in targets:
        for s in GRAPH:
            sc = rng.random()
            rows.append(dict(target=t, site=s, score=sc, lo=sc - 0.02, hi=sc + 0.02, **(extra or {})))
    return pd.DataFrame(rows)


def test_make_all(tmp_path):
    rng = np.random.default_rng(0)
    p5_targets = ["levels_remaining", "final_depth", "is_last", "tree/tallest_reach", "tree/apical_reach_rel",
                  "tree/apical_is_tallest", "tree/log_n_visible"]
    p3 = pd.concat([_graph(rng, ["cell_class", "tmd_pc1"], dict(v_lo=v, v_hi=v)) for v in range(1, 6)])
    p6 = pd.DataFrame([dict(part="a_own_lineage", target="is_apical_subtree", site=s, bin_lo=d, bin_hi=d,
                            metric="binary", score=0.8, lo=0.75, hi=0.85)
                       for s in ("in", "L1", "L11") for d in range(2, 8)])
    for run in ("trained_v3", "random_v3"):
        d = tmp_path / "runs" / "uncond" / run
        d.mkdir(parents=True)
        _p2_bins(rng).to_csv(d / "p2_bins_val_t0.csv", index=False)
        _graph(rng, p5_targets).to_csv(d / "p5_val_t0.csv", index=False)
        p3.to_csv(d / "p3_val_t0.csv", index=False)
    (tmp_path / "runs" / "uncond" / "trained_v2").mkdir()
    p6.to_csv(tmp_path / "runs" / "uncond" / "trained_v2" / "p6_val_t0.csv", index=False)

    made = make_all(tmp_path / "runs", "uncond", tmp_path / "figs")
    assert [p.name[:2] for p in made] == ["F1", "F2", "F3", "F5", "F4", "F6"]
    assert all(p.stat().st_size > 0 for p in made)
