"""Path geometry and visible Strahler order on a hand-built tree (P1 path / morpho targets)."""
import numpy as np
import pandas as pd

from interp.targets import add_path_geometry, final_node_targets, level_targets

#        0 (soma, origin)
#        |
#        1 (1,0,0)
#       / \
#  (2,0,0) 2   3 (1,1,0)
#     / \       \
# (3,0,0) 4  5 (2,1,0)  6 (1,2,0)
PARENT = np.array([-1, 0, 1, 1, 2, 2, 3])
DEPTH = np.array([0, 1, 2, 2, 3, 3, 3])
POS = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0], [3, 0, 0], [2, 1, 0], [1, 2, 0]], dtype=float)
S2, S5 = np.sqrt(2), np.sqrt(5)


def _final():
    return final_node_targets(PARENT, DEPTH, POS, None, np.array([0.0, 0.0, 1.0]))


def test_path_geometry():
    f = _final()
    np.testing.assert_allclose(f.pathdist_soma, [0, 1, 2, 2, 3, 3, 3])
    np.testing.assert_allclose(f.radial, [0, 1, 2, S2, 3, S5, S5])
    np.testing.assert_allclose(f.log_tort, [np.nan, 0, 0, np.log(2 / S2), 0, np.log(3 / S5), np.log(3 / S5)])
    np.testing.assert_allclose(f.in_edge_radial_cos, [np.nan, np.nan, 1, 1 / S2, 1, 1 / S5, 2 / S5])


def test_path_geometry_on_old_targets_frame():
    """targets_final.parquet written before these columns existed: recomputed from parent + px/py/pz."""
    f = _final()
    old = f.drop(columns=["pathdist_soma", "radial", "log_tort", "in_edge_radial_cos"]).sample(frac=1, random_state=0)
    pd.testing.assert_frame_equal(add_path_geometry(old)[f.columns], f)


def test_vis_strahler():
    f = _final()
    np.testing.assert_array_equal(f.strahler_final, [2, 2, 2, 1, 1, 1, 1])
    D = int(DEPTH.max())
    lv0 = level_targets(f, D, 0)                      # everything visible
    np.testing.assert_array_equal(lv0.vis_strahler, f.strahler_final)
    lv1 = level_targets(f, D, 1).set_index("node")    # depth <= 2 visible: 2 and 3 are leaves
    assert lv1.vis_strahler.to_dict() == {0: 2, 1: 2, 2: 1, 3: 1}
    lv2 = level_targets(f, D, 2).set_index("node")    # depth <= 1 visible: a single path
    assert lv2.vis_strahler.to_dict() == {0: 1, 1: 1}
