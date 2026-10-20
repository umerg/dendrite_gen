"""Small helpers of the analysis engine (no data needed)."""
import pandas as pd

from interp.analyses.common import min_depth_rows, site_depth, with_min_depth


def test_site_depth_orders_node_and_graph_sites():
    order = ["in", "L0", "A1", "G1", "L1", "pool_L1", "A3", "G3", "L3"]
    assert sorted(order[::-1], key=site_depth)[:3] == ["in", "L0", "A1"]
    assert site_depth("A1") < site_depth("G1") < site_depth("L1") == site_depth("pool_L1")
    assert site_depth("pool_in") == site_depth("in")


def test_min_depth_rows_and_signature():
    rows = pd.DataFrame({"depth": [1, 2, 5, 7]})
    assert min_depth_rows(rows, 1) is rows
    assert list(min_depth_rows(rows, 5).depth) == [5, 7]
    assert with_min_depth({"a": 1}, 1) == {"a": 1}          # unfiltered runs keep old checkpoints
    assert with_min_depth({"a": 1}, 5) == {"a": 1, "min_depth": 5}
