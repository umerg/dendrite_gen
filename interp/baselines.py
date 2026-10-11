"""Row table (index ⨝ targets) and the non-learned baselines every probe is compared to.

``row_table(store)`` returns one row per stored activation row with:
  * level targets (visible / hidden split, look-ahead labels)        — interp.targets
  * cross-dendrite targets for frontier rows (P6b)
  * observables: what the model can see at this step without any learning
  * a k-hop neighbourhood profile of the visible tree (counts / frontier counts / mean
    |axial| at hop distance 1..K) — the strongest structural baseline within the
    message-passing receptive field.

Observables never include the reduction level, the total depth D, or anything derived
from the hidden future. For frontier rows they never include the row's own position
(it is noised by the flow); the flowed expansion input ``e_t`` and the model's own
implied label ``e1_implied`` are included, so look-ahead is measured *beyond* them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from .store import Store
from .targets import add_path_geometry, apical_final_reach, cross_dendrite_targets, level_targets

K_HOP = 12
CACHE_VERSION = 3   # bump whenever graph_table columns change (old caches are ignored)
ROWS_VERSION = 4    # bump whenever row_table columns change
KHOP_COLS = ([f"hop{k}_n" for k in range(1, K_HOP + 1)]
             + [f"hop{k}_front" for k in range(1, K_HOP + 1)]
             + [f"hop{k}_axial" for k in range(1, K_HOP + 1)])

OBS_COMMON = ["depth", "slot", "is_root_child", "t", "n_visible", "n_frontier",
              "own_arm_vis_size", "own_arm_vis_reach", "own_arm_vis_height",
              "n_arms", "tallest_vis_reach"]
OBS_FRONTIER = OBS_COMMON + ["e_t", "e1_implied", "parent_edge", "parent_angle", "n_front_siblings"]
# Local-only observables for P6b: nothing that summarises *other* dendrites (no n_visible,
# n_frontier, n_arms, tallest_vis_reach), so cross-dendrite information has to come from the net.
OBS_LOCAL = ["depth", "slot", "t", "e_t", "e1_implied", "parent_edge", "parent_angle", "n_front_siblings",
             "own_arm_vis_size", "own_arm_vis_reach", "own_arm_vis_height"]
OBS_CONTEXT = OBS_COMMON + ["parent_dist", "branch_angle", "axial_coord", "vis_desc", "vis_height",
                            "n_vis_children"]


def khop_profile(lt: pd.DataFrame, K: int = K_HOP) -> pd.DataFrame:
    """Per visible node: neighbourhood counts by hop distance in the visible tree.

    Counts include frontier nodes (their existence is known); the mean |axial| only uses
    already-placed nodes, because frontier positions are noised flow inputs (pure noise at t=0).
    """
    nodes = lt.node.to_numpy()
    n = len(nodes)
    loc = {v: i for i, v in enumerate(nodes)}
    par = lt.parent.to_numpy()
    rows = [loc[v] for v, p in zip(nodes, par) if p >= 0]
    cols = [loc[p] for p in par if p >= 0]
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    D = shortest_path(A, directed=False, unweighted=True)
    front = lt.is_frontier.to_numpy().astype(float)
    placed = 1.0 - front                     # frontier positions are noised in the model input
    ax = lt.axial_coord.abs().to_numpy() * placed
    out = np.zeros((n, 3 * K))
    for k in range(1, K + 1):
        M = (D == k).astype(float)
        cnt = M.sum(1)
        n_placed = M @ placed
        out[:, k - 1] = cnt
        out[:, K + k - 1] = M @ front
        out[:, 2 * K + k - 1] = np.where(n_placed > 0, (M @ ax) / np.maximum(n_placed, 1), 0.0)
    df = pd.DataFrame(out, columns=KHOP_COLS)
    df["node"] = nodes
    return df


def _item_rows(rows: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    n_levels, level = int(rows.n_levels.iloc[0]), int(rows.level.iloc[0])
    lt = level_targets(final, n_levels, level)
    vis = lt[lt.owner >= 0]
    # sizes / heights count every visible node (structure is known); reach only uses
    # already-placed nodes (frontier positions are noised flow inputs).
    arm = vis.groupby("owner").agg(own_arm_vis_size=("node", "size"), own_arm_vis_height=("depth", "max"))
    placed = vis[~vis.is_frontier]
    arm = arm.join(placed.assign(r=placed.axial_coord.abs()).groupby("owner").r.max().rename("own_arm_vis_reach"))
    arm["own_arm_vis_reach"] = arm.own_arm_vis_reach.fillna(0.0)
    lt = lt.merge(arm, left_on="owner", right_index=True, how="left")
    lt["n_visible"] = len(lt)
    lt["n_frontier"] = int(lt.is_frontier.sum())
    lt["n_arms"] = len(arm)
    lt["tallest_vis_reach"] = float(arm.own_arm_vis_reach.max()) if len(arm) else 0.0
    ch_vis = lt.groupby("parent").size()
    lt["n_vis_children"] = lt.node.map(ch_vis).fillna(0).astype(int)
    front_sib = lt[lt.is_frontier].groupby("parent").size()
    lt["n_front_siblings"] = lt.parent.map(front_sib).fillna(0).astype(int)
    # parent's (visible, clean) incoming edge length and branch angle for frontier rows
    pe = final.parent_dist.to_numpy()
    pa = final.branch_angle.to_numpy()
    p = lt.parent.to_numpy()
    lt["parent_edge"] = np.where(p >= 0, pe[np.maximum(p, 0)], 0.0)
    lt["parent_angle"] = np.where(p >= 0, pa[np.maximum(p, 0)], np.nan)
    # parent's soma-relative path geometry (P1 path/ctx targets; parents are always placed)
    par = final.parent.to_numpy()
    pp = np.maximum(p, 0)
    for src, dst in [("pathdist_soma", "parent_pathdist"), ("radial", "parent_radial"),
                     ("axial_coord", "parent_axial"), ("log_tort", "parent_log_tort"),
                     ("in_edge_radial_cos", "parent_radial_cos")]:
        lt[dst] = np.where(p >= 0, final[src].to_numpy()[pp], np.nan)
    gp = np.where(p >= 0, par[pp], -1)
    lt["gp_edge"] = np.where(gp >= 0, pe[np.maximum(gp, 0)], np.nan)   # grandparent's incoming edge

    cd = cross_dendrite_targets(lt)
    cd_cols = ["apical_vis_reach", "apical_is_tallest_vis", "other_arm_tall_vis", "n_tall_vis",
               "own_rank", "dAB"]
    lt = lt.merge(cd[["node"] + [c for c in cd_cols if c in cd]], on="node", how="left")
    lt["apical_final_reach"] = apical_final_reach(final)
    lt = lt.merge(khop_profile(lt), on="node", how="left")
    dup = [c for c in lt.columns if c in rows.columns and c != "node"]
    return rows.merge(lt.drop(columns=dup), on="node", how="left", validate="one_to_one")


def row_table(store: Store, cache: bool = True) -> pd.DataFrame:
    path = store.path / f"rows_v{ROWS_VERSION}.parquet"
    if cache and path.exists():
        return pd.read_parquet(path)
    tf = store.targets_final()
    finals = {k: add_path_geometry(g) for k, g in tf.groupby("neuron")}
    parts = [_item_rows(rows, finals[n]) for (n, _), rows in store.index.groupby(["neuron", "level"], sort=False)]
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values("row_sparse").reset_index(drop=True)
    if cache:
        out.to_parquet(path)
    return out


def matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Float matrix with NaN → 0 plus a missing-indicator column for any col that had NaNs."""
    X = df[cols].astype(float)
    miss = X.isna()
    extra = [miss[c].to_numpy()[:, None].astype(float) for c in cols if miss[c].any()]
    return np.hstack([X.fillna(0.0).to_numpy()] + extra) if extra else X.to_numpy()


# ----------------------------------------------------------------------------- graph level (P3, P5)

GRAPH_OBS = ["visible_depth", "n_visible", "n_frontier", "frontier_frac", "cont_rate", "n_arms",
             "arm_size_max", "arm_size_mean", "arm_reach_max", "arm_reach_mean", "arm_height_max"]


def _partial_graph(final: pd.DataFrame, v: int):
    import networkx as nx

    vis = final[final.depth <= v]
    G = nx.Graph()
    for r in vis.itertuples(index=False):
        G.add_node(int(r.node), pos=np.array([r.px, r.py, r.pz], dtype=np.float64))
    G.add_edges_from((int(r.node), int(r.parent)) for r in vis.itertuples(index=False) if r.parent >= 0)
    G.graph["root"] = 0
    return G


def graph_table(store: Store, uhat, partial_tmd_cfg=None, cache: bool = True) -> tuple[pd.DataFrame, np.ndarray | None]:
    """One row per reduction item: ids, visible-tree observables, partial-tree morphometrics.

    ``partial_tmd_cfg`` (a model cfg) additionally computes the TMD vector of the visible
    partial tree with the conditioning filtrations — the strongest "what is already
    knowable" baseline for P3. Returned separately as [items, D] (rows aligned to the table).
    Cached as graph_rows_v<N>.parquet / partial_tmd_v<N>.npy in the store directory.
    """
    from validation.dist_metrics import MORPHO_KEYS, assemble_morpho_vector

    path, tpath = (store.path / f"graph_rows_v{CACHE_VERSION}.parquet",
                   store.path / f"partial_tmd_v{CACHE_VERSION}.npy")
    want_tmd = partial_tmd_cfg is not None
    if cache and path.exists() and (not want_tmd or tpath.exists()):
        return pd.read_parquet(path), (np.load(tpath) if want_tmd else None)

    gi = store.graph_index.sort_values("item").reset_index(drop=True)
    tf = store.targets_final()
    finals = {k: g.sort_values("node").reset_index(drop=True) for k, g in tf.groupby("neuron")}
    front = store.index[store.index.is_frontier]
    cont = front.groupby("item").e1_implied.apply(lambda e: float((e > 0).mean()))
    rows, tmds = [], []
    for r in gi.itertuples(index=False):
        final = finals[r.neuron]
        v = int(r.n_levels - r.level)
        vis = final[(final.depth <= v) & (final.owner >= 0)]
        arm = vis.groupby("owner").agg(size=("node", "size"), height=("depth", "max"))
        placed = vis[vis.depth < v]          # frontier positions are noised flow inputs
        arm = arm.join(placed.assign(a=placed.axial_coord.abs()).groupby("owner").a.max().rename("reach"))
        arm["reach"] = arm.reach.fillna(0.0)
        # geometry (morphometrics, partial TMD) only from placed nodes; counts use all visible
        G = _partial_graph(final, max(v - 1, 0))
        try:
            mv = assemble_morpho_vector(G, uhat=uhat)
        except Exception:  # noqa: BLE001 — degenerate partial tree (e.g. soma only at the last step)
            mv = np.full(len(MORPHO_KEYS), np.nan)
        rec = dict(item=r.item, neuron=r.neuron, level=r.level, n_levels=r.n_levels, cell_class=r.cell_class,
                   visible_depth=v, frac_visible=v / r.n_levels, n_visible=r.n_nodes, n_frontier=r.n_frontier,
                   frontier_frac=r.n_frontier / max(r.n_nodes, 1), cont_rate=cont.get(r.item, np.nan),
                   n_arms=len(arm), arm_size_max=arm["size"].max(), arm_size_mean=arm["size"].mean(),
                   arm_reach_max=arm.reach.max(), arm_reach_mean=arm.reach.mean(), arm_height_max=arm.height.max())
        rec.update({f"morpho_{k}": float(x) for k, x in zip(MORPHO_KEYS, mv)})
        rows.append(rec)
        if want_tmd:
            from .load import tmd_vector
            try:
                tmds.append(tmd_vector(G, partial_tmd_cfg))
            except Exception:  # noqa: BLE001 — tiny partial trees can be degenerate
                tmds.append(None)
    out = pd.DataFrame(rows)
    ptmd = None
    if want_tmd:
        dim = next((len(x) for x in tmds if x is not None), None)
        if dim is None:
            raise RuntimeError("partial TMD failed for every item")
        ptmd = np.stack([x if x is not None else np.full(dim, np.nan) for x in tmds]).astype(np.float32)
    if cache:
        out.to_parquet(path)
        if want_tmd:
            np.save(tpath, ptmd)
    return out, ptmd


def morpho_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("morpho_")]
