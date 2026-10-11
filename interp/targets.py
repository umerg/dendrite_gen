"""Probe targets, computed on the real (level-0) neuron and projected to every level.

All arrays are indexed by level-0 node index (root = 0), i.e. the ``fine_idx`` space of
``reduction_ids``. Positions are in model units (µm / pos_scale_factor).

Final-tree targets (``final_node_targets``) describe what each node *becomes*; level
targets (``level_targets``) split each quantity into the part visible at level ℓ and the
hidden future — the look-ahead targets of P2. Cross-dendrite targets (P6b) describe the
*other* primary dendrites as seen at level ℓ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def children_lists(parent: np.ndarray) -> list[list[int]]:
    ch: list[list[int]] = [[] for _ in range(parent.shape[0])]
    for v, p in enumerate(parent):
        if p >= 0:
            ch[p].append(v)
    return ch


def _postorder(parent: np.ndarray) -> np.ndarray:
    """Nodes ordered so every child precedes its parent."""
    ch = children_lists(parent)
    pre, stack = [], [int(np.nonzero(parent < 0)[0][0])]
    while stack:
        u = stack.pop()
        pre.append(u)
        stack.extend(ch[u])
    return np.asarray(pre[::-1], dtype=np.int64)


def final_node_targets(parent: np.ndarray, depth: np.ndarray, pos: np.ndarray,
                       is_apical_root_child: np.ndarray | None, uhat: np.ndarray) -> pd.DataFrame:
    """Per-node targets on the finished neuron (one row per level-0 node)."""
    n = parent.shape[0]
    ch = children_lists(parent)
    order = _postorder(parent)
    root = int(np.nonzero(parent < 0)[0][0])

    edge_len = np.zeros(n)
    nonroot = parent >= 0
    edge_len[nonroot] = np.linalg.norm(pos[nonroot] - pos[parent[nonroot]], axis=1)

    desc = np.zeros(n, dtype=np.int64)        # descendants, excluding self
    height = np.zeros(n, dtype=np.int64)      # longest downward hop path
    leaves = np.zeros(n, dtype=np.int64)
    strahler = np.zeros(n, dtype=np.int64)
    sub_len = np.zeros(n)                     # summed edge length below the node
    for u in order:
        c = ch[u]
        if not c:
            leaves[u], strahler[u] = 1, 1
            continue
        desc[u] = sum(desc[v] + 1 for v in c)
        height[u] = 1 + max(height[v] for v in c)
        leaves[u] = sum(leaves[v] for v in c)
        sub_len[u] = sum(sub_len[v] + edge_len[v] for v in c)
        so = [strahler[v] for v in c]
        m = max(so)
        strahler[u] = m + 1 if so.count(m) >= 2 else m  # same rule as structural_metrics

    # Branch angle: angle between the parent's incoming edge and this node's edge.
    angle = np.full(n, np.nan)
    gp_ok = nonroot & (parent >= 0)
    for v in np.nonzero(gp_ok)[0]:
        p = parent[v]
        if parent[p] < 0:
            continue  # root children have no incoming reference edge
        a = pos[p] - pos[parent[p]]
        b = pos[v] - pos[p]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            angle[v] = np.arccos(np.clip(a @ b / (na * nb), -1.0, 1.0))

    # Owning primary dendrite (root child ancestor).
    owner = np.full(n, -1, dtype=np.int64)
    for u in order[::-1]:  # parents before children
        p = parent[u]
        if p < 0:
            continue
        owner[u] = u if p == root else owner[p]
    is_apical_sub = np.zeros(n, dtype=bool)
    if is_apical_root_child is not None and is_apical_root_child.any():
        ap = int(np.nonzero(is_apical_root_child)[0][0])
        is_apical_sub = owner == ap

    axial = (pos - pos[root]) @ np.asarray(uhat, dtype=float)
    df = pd.DataFrame(dict(
        node=np.arange(n), parent=parent, depth=depth, n_children=[len(c) for c in ch],
        desc_final=desc, height_final=height, leaves_final=leaves, strahler_final=strahler,
        subtree_pathlen=sub_len, parent_dist=edge_len, branch_angle=angle,
        owner=owner, is_apical_subtree=is_apical_sub, axial_coord=axial,
        px=pos[:, 0], py=pos[:, 1], pz=pos[:, 2],
    ))
    return add_path_geometry(df)


def add_path_geometry(final: pd.DataFrame) -> pd.DataFrame:
    """Soma-relative path geometry from ``parent`` and ``px/py/pz`` (P1 path targets).

    Works on any ``final_node_targets`` frame, including ``targets_final.parquet`` written
    before these columns existed. Adds:
    pathdist_soma   summed edge length along the path soma → node (0 at the root)
    radial          Euclidean distance to the root
    log_tort        log(pathdist_soma / radial), NaN at the root
    in_edge_radial_cos  cos(incoming edge, root→node direction): 1 = the edge points straight
                    away from the soma; NaN for the root and root children (trivially 1)
    """
    df = final.sort_values("node").reset_index(drop=True)
    parent = df.parent.to_numpy()
    pos = df[["px", "py", "pz"]].to_numpy(float)
    root = int(np.nonzero(parent < 0)[0][0])
    rel = pos - pos[root]
    nonroot = parent >= 0
    edge = np.zeros(len(df))
    edge[nonroot] = np.linalg.norm(pos[nonroot] - pos[parent[nonroot]], axis=1)
    path = np.zeros(len(df))
    for u in _postorder(parent)[::-1]:             # parents before children
        if parent[u] >= 0:
            path[u] = path[parent[u]] + edge[u]
    radial = np.linalg.norm(rel, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tort = np.where((radial > 0) & (path > 0), np.log(path / radial), np.nan)
    cos = np.full(len(df), np.nan)
    p = np.where(nonroot, parent, root)
    ok = nonroot & (parent[p] >= 0) & (radial > 0) & (edge > 0)
    cos[ok] = np.einsum("ij,ij->i", pos[ok] - pos[p[ok]], rel[ok]) / (edge[ok] * radial[ok])
    df["pathdist_soma"], df["radial"], df["log_tort"], df["in_edge_radial_cos"] = path, radial, tort, cos
    return df


def level_targets(final: pd.DataFrame, n_levels: int, level: int) -> pd.DataFrame:
    """Visible / hidden split at reduction level ``level`` (rows = visible nodes only).

    Visible depth v = D - level with D = n_levels; frontier = depth == v.
    """
    v = n_levels - level
    df = final[final.depth <= v].copy()
    parent = final.parent.to_numpy()
    depth = final.depth.to_numpy()

    # Visible descendants: count descendants with depth <= v (bottom-up over visible nodes).
    vis_desc = np.zeros(len(final), dtype=np.int64)
    for u in np.argsort(-depth, kind="stable"):
        if depth[u] > v or parent[u] < 0:
            continue
        vis_desc[parent[u]] += vis_desc[u] + 1
    # Strahler order of the visible tree (frontier nodes and finished tips are its leaves).
    vis_str = np.zeros(len(final), dtype=np.int64)
    best = np.zeros(len(final), dtype=np.int64)    # highest child order seen so far
    nbest = np.zeros(len(final), dtype=np.int64)   # how many children attain it
    for u in np.argsort(-depth, kind="stable"):
        if depth[u] > v:
            continue
        vis_str[u] = 1 if nbest[u] == 0 else best[u] + int(nbest[u] >= 2)
        p = parent[u]
        if p >= 0:
            if vis_str[u] > best[p]:
                best[p], nbest[p] = vis_str[u], 1
            elif vis_str[u] == best[p]:
                nbest[p] += 1
    df["vis_strahler"] = vis_str[df.node.to_numpy()]
    df["level"] = level
    df["visible_depth"] = v
    df["is_frontier"] = df.depth == v
    df["vis_desc"] = vis_desc[df.node.to_numpy()]
    df["hidden_desc"] = df.desc_final - df.vis_desc
    df["vis_height"] = np.minimum(df.height_final, v - df.depth)
    df["hidden_height"] = df.height_final - df.vis_height
    df["label2"] = df.height_final >= 1        # 1-step expansion label (frontier rows)
    df["y_h2"] = df.height_final >= 2          # look-ahead: will grow ≥2 more generations
    return df


def cross_dendrite_targets(lvl: pd.DataFrame, tall_frac: float = 0.6) -> pd.DataFrame:
    """Per frontier row: what the *other* primary dendrites look like at this level (P6b).

    Reach of a dendrite = max |axial_coord| over its already-placed nodes (depth < visible
    depth; the frontier's positions are noised flow inputs, so "currently visible" excludes
    them). Adds:
    apical_vis_reach (normalised by tallest visible reach), apical_is_tallest_vis,
    other_arm_tall_vis, n_tall_vis, own_rank, apical_final_reach (normalised), and dAB =
    depth(row) + depth of the apical's farthest-reaching visible node (hop distance via soma).
    """
    vis = lvl[lvl.owner >= 0]
    placed = vis[vis.depth < vis.visible_depth]
    reach = placed.assign(r=placed.axial_coord.abs()).groupby("owner").r.max()
    arg = placed.assign(r=placed.axial_coord.abs()).sort_values("r").groupby("owner").tail(1)
    arg_depth = arg.set_index("owner").depth
    ap = vis.loc[vis.is_apical_subtree, "owner"]
    out = lvl[lvl.is_frontier & (lvl.owner >= 0)].copy()
    if reach.empty:
        return out
    tallest = reach.max()
    tall = reach >= tall_frac * tallest
    ap_owner = int(ap.iloc[0]) if len(ap) else -1
    out["apical_vis_reach"] = (reach.get(ap_owner, np.nan) / tallest) if tallest > 0 else np.nan
    out["apical_is_tallest_vis"] = ap_owner >= 0 and reach.idxmax() == ap_owner
    rank = reach.rank(ascending=False, method="min")
    out["own_rank"] = out.owner.map(rank).to_numpy()
    out["n_tall_vis"] = int(tall.sum())
    out["other_arm_tall_vis"] = [bool((tall & (tall.index != o)).any()) for o in out.owner]
    out["dAB"] = out.depth + (arg_depth.get(ap_owner, np.nan) if ap_owner >= 0 else np.nan)
    out.loc[out.owner == ap_owner, "dAB"] = np.nan   # no soma-crossing path to its own dendrite
    return out


def apical_final_reach(final: pd.DataFrame) -> float:
    """Apical reach / tallest reach on the finished neuron (P6b future target)."""
    f = final[final.owner >= 0]
    reach = f.assign(r=f.axial_coord.abs()).groupby("owner").r.max()
    ap = f.loc[f.is_apical_subtree, "owner"]
    if reach.empty or not len(ap) or reach.max() <= 0:
        return np.nan
    return float(reach[int(ap.iloc[0])] / reach.max())
