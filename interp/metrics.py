"""Per-tree readouts for generated (and real) neurons (methodology §11b.2).

Tall-arm statistics follow ``docs/APICAL_AXIAL_ERROR_MODE.md`` §3 (the scripts cited there are
lost; this re-implements the definition):
  arm reach  r_a = max over nodes v of soma child a's subtree of |(p_v − p_root)·û|
  tall       r_a ≥ frac · max_b r_b   (frac = 0.6)
  dominance  r_(1) / r_(2)

The slot of a soma child is its rank by node id among the root's children. In generated
trees node ids follow creation order and all soma children are spawned at step 0, so slot 0 is
the child the model treats as apical. GT trees have no slot (node ids follow the SWC file); for
them only the slot-free statistics are meaningful.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def _root(G: nx.Graph) -> int:
    r = G.graph.get("root", None)
    return r if r is not None and r in G else min(G.nodes)


def _pos(G, n) -> np.ndarray:
    return np.asarray(G.nodes[n]["pos"], dtype=np.float64).reshape(3)


def arms(G: nx.Graph) -> dict[int, list[int]]:
    """Soma child → nodes of its subtree (including itself)."""
    r = _root(G)
    out = {}
    H = G.copy()
    H.remove_node(r)
    for c in sorted(G.neighbors(r)):
        out[c] = sorted(nx.node_connected_component(H, c))
    return out


def arm_table(G: nx.Graph, uhat) -> pd.DataFrame:
    """One row per soma child: slot, reach (|axial| max), signed axial at the reach node, size."""
    u = np.asarray(uhat, dtype=np.float64).reshape(3)
    u = u / np.linalg.norm(u)
    p0 = _pos(G, _root(G))
    rows = []
    for rank, (c, nodes) in enumerate(arms(G).items()):
        slot = int(G.nodes[c].get("owner", rank))   # rollouts record the spawn slot
        ax = np.array([(_pos(G, n) - p0) @ u for n in nodes])
        k = int(np.argmax(np.abs(ax)))
        rows.append(dict(child=c, slot=slot, reach=float(abs(ax[k])), signed=float(ax[k]), n_nodes=len(nodes)))
    return pd.DataFrame(rows, columns=["child", "slot", "reach", "signed", "n_nodes"])


def tall_arm_stats(G: nx.Graph, uhat, frac: float = 0.6) -> dict:
    a = arm_table(G, uhat)
    nan = float("nan")
    if a.empty or a.reach.max() <= 0:
        return dict(n_arms=len(a), n_tall=0, one_tall=False, ge2_tall=False, dominance=nan,
                    tallest_slot=-1, tallest_is_slot0=False, tallest_neg=False,
                    apical_reach=nan, basal_reach_max=nan, apical_nodes=0, basal_nodes=0)
    top = a.reach.max()
    tall = a.reach >= frac * top
    srt = np.sort(a.reach.to_numpy())[::-1]
    best = a.loc[a.reach.idxmax()]
    s0, basal = a[a.slot == 0], a[a.slot >= 1]
    return dict(n_arms=len(a), n_tall=int(tall.sum()), one_tall=bool(tall.sum() == 1), ge2_tall=bool(tall.sum() >= 2),
                dominance=float(srt[0] / srt[1]) if len(srt) > 1 and srt[1] > 0 else nan,
                tallest_slot=int(best.slot), tallest_is_slot0=bool(best.slot == 0), tallest_neg=bool(best.signed < 0),
                apical_reach=float(s0.reach.iloc[0]) if len(s0) else nan,
                basal_reach_max=float(basal.reach.max()) if len(basal) else nan,
                apical_nodes=int(s0.n_nodes.sum()), basal_nodes=int(basal.n_nodes.sum()))


def size_stats(G: nx.Graph, uhat) -> dict:
    u = np.asarray(uhat, dtype=np.float64).reshape(3)
    u = u / np.linalg.norm(u)
    r = _root(G)
    P = np.stack([_pos(G, n) for n in G.nodes])
    ax = P @ u
    depth = nx.single_source_shortest_path_length(G, r)
    tips = sum(1 for n in G.nodes if n != r and G.degree[n] == 1)
    length = float(sum(np.linalg.norm(_pos(G, a) - _pos(G, b)) for a, b in G.edges))
    return dict(n_nodes=G.number_of_nodes(), n_tips=tips, max_depth=int(max(depth.values())),
                axial_extent=float(ax.max() - ax.min()), total_length=length)


def tree_row(G: nx.Graph, uhat, frac: float = 0.6) -> dict:
    return {**size_stats(G, uhat), **tall_arm_stats(G, uhat, frac)}


def tree_table(graphs, uhat, frac: float = 0.6) -> pd.DataFrame:
    return pd.DataFrame([tree_row(G, uhat, frac) for G in graphs])


# ----------------------------------------------------------------------------- E3: one node's subtree

def _children(G: nx.Graph) -> dict[int, list[int]]:
    r = _root(G)
    par = nx.bfs_predecessors(G, r)
    ch: dict[int, list[int]] = {n: [] for n in G.nodes}
    for c, p in par:
        ch[p].append(c)
    return ch


def _parents(G: nx.Graph) -> dict[int, int]:
    return dict(nx.bfs_predecessors(G, _root(G)))


def _subtree(ch: dict, n: int) -> list[int]:
    out, stack = [], [n]
    while stack:
        v = stack.pop()
        out.append(v)
        stack.extend(ch[v])
    return out


def subtree_stats(G: nx.Graph, node: int, uhat, ch: dict | None = None) -> dict:
    """Descendants of ``node`` (excluding itself): count, height (generations), path length,
    Strahler order of the subtree rooted at ``node``, axial reach of the subtree from ``node``."""
    ch = ch or _children(G)
    nodes = _subtree(ch, node)
    u = np.asarray(uhat, dtype=np.float64).reshape(3)
    u = u / np.linalg.norm(u)
    p0 = _pos(G, node)
    height, stack = 0, [(node, 0)]
    while stack:
        v, d = stack.pop()
        height = max(height, d)
        stack.extend((c, d + 1) for c in ch[v])
    order = {}
    for v in reversed(nodes):           # children before parents (DFS pre-order reversed)
        if not ch[v]:
            order[v] = 1
        else:
            o = sorted((order[c] for c in ch[v]), reverse=True)
            order[v] = o[0] + 1 if len(o) > 1 and o[0] == o[1] else o[0]
    length = float(sum(np.linalg.norm(_pos(G, c) - _pos(G, v)) for v in nodes for c in ch[v]))
    reach = float(max(abs((_pos(G, v) - p0) @ u) for v in nodes))
    return dict(n_desc=len(nodes) - 1, height=height, path_length=length, strahler=order[node], axial_reach=reach)


def growth_row(G: nx.Graph, uhat) -> dict:
    """E3 readout for a rollout graph carrying ``graph['target_key']`` and node ``key`` attrs.
    Target subtree, its sibling's subtree, the rest of its dendrite, the other basal dendrites."""
    key = G.graph.get("target_key")
    nan = float("nan")
    base = dict(has_target=key is not None)
    if key is None:
        return base
    hit = [n for n in G.nodes if G.nodes[n].get("key") == key]
    if len(hit) != 1:
        raise ValueError(f"target key found {len(hit)} times")
    t = hit[0]
    ch, par = _children(G), _parents(G)
    st = subtree_stats(G, t, uhat, ch)
    out = {**base, "t_expanded": bool(ch[t]), **{f"t_{k}": v for k, v in st.items()}}
    sib = [c for c in ch[par[t]] if c != t] if t in par else []
    if sib:
        ss = subtree_stats(G, sib[0], uhat, ch)
        out.update(s_expanded=bool(ch[sib[0]]), s_n_desc=ss["n_desc"], s_height=ss["height"])
    else:
        out.update(s_expanded=False, s_n_desc=nan, s_height=nan)
    own = G.nodes[t].get("owner", -1)
    in_t = set(_subtree(ch, t))
    dend = [n for n in G.nodes if G.nodes[n].get("owner", -2) == own]
    out["dend_rest_nodes"] = len([n for n in dend if n not in in_t])
    out["other_basal_nodes"] = sum(1 for n in G.nodes if G.nodes[n].get("owner", -1) >= 1 and G.nodes[n]["owner"] != own)
    out["target_owner"] = own
    return out


def main(argv=None):
    """GT calibration: tall-arm rates of real neurons (doc reference: 80.3 % one tall, 19.7 % ≥ 2)."""
    import argparse

    from .load import compose_cfg, load_split, uhat_of

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--config", default="parity_neurons_uncond")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--frac", type=float, default=0.6)
    a = ap.parse_args(argv)
    cfg = compose_cfg(a.config)
    for sp in a.splits.split(","):
        df = tree_table(load_split(cfg, sp, data_dir=a.data_dir), uhat_of(cfg), a.frac)
        print(f"[{sp}] n={len(df)}  one_tall={df.one_tall.mean():.3f}  ge2_tall={df.ge2_tall.mean():.3f}  "
              f"dominance(median)={df.dominance.median():.2f}  tallest_neg={df.tallest_neg.mean():.3f}  "
              f"n_nodes(median)={df.n_nodes.median():.0f}", flush=True)


if __name__ == "__main__":
    main()
