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
