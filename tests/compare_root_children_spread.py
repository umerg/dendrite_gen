"""Compare root children offset vectors and subtree sizes between GT and predicted trees."""
from __future__ import annotations
import sys, pickle, numpy as np, networkx as nx
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_loading import load_swc_graph


def root_tree(G, root):
    pm = {root: None}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in G.neighbors(u):
            if v not in pm:
                pm[v] = u
                q.append(v)
    return pm


def get_children(pm):
    cm = {n: [] for n in pm}
    for n, p in pm.items():
        if p is not None:
            cm[p].append(n)
    return cm


def subtree_size_and_depth(root_node, cm):
    nodes = []
    max_d = 0
    q = deque([(root_node, 0)])
    while q:
        u, d = q.popleft()
        nodes.append(u)
        max_d = max(max_d, d)
        for c in cm.get(u, []):
            q.append((c, d + 1))
    return len(nodes), max_d


def analyse(G, root, label):
    pm = root_tree(G, root)
    cm = get_children(pm)
    root_kids = cm[root]
    k = len(root_kids)
    root_pos = np.array(G.nodes[root]["pos"])

    print(f"  {label}: N={G.number_of_nodes()}, K={k}")

    child_data = []
    for i, c in enumerate(root_kids):
        c_pos = np.array(G.nodes[c]["pos"])
        offset = c_pos - root_pos
        dist = np.linalg.norm(offset)
        n_sub, max_d = subtree_size_and_depth(c, cm)
        child_data.append(dict(idx=i, offset=offset, dist=dist, n_nodes=n_sub, max_depth=max_d))

    # Sort by subtree size descending
    child_data.sort(key=lambda x: -x["n_nodes"])

    print("  (sorted by subtree size)")
    print(f"  {'idx':>3} {'dist':>7} {'nodes':>5} {'depth':>5}   offset_vector")
    for cd in child_data:
        o = cd["offset"]
        print(f"  {cd['idx']:>3} {cd['dist']:>7.4f} {cd['n_nodes']:>5} {cd['max_depth']:>5}   "
              f"[{o[0]:+.4f}, {o[1]:+.4f}, {o[2]:+.4f}]")

    # Pairwise angles
    offsets = np.array([cd["offset"] for cd in child_data])
    norms = np.linalg.norm(offsets, axis=1, keepdims=True).clip(1e-12)
    unit = offsets / norms
    cos_mat = np.clip(unit @ unit.T, -1, 1)
    angle_mat = np.degrees(np.arccos(cos_mat))

    top_n = min(5, k)
    print(f"\n  Pairwise angles (top {top_n} by size):")
    hdr = "       " + "".join(f"  c{child_data[j]['idx']:<4}" for j in range(top_n))
    print(hdr)
    for i in range(top_n):
        row = f"  c{child_data[i]['idx']:<3} "
        for j in range(top_n):
            if j <= i:
                row += "  ---  "
            else:
                row += f" {angle_mat[i][j]:>4.0f}° "
        row += f"  (n={child_data[i]['n_nodes']})"
        print(row)

    # Flag indistinguishable pairs: close angle AND similar subtree size
    print(f"\n  Indistinguishable pairs (angle<30° AND size_ratio>0.5, both>1 node):")
    found = False
    for i in range(len(child_data)):
        for j in range(i + 1, len(child_data)):
            angle = angle_mat[i][j]
            si, sj = child_data[i]["n_nodes"], child_data[j]["n_nodes"]
            sr = min(si, sj) / max(si, sj) if max(si, sj) > 0 else 1
            if angle < 30 and sr > 0.5 and min(si, sj) > 1:
                found = True
                print(f"    c{child_data[i]['idx']} & c{child_data[j]['idx']}: "
                      f"angle={angle:.1f}°, sizes={si}&{sj} (ratio={sr:.2f})")
    if not found:
        print("    None found")

    # Also flag pairs with similar size but ANY angle (to see if sizes cluster)
    print(f"\n  Same-size pairs (size_ratio>0.7, both>2 nodes, any angle):")
    found2 = False
    for i in range(len(child_data)):
        for j in range(i + 1, len(child_data)):
            si, sj = child_data[i]["n_nodes"], child_data[j]["n_nodes"]
            sr = min(si, sj) / max(si, sj) if max(si, sj) > 0 else 1
            if sr > 0.7 and min(si, sj) > 2:
                found2 = True
                angle = angle_mat[i][j]
                print(f"    c{child_data[i]['idx']} & c{child_data[j]['idx']}: "
                      f"sizes={si}&{sj} (ratio={sr:.2f}), angle={angle:.1f}°")
    if not found2:
        print("    None found")
    print()


def main():
    with open("outputs/2026-04-01/17-02-21/validation/step_30001.pkl", "rb") as f:
        data = pickle.load(f)
    pred_graphs = data["ema_1"]["pred_graphs"]

    gt_files = sorted([f for f in Path("/Volumes/Seagate/neurons_v1/train/").glob("*.csv.swc")
                        if not f.name.startswith("._")])
    gt_graphs = [load_swc_graph(f) for f in gt_files]

    for gi in range(len(gt_graphs)):
        print("=" * 70)
        print(f"GRAPH {gi}")
        print("=" * 70)
        gt_root = gt_graphs[gi].graph.get("root", 0)
        pred_root = max(pred_graphs[gi].nodes(), key=lambda n: pred_graphs[gi].degree(n))
        analyse(gt_graphs[gi], gt_root, "GT")
        analyse(pred_graphs[gi], pred_root, "Pred")


if __name__ == "__main__":
    main()
