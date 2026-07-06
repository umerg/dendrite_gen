"""Structural + loss accounting for a neuron SWC dataset (train/val/test).

Runs over ALL files in each split. For each file (conceptually re-rooted at the soma =
node id 1, the pipeline convention) it computes structure and the exact losses the
clean_trees + rare-class + root-degree>cap pipeline incurs:

  SAMPLE-level loss (whole neuron dropped):
    - broken / disconnected / missing-soma  (data quality)
    - cell type in --drop-classes           (rare test-only classes)
    - soma degree > --max-children          (one-hot ordinal limit)
  NODE-level loss (within kept neurons):
    - multifurcation PRUNE case (>=4 children / undirected deg>=5): keep 2 thickest, delete rest
    - multifurcation SPLIT case (3 children / deg 4): +1 inserted node (lossless)

Point --root at the RAW swc_simplified to size the losses, or at the CLEANED
neurons_conditional to verify it (expect 0 multifurcations, root degree <= cap).

Usage:
  conda run -n NEURO2 python data_analysis/dataset_loss_accounting.py                       # raw, cap 16
  conda run -n NEURO2 python data_analysis/dataset_loss_accounting.py \
      --root /Users/umer/Documents/neurons_conditional --drop-classes ""                    # verify cleaned
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict, deque

import numpy as np


def load_celltypes(csv_path):
    m = {}
    if csv_path and os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                m[row["filename"]] = row["cell_type"]
    return m


def ctype_of(fname, celltype):
    return celltype.get(fname.replace("_simplified.swc", ".swc"), "UNKNOWN")


def analyse_file(path):
    typ, rad, parent_raw = {}, {}, {}
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            if len(p) < 7:
                p = s.replace(",", " ").split()
            if len(p) < 7:
                continue
            nid = int(p[0])
            typ[nid] = int(float(p[1]))
            rad[nid] = float(p[5])
            parent_raw[nid] = int(p[6])
    N = len(typ)
    r = {"N": N, "broken": False, "disconnected": False,
         "soma_present": (typ.get(1) == 1), "node1_present": (1 in typ)}
    nbr = {n: set() for n in typ}
    for nid, pa in parent_raw.items():
        if pa > 0:
            if pa not in nbr:
                r["broken"] = True
                return r
            nbr[nid].add(pa)
            nbr[pa].add(nid)
    root = 1 if 1 in typ else min(typ)
    r["soma_deg"] = len(nbr[root])

    parent = {root: None}
    order = []
    seen = {root}
    q = deque([root])
    while q:
        u = q.popleft()
        order.append(u)
        for v in nbr[u]:
            if v not in seen:
                parent[v] = u
                seen.add(v)
                q.append(v)
    r["disconnected"] = len(seen) != N

    depth = {root: 0}
    children = defaultdict(list)
    for u in order:
        if parent[u] is not None:
            depth[u] = depth[parent[u]] + 1
            children[parent[u]].append(u)
    r["max_depth"] = max(depth.values()) if depth else 0
    r["depth_hist"] = Counter(depth.values())

    subsize = {n: 1 for n in typ}
    for u in reversed(order):
        if parent[u] is not None:
            subsize[parent[u]] += subsize[u]

    leaves = bifurc = passthrough = 0
    split_ct = prune_ct = 0
    dropped_branches = dropped_nodes = inserted_nodes = 0
    prune_child = Counter()
    for u in typ:
        if u == root:
            continue
        d = len(nbr[u])
        if d == 1:
            leaves += 1
        elif d == 2:
            passthrough += 1
        elif d == 3:
            bifurc += 1
        k = len(children[u])
        if k == 3:
            split_ct += 1
            inserted_nodes += 1
        elif k >= 4:
            prune_ct += 1
            prune_child[k] += 1
            dropped_branches += k - 2
            kept = set(sorted(children[u], key=lambda x: rad[x], reverse=True)[:2])
            for c in children[u]:
                if c not in kept:
                    dropped_nodes += subsize[c]
    r.update(leaves=leaves, bifurc=bifurc, passthrough=passthrough,
             split_ct=split_ct, prune_ct=prune_ct, dropped_branches=dropped_branches,
             dropped_nodes=dropped_nodes, inserted_nodes=inserted_nodes,
             prune_child=prune_child)
    return r


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def stats(a):
    a = np.array(a)
    return f"mean={a.mean():.1f} median={np.median(a):.0f} min={a.min()} max={a.max()} p95={np.percentile(a,95):.0f}"


def main():
    ap = argparse.ArgumentParser(description="Structural + loss accounting for a neuron SWC dataset.")
    ap.add_argument("--root", default="/Users/umer/Documents/neurons_simplified/swc_simplified")
    ap.add_argument("--csv", default="/Users/umer/Downloads/targets_cell_type.csv")
    ap.add_argument("--max-children", type=int, default=16)
    ap.add_argument("--drop-classes", default="WM-P,MC,BPC",
                    help="Comma-separated cell types dropped wholesale (empty string = none).")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    CAP = args.max_children
    DROP = {c.strip() for c in args.drop_classes.split(",") if c.strip()}
    celltype = load_celltypes(args.csv)

    report = {}
    grand_prune_child = Counter()
    grand_depth = Counter()
    grand_ct_all = Counter()
    grand_ct_dropped_cap = Counter()
    grand_ct_dropped_rare = Counter()
    grand_soma_deg = Counter()

    for split in args.splits:
        d = os.path.join(args.root, split)
        if not os.path.isdir(d):
            print(f"[skip] {d} not found")
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".swc") and not f.startswith("._"))
        S = {
            "n_files": len(files), "n_broken": 0, "n_disconnected": 0,
            "n_no_soma": 0, "n_rare": 0, "n_soma_gt_cap": 0,
            "nodes": [], "max_depth": [], "soma_deg": [],
            "sum_nodes_kept": 0, "leaves": 0, "bifurc": 0, "passthrough": 0,
            "split_ct": 0, "prune_ct": 0, "dropped_branches": 0,
            "dropped_nodes": 0, "inserted_nodes": 0,
            "ct_kept": Counter(), "prune_child": Counter(), "depth_hist": Counter(),
        }
        for fn in files:
            ct = ctype_of(fn, celltype)
            grand_ct_all[ct] += 1
            if ct in DROP:
                S["n_rare"] += 1
                grand_ct_dropped_rare[ct] += 1
                continue
            r = analyse_file(os.path.join(d, fn))
            if r["broken"]:
                S["n_broken"] += 1
                continue
            if r["disconnected"]:
                S["n_disconnected"] += 1
            if not r["soma_present"]:
                S["n_no_soma"] += 1
            S["nodes"].append(r["N"])
            S["max_depth"].append(r["max_depth"])
            S["soma_deg"].append(r["soma_deg"])
            grand_soma_deg[r["soma_deg"]] += 1
            S["depth_hist"].update(r["depth_hist"])
            grand_depth.update(r["depth_hist"])
            if r["soma_deg"] > CAP:
                S["n_soma_gt_cap"] += 1
                grand_ct_dropped_cap[ct] += 1
                continue
            S["ct_kept"][ct] += 1
            S["sum_nodes_kept"] += r["N"]
            for k in ("leaves", "bifurc", "passthrough", "split_ct", "prune_ct",
                      "dropped_branches", "dropped_nodes", "inserted_nodes"):
                S[k] += r[k]
            S["prune_child"].update(r["prune_child"])
            grand_prune_child.update(r["prune_child"])
        report[split] = S

    print("=" * 90)
    print(f"DATASET LOSS REPORT  root={args.root}  cap={CAP}  drop_classes={sorted(DROP) or 'none'}")
    print("=" * 90)
    tot_files = tot_kept = tot_dropped = 0
    for split, S in report.items():
        kept = S["n_files"] - S["n_rare"] - S["n_soma_gt_cap"] - S["n_broken"]
        tot_files += S["n_files"]
        tot_dropped += S["n_rare"] + S["n_soma_gt_cap"] + S["n_broken"]
        tot_kept += kept
        print(f"\n### {split}  ({S['n_files']} files)")
        print(f"  data quality: broken={S['n_broken']}  disconnected={S['n_disconnected']}  missing-soma(type1)={S['n_no_soma']}")
        if S["nodes"]:
            print(f"  nodes/file : {stats(S['nodes'])}")
            print(f"  soma-rooted max depth : {stats(S['max_depth'])}")
            print(f"  soma degree (#primary dendrites) : {stats(S['soma_deg'])}")
        print(f"  --- SAMPLE LOSS: rare-class={S['n_rare']}  soma>{CAP}={S['n_soma_gt_cap']}  ->  KEPT={kept}")
        tot_multi = S["split_ct"] + S["prune_ct"]
        print(f"  --- within KEPT ({kept}): multifurcations={tot_multi}  "
              f"split(3ch,lossless)={S['split_ct']} ({pct(S['split_ct'],tot_multi):.1f}%)  "
              f"prune(>=4ch,lossy)={S['prune_ct']} ({pct(S['prune_ct'],tot_multi):.1f}%)")
        print(f"      NODE LOSS (prune): branches={S['dropped_branches']}  nodes={S['dropped_nodes']} "
              f"({pct(S['dropped_nodes'],S['sum_nodes_kept']):.2f}% of kept)   NODE GAIN (split): {S['inserted_nodes']}")

    print("\n" + "=" * 90)
    print(f"TOTALS: files={tot_files}  kept={tot_kept} ({pct(tot_kept,tot_files):.2f}%)  dropped={tot_dropped} ({pct(tot_dropped,tot_files):.2f}%)")
    print("=" * 90)

    print("\nSAMPLE LOSS BY CELL TYPE:")
    print(f"  {'type':<8}{'total':>8}{'rare':>7}{'soma>cap':>10}{'kept':>8}{'drop%':>8}")
    for ct in sorted(grand_ct_all, key=lambda c: -grand_ct_all[c]):
        tot = grand_ct_all[ct]
        rare = grand_ct_dropped_rare.get(ct, 0)
        capd = grand_ct_dropped_cap.get(ct, 0)
        kept = tot - rare - capd
        print(f"  {ct:<8}{tot:>8}{rare:>7}{capd:>10}{kept:>8}{pct(rare+capd,tot):>7.2f}%")

    if grand_soma_deg:
        print("\nSOMA DEGREE DISTRIBUTION (analysed files):")
        mx = max(grand_soma_deg.values())
        for k in sorted(grand_soma_deg):
            bar = "#" * int(60 * grand_soma_deg[k] / mx)
            flag = f"  <-- DROPPED (>{CAP})" if k > CAP else ""
            print(f"  deg {k:>2}: {grand_soma_deg[k]:>6}  {bar}{flag}")

    print("\nPRUNE-NODE child-count distribution (kept):", dict(sorted(grand_prune_child.items())))

    if grand_depth:
        total_depth_nodes = sum(grand_depth.values())
        print("\nSOMA-ROOTED DEPTH: nodes beyond a hypothetical max-depth cap (none applied):")
        for cap in [10, 12, 14, 16, 20, 24]:
            beyond = sum(v for dd, v in grand_depth.items() if dd > cap)
            print(f"  cap={cap:>2}: {beyond:>7} nodes beyond ({pct(beyond,total_depth_nodes):.2f}%)")

    if args.out_json:
        ser = {
            "root": args.root, "cap": CAP, "drop_classes": sorted(DROP),
            "totals": {"files": tot_files, "kept": tot_kept, "dropped": tot_dropped},
            "by_celltype": {ct: {"total": grand_ct_all[ct],
                                 "rare": grand_ct_dropped_rare.get(ct, 0),
                                 "soma_gt_cap": grand_ct_dropped_cap.get(ct, 0)}
                            for ct in grand_ct_all},
            "soma_deg_dist": dict(grand_soma_deg),
            "prune_child_dist": dict(grand_prune_child),
        }
        with open(args.out_json, "w") as f:
            json.dump(ser, f, indent=2)
        print(f"\nWrote JSON to {args.out_json}")


if __name__ == "__main__":
    main()
