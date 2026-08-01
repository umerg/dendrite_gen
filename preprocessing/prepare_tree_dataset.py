#!/usr/bin/env python3
"""Produce the trainable, genus-labelled biological-tree datasets `trees_genus_d<CAP>`.

Input:  BioDiv-3DTrees `Graphs.tar.zst` (3,386 `Graph/<treeID>_cor_graph.graphml` QSM graphs)
        + `labels.csv` (species + official 80/10/10 species-stratified split).
Output: trees_genus_d<CAP>/{train,val,test}/<treeID>.swc  (base-rooted, strictly binary,
        depth-capped, radii kept, genus integer embedded as a header comment).

The raw graphs are FULL CYLINDER RESOLUTION (mean ~14.9k nodes/tree, ~66% of them degree-2
chain nodes) and undirected with no parent pointers, so this script must root and orient
them itself. Only broadleaf trees have QSM reconstructions, so the labels' broadleaf/conifer
`type` column carries no signal -- `species` -> genus is the class axis.

Per tree, reusing preprocessing/clean_trees.py:
  1. Root at the QSM base cylinder (graphml node '0') and BFS-orient to get parent pointers.
  2. Look up genus + split from labels.csv. Drop the rare tail genera (Tilia, Prunus, Ulmus)
     and the handful of trees with a blank split.
  3. clean_swc_tree(root_mode="parent"): collapse degree-2 chains, trim to `max_depth`,
     binarize (3-child = lossless insert; >=4-child = keep the 2 THICKEST, delete the rest),
     final re-collapse.
  4. Write with a '# cell_class N' header.

`keep_attrs=True` matters: it carries the real radii through the FIRST collapse so that the
">=4-child keeps the 2 thickest" prune is a genuine morphological choice. The legacy
`small_trees` set used --drop-attrs, which zeroes every radius to 1.0 before that prune runs,
making the surviving pair arbitrary node order. Radii are inert downstream (nx_graph_to_adj_pos
keeps only positions + adjacency), exactly as for neurons_conditional_full.

All requested depth caps are produced in a SINGLE pass over the archive -- parsing the graphml
dominates the runtime, so the per-cap cleaning is amortised.

Usage:
  conda run -n NEURO2 python preprocessing/prepare_tree_dataset.py --max-depth 10 15 20
  conda run -n NEURO2 python preprocessing/prepare_tree_dataset.py --max-depth 10 --limit 50 --dry-run
  conda run -n NEURO2 python preprocessing/prepare_tree_dataset.py --verify /Users/umer/Documents/trees_genus_d10
"""
from __future__ import annotations

import argparse
import io
import sys
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd

# Import clean_trees helpers regardless of how this script is launched.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
from clean_trees import (  # noqa: E402
    SWC_COLS,
    SWCTree,
    clean_swc_tree,
    collapse_degree2,
    trim_to_max_depth,
    write_swc,
)
from utils.data_loading import TREE_GENUS_NAMES  # noqa: E402

# Genus-integer mapping, frequency-ordered. Derived from the canonical TREE_GENUS_NAMES so
# the writer and the per-class metrics never desync (mirrors prepare_conditional_dataset.py).
CLASS_MAP = {name: idx for idx, name in enumerate(TREE_GENUS_NAMES)}
# Rare tail genera, dropped: Tilia 13, Prunus 2, Ulmus 1 trees (0.47% of the graph corpus).
DROP_GENERA = {"Tilia", "Prunus", "Ulmus"}

ROOT_PARENT = -1  # root parent sentinel written in the SWC (matches prepare_conditional_dataset)
# The model encodes each root child's sibling rank as a one-hot of width MAX_CHILDREN, so a
# root out-degree above it would collide. Trees fork at the base far less than somata do
# (max observed: 3), so this is only a guard -- imported rather than copied to stay in lockstep.
from graph_generation.method.expansion import MAX_CHILDREN as MAX_ROOT_CHILDREN  # noqa: E402
GRAPH_SUFFIX = "_cor_graph.graphml"
# labels.csv spells the validation split "validate"; the loader expects a `val/` directory.
SPLIT_MAP = {"train": "train", "validate": "val", "test": "test"}


# ---------- graphml reading ----------

def parse_graphml(data: bytes):
    """Lean reader for the QSM graphml: returns (node_ids, xyz[N,3], radius[N], edges).

    networkx.read_graphml is several times slower on these files (type coercion + graph
    construction) and we only need four node attributes, so parse the XML directly. The
    `for="node"` key declarations are read from the file rather than assumed, since the
    d0..d3 -> attr.name mapping is not guaranteed to be stable across files.
    """
    keymap: dict[str, str] = {}
    ids: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    rs: list[float] = []
    raw_edges: list[tuple[str, str]] = []

    for _event, elem in ET.iterparse(io.BytesIO(data), events=("end",)):
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag == "key":
            if elem.get("for") == "node":
                keymap[elem.get("id")] = elem.get("attr.name")
            elem.clear()
        elif tag == "node":
            vals: dict[str, str] = {}
            for d in elem:
                if d.tag.rsplit("}", 1)[-1] == "data":
                    vals[keymap.get(d.get("key"))] = d.text
            ids.append(elem.get("id"))
            xs.append(float(vals["pos_x"]))
            ys.append(float(vals["pos_y"]))
            zs.append(float(vals["pos_z"]))
            rs.append(float(vals.get("radius", 1.0)))
            elem.clear()
        elif tag == "edge":
            raw_edges.append((elem.get("source"), elem.get("target")))
            elem.clear()

    idx_of = {nid: i for i, nid in enumerate(ids)}
    edges = [(idx_of[u], idx_of[v]) for u, v in raw_edges]
    xyz = np.column_stack([xs, ys, zs]).astype(np.float64)
    return ids, xyz, np.asarray(rs, dtype=np.float64), edges


def orient_from_base(ids, xyz, radius, edges):
    """Root at the QSM base cylinder and BFS-orient into an SWC dataframe.

    The dataset README defines cylinder id 1 -- graphml node '0' -- as the base of the tree,
    so that is the root. A lowest-z heuristic is only a fallback for a graph missing node '0'
    (none in this corpus). The two rules disagree for 24 of the 3,386 graphs, and there the
    lowest-z node is the wrong one: it is an interior branch node (degree 2 or 3) in 17 of
    them, and sits up to 85 m below the trunk base on stray QSM cylinders. Rooting at an
    interior node re-orients every edge on the path to the real base, inverting the topology
    of a large part of the tree.

    Returns (df, info). `info` reports the node count reached from the root (< N means the
    graph is disconnected) and whether the base also happened to be the lowest-z node.
    """
    n = len(xyz)
    try:
        root = ids.index("0")
    except ValueError:
        root = int(np.argmin(xyz[:, 2]))
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    parent = np.full(n, -1, dtype=np.int64)
    seen = np.zeros(n, dtype=bool)
    seen[root] = True
    order = [root]
    dq = deque([root])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                parent[v] = u
                order.append(v)
                dq.append(v)

    new_id = {old: i + 1 for i, old in enumerate(order)}
    rows = [
        {
            "id": new_id[o],
            "type": 3,
            "x": xyz[o, 0],
            "y": xyz[o, 1],
            "z": xyz[o, 2],
            "radius": radius[o],
            "parent": ROOT_PARENT if o == root else new_id[parent[o]],
        }
        for o in order
    ]
    df = pd.DataFrame(rows, columns=SWC_COLS)
    info = {
        "n_raw": n,
        "n_reached": len(order),
        "base_is_lowest_z": root == int(np.argmin(xyz[:, 2])),
        "n_zero_radius": int((radius <= 0.0).sum()),
    }
    return df, info


# ---------- exact loss counters ----------

def _subtree_size(tree: SWCTree, v: int) -> int:
    n = 0
    stack = [v]
    while stack:
        w = stack.pop()
        n += 1
        stack.extend(tree.children.get(w, []))
    return n


def measure_prune(tree: SWCTree, root_parent_value: int) -> dict:
    """Replicate `normalize_high_degree`'s prune pass on `tree`, counting exactly what it loses.

    MUTATES `tree` (it performs the same deletions), so only ever call this on a throwaway
    copy. Counting analytically on the untrimmed skeleton instead would over-count, because
    multifurcations that sit inside a deleted subtree are never reached by the real pass --
    hence this replays the same iteration order and the same dynamic `len(kids) <= 3` recheck
    that `clean_trees` uses, then counts the 3-child splits that survive the pruning.
    """
    rp = root_parent_value
    nodes_deleted = 0
    branches_deleted = 0
    lossy_prunes = 0

    candidates = [u for u in list(tree.nodes.keys())
                  if (tree.nodes[u]["parent"] > rp and len(tree.children.get(u, [])) >= 3)]
    for u in candidates:
        kids = list(tree.children.get(u, []))
        if len(kids) <= 3:
            continue  # already pruned away, or only a 3-way split (handled losslessly)
        lossy_prunes += 1
        kids_sorted = sorted(kids, key=lambda c: tree.nodes[c]["radius"], reverse=True)
        keep = set(kids_sorted[:2])
        for c in kids:
            if c not in keep:
                branches_deleted += 1
                nodes_deleted += _subtree_size(tree, c)
                tree.delete_subtree(c)

    # Lossless 3-child splits, counted AFTER pruning (each inserts exactly one node).
    lossless_splits = sum(
        1 for u in list(tree.nodes.keys())
        if tree.nodes[u]["parent"] > rp and len(tree.children.get(u, [])) == 3
    )
    return {
        "nodes_deleted": nodes_deleted,
        "branches_deleted": branches_deleted,
        "lossy_prunes": lossy_prunes,
        "lossless_splits": lossless_splits,
    }


def swc_shape(df):
    """(node count, max depth, root out-degree, #non-root multifurcations, #degree-2, #roots).

    `n_multi` counts NON-ROOT multifurcations only, which must be 0: `normalize_high_degree`
    deliberately skips the root (its `parent > rp` filter), exactly as the neuron pipeline
    leaves a soma's primary dendrites alone. A tree that forks at the trunk base therefore
    keeps a root out-degree of 2 or 3 -- reported via `root_deg`, and handled downstream by
    the MAX_CHILDREN-wide root-child ordinal one-hot.
    """
    par = dict(zip(df["id"].astype(int), df["parent"].astype(int)))
    kids: dict[int, list[int]] = {}
    roots = []
    for nid, p in par.items():
        if p <= 0:
            roots.append(nid)
        else:
            kids.setdefault(p, []).append(nid)
    max_depth = 0
    for r in roots:
        dq = deque([(r, 0)])
        while dq:
            u, d = dq.popleft()
            max_depth = max(max_depth, d)
            for c in kids.get(u, []):
                dq.append((c, d + 1))
    rootset = set(roots)
    n_multi = sum(1 for u, ch in kids.items() if len(ch) > 2 and u not in rootset)
    n_deg2 = sum(1 for u, ch in kids.items() if len(ch) == 1 and u not in rootset)
    root_deg = len(kids.get(roots[0], [])) if len(roots) == 1 else -1
    return len(par), max_depth, root_deg, n_multi, n_deg2, len(roots)


# ---------- labels ----------

def load_labels(csv_path: Path) -> dict[str, dict]:
    df = pd.read_csv(csv_path)
    df["treeID"] = df["treeID"].astype(str)
    out = {}
    for r in df.itertuples(index=False):
        species = str(r.species)
        out[r.treeID] = {
            "species": species,
            "genus": species.split("_")[0],
            "split": SPLIT_MAP.get(str(r.split), None),
        }
    return out


# ---------- archive iteration ----------

def iter_graphs(archive: Path | None, graph_dir: Path | None):
    """Yield (treeID, raw_bytes). Streams the .tar.zst (never extracts ~15 GB to disk)."""
    if graph_dir is not None:
        for p in sorted(graph_dir.iterdir()):
            if p.is_file() and p.name.endswith(GRAPH_SUFFIX):
                yield p.name[: -len(GRAPH_SUFFIX)], p.read_bytes()
        return

    import zstandard

    dctx = zstandard.ZstdDecompressor()
    with open(archive, "rb") as fh, dctx.stream_reader(fh) as reader:
        with tarfile.open(mode="r|", fileobj=reader) as tf:
            for member in tf:
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if not name.endswith(GRAPH_SUFFIX):
                    continue
                buf = tf.extractfile(member)
                if buf is None:
                    continue
                yield name[: -len(GRAPH_SUFFIX)], buf.read()


# ---------- main build ----------

def build(args) -> int:
    labels = load_labels(args.labels)
    caps = sorted(set(args.max_depth))
    out_dirs = {c: args.out_parent / f"{args.prefix}{c}" for c in caps}

    print(f"genus mapping: {CLASS_MAP}")
    print(f"dropped genera: {sorted(DROP_GENERA)}   root parent sentinel: {ROOT_PARENT}")
    print(f"caps: {caps}")
    for c in caps:
        print(f"  d{c} -> {out_dirs[c]}")
    print(f"labels: {args.labels}  ({len(labels)} rows)")
    print(f"source: {args.graph_dir or args.archive}   dry_run={args.dry_run}\n")

    if not args.dry_run:
        for c in caps:
            for sp in ("train", "val", "test"):
                (out_dirs[c] / sp).mkdir(parents=True, exist_ok=True)

    tally = Counter()
    kept_by_genus = Counter()
    dropped_by_genus = Counter()
    split_genus = Counter()
    per_tree: dict[int, list[dict]] = {c: [] for c in caps}
    raw_rows: list[dict] = []
    identity_mismatch = 0

    for i, (tree_id, data) in enumerate(iter_graphs(args.archive, args.graph_dir)):
        if args.limit and i >= args.limit:
            break
        tally["seen"] += 1
        if tally["seen"] % 250 == 0:
            print(f"  ... {tally['seen']} graphs processed", flush=True)

        meta = labels.get(tree_id)
        if meta is None:
            tally["drop_unlabelled"] += 1
            continue
        genus = meta["genus"]
        if genus in DROP_GENERA:
            tally["drop_rare_genus"] += 1
            dropped_by_genus[genus] += 1
            continue
        if genus not in CLASS_MAP:
            tally["drop_unknown_genus"] += 1
            dropped_by_genus[genus] += 1
            continue
        if meta["split"] is None:
            tally["drop_no_split"] += 1
            dropped_by_genus[f"{genus} (no split)"] += 1
            continue

        try:
            ids, xyz, radius, edges = parse_graphml(data)
            df_raw, info = orient_from_base(ids, xyz, radius, edges)
        except Exception as exc:  # noqa: BLE001
            tally["errors"] += 1
            print(f"[FAIL] {tree_id}: {type(exc).__name__}: {exc}")
            continue

        if info["n_reached"] != info["n_raw"]:
            tally["drop_disconnected"] += 1
            print(f"[SKIP] {tree_id}: disconnected "
                  f"({info['n_reached']}/{info['n_raw']} nodes reached from base)")
            continue
        if not info["base_is_lowest_z"]:
            tally["base_not_lowest_z"] += 1

        # Degree-2 collapse once per tree (cap-independent, and the expensive step): gives the
        # full branching skeleton the depth cap is applied to, and the base for the exact
        # per-cap loss counters below.
        skel_df = collapse_degree2(
            SWCTree(df_raw, root_parent_value=ROOT_PARENT), keep_attrs=True,
            root_parent_value=ROOT_PARENT, keep_parent_value=ROOT_PARENT,
        )
        n_skel, skel_depth = swc_shape(skel_df)[:2]
        raw_rows.append({
            "treeID": tree_id, "genus": genus, "split": meta["split"],
            "n_raw": info["n_raw"], "n_skeleton": n_skel,
            "skeleton_max_depth": skel_depth,
            "n_zero_radius": info["n_zero_radius"],
        })

        ok = True
        cleaned: dict[int, pd.DataFrame] = {}
        for c in caps:
            try:
                cleaned[c] = clean_swc_tree(
                    df_raw.copy(), root_parent_value=ROOT_PARENT, keep_parent_value=ROOT_PARENT,
                    max_depth=c, keep_attrs=True, root_mode="parent",
                )
            except Exception as exc:  # noqa: BLE001
                ok = False
                tally["errors"] += 1
                print(f"[FAIL] {tree_id} @cap{c}: {type(exc).__name__}: {exc}")
                break
        if not ok:
            continue

        tally["written"] += 1
        kept_by_genus[genus] += 1
        split_genus[(meta["split"], genus)] += 1

        for c in caps:
            df_c = cleaned[c]
            n, md, rdeg, n_multi, n_deg2, n_roots = swc_shape(df_c)
            # Throwaway trimmed copy, measured then discarded (measure_prune mutates it).
            t = SWCTree(skel_df, root_parent_value=ROOT_PARENT)
            trim_to_max_depth(t, max_depth=c)
            n_skel_cap = len(t.nodes)
            ctr = measure_prune(t, ROOT_PARENT)
            # Self-check that the measurement path agrees with clean_swc_tree's output:
            # final == capped_skeleton - deleted + inserted (pruning removes whole subtrees
            # and never creates new degree-2 nodes, so nothing else moves the count).
            if n_skel_cap - ctr["nodes_deleted"] + ctr["lossless_splits"] != n:
                identity_mismatch += 1
            per_tree[c].append({
                "treeID": tree_id, "genus": genus, "split": meta["split"],
                "nodes": n, "max_depth": md, "root_deg": rdeg,
                "n_multi": n_multi, "n_deg2": n_deg2, "n_roots": n_roots,
                "n_raw": info["n_raw"], "n_skeleton": n_skel,
                "skeleton_max_depth": skel_depth,
                "n_skel_cap": n_skel_cap,
                "lossless_splits": ctr["lossless_splits"],
                "lossy_prunes": ctr["lossy_prunes"],
                "branches_deleted": ctr["branches_deleted"],
                "nodes_deleted": ctr["nodes_deleted"],
            })
            if not args.dry_run:
                write_swc(
                    df_c, out_dirs[c] / meta["split"] / f"{tree_id}.swc",
                    root_parent_value=ROOT_PARENT,
                    header_lines=[
                        f"cleaned by prepare_tree_dataset.py: base-rooted, binarized, "
                        f"max_depth={c}, radii kept",
                        f"cell_class {CLASS_MAP[genus]}",
                        f"cell_type {genus}",
                        f"species {meta['species']}",
                        f"tree_id {tree_id}",
                    ],
                )

    # ---------- report ----------
    print("\n" + "=" * 78)
    print(f"seen={tally['seen']}  WRITTEN={tally['written']}  "
          f"drop_rare_genus={tally['drop_rare_genus']}  drop_no_split={tally['drop_no_split']}  "
          f"drop_unlabelled={tally['drop_unlabelled']}  drop_unknown={tally['drop_unknown_genus']}  "
          f"drop_disconnected={tally['drop_disconnected']}  errors={tally['errors']}")
    print(f"QSM base (node '0') is not the lowest-z node: {tally['base_not_lowest_z']}  "
          f"(informational; node '0' is authoritative)   "
          f"node-identity mismatches: {identity_mismatch}")
    print("=" * 78)

    print(f"\n{'genus':<12}{'id':>4}{'kept':>8}")
    for gname, gid in sorted(CLASS_MAP.items(), key=lambda kv: kv[1]):
        print(f"{gname:<12}{gid:>4}{kept_by_genus.get(gname, 0):>8}")
    print(f"dropped: {dict(dropped_by_genus)}")

    if raw_rows:
        rdf = pd.DataFrame(raw_rows)
        print(f"\nraw graphs: nodes mean {rdf.n_raw.mean():.0f} median {rdf.n_raw.median():.0f} "
              f"max {rdf.n_raw.max()} | collapsed skeleton mean {rdf.n_skeleton.mean():.0f} "
              f"median {rdf.n_skeleton.median():.0f} max {rdf.n_skeleton.max()} "
              f"| skeleton depth mean {rdf.skeleton_max_depth.mean():.1f} "
              f"max {rdf.skeleton_max_depth.max()}")
        print(f"zero-radius raw nodes: {int(rdf.n_zero_radius.sum())} "
              f"({rdf.n_zero_radius.gt(0).sum()} trees affected)")

    for c in caps:
        df = pd.DataFrame(per_tree[c])
        if df.empty:
            continue
        print("\n" + "-" * 78)
        print(f"### cap {c}  ({len(df)} trees)")
        print(f"  nodes      mean {df.nodes.mean():7.1f}  median {df.nodes.median():6.0f}  "
              f"p95 {df.nodes.quantile(.95):6.0f}  max {df.nodes.max():6.0f}")
        print(f"  max_depth  mean {df.max_depth.mean():7.1f}  median {df.max_depth.median():6.0f}  "
              f"p95 {df.max_depth.quantile(.95):6.0f}  max {df.max_depth.max():6.0f}")
        print(f"  structure  non-root multifurcations {int(df.n_multi.sum())}  "
              f"degree-2 {int(df.n_deg2.sum())}  n_roots!=1 {(df.n_roots != 1).sum()}")
        print(f"  root degree      " + "  ".join(
            f"{k}:{v}" for k, v in sorted(df.root_deg.value_counts().items()))
            + "   (root is exempt from binarization, as the soma is for neurons)")
        tot_skel = int(df.n_skeleton.sum())
        tot_cap = int(df.n_skel_cap.sum())
        tot_del = int(df.nodes_deleted.sum())
        tot_ins = int(df.lossless_splits.sum())
        print(f"  depth-cap loss   {tot_skel - tot_cap:>9} / {tot_skel} skeleton nodes = "
              f"{100 * (tot_skel - tot_cap) / max(tot_skel, 1):.2f}%")
        print(f"  prune loss       {tot_del:>9} nodes = "
              f"{100 * tot_del / max(int(df.nodes.sum()), 1):.2f}% of kept nodes; "
              f"{int(df.branches_deleted.sum())} branches at {int(df.lossy_prunes.sum())} junctions")
        print(f"  lossless splits  {tot_ins:>9} inserted nodes "
              f"({100 * tot_ins / max(tot_ins + int(df.lossy_prunes.sum()), 1):.1f}% of multifurcations)")
        print(f"  split counts     " + "  ".join(
            f"{k}={v}" for k, v in sorted(df.split.value_counts().items())))
        if not args.dry_run:
            stats_csv = out_dirs[c] / "dataset_stats.csv"
            df.to_csv(stats_csv, index=False)
            print(f"  per-tree stats -> {stats_csv}")

    print("\n--- split x genus (kept) ---")
    tbl = pd.DataFrame(
        [{"split": s, "genus": g, "n": n} for (s, g), n in split_genus.items()]
    )
    if not tbl.empty:
        piv = tbl.pivot_table(index="genus", columns="split", values="n",
                              aggfunc="sum", fill_value=0)
        cols = [c for c in ("train", "val", "test") if c in piv.columns]
        piv = piv.reindex([g for g in TREE_GENUS_NAMES if g in piv.index])[cols]
        piv["total"] = piv.sum(axis=1)
        piv.loc["TOTAL"] = piv.sum()
        print(piv.to_string())
    return 0


# ---------- verify ----------

def verify(root: Path) -> int:
    """Structural re-check of a written dataset (mirrors dataset_loss_accounting.py's role)."""
    from utils.data_loading import load_swc_graph
    import networkx as nx

    print(f"verifying {root}\n" + "=" * 78)
    bad = Counter()
    per_split = {}
    genus_counts = Counter()
    root_degs = Counter()
    depths = []
    nodes = []
    for sp in ("train", "val", "test"):
        d = root / sp
        if not d.is_dir():
            print(f"[MISS] {d}")
            bad["missing_split"] += 1
            continue
        files = [f for f in sorted(d.iterdir())
                 if f.is_file() and f.name.endswith(".swc") and not f.name.startswith("._")]
        per_split[sp] = len(files)
        for f in files:
            try:
                G = load_swc_graph(f)
            except Exception as exc:  # noqa: BLE001
                bad["load_error"] += 1
                print(f"[FAIL] {f.name}: {exc}")
                continue
            cc = G.graph.get("cell_class")
            if cc is None or not (0 <= int(cc) < len(TREE_GENUS_NAMES)):
                bad["bad_cell_class"] += 1
            else:
                genus_counts[TREE_GENUS_NAMES[int(cc)]] += 1
            if not nx.is_tree(G):
                bad["not_tree"] += 1
            r = G.graph["root"]
            deg = {n: G.degree(n) for n in G.nodes}
            # child counts: degree minus the edge to the parent (root has no parent). The root
            # is exempt from binarization (as a neuron soma is), so it is excluded here and
            # tracked via root_degree instead.
            n_multi = sum(1 for n in G.nodes if n != r and (deg[n] - 1) > 2)
            n_deg2 = sum(1 for n in G.nodes if n != r and deg[n] == 2)
            if n_multi:
                bad["nonroot_multifurcation"] += 1
            if n_deg2:
                bad["degree2"] += 1
            if deg[r] > MAX_ROOT_CHILDREN:
                bad["root_deg_over_cap"] += 1
            root_degs[deg[r]] += 1
            depths.append(max(nx.shortest_path_length(G, r).values()))
            nodes.append(G.number_of_nodes())

    print(f"files: {per_split}  total={sum(per_split.values())}")
    print(f"nodes: mean {np.mean(nodes):.1f} median {np.median(nodes):.0f} "
          f"p95 {np.percentile(nodes, 95):.0f} max {max(nodes)}")
    print(f"depth: mean {np.mean(depths):.1f} median {np.median(depths):.0f} "
          f"p95 {np.percentile(depths, 95):.0f} max {max(depths)}")
    print(f"genus: {dict(sorted(genus_counts.items(), key=lambda kv: -kv[1]))}")
    print(f"root degree: {dict(sorted(root_degs.items()))}  (cap {MAX_ROOT_CHILDREN})")
    if bad:
        print(f"\nPROBLEMS: {dict(bad)}")
    else:
        print(f"\nALL CHECKS PASS: strictly binary away from the root, no degree-2 nodes, "
              f"single root, root degree <= {MAX_ROOT_CHILDREN}, cell_class present and in range.")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build genus-labelled, depth-capped biological-tree SWC datasets.")
    ap.add_argument("--archive", type=Path, default=Path("/Users/umer/Documents/trees/Graphs.tar.zst"),
                    help="BioDiv-3DTrees Graphs.tar.zst (streamed, never fully extracted).")
    ap.add_argument("--graph-dir", type=Path, default=None,
                    help="Alternative to --archive: a directory of already-extracted *_cor_graph.graphml.")
    ap.add_argument("--labels", type=Path, default=Path("/Users/umer/Documents/trees/labels.csv"))
    ap.add_argument("--out-parent", type=Path, default=Path("/Users/umer/Documents"))
    ap.add_argument("--prefix", default="trees_genus_d",
                    help="Output dir name prefix; the depth cap is appended (default trees_genus_d).")
    ap.add_argument("--max-depth", type=int, nargs="+", default=[10, 15, 20],
                    help="Depth caps to build, all in one pass over the archive.")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N graphs (smoke test).")
    ap.add_argument("--dry-run", action="store_true", help="Count and report only; write nothing.")
    ap.add_argument("--verify", type=Path, default=None,
                    help="Skip building; structurally verify an existing output root.")
    args = ap.parse_args()

    if args.verify is not None:
        return verify(args.verify)
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
