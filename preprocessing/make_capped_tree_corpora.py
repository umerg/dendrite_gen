#!/usr/bin/env python
"""Derive node-count-capped, sample-matched duplicates of the tree corpora.

SemlaFlow's activation memory is O(N^2) (~57 KB per node-pair in fp32, 28.5 KB in bf16,
4.0 KB with gradient checkpointing), so the long node-count tail of `trees_genus_d15` and
`trees_genus_d20` (max 1456 and 3056 nodes) is what forces bf16 + gradient checkpointing and
puts d20 at ~38.5 GB of a 40 GB card. The tail is cheap to give up: the top 5% of trees hold
~18% of the nodes but ~39% of the N^2 compute, and 96% of them are Fagus -- the genus that is
already 88.2% of the corpus.

This script drops every tree whose node count at ONE reference depth (`--cap-source`, i.e. d20)
exceeds `--node-cap`, and removes that same treeID set from ALL depths. Applying one shared ID
set is what keeps the capped corpora sample-matched, so a d10-vs-d15-vs-d20 comparison stays a
depth ablation rather than a depth-plus-composition one. It is free to do: d15's own tail above
785 nodes is a strict subset of d20's tail above 1110.

The source corpora are opened read-only and never written to; each output directory is a sibling
of its source carrying `--suffix`. See docs/TREE_DATASET_STATS.md for the loss accounting and
semla-flow/RUN.md section 3a for the training flags the cap buys.

Usage:
    python preprocessing/make_capped_tree_corpora.py                       # d10/d15/d20 @ 1110
    python preprocessing/make_capped_tree_corpora.py --dry-run
    python preprocessing/make_capped_tree_corpora.py --force               # rebuild in place
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data_loading import TREE_GENUS_NAMES  # noqa: E402  single source of truth for ids

SCRIPT_VERSION = 1
SPLITS = ("train", "val", "test")
GENUS_IDS = {name: i for i, name in enumerate(TREE_GENUS_NAMES)}


# --------------------------------------------------------------------------------------- read


def read_stats(root: Path) -> tuple[dict[str, dict], list[str]]:
    """Parse `<root>/dataset_stats.csv` into {treeID: row}, preserving the field order."""
    path = root / "dataset_stats.csv"
    with path.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    stats = {r["treeID"]: r for r in rows}
    if len(stats) != len(rows):
        raise SystemExit(f"{path}: duplicate treeIDs")
    return stats, fields


def list_swc(d: Path) -> set[str]:
    """Stems of the real SWC files in `d`, applying prepare_tree_dataset.py's `._` skip."""
    return {
        f.stem for f in d.iterdir()
        if f.is_file() and f.name.endswith(".swc") and not f.name.startswith("._")
    }


def scan_swc(path: Path) -> tuple[dict[str, str], int, np.ndarray]:
    """One read-only pass: header comments, node count, and xyz coordinates."""
    header: dict[str, str] = {}
    coords: list[tuple[float, float, float]] = []
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                body = line[1:].strip()
                if body.startswith("cell_class "):
                    header["cell_class"] = body.split(None, 1)[1]
                elif body.startswith("cell_type "):
                    header["cell_type"] = body.split(None, 1)[1]
                elif body.startswith("tree_id "):
                    header["tree_id"] = body.split(None, 1)[1]
                else:
                    m = re.search(r"max_depth=(\d+)", body)
                    if m:
                        header["max_depth"] = m.group(1)
                continue
            if not line.strip():
                continue
            parts = line.split()
            coords.append((float(parts[2]), float(parts[3]), float(parts[4])))
    return header, len(coords), np.asarray(coords, dtype=np.float32)


def coord_std(per_tree: list[np.ndarray]) -> float:
    """Std over per-tree zero-CoM'd coords -- byte-for-byte the arithmetic of
    semlaflow.preprocess_neurons._coord_std, so the value can be pasted into DATASET_CONFIGS."""
    centred = [c - c.mean(axis=0, keepdims=True) for c in per_tree]
    return float(np.concatenate(centred, axis=0).std())


# ------------------------------------------------------------------------------------- verify


def preflight(sources: list[Path], cap_source: Path) -> tuple[dict[str, dict], list[str]]:
    """Assert the sources are the same corpus at different depths. Nothing is written yet."""
    cap_stats, cap_fields = read_stats(cap_source)
    for src in sources:
        if not src.is_dir():
            raise SystemExit(f"missing source corpus: {src}")
        stats, _ = read_stats(src)
        if set(stats) != set(cap_stats):
            raise SystemExit(f"{src}: treeID set differs from {cap_source}")
        for tid, row in stats.items():
            if row["split"] != cap_stats[tid]["split"]:
                raise SystemExit(f"{src}: split for {tid} differs from {cap_source}")
            if row["genus"] != cap_stats[tid]["genus"]:
                raise SystemExit(f"{src}: genus for {tid} differs from {cap_source}")
        for split in SPLITS:
            d = src / split
            if not d.is_dir():
                raise SystemExit(f"missing split dir: {d}")
            on_disk = list_swc(d)
            in_csv = {t for t, r in stats.items() if r["split"] == split}
            if on_disk != in_csv:
                raise SystemExit(
                    f"{d}: disk/CSV mismatch -- {len(on_disk - in_csv)} disk-only, "
                    f"{len(in_csv - on_disk)} csv-only"
                )
        print(f"  [ok] {src.name}: 3 splits, {len(stats)} trees, disk == dataset_stats.csv")
    return cap_stats, cap_fields


def git_provenance(repo: Path) -> dict:
    def run(*args):
        try:
            return subprocess.run(
                ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:  # noqa: BLE001  provenance is best-effort
            return ""
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


# -------------------------------------------------------------------------------------- build


def build_one(src: Path, dest: Path, keep: list[str], drop: set[str],
              cap_source: Path, node_cap: int, dry_run: bool) -> dict:
    """Copy the kept SWCs of one depth into `dest` and return its manifest."""
    depth_m = re.search(r"_d(\d+)$", src.name)
    if not depth_m:
        raise SystemExit(f"cannot infer depth from source name: {src.name}")
    depth = int(depth_m.group(1))

    stats, fields = read_stats(src)
    if not dry_run:
        for split in SPLITS:
            (dest / split).mkdir(parents=True, exist_ok=True)

    per_split_n: Counter[str] = Counter()
    per_split_max: dict[str, int] = {}
    genus_by_split: dict[str, Counter[str]] = {s: Counter() for s in SPLITS}
    train_coords: list[np.ndarray] = []
    all_nodes: list[int] = []

    for tid in keep:
        row = stats[tid]
        split = row["split"]
        s_path = src / split / f"{tid}.swc"
        header, n_nodes, coords = scan_swc(s_path)

        # Every depth holds files with identical names, so a loop bug that copied d20 files
        # into d10_capped would only surface much later as a wrong coord_std. These four
        # asserts are the cheap net for that.
        if header.get("tree_id") != tid:
            raise SystemExit(f"{s_path}: header tree_id {header.get('tree_id')!r} != filename")
        if header.get("max_depth") != str(depth):
            raise SystemExit(
                f"{s_path}: header max_depth={header.get('max_depth')} != {depth} "
                f"-- wrong source corpus"
            )
        if n_nodes != int(row["nodes"]):
            raise SystemExit(f"{s_path}: {n_nodes} nodes != dataset_stats {row['nodes']}")
        if int(header.get("cell_class", -1)) != GENUS_IDS[row["genus"]]:
            raise SystemExit(f"{s_path}: cell_class != id of genus {row['genus']}")

        if not dry_run:
            shutil.copy2(s_path, dest / split / f"{tid}.swc")

        per_split_n[split] += 1
        per_split_max[split] = max(per_split_max.get(split, 0), n_nodes)
        genus_by_split[split][row["genus"]] += 1
        all_nodes.append(n_nodes)
        if split == "train":
            train_coords.append(coords)

    cs = coord_std(train_coords)
    max_all = max(per_split_max.values())
    manifest = {
        "schema": SCRIPT_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "preprocessing/make_capped_tree_corpora.py",
        "git": git_provenance(Path(__file__).resolve().parents[1]),
        "rule": {
            "cap_source": str(cap_source),
            "column": "nodes",
            "op": ">",
            "node_cap": node_cap,
            "note": "the same treeID set is dropped from every depth, keeping them sample-matched",
        },
        "source": {"path": str(src), "depth": depth, "n_trees": len(stats)},
        "dropped": {"n": len(drop)},
        "kept": {
            "n": len(keep),
            "by_split": dict(per_split_n),
            "genus_by_split": {s: dict(genus_by_split[s]) for s in SPLITS},
        },
        "node_counts": {
            "max_all_splits": max_all,
            "max_by_split": per_split_max,
            "total": int(sum(all_nodes)),
            "median": int(np.median(all_nodes)),
        },
        "coord_std_train": round(cs, 4),
        "for_registry": {"coord_std": round(cs, 4), "max_nodes": max_all},
        "for_cli": {"preprocess_max_atoms": max_all, "train_max_atoms": max_all + 1},
    }

    if not dry_run:
        kept_ids = set(keep)
        with (dest / "dataset_stats.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for tid, row in stats.items():          # source row order, kept rows only
                if tid in kept_ids:
                    writer.writerow(row)
        (dest / "BUILD_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# --------------------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Derive node-count-capped, sample-matched duplicates of the tree corpora.")
    ap.add_argument("--src-parent", type=Path, default=Path("/Users/umer/Documents"))
    ap.add_argument("--prefix", default="trees_genus_d")
    ap.add_argument("--depths", type=int, nargs="+", default=[10, 15, 20])
    ap.add_argument("--cap-depth", type=int, default=20,
                    help="Depth whose node counts define the drop set (default 20).")
    ap.add_argument("--node-cap", type=int, default=1110,
                    help="Drop trees with more than this many nodes AT --cap-depth.")
    ap.add_argument("--suffix", default="_capped")
    ap.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    ap.add_argument("--force", action="store_true",
                    help="Rebuild even if a destination exists (the old tree is moved aside).")
    ap.add_argument("--expect-kept", type=int, default=None)
    ap.add_argument("--expect-dropped", type=int, default=None)
    args = ap.parse_args()

    sources = [args.src_parent / f"{args.prefix}{d}" for d in args.depths]
    cap_source = args.src_parent / f"{args.prefix}{args.cap_depth}"
    if cap_source not in sources:
        raise SystemExit(f"--cap-depth {args.cap_depth} is not among --depths {args.depths}")

    print(f"pre-flight ({len(sources)} corpora)")
    cap_stats, _ = preflight(sources, cap_source)

    drop = {t for t, r in cap_stats.items() if int(r["nodes"]) > args.node_cap}
    keep = sorted(set(cap_stats) - drop)
    if not drop:
        raise SystemExit(f"node cap {args.node_cap} drops nothing at d{args.cap_depth}")
    if args.expect_dropped is not None and len(drop) != args.expect_dropped:
        raise SystemExit(f"dropped {len(drop)} != --expect-dropped {args.expect_dropped}")
    if args.expect_kept is not None and len(keep) != args.expect_kept:
        raise SystemExit(f"kept {len(keep)} != --expect-kept {args.expect_kept}")

    d_split = Counter(cap_stats[t]["split"] for t in drop)
    d_genus = Counter(cap_stats[t]["genus"] for t in drop)
    print(f"\nrule: drop d{args.cap_depth} nodes > {args.node_cap}"
          f"  ->  {len(drop)} dropped, {len(keep)} kept")
    print(f"  dropped by split: {dict(d_split)}")
    print(f"  dropped by genus: {dict(d_genus)}")

    # Destination safety: never inside a source, never a source itself.
    dests = []
    for src in sources:
        dest = src.parent / f"{src.name}{args.suffix}"
        rs, rd = src.resolve(), dest.resolve()
        if rd == rs or rs in rd.parents or rd in rs.parents:
            raise SystemExit(f"refusing: destination {dest} overlaps source {src}")
        if dest.exists():
            if not args.force:
                raise SystemExit(f"{dest} already exists -- pass --force to rebuild")
            if (dest / "smol").exists() and not args.dry_run:
                raise SystemExit(
                    f"STALE: {dest / 'smol'} was built from the previous contents. Delete it "
                    f"and re-run semlaflow.preprocess_neurons after this rebuild."
                )
        dests.append(dest)

    manifests = {}
    for src, dest in zip(sources, dests):
        print(f"\nbuilding {dest.name}")
        if dest.exists() and args.force and not args.dry_run:
            aside = dest.with_name(f"{dest.name}.bak-{int(datetime.now().timestamp())}")
            dest.rename(aside)
            print(f"  moved previous tree aside -> {aside.name}")
        m = build_one(src, dest, keep, drop, cap_source, args.node_cap, args.dry_run)
        manifests[dest.name] = m
        nc, ks = m["node_counts"], m["kept"]
        print(f"  kept {ks['n']} {ks['by_split']}")
        print(f"  max N {nc['max_by_split']} -> {nc['max_all_splits']}  median {nc['median']}"
              f"  total {nc['total']:,}")
        print(f"  coord_std={m['coord_std_train']:.4f}  "
              f"preprocess --max_atoms {m['for_cli']['preprocess_max_atoms']}  "
              f"train --max_atoms {m['for_cli']['train_max_atoms']}")

    if not args.dry_run:
        for dest in dests:
            lines = [f"# dropped: d{args.cap_depth} nodes > {args.node_cap}; n={len(drop)}",
                     "treeID,genus,split,nodes_at_cap_source"]
            lines += [f"{t},{cap_stats[t]['genus']},{cap_stats[t]['split']},"
                      f"{cap_stats[t]['nodes']}" for t in sorted(drop)]
            (dest / "DROPPED_IDS.csv").write_text("\n".join(lines) + "\n")

    # Sample-matched invariant: identical kept sets and genus tables at every depth.
    ref = manifests[dests[0].name]["kept"]
    for name, m in manifests.items():
        if m["kept"] != ref:
            raise SystemExit(f"{name}: kept counts differ across depths -- not sample-matched")
    print(f"\nsample-matched across {len(dests)} depths: {ref['n']} trees {ref['by_split']}")
    print("DRY RUN -- nothing written" if args.dry_run else "done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
