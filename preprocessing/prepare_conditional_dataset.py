#!/usr/bin/env python3
"""Produce the trainable, class-labelled neuron dataset `neurons_conditional`.

Input:  swc_simplified/{train,val,test}/*_simplified.swc  (degree-2 collapsed, tip-rooted,
        NOT binarized) + targets_cell_type.csv.
Output: neurons_conditional/{train,val,test}/<id>.swc  (soma-rooted, strictly binary,
        radii/types kept, class integer embedded as a header comment).

Per file, reusing preprocessing/clean_trees.py:
  1. Look up cell type. Drop the rare test-only classes (WM-P, MC, BPC).
  2. clean_swc_tree(root_mode="index"): re-root at the soma (node id 1), collapse degree-2,
     binarize keeping the 2 THICKEST children (real radii, because keep_attrs=True), no depth cap.
  3. Drop neurons whose soma has > MAX_CHILDREN children (one-hot ordinal limit).
  4. Write with a '# cell_class N' header.

The train/val/test split from the input is preserved (no re-split).

Usage:
  conda run -n NEURO2 python preprocessing/prepare_conditional_dataset.py
  conda run -n NEURO2 python preprocessing/prepare_conditional_dataset.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Import clean_trees helpers regardless of how this script is launched.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
from clean_trees import read_swc, clean_swc_tree, write_swc  # noqa: E402
from utils.data_loading import CELL_CLASS_NAMES  # noqa: E402

# Keep in lockstep with graph_generation/method/expansion.py::MAX_CHILDREN.
MAX_CHILDREN = 16

# Cortical-layer-ordered class ids for the 7 kept pyramidal types. Derived from the
# canonical CELL_CLASS_NAMES so the writer and the per-class metrics never desync.
CLASS_MAP = {name: idx for idx, name in enumerate(CELL_CLASS_NAMES)}
DROP_CLASSES = {"WM-P", "MC", "BPC"}

ROOT_PARENT = -1  # root parent sentinel written in the SWC (matches neurons_final)


def load_celltypes(csv_path: Path) -> dict[str, str]:
    m = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            m[row["filename"]] = row["cell_type"]
    return m


def ctype_of(fname: str, celltype: dict[str, str]) -> str:
    return celltype.get(fname.replace("_simplified.swc", ".swc"), "UNKNOWN")


def root_degree(df_clean) -> int:
    """Number of children of the (single) root after cleaning/re-indexing."""
    root_ids = df_clean.loc[df_clean["parent"] <= ROOT_PARENT, "id"].tolist()
    if len(root_ids) != 1:
        return -1  # unexpected; treat as a structural failure
    return int((df_clean["parent"] == root_ids[0]).sum())


def process_split(split: str, in_dir: Path, out_dir: Path, celltype: dict[str, str],
                  dry_run: bool) -> dict:
    files = sorted(f for f in in_dir.iterdir()
                   if f.is_file() and f.name.endswith("_simplified.swc") and not f.name.startswith("._"))
    tally = {
        "n_in": len(files), "written": 0,
        "dropped_rare_class": 0, "dropped_unknown_class": 0,
        "dropped_soma_gt_cap": 0, "errors": 0,
        "kept_by_class": Counter(), "dropped_gt_cap_by_class": Counter(),
        "rare_by_class": Counter(),
    }
    if not dry_run:
        (out_dir / split).mkdir(parents=True, exist_ok=True)
    for src in files:
        ct = ctype_of(src.name, celltype)
        if ct in DROP_CLASSES:
            tally["dropped_rare_class"] += 1
            tally["rare_by_class"][ct] += 1
            continue
        if ct not in CLASS_MAP:
            tally["dropped_unknown_class"] += 1
            continue
        try:
            df = read_swc(src)
            df_clean = clean_swc_tree(
                df, root_parent_value=ROOT_PARENT, keep_parent_value=ROOT_PARENT,
                max_depth=None, keep_attrs=True, root_mode="index",
            )
            rdeg = root_degree(df_clean)
        except Exception as e:  # noqa: BLE001
            tally["errors"] += 1
            print(f"[FAIL] {split}/{src.name}: {type(e).__name__}: {e}")
            continue
        if rdeg < 0:
            tally["errors"] += 1
            print(f"[FAIL] {split}/{src.name}: no unique root after cleaning")
            continue
        if rdeg > MAX_CHILDREN:
            tally["dropped_soma_gt_cap"] += 1
            tally["dropped_gt_cap_by_class"][ct] += 1
            continue
        tally["written"] += 1
        tally["kept_by_class"][ct] += 1
        if not dry_run:
            dst = out_dir / split / src.name.replace("_simplified.swc", ".swc")
            write_swc(
                df_clean, dst, root_parent_value=ROOT_PARENT,
                header_lines=[
                    "cleaned by prepare_conditional_dataset.py: soma-rooted, binarized, radii kept",
                    f"cell_class {CLASS_MAP[ct]}",
                    f"cell_type {ct}",
                ],
            )
    return tally


def main() -> int:
    ap = argparse.ArgumentParser(description="Clean swc_simplified into class-labelled neurons_conditional.")
    ap.add_argument("--in-root", type=Path,
                    default=Path("/Users/umer/Documents/neurons_simplified/swc_simplified"))
    ap.add_argument("--out-root", type=Path,
                    default=Path("/Users/umer/Documents/neurons_conditional"))
    ap.add_argument("--csv", type=Path, default=Path("/Users/umer/Downloads/targets_cell_type.csv"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--dry-run", action="store_true", help="Count only; write nothing.")
    args = ap.parse_args()

    celltype = load_celltypes(args.csv)
    print(f"class mapping: {CLASS_MAP}   dropped classes: {sorted(DROP_CLASSES)}   cap: {MAX_CHILDREN}")
    print(f"in={args.in_root}  out={args.out_root}  dry_run={args.dry_run}\n")

    grand = defaultdict(int)
    grand_kept = Counter()
    grand_gt = Counter()
    grand_rare = Counter()
    for split in args.splits:
        in_dir = args.in_root / split
        if not in_dir.is_dir():
            print(f"[skip] {in_dir} not found")
            continue
        t = process_split(split, in_dir, args.out_root, celltype, args.dry_run)
        print(f"### {split}: in={t['n_in']}  written={t['written']}  "
              f"drop_rare={t['dropped_rare_class']}  drop_soma>{MAX_CHILDREN}={t['dropped_soma_gt_cap']}  "
              f"drop_unknown={t['dropped_unknown_class']}  errors={t['errors']}")
        for k in ("n_in", "written", "dropped_rare_class", "dropped_unknown_class",
                  "dropped_soma_gt_cap", "errors"):
            grand[k] += t[k]
        grand_kept.update(t["kept_by_class"])
        grand_gt.update(t["dropped_gt_cap_by_class"])
        grand_rare.update(t["rare_by_class"])

    print("\n" + "=" * 70)
    print(f"TOTAL in={grand['n_in']}  WRITTEN={grand['written']}  "
          f"dropped_rare={grand['dropped_rare_class']}  dropped_soma>{MAX_CHILDREN}={grand['dropped_soma_gt_cap']}  "
          f"dropped_unknown={grand['dropped_unknown_class']}  errors={grand['errors']}")
    print("=" * 70)
    print(f"{'type':<8}{'id':>4}{'kept':>8}{'drop_soma>cap':>15}")
    for ct, cid in sorted(CLASS_MAP.items(), key=lambda kv: kv[1]):
        print(f"{ct:<8}{cid:>4}{grand_kept.get(ct,0):>8}{grand_gt.get(ct,0):>15}")
    print(f"rare dropped: {dict(grand_rare)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
