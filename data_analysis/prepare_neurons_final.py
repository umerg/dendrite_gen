#!/usr/bin/env python3
"""Filter neurons by root degree and split into train/val sets.

Drops neurons where root degree > MAX_CHILDREN (10), matching the one-hot
encoding constraint in expansion.py. Copies surviving SWC files into
neurons_final/{train,val}/ with a 90/10 split.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.data_loading import load_swc_graph

MAX_CHILDREN = 10  # from graph_generation/method/expansion.py


def get_swc_files(dir_path: Path) -> list[Path]:
    """Return sorted list of SWC files, matching load_swc_graphs_from_dir filter."""
    files = []
    for f in sorted(dir_path.iterdir()):
        if not f.is_file():
            continue
        if f.name.startswith("._"):
            continue
        if not f.name.endswith(".swc"):
            continue
        files.append(f)
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter and split cleaned neurons.")
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("/Users/umer/Documents/neurons_cleaned"),
        help="Directory of cleaned SWC files.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/Users/umer/Documents/neurons_final"),
        help="Output directory (will contain train/ and val/ subdirs).",
    )
    parser.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction (default 0.1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    swc_files = get_swc_files(input_dir)
    print(f"Found {len(swc_files)} SWC files in {input_dir}")

    # Filter by root degree
    kept: list[Path] = []
    dropped = 0
    degree_dropped: dict[int, int] = {}
    for f in swc_files:
        G = load_swc_graph(f)
        root = G.graph.get("root")
        deg = G.degree(root) if root is not None else 0
        if deg > MAX_CHILDREN:
            dropped += 1
            degree_dropped[deg] = degree_dropped.get(deg, 0) + 1
        else:
            kept.append(f)

    print(f"\nFiltered: kept {len(kept)}, dropped {dropped} (root degree > {MAX_CHILDREN})")
    if degree_dropped:
        for deg in sorted(degree_dropped):
            print(f"  degree {deg}: {degree_dropped[deg]} dropped")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(kept)
    n_val = int(len(kept) * args.val_frac)
    val_files = kept[:n_val]
    train_files = kept[n_val:]
    print(f"\nSplit: {len(train_files)} train, {len(val_files)} val")

    # Copy files
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for f in train_files:
        shutil.copy2(f, train_dir / f.name)
    for f in val_files:
        shutil.copy2(f, val_dir / f.name)

    print(f"\nDone. Output at {output_dir}")
    print(f"  train/ : {len(train_files)} files")
    print(f"  val/   : {len(val_files)} files")


if __name__ == "__main__":
    main()
