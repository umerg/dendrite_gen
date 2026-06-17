#!/usr/bin/env python3
"""Re-split a directory-based SWC dataset into a clean train/val split (+ 1 test file).

The dataset is laid out as ``data_dir/{train,val,test}`` and the training pipeline
(main.py) treats whatever lives in those folders as the split. This script pools the
*unique* SWC files across all three folders (deduping by filename -- the original
`small_trees` val set was just byte-identical copies of train files), shuffles them
deterministically, and rebuilds the split as ``train_frac`` / ``1 - train_frac``.

It then copies one val file into ``test/`` so the (otherwise empty) test split is
non-degenerate, and writes ``{train,val,test}_files.txt`` manifests into the dataset
root for reproducibility.

macOS AppleDouble files (``._*``) are ignored throughout (the loader skips them too).
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

SPLITS = ("train", "val", "test")


def _is_real_swc(p: Path) -> bool:
    return p.is_file() and p.name.endswith(".swc") and not p.name.startswith("._")


def _collect_canonical(data_dir: Path) -> dict[str, Path]:
    """Map basename -> canonical source path, deduping across splits (train wins)."""
    canonical: dict[str, Path] = {}
    for split in SPLITS:
        d = data_dir / split
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if _is_real_swc(p) and p.name not in canonical:
                canonical[p.name] = p
    return canonical


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Dataset root containing train/ val/ test/.")
    ap.add_argument("--train-frac", type=float, default=0.9,
                    help="Fraction of unique files assigned to train (default 0.9).")
    ap.add_argument("--seed", type=int, default=0, help="Shuffle seed (default 0).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the planned split without touching any files.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise NotADirectoryError(f"{data_dir} is not a directory.")

    canonical = _collect_canonical(data_dir)
    names = sorted(canonical)
    n = len(names)
    if n == 0:
        raise RuntimeError(f"No .swc files found under {data_dir}/{{train,val,test}}.")

    rng = random.Random(args.seed)
    rng.shuffle(names)
    n_train = round(args.train_frac * n)
    train_names = sorted(names[:n_train])
    val_names = sorted(names[n_train:])
    if not val_names:
        raise RuntimeError("Val split is empty; lower --train-frac.")
    test_name = val_names[0]  # deterministic: a redundant copy of one val tree

    assign = {nm: "train" for nm in train_names}
    assign.update({nm: "val" for nm in val_names})

    # keep-set: the canonical destination path for every file we want to retain.
    keep = {data_dir / sub / nm for nm, sub in assign.items()}
    keep.add(data_dir / "test" / test_name)

    print(f"Unique files: {n}  ->  train {len(train_names)} / val {len(val_names)} "
          f"(frac={args.train_frac}, seed={args.seed})")
    print(f"test/ will hold a copy of: {test_name}")

    if args.dry_run:
        existing = {p for split in SPLITS for p in (data_dir / split).glob("*")
                    if _is_real_swc(p)}
        n_move = sum(1 for nm, sub in assign.items()
                     if canonical[nm] != data_dir / sub / nm)
        n_del = sum(1 for p in existing if p not in keep)
        print(f"[dry-run] would move ~{n_move} files, delete ~{n_del} stale duplicates, "
              f"copy 1 test file. No changes made.")
        return

    # Pass 1: place each canonical file at its target path.
    moved = 0
    for nm, sub in assign.items():
        target = data_dir / sub / nm
        src = canonical[nm]
        if src == target:
            continue
        if target.exists():
            target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(target))
        moved += 1

    # Pass 2: delete stale real-swc duplicates not in the keep-set.
    deleted = 0
    for split in SPLITS:
        for p in (data_dir / split).glob("*"):
            if _is_real_swc(p) and p not in keep:
                p.unlink()
                deleted += 1

    # Pass 3: ensure test/ has the chosen file (copied from val).
    test_dir = data_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_dst = test_dir / test_name
    if not test_dst.exists():
        shutil.copy2(data_dir / "val" / test_name, test_dst)

    # Manifests.
    (data_dir / "train_files.txt").write_text("\n".join(train_names) + "\n")
    (data_dir / "val_files.txt").write_text("\n".join(val_names) + "\n")
    (data_dir / "test_files.txt").write_text(test_name + "\n")

    # Verify.
    def count(split: str) -> int:
        return sum(1 for p in (data_dir / split).iterdir() if _is_real_swc(p))

    ct, cv, cte = count("train"), count("val"), count("test")
    print(f"Done: moved {moved}, deleted {deleted} stale duplicates.")
    print(f"Final counts -> train {ct} / val {cv} / test {cte}")
    assert ct == len(train_names), (ct, len(train_names))
    assert cv == len(val_names), (cv, len(val_names))
    assert cte == 1, cte
    assert ct + cv == n, (ct, cv, n)
    print("Verified: train+val == unique total, test == 1.")


if __name__ == "__main__":
    main()
