"""Training-budget accounting for the iterative-expansion method vs the SemlaFlow baseline.

Answers, for a given corpus, the questions needed to train both methods on a comparable
budget (see docs/BUDGET_PARITY_SEMLAFLOW.md):

  OUR SIDE   how many (graph, reduction-level) items does one pass over the corpus produce,
             how many node forward-passes does that cost, and how many supervised targets
             does it deliver? (Runs the real deterministic depth reducer.)
  SEMLA SIDE how many optimizer steps is one epoch under SemlaFlow's cost-based bucket
             sampler, how many graphs actually get seen, and what is the peak dense-bond
             tensor per batch? (Replicates semlaflow.data.util.BucketBatchSampler.)
  PARITY     the step count that matches SemlaFlow's `--epochs`, our batch size that makes
             steps/epoch agree, and the compute asymmetry in both currencies.

Usage:
    conda run -n NEURO2 python data_analysis/budget_accounting.py --dataset neurons
    conda run -n NEURO2 python data_analysis/budget_accounting.py --dataset trees_d20 --sample 400
    ... --root /path/to/corpus --batch-size 256 --semla-batch-cost 1024 --semla-epochs 300
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_generation.depth_reduction import DepthReductionFactory  # noqa: E402
from utils.data_loading import load_swc_graph, nx_graph_to_adj_pos  # noqa: E402

# SemlaFlow constants (semlaflow/scriptutil.py::get_n_bond_types, uniform-sample strategy).
N_BOND_TYPES = 5

# Per-corpus settings. `buckets` and `batch_cost` are NOT ours to choose -- they mirror the
# bucket_limits registered in semlaflow/scriptutil.py (DATASET_CONFIGS) and the --batch_cost that
# semla-flow/RUN.md 0/3a prescribes for each corpus. `batch_size` is *our* training.batch_size for
# the config that trains on it. Keep these in sync with RUN.md; see
# docs/BUDGET_PARITY_SEMLAFLOW.md 4/7.3/7.4.
#
# NOTE the live SemlaFlow sampler is stock (drop_last=True, no dead-bucket clamp), so the numbers
# in docs 4 come from `--no-clamp --stock-drop-last`; this script's defaults are the *patched*
# sampler.
_SWC_BUCKET_PREFIX = [24, 40, 56, 72, 96, 128, 160, 200]

PRESETS = {
    "neurons": dict(
        root="/Users/umer/Documents/neurons_conditional_full",
        buckets=[40, 56, 72, 96, 128, 160, 200, 256, 537],
        batch_cost=1024,
        batch_size=256,
    ),
    "trees_d10": dict(
        root="/Users/umer/Documents/trees_genus_d10",
        buckets=[96, 128, 160, 200, 256, 320, 384],
        batch_cost=1024,
        batch_size=256,
    ),
    "trees_d10_capped": dict(
        root="/Users/umer/Documents/trees_genus_d10_capped",
        buckets=[96, 128, 160, 200, 240, 268],
        batch_cost=1024,
        batch_size=128,
    ),
    "trees_d15_capped": dict(
        root="/Users/umer/Documents/trees_genus_d15_capped",
        buckets=[128, 200, 264, 336, 416, 512, 592, 666],
        batch_cost=2048,
        batch_size=128,
    ),
    "trees_d20_capped": dict(
        root="/Users/umer/Documents/trees_genus_d20_capped",
        buckets=[160, 200, 264, 336, 424, 528, 648, 784, 928, 1110],
        batch_cost=16384,
        batch_size=128,
    ),
    # Superseded by the capped corpora (RUN.md 0a). They keep the fine-grained prefix ladder and
    # are lossy at every batch_cost (2.2-22.3% of the train split never sampled) -- kept so old
    # runs stay reproducible.
    "trees_d15": dict(
        root="/Users/umer/Documents/trees_genus_d15",
        buckets=_SWC_BUCKET_PREFIX + [264, 336, 416, 512, 640, 768, 1024, 1280, 1536],
        batch_cost=16384,
        batch_size=128,
    ),
    "trees_d20": dict(
        root="/Users/umer/Documents/trees_genus_d20",
        buckets=_SWC_BUCKET_PREFIX + [264, 336, 424, 528, 648, 784, 1024, 1408, 1792, 2304, 3072],
        batch_cost=16384,
        batch_size=128,
    ),
}


def swc_files(split_dir: Path) -> list[Path]:
    return [
        p for p in sorted(split_dir.iterdir())
        if p.is_file() and p.name.endswith(".swc") and not p.name.startswith("._")
    ]


def node_count(path: Path) -> int:
    with open(path) as fh:
        return sum(1 for line in fh if line.strip() and not line.startswith("#"))


# --------------------------------------------------------------------------- our side


def reduction_stats(files, contract_root: bool = False):
    """Run the real deterministic depth reducer; return per-graph arrays."""
    factory = DepthReductionFactory(
        mode="deterministic", cherry_p=1.0, ensure_progress=True,
        root=0, contract_root=contract_root,
    )
    nodes, levels, visits, targets = [], [], [], []
    for f in files:
        G = load_swc_graph(f)
        adj, _pos, _ = nx_graph_to_adj_pos(G)
        g = factory(adj, rng=np.random.default_rng(0))
        ns, tg = [], []
        while True:
            ns.append(g.n)
            reduced = g.get_reduced_graph()
            if not reduced.did_contract:
                # Terminal record supervises the root's children (see
                # RandRedDataset.get_random_reduction_sequence's `forced_new`).
                st = g._state
                tg.append(len(st.children.get(st.root, [])))
                break
            tg.append(int(np.asarray(getattr(g, "new_leaves_from_next", [])).size))
            g = reduced
        nodes.append(ns[0])
        levels.append(len(ns))
        visits.append(sum(ns))
        targets.append(sum(tg))
    return (np.array(nodes, float), np.array(levels, float),
            np.array(visits, float), np.array(targets, float))


# ------------------------------------------------------------------------- semla side


def round8(x: float) -> int:
    """semlaflow.data.util.BucketBatchSampler._round_batch_size(round_batch_to_8=True)."""
    bs = 8 * round(x / 8)
    return 1 if bs == 0 else bs


def semla_epoch(sizes, buckets, batch_cost, clamp=True, drop_last=False):
    """Replicate BucketBatchSampler's accounting (quadratic cost).

    Defaults are the *recommended* baseline config (docs/BUDGET_PARITY_SEMLAFLOW.md 7.5/7.8):
    batch size clamped to bucket population, and drop_last off so every graph is seen every
    epoch. Pass clamp=False, drop_last=True for stock SemlaFlow.
    """
    costs = [(b ** 2) / 256 + 1 for b in buckets]
    batch = [round8(batch_cost / c) for c in costs]
    items = [0] * len(buckets)
    over = 0
    for n in sizes:
        for i, lim in enumerate(buckets):
            if lim >= n:
                items[i] += 1
                break
        else:
            over += 1
    dead = [(lim, it) for lim, b, it in zip(buckets, batch, items) if it and it < b]
    if clamp:
        # See docs: without this, a bucket holding fewer graphs than its batch size
        # yields 0 batches and those graphs are never trained on.
        batch = [min(b, it) if it else b for b, it in zip(batch, items)]
    if drop_last:
        batches = [it // b for it, b in zip(items, batch)]
        seen = sum(b * nb for b, nb in zip(batch, batches))
    else:
        # BucketBatchSampler adds one partial batch per bucket and sizes it from the
        # remaining items (util.py:49-51, :74-77), so nothing is dropped.
        batches = [math.ceil(it / b) if it else 0 for it, b in zip(items, batch)]
        seen = sum(items)
    steps = sum(batches)
    peak_mb = max(b * (lim ** 2) * N_BOND_TYPES * 4 / 1e6 for b, lim in zip(batch, buckets))
    return dict(buckets=buckets, costs=costs, batch=batch, items=items, batches=batches,
                steps=steps, seen=seen, over=over, dead=dead, peak_mb=peak_mb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(PRESETS), default="neurons")
    ap.add_argument("--root", type=str, default=None, help="Corpus root (expects train/).")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--sample", type=int, default=0, help="Subsample N graphs for the reducer pass (0 = all).")
    ap.add_argument("--batch-size", type=int, default=None, help="Our training.batch_size (level-samples/step).")
    ap.add_argument("--semla-batch-cost", type=int, default=None)
    ap.add_argument("--semla-epochs", type=int, default=300)
    ap.add_argument("--no-clamp", action="store_false", dest="clamp",
                    help="Do NOT clamp bucket batch size to bucket population (stock SemlaFlow).")
    ap.add_argument("--stock-drop-last", action="store_true", dest="drop_last",
                    help="Stock SemlaFlow train sampler (drop_last=True): drops each bucket's "
                         "remainder every epoch. Default off -- see docs section 7.8.")
    args = ap.parse_args()

    preset = PRESETS[args.dataset]
    root = Path(args.root or preset["root"])
    batch_size = args.batch_size or preset["batch_size"]
    batch_cost = args.semla_batch_cost or preset["batch_cost"]
    buckets = preset["buckets"]

    files = swc_files(root / args.split)
    sizes = np.array([node_count(f) for f in files], dtype=float)
    print(f"\n=== {args.dataset}  {root}/{args.split}: {len(files)} graphs, "
          f"nodes mean {sizes.mean():.1f} median {np.median(sizes):.0f} "
          f"p95 {np.percentile(sizes, 95):.0f} max {sizes.max():.0f}")

    red_files = files
    if args.sample and args.sample < len(files):
        random.seed(0)
        red_files = random.sample(files, args.sample)
    nodes, levels, visits, targets = reduction_stats(red_files)
    scale = len(files) / len(red_files)
    items_epoch = levels.sum() * scale
    visits_epoch = visits.sum() * scale
    nodes_per_level = visits.sum() / levels.sum()

    print(f"\n-- OUR SIDE (deterministic depth reduction, {len(red_files)} graphs"
          f"{' extrapolated' if scale > 1 else ''})")
    print(f"   levels / graph          mean {levels.mean():7.2f}  median {np.median(levels):5.0f}  "
          f"p95 {np.percentile(levels, 95):5.0f}  max {levels.max():5.0f}")
    print(f"   node-visits / graph     mean {visits.mean():7.1f}   ({visits.sum()/nodes.sum():.2f}x nodes)")
    print(f"   supervised targets      mean {targets.mean():7.2f}   "
          f"({targets.sum()/nodes.sum():.3f} per node == N-1 per graph)")
    print(f"   items / epoch           {items_epoch:,.0f}   node-visits / epoch {visits_epoch/1e6:.2f}M")
    print(f"   mean nodes / item       {nodes_per_level:.2f}")
    print(f"   steps / epoch @ B={batch_size:<4d}  {items_epoch/batch_size:,.0f}   "
          f"nodes / step {batch_size*nodes_per_level:,.0f}")

    sm = semla_epoch(sizes, buckets, batch_cost, clamp=args.clamp, drop_last=args.drop_last)
    print(f"\n-- SEMLAFLOW SIDE (batch_cost={batch_cost}, quadratic, "
          f"clamp={'on' if args.clamp else 'OFF (stock)'}, "
          f"drop_last={'ON (stock)' if args.drop_last else 'off'})")
    print(f"   {'bucket':>7} {'cost':>9} {'batch':>6} {'items':>7} {'batches':>8} {'bond MB':>9}")
    for lim, c, b, it, nb in zip(buckets, sm["costs"], sm["batch"], sm["items"], sm["batches"]):
        print(f"   {lim:7d} {c:9.1f} {b:6d} {it:7d} {nb:8d} "
              f"{b*(lim**2)*N_BOND_TYPES*4/1e6:9.1f}")
    if sm["over"]:
        print(f"   !! {sm['over']} graphs exceed the top bucket -- SmolDM raises on this")
    if sm["dead"]:
        print(f"   !! buckets with fewer graphs than their batch size: {sm['dead']} "
              f"(never trained without the clamp)")
    print(f"   steps / epoch {sm['steps']}   graphs seen {sm['seen']}/{len(files)} "
          f"({sm['seen']/len(files)*100:.1f}%)   graphs/step {sm['seen']/sm['steps']:.1f}   "
          f"nodes/step {sm['seen']/sm['steps']*sizes.mean():,.0f}")
    semla_visits_epoch = sm["seen"] * sizes.mean()
    semla_pairs_epoch = (sizes ** 2).sum() * sm["seen"] / len(files)
    print(f"   node-visits / epoch {semla_visits_epoch/1e6:.2f}M   "
          f"dense pair-interactions / epoch {semla_pairs_epoch/1e6:.1f}M")

    # Budget hierarchy: (1) denoising events per node == epochs, (2) gradient steps,
    # (3) measured GPU-hours. See docs/BUDGET_PARITY_SEMLAFLOW.md section 5.
    E = args.semla_epochs
    total_steps = E * sm["steps"]
    b_star = items_epoch / sm["steps"]
    steps_C = E * items_epoch / batch_size
    eff_epochs = E * sm["seen"] / len(files)
    print(f"\n-- PARITY at E = {E} epochs (= {E} coordinate-denoising events per node)")
    caveat = (f"  (drop_last makes its E effectively {eff_epochs:.1f}, unevenly across buckets)"
              if args.drop_last else "  (both exact -- nothing dropped)")
    print(f"   total denoising events   {E*sizes.sum()/1e6:.1f}M ours   "
          f"{E*sm['seen']*sizes.mean()/1e6:.1f}M SemlaFlow{caveat}")
    print(f"   SemlaFlow               {total_steps:,} steps")
    print(f"   [B] equal epochs AND steps -> training.batch_size = {b_star:.0f}, "
          f"num_steps = {total_steps:,}")
    print(f"   [C] equal epochs, B={batch_size:<4d}      -> num_steps = {steps_C:,.0f} "
          f"({steps_C/total_steps:.2f}x SemlaFlow's steps)")
    print(f"   [A] equal steps, B={batch_size:<4d} (avoid) -> {total_steps*batch_size/items_epoch:.0f} "
          f"epochs for us, not {E}")
    print(f"   compute per matched epoch: node-visits {visits_epoch/semla_visits_epoch:.2f}x ours, "
          f"pair/edge interactions {semla_pairs_epoch/visits_epoch:.1f}x theirs")
    print("   -> the two currencies disagree; measure GPU-hours, do not derive them.\n")


if __name__ == "__main__":
    main()
