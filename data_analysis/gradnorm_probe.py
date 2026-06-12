"""Gradient-norm probe: reproduce (or rule out) the training spike on a checkpoint.

Background. The flow run's loss spiked hard and transiently around 99k/125k in the
original run, but a resume from the 75k checkpoint (with grad_clip_norm=1.0) stayed
stable and the logged grad_norm never exceeded ~0.8. That log is sparse, though
(one point per 1000 steps) and the resumed run draws a different batch trajectory, so
we have NOT actually observed whether any batch produces a spike-sized gradient on
this model — nor whether clip=1.0 would neutralise it.

This script answers that deterministically, in seconds on a GPU, against the real
checkpoint weights. It builds the exact training model/method/dataset (by composing
the run config and reusing main.get_expansion_items), loads the checkpoint, and
measures the PRE-clip gradient norm of one forward+backward for several batch
compositions:

  - random-sweep : N random 512-sample batches  -> the faithful training distribution
  - all-worst    : the 512 samples with the largest ||C_0|| (parent->child offset)
                   -> removes the mean-reduction dilution that hides single outliers
  - largest      : the 512 samples with the most nodes
  - enriched     : a random batch with the worst-K samples injected (an "unlucky" batch)

Because the position loss is mean-reduced over all leaves in a batch, a handful of
||C_0|| outliers among ~5k leaves get diluted — so testing several compositions (and
sweeping many random batches) is the point: it tells us whether the spike is a
single-batch event (clipping is the lever) or a multi-step trajectory effect (it isn't).

Usage (cluster):
    conda run -n NEURO2 python data_analysis/gradnorm_probe.py \
        --checkpoint /path/orig_run/checkpoints/step_75000.pt \
        --data-dir   /path/neurons_final/train \
        --worst-csv  data_analysis/so2_degeneracy_worst.csv

Sanity: random-sweep median grad_norm should land near the run's logged ~0.65.
Verdict: if all modes stay <~2, single batches don't spike (re-open optimizer-trajectory
hypotheses); if any mode is >>1 (10-100x), the spike is reproduced and clip=1.0 (post-clip
norm = 1.0) confirms the fix catches it.
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch as th

# Ensure repository root is on sys.path when running the script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import graph_generation as gg  # noqa: E402
from main import get_expansion_items  # noqa: E402
from utils.data_loading import load_swc_graph  # noqa: E402


def build_diffusion(cfg):
    """Replicate main.py's diffusion branch for the configured diffusion."""
    dc = cfg.diffusion
    name = getattr(dc, "name", None)
    if name == "flow":
        ps = getattr(dc, "prior_std_pos", None)
        return gg.diffusion.FlowMatchingModel(
            num_steps=dc.num_steps,
            prior_std=getattr(dc, "prior_std", 1.0),
            time_dist=getattr(dc, "time_dist", "uniform"),
            beta_a=getattr(dc, "beta_a", 2.0),
            beta_b=getattr(dc, "beta_b", 1.0),
            sigma_min=getattr(dc, "sigma_min", 0.0),
            prior_std_pos=(list(ps) if ps is not None else None),
        )
    if name == "basic":
        return gg.diffusion.DenoisingDiffusionModel(num_steps=dc.num_steps)
    if name == "edm":
        return gg.diffusion.EDMDiffusionModel(num_steps=dc.num_steps)
    raise ValueError(f"Unsupported diffusion name for probe: {name!r}")


def load_pool(data_dir: Path, n: int, worst_csv: Path | None, seed: int):
    """Load up to n SWC graphs; force-include files named in worst_csv if given."""
    files = sorted(f for f in data_dir.iterdir() if f.suffix == ".swc")
    forced = []
    if worst_csv is not None and worst_csv.exists():
        names = set()
        with open(worst_csv) as fh:
            for row in csv.DictReader(fh):
                fn = row.get("file")
                if fn:
                    names.add(fn.strip())
        forced = [data_dir / fn for fn in names if (data_dir / fn).exists()]
        print(f"  worst-csv: {len(names)} files referenced, {len(forced)} found in data-dir")

    rng = random.Random(seed)
    others = [f for f in files if f not in set(forced)]
    rng.shuffle(others)
    n_rand = max(0, n - len(forced))
    chosen = forced + others[:n_rand]

    graphs, skipped = [], 0
    for f in chosen:
        try:
            graphs.append(load_swc_graph(f))
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            if skipped <= 3:
                print(f"  skip {f.name}: {exc}")
    print(f"Loaded {len(graphs)} graphs ({len(forced)} forced from worst-csv, {skipped} skipped).")
    return graphs


def sample_max_offset(s) -> float:
    """Max parent->child offset norm in a reduced sample (== max ||C_0|| up to frame)."""
    pos = s.pos
    pidx = s.parent_idx_1b.long() - 1  # 1-based -> 0-based, root = -1
    hp = pidx >= 0
    if int(hp.sum()) == 0:
        return 0.0
    return float((pos[hp] - pos[pidx[hp]]).norm(dim=-1).max())


def clip_outcome(grad_norm: float, thr: float) -> str:
    if grad_norm > thr:
        return f"CLIPPED (scale {thr / grad_norm:.3f} -> {thr:.2f})"
    return f"not clipped ({grad_norm:.3f} <= {thr})"


def measure(method, model, samples, device, seed) -> dict:
    from torch_geometric.data import Batch
    batch = Batch.from_data_list(list(samples)).to(device)
    th.manual_seed(seed)
    model.zero_grad(set_to_none=True)
    loss, metrics = method.get_loss(batch=batch, model=model)
    loss.backward()
    gnorm = float(th.nn.utils.clip_grad_norm_(model.parameters(), float("inf")))
    model.zero_grad(set_to_none=True)
    return {
        "grad_norm": gnorm,
        "pos_loss": metrics.get("leaf_pos_loss", float("nan")),
        "exp_loss": metrics.get("leaf_expansion_loss", float("nan")),
        "cum_loss": metrics.get("cumulative_loss", float("nan")),
        "num_leaves": metrics.get("num_leaves", -1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="abs path to step_XXXXX.pt; if omitted, runs with RANDOM weights (plumbing test only)")
    ap.add_argument("--data-dir", type=Path, default=Path("/Users/umer/Documents/neurons_final/train"))
    ap.add_argument("--config-name", default="neuron_dataset_run_2")
    ap.add_argument("--pool-graphs", type=int, default=200)
    ap.add_argument("--n-batches", type=int, default=12)
    ap.add_argument("--worst-csv", type=Path, default=None)
    ap.add_argument("--enrich-k", type=int, default=8, help="# worst samples injected into the enriched batch")
    ap.add_argument("--eval", action="store_true", help="run model.eval() (no dropout) instead of train()")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = ("cuda" if th.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"[device] {device}" + ("  (CPU: expect minutes; use a free GPU for seconds)" if device == "cpu" else ""))
    t_start = time.perf_counter()

    # --- compose the exact run config (model dims, flow params, reduction, pos_scale) ---
    from hydra import initialize_config_dir, compose
    with initialize_config_dir(config_dir=str(REPO_ROOT / "config"), version_base=None):
        cfg = compose(config_name=args.config_name)
    print(f"Composed config '{args.config_name}': model={cfg.model.name} "
          f"layers={cfg.model.num_layers} feats={cfg.model.feats_dim} m_dim={cfg.model.m_dim} "
          f"batch_size={cfg.training.batch_size} pos_scale={getattr(cfg.dataset,'pos_scale_factor',None)} "
          f"diffusion={cfg.diffusion.name}")

    # --- build model/method/dataset via training's own path (arch matches checkpoint) ---
    _t = time.perf_counter()
    pool = load_pool(args.data_dir, args.pool_graphs, args.worst_csv, args.seed)
    if not pool:
        print("No graphs loaded — aborting.")
        sys.exit(1)
    print(f"[timing] loaded pool in {time.perf_counter()-_t:.1f}s")

    # tmd is unused for this run (tmd_in_dim/tmd_hidden_dim == 0) but get_expansion_items
    # computes a TMD per graph — stub it to avoid that wasted setup cost.
    import main as _main
    if getattr(cfg.model, "tmd_hidden_dim", 0) == 0 and getattr(cfg.model, "tmd_in_dim", 0) == 0:
        _main.compute_tmd_mixed = lambda G: np.zeros((1,), dtype=np.float32)
        print("[setup] tmd dims are 0 -> skipping TMD computation")

    _t = time.perf_counter()
    diffusion = build_diffusion(cfg)
    items = get_expansion_items(cfg, pool, diffusion=diffusion)
    model, method = items["model"], items["method"]
    print(f"[timing] built model/method/dataset (reductions precomputed) in {time.perf_counter()-_t:.1f}s")

    if args.checkpoint is not None:
        ckpt = th.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"Loaded checkpoint {args.checkpoint} (step {ckpt.get('step', '?')}).")
    else:
        print("WARNING: no --checkpoint given -> RANDOM weights. grad_norm values are NOT "
              "meaningful; this only validates the pipeline.")

    model = model.to(device)
    method = method.to(device)
    (model.eval() if args.eval else model.train())

    samples = list(items["train_dataloader"].dataset.samples)
    bs = min(cfg.training.batch_size, len(samples))
    print(f"{len(samples)} reduced samples in pool; batch size = {bs} "
          f"(mode={'eval' if args.eval else 'train'}, device={device})")

    # --- rank samples (cheap, no model) ---
    offsets = np.array([sample_max_offset(s) for s in samples])
    nodes = np.array([int(s.pos.size(0)) for s in samples])
    order_worst = np.argsort(offsets)[::-1]
    order_large = np.argsort(nodes)[::-1]
    print(f"per-sample max ||C_0||: median={np.median(offsets):.2f} max={offsets.max():.2f}; "
          f"nodes: median={int(np.median(nodes))} max={int(nodes.max())}")

    rng = random.Random(args.seed)
    rows = []

    # random sweep
    print(f"\n[measuring] {args.n_batches} random batches + 3 targeted "
          f"(forward+backward each, no progress bar in get_loss)...")
    sweep = []
    for b in range(args.n_batches):
        _t = time.perf_counter()
        pick = rng.sample(range(len(samples)), bs)
        r = measure(method, model, [samples[i] for i in pick], device, args.seed + b)
        sweep.append(r["grad_norm"])
        print(f"  sweep {b+1}/{args.n_batches}: grad_norm={r['grad_norm']:.4f} "
              f"pos_loss={r['pos_loss']:.4f} ({time.perf_counter()-_t:.1f}s/batch)")
    sweep = np.array(sweep)
    rows.append(("random-sweep (mean)", float(sweep.mean()), None, None, None))
    rows.append(("random-sweep (min)", float(sweep.min()), None, None, None))
    rows.append(("random-sweep (median)", float(np.median(sweep)), None, None, None))
    rows.append(("random-sweep (MAX)", float(sweep.max()), None, None, None))

    # targeted batches
    def take(order):
        return [samples[i] for i in order[:bs].tolist()]

    targeted = {
        "all-worst-||C0||": take(order_worst),
        "largest-neuron": take(order_large),
    }
    # enriched: random batch with worst-K injected
    enr = rng.sample(range(len(samples)), bs)
    for j in range(min(args.enrich_k, bs)):
        enr[j] = int(order_worst[j])
    targeted["enriched(random+worst%d)" % args.enrich_k] = [samples[i] for i in enr]

    for name, smp in targeted.items():
        _t = time.perf_counter()
        r = measure(method, model, smp, device, args.seed)
        rows.append((name, r["grad_norm"], r["pos_loss"], r["exp_loss"], r["num_leaves"]))
        print(f"  {name}: grad_norm={r['grad_norm']:.4f} pos_loss={r['pos_loss']:.4f} "
              f"({time.perf_counter()-_t:.1f}s)")
    print(f"[timing] total {time.perf_counter()-t_start:.1f}s")

    # --- report ---
    line = "=" * 86
    print("\n" + line)
    print("GRAD-NORM PROBE" + ("  [RANDOM WEIGHTS — not meaningful]" if args.checkpoint is None else ""))
    print(line)
    print(f"{'batch mode':<30}{'grad_norm':>12}{'pos_loss':>11}{'exp_loss':>11}{'leaves':>9}")
    for name, gn, pl, el, nl in rows:
        pls = f"{pl:.4f}" if pl is not None else ""
        els = f"{el:.4f}" if el is not None else ""
        nls = f"{nl}" if nl is not None else ""
        print(f"{name:<30}{gn:>12.4f}{pls:>11}{els:>11}{nls:>9}")

    print("\nclip outcomes (worst single batch observed):")
    worst_gn = max(r[1] for r in rows)
    for thr in (1.0, 5.0):
        print(f"  threshold {thr}: {clip_outcome(worst_gn, thr)}")

    print("\nverdict:")
    if worst_gn <= 2.0:
        print(f"  No batch spiked (max grad_norm {worst_gn:.2f} <= 2). Single batches do NOT produce")
        print("  spike-sized gradients on this model -> the instability is likely a multi-step /")
        print("  optimizer-trajectory effect, not a single-batch event. Per-step clipping is not the")
        print("  primary lever; revisit Adam beta2 / warmup / accumulation, or a longer dense-logged run.")
    else:
        print(f"  Spike reproduced (max grad_norm {worst_gn:.2f} > 2). clip=1.0 would clamp it to 1.0,")
        print("  confirming clipping neutralises the spike. The triggering batch mode is shown above.")
    print(line)


if __name__ == "__main__":
    main()
