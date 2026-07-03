#!/usr/bin/env python
"""
Does the teacher-forced sampler's over-production shrink with more ODE steps?

The flow model is DATA-PREDICTION (net regresses the clean offset C_0; the velocity is
derived as (C1_pred - C_t)/(1-t) and integrated with explicit Euler). The teacher-forced
forward (one-step prediction) beats the sampler because the sampler feeds the net its own
off-path estimate over only `num_steps` Euler steps -> a flow-time train/sample mismatch +
integration error.

This sweeps `diffusion.num_steps` at a fixed checkpoint and measures the teacher-forced
SAMPLING distribution distance (branch_length KS/W1 vs the GT |C_0| pool, via the existing
suite). Reading:
  - KS falls toward the floor as steps increase  -> mostly integration error (cheap fix:
    more steps / better solver).
  - KS plateaus well above the floor              -> prediction quality on off-path inputs
    (needs t-weighting / consistency / training-side changes).

    conda run -n NEURO2 python data_analysis/ode_steps_audit.py \
        --ckpt-dir <run>/checkpoints --eval-dir .../val --step 9000 \
        --n-graphs 400 --steps-grid 5 10 20 50 100 --out ode_steps.pkl
"""
import argparse, sys, pickle, time
from pathlib import Path
import numpy as np
import torch as th

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "data_analysis"))
from hydra import initialize_config_dir, compose
from seed_variance_probe import build_model_diffusion_method, load_ckpt
from validation.teacher_forced_eval import _build_eval_batches, evaluate_teacher_forced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--config-name", default="neuron_dataset_run_3")
    ap.add_argument("--config-dir", default=str(REPO / "config"))
    ap.add_argument("--step", type=int, default=9000, help="checkpoint step to audit")
    ap.add_argument("--steps-grid", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20, 50, 100],
                    help="num_steps=1 IS the one-step-from-complete-noise baseline (clean §16 analog, in §18 KS units)")
    ap.add_argument("--n-graphs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="ode_steps.pkl")
    args = ap.parse_args()

    th.set_float32_matmul_precision("high")
    with initialize_config_dir(version_base="1.3", config_dir=args.config_dir):
        cfg = compose(config_name=args.config_name)
    uhat = np.asarray(getattr(cfg.model, "so2_axis", [0., 1., 0.]), dtype=float).reshape(3)
    base_steps = int(getattr(cfg.diffusion, "num_steps", 10))

    batches, n = _build_eval_batches(cfg, args.eval_dir, args.n_graphs, args.batch_size)
    print(f"{n} eval graphs -> {len(batches)} batches; config num_steps={base_steps}; auditing step {args.step}")
    model, method = build_model_diffusion_method(cfg)
    model = model.to(args.device); method = method.to(args.device)
    load_ckpt(model, str(Path(args.ckpt_dir) / f"step_{args.step}.pt"), args.device)

    results = {"step": args.step, "base_num_steps": base_steps, "by_num_steps": {}}
    print(f"\n{'num_steps':>10}{'branch_len_KS':>15}{'branch_len_W1µm':>17}{'bif_angle_KS':>14}{'time':>7}")
    for ns in args.steps_grid:
        method.diffusion.num_steps = int(ns)
        t0 = time.time()
        res = evaluate_teacher_forced(method, model, batches, uhat, device=args.device)
        d = res["dist"]
        results["by_num_steps"][ns] = d
        print(f"{ns:>10}{d.get('branch_length_ks', float('nan')):>15.4f}"
              f"{d.get('branch_length_w1', float('nan'))*45.1:>17.2f}"
              f"{d.get('bifurcation_angle_ks', float('nan')):>14.4f}{time.time()-t0:>6.0f}s")
        with open(args.out, "wb") as f:
            pickle.dump(results, f)
    method.diffusion.num_steps = base_steps
    print(f"\nfloor branch_length_ks ~= 0.006. Saved -> {args.out}")
    print("num_steps=1 = one-step-from-complete-noise (the clean §16 analog, in §18 KS units):")
    print("  (1-step KS - floor) = irreducible 'predict offset from pure noise' hardness.")
    print("  1 -> 10 change      = the multi-step rollout/integration effect, isolated:")
    print("     drops  => integration helps (10 may be too few);")
    print("     flat   => one-shot is the ceiling (prediction-quality / training-side fix);")
    print("     RISES  => integration HURTS (pathological velocity field — flow is wrong).")


if __name__ == "__main__":
    main()
