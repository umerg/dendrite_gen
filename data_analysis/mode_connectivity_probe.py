#!/usr/bin/env python
"""Linear-mode-connectivity (loss-barrier) probe — why does EMA make fluctuation WORSE?

EMA averages WEIGHTS. It only helps if the weights jitter locally inside one basin, so
that the average weight is still a good model. We measured the free-running output is
white-noise across checkpoints (decorrelates in ~100 steps) and EMA empirically HURTS,
which is the signature of weight steps that are NOT linearly mode-connected: the average
of two good checkpoints lands in the high-loss region BETWEEN them.

This probe takes two checkpoints (ideally two *good* converged ones, e.g. ~100-500 steps
apart = the window EMA averages over), linearly interpolates their weights
``w(alpha) = (1-alpha)*w_A + alpha*w_B``, and evaluates the teacher-forced metric along
alpha in [0,1]. A hump (interior worse than both endpoints) = a loss barrier = no linear
mode connectivity = EMA cannot work by construction.

It evaluates at BOTH num_steps=1 (one-shot prediction = pure weight/prediction effect) and
num_steps=10 (full sampler) to localize the barrier:
  - barrier at num_steps=1  -> the WEIGHTS aren't mode-connected (optimization/landscape;
    no weight decay, LR). EMA is destructive; fix is optimization-side.
  - flat at num_steps=1 but barrier at num_steps=10 -> weights ARE mode-connected; the
    barrier is the SAMPLER gain amplifying tiny prediction diffs. Fix is sampler-side.
  - flat at both -> weights mode-connected & sampler stable along this path -> EMA *should*
    help; revisit the EMA config / which curve was read.

    conda run -n NEURO2 python data_analysis/mode_connectivity_probe.py \
        --ckpt-dir <run>/checkpoints --eval-dir .../val \
        --step-a 5600 --step-b 5700 --n-graphs 400 --out mode_conn.pkl
"""
import argparse, sys, pickle
from pathlib import Path
import numpy as np
import torch as th

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "data_analysis"))
from hydra import initialize_config_dir, compose
from seed_variance_probe import build_model_diffusion_method
from validation.teacher_forced_eval import _build_eval_batches, evaluate_teacher_forced


def _load_model_state(path, device, key):
    state = th.load(path, map_location=device)
    if key not in state:
        raise KeyError(f"'{key}' not in {path}; keys={list(state)[:8]}")
    return state[key]


def _interp_state(sa, sb, alpha):
    """Linear blend of two state_dicts; float tensors interpolated, others taken from A."""
    out = {}
    for k, va in sa.items():
        vb = sb[k]
        if th.is_floating_point(va):
            out[k] = (1.0 - alpha) * va + alpha * vb
        else:
            out[k] = va.clone()  # int buffers (e.g. num_batches_tracked) — not interpolable
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--config-name", default="neuron_dataset_run_3")
    ap.add_argument("--config-dir", default=str(REPO / "config"))
    ap.add_argument("--step-a", type=int, required=True, help="endpoint A (a good converged ckpt)")
    ap.add_argument("--step-b", type=int, required=True, help="endpoint B (another good ckpt)")
    ap.add_argument("--ckpt-key", default="model", help="state_dict key to interpolate ('model' or e.g. 'model_ema_0.999')")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--num-steps-list", type=int, nargs="+", default=[1, 10],
                    help="1=one-shot (weights only); 10=full sampler (weights+gain)")
    ap.add_argument("--n-graphs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="mode_conn.pkl")
    args = ap.parse_args()

    th.set_float32_matmul_precision("high")
    with initialize_config_dir(version_base="1.3", config_dir=args.config_dir):
        cfg = compose(config_name=args.config_name)
    uhat = np.asarray(getattr(cfg.model, "so2_axis", [0., 1., 0.]), dtype=float).reshape(3)

    batches, n = _build_eval_batches(cfg, args.eval_dir, args.n_graphs, args.batch_size)
    print(f"{n} eval graphs; interpolating {args.ckpt_key}: step {args.step_a} <-> {args.step_b}")
    model, method = build_model_diffusion_method(cfg)
    model = model.to(args.device); method = method.to(args.device)

    cd = Path(args.ckpt_dir)
    sa = _load_model_state(cd / f"step_{args.step_a}.pt", args.device, args.ckpt_key)
    sb = _load_model_state(cd / f"step_{args.step_b}.pt", args.device, args.ckpt_key)

    results = {"step_a": args.step_a, "step_b": args.step_b, "ckpt_key": args.ckpt_key,
               "alphas": args.alphas, "by_alpha": {}}
    print(f"\n{'alpha':>6}" + "".join(f"{'ns='+str(ns)+' br_ks':>16}" for ns in args.num_steps_list)
          + f"{'fwd_r(ns=' + str(args.num_steps_list[-1]) + ')':>16}")
    for a in args.alphas:
        model.load_state_dict(_interp_state(sa, sb, a)); model.eval()
        results["by_alpha"][a] = {}
        row = f"{a:>6.2f}"
        for ns in args.num_steps_list:
            method.diffusion.num_steps = int(ns)
            d = evaluate_teacher_forced(method, model, batches, uhat, device=args.device)["dist"]
            results["by_alpha"][a][ns] = d
            row += f"{d.get('branch_length_ks', float('nan')):>16.4f}"
        dlast = results["by_alpha"][a][args.num_steps_list[-1]]
        fr = dlast.get("fwd_mag_mean_samp", float("nan")) / dlast.get("fwd_mag_mean_gt", float("nan"))
        row += f"{fr:>16.3f}"
        print(row)
        with open(args.out, "wb") as f:
            pickle.dump(results, f)

    # barrier summary per num_steps: interior max minus the worse endpoint
    print("\nBARRIER (interior_max - max(endpoints); >0 => loss barrier => NOT mode-connected):")
    al = np.array(args.alphas)
    interior = (al > 0) & (al < 1)
    for ns in args.num_steps_list:
        v = np.array([results["by_alpha"][a][ns]["branch_length_ks"] for a in args.alphas])
        endpts = max(v[0], v[-1]); barrier = float(v[interior].max() - endpts)
        rel = barrier / endpts if endpts > 0 else float("nan")
        print(f"  num_steps={ns:<3} barrier={barrier:+.4f} ({rel:+.0%} of endpoint)  "
              f"{'<-- BARRIER (weights not mode-connected)' if (ns==1 and barrier>0.005) else ('<-- sampler-gain barrier' if barrier>0.005 else 'flat (mode-connected)')}")
    print(f"\nSaved -> {args.out}")
    print("Read: ns=1 barrier => optimization/landscape (EMA destructive; fix LR/weight-decay).")
    print("      ns=1 flat + ns=10 barrier => sampler gain (fix sampler, not optimizer).")
    print("      both flat => EMA should work; revisit EMA config / which curve was read.")


if __name__ == "__main__":
    main()
