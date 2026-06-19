#!/usr/bin/env python
"""
Audit the expansion (spawn/stop) head, teacher-forced, as a function of flow-time t.

Why: the teacher-forced suite reported expansion AUC ~0.65, flat across training. Before
trusting that, this isolates the model's CLEAN forward expansion prediction at CONTROLLED t:

  - At high t (e_t ~= e_0) the prediction is trivial -> accuracy MUST be ~1.0. If it is not,
    the measurement pipeline is buggy.  [sanity]
  - At low t (e_t ~= prior noise) the model must decide expansion FROM CONTEXT. That low-t
    accuracy/AUC is the real skill; whether IT rises across checkpoints tells us if the head
    learns at all (vs. is information-starved, e.g. size is unconditioned: use_size_ratio).

It runs the real `get_loss` FORWARD (not sampling), forcing a fixed t via `_sample_time`, and
captures the clean `e_pred`/`e_0` via the `compute_flow_diagnostics` call. Run on the cluster:

    conda run -n NEURO2 python data_analysis/expansion_audit.py \
        --ckpt-dir <run>/checkpoints --eval-dir .../val \
        --steps 2000 9000 30000 60000 --n-graphs 400 --out exp_audit.pkl
"""
import argparse, sys, pickle
from pathlib import Path
import numpy as np
import torch as th
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "data_analysis"))
from hydra import initialize_config_dir, compose
import graph_generation.diffusion.flow as flowmod
from seed_variance_probe import build_model_diffusion_method, load_ckpt
from validation.teacher_forced_eval import _build_eval_batches, _auc


def expansion_by_t(method, model, batches, device, t_grid):
    """For each fixed t: forward get_loss, capture clean e_pred vs e_0, return acc/auc/base."""
    stash = {}
    orig_time = method.diffusion._sample_time
    orig_diag = flowmod.compute_flow_diagnostics

    def cap_diag(*, C_pred, C_0, e_pred, e_0, t_leaf, is_root_child, prior_var):
        stash.setdefault("ep", []).append(e_pred.detach().cpu().numpy().reshape(-1))
        stash.setdefault("e0", []).append(e_0.detach().cpu().numpy().reshape(-1))
        return orig_diag(C_pred=C_pred, C_0=C_0, e_pred=e_pred, e_0=e_0,
                         t_leaf=t_leaf, is_root_child=is_root_child, prior_var=prior_var)

    out = {}
    flowmod.compute_flow_diagnostics = cap_diag
    try:
        for t in t_grid:
            method.diffusion._sample_time = (lambda n, dev, _t=t: th.full((n,), float(_t), device=dev))
            stash.clear()
            with th.no_grad():
                for batch in batches:
                    method.get_loss(batch.to(device), model)
            ep = np.concatenate(stash["ep"]); e0 = np.concatenate(stash["e0"])
            pred = ep > 0.0; true = e0 > 0.0
            out[round(float(t), 3)] = {
                "acc": float((pred == true).mean()),
                "auc": _auc(ep.astype(np.float64), true),
                "base_rate": float(true.mean()),
                "mean_abs_epred": float(np.abs(ep).mean()),
                "n": int(e0.size),
            }
    finally:
        method.diffusion._sample_time = orig_time
        flowmod.compute_flow_diagnostics = orig_diag
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--config-name", default="neuron_dataset_run_3")
    ap.add_argument("--config-dir", default=str(REPO / "config"))
    ap.add_argument("--steps", type=int, nargs="+", default=[2000, 9000, 30000, 60000])
    ap.add_argument("--t-grid", type=float, nargs="+", default=[0.05, 0.25, 0.5, 0.75, 0.95])
    ap.add_argument("--n-graphs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="exp_audit.pkl")
    args = ap.parse_args()

    th.set_float32_matmul_precision("high")
    with initialize_config_dir(version_base="1.3", config_dir=args.config_dir):
        cfg = compose(config_name=args.config_name)
    batches, n = _build_eval_batches(cfg, args.eval_dir, args.n_graphs, args.batch_size)
    print(f"{n} eval graphs -> {len(batches)} batches; t-grid={args.t_grid}")
    model, method = build_model_diffusion_method(cfg)
    model = model.to(args.device); method = method.to(args.device)

    results = {"by_step": {}, "t_grid": args.t_grid}
    for step in args.steps:
        ck = Path(args.ckpt_dir) / f"step_{step}.pt"
        if not ck.exists():
            print(f"[skip] {ck}"); continue
        load_ckpt(model, str(ck), args.device)
        res = expansion_by_t(method, model, batches, args.device, args.t_grid)
        results["by_step"][step] = res
        print(f"\nstep {step} (base_rate={list(res.values())[0]['base_rate']:.3f}):")
        print("   t    acc    auc   mean|e_pred|")
        for t, m in sorted(res.items()):
            print(f"  {t:.2f}  {m['acc']:.3f}  {m['auc']:.3f}   {m['mean_abs_epred']:.2f}")
        with open(args.out, "wb") as f:
            pickle.dump(results, f)
    with open(args.out, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved -> {args.out}")
    print("Read: high-t acc≈1.0 = pipeline OK. low-t acc/auc = real skill; does it rise across steps?")


if __name__ == "__main__":
    main()
