#!/usr/bin/env python
"""
Airtight sampling-noise vs weight-wander probe.

Run this ON THE CLUSTER, where the checkpoints (step_*.pt) and the SWC data live.
It reuses the *exact* validation pipeline (same model, flow sampler, sample_graphs,
build_gt_cache, compute_distribution_metrics) so the numbers are directly comparable
to the logged validation metrics — no bootstrap proxy.

Two experiments:

  EXP-A  (the airtight sampling-noise measurement; this is what makes the claim watertight)
      Load ONE frozen checkpoint, generate the full eval set N times with N different
      random seeds (weights held fixed), recompute the validation metrics each time.
      The std across the N runs = the TRUE per-eval sampling (seed) noise, sigma_seed.

  EXP-B  (optional, --wander-steps)
      Re-evaluate a sweep of checkpoints, ONE fresh seed each. The std across checkpoints
      = total checkpoint-to-checkpoint variance, sigma_total, measured consistently.

  Decomposition:  sigma_weight_wander = sqrt(max(0, sigma_total^2 - sigma_seed^2)).
  (If you skip EXP-B, compare sigma_seed to the sigma_obs already measured from the
   60 logged pkls — that alone settles whether the swings are sampling noise.)

Example (start small to gauge runtime, then scale up):
    conda run -n NEURO2 python data_analysis/seed_variance_probe.py \
        --ckpt-dir /scratch/guptau/<run>/checkpoints \
        --eval-dir /scratch/guptau/neurons_final/val_extended \
        --config-name neuron_dataset_run_3 \
        --seedvar-steps 9000 30000 60000 \
        --n-seeds 16 \
        --out seed_variance_results.pkl

Then send me seed_variance_results.pkl and I'll fold it into the report.
"""
import argparse, sys, time, pickle
from pathlib import Path

import numpy as np
import torch as th

# --- repo imports (run from the repo root) ---
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import graph_generation as gg
from hydra import initialize_config_dir, compose
from utils.data_loading import load_swc_graphs_from_dir
from utils.tmd import compute_tmd_mixed, compute_tmd_embedding
from validation.dist_metrics import compute_distribution_metrics, build_gt_cache

# Metrics we report variance for (must exist in the dist dict).
HEADLINE = [
    "w1_pooled_mean_normalized", "w1_pertree_mean_normalized",
    "mmd_morpho", "mmd_tmd", "coverage_morpho", "coverage_tmd",
    "node_count_w1", "leaf_count_w1", "branch_length_w1",
    "bifurcation_angle_w1", "radial_span_w1", "axial_extent_w1",
    "total_extent_w1", "tmd_barlen_w1", "strahler_w1", "partition_asymmetry_w1",
]
# Late-phase sigma_obs from the 60 logged pkls (for an immediate verdict; see report §7).
LATE_SIGMA_OBS = {
    "node_count_w1": 2.12, "leaf_count_w1": 1.07, "branch_length_w1": 2.31,
    "radial_span_w1": 9.93, "axial_extent_w1": 7.71, "total_extent_w1": 11.42,
    "w1_pooled_mean_normalized": 0.0305,
}


def build_model_diffusion_method(cfg):
    """Replicates main.py construction WITHOUT loading train data."""
    # ---- diffusion ----
    dcfg = cfg.diffusion
    name = getattr(dcfg, "name", None)
    if name == "basic":
        diffusion = gg.diffusion.DenoisingDiffusionModel(num_steps=dcfg.num_steps)
    elif name == "edm":
        diffusion = gg.diffusion.EDMDiffusionModel(num_steps=dcfg.num_steps)
    elif name in ("flow", "flow_v"):
        psp = getattr(dcfg, "prior_std_pos", None)
        flow_cls = (
            gg.diffusion.VFlowMatchingModel if name == "flow_v"
            else gg.diffusion.FlowMatchingModel
        )
        diffusion = flow_cls(
            num_steps=dcfg.num_steps,
            prior_std=getattr(dcfg, "prior_std", 1.0),
            time_dist=getattr(dcfg, "time_dist", "uniform"),
            beta_a=getattr(dcfg, "beta_a", 2.0),
            beta_b=getattr(dcfg, "beta_b", 1.0),
            sigma_min=getattr(dcfg, "sigma_min", 0.0),
            prior_std_pos=(list(psp) if psp is not None else None),
        )
    else:
        raise ValueError(f"Unknown diffusion name: {name}")

    # ---- model (egnn) ----
    edge_embedding_nums, edge_embedding_dims, edge_attr_dim = [2], [4], 1
    if cfg.method.name == "expansion_augmented":
        edge_embedding_nums = [3]
    m = cfg.model
    model = gg.model.SO2_EGNN_Network(
        n_layers=m.num_layers, feats_dim=m.feats_dim, pos_dim=3, m_dim=m.m_dim,
        edge_embedding_nums=edge_embedding_nums, edge_embedding_dims=edge_embedding_dims,
        edge_attr_dim=edge_attr_dim, dropout=m.dropout, norm_feats=m.norm_feats,
        global_linear_attn_every=m.global_linear_attn_every,
        global_linear_attn_heads=m.global_linear_attn_heads,
        global_linear_attn_dim_head=m.global_linear_attn_dim_head,
        num_global_tokens=m.num_global_tokens, offset_head_hidden=m.offset_head_hidden,
        tmd_in_dim=getattr(m, "tmd_in_dim", 0), tmd_hidden_dim=getattr(m, "tmd_hidden_dim", 0),
        so2_axis=m.so2_axis,
    )

    # ---- method ----
    method = gg.method.Expansion(
        diffusion=diffusion,
        red_threshold=cfg.reduction.red_threshold,
        expansion_loss_weight=getattr(cfg.method, "expansion_loss_weight", 1.0),
        use_size_ratio=getattr(cfg.method, "use_size_ratio", True),
        max_tree_size=getattr(cfg.method, "max_tree_size", 500),
        # Variant 1: must mirror main.py so the TF/sampling path pins clean GT expansion.
        predict_positions_only=getattr(cfg.method, "predict_positions_only", False),
        given_topology=getattr(cfg.method, "given_topology", False),
    )
    return model, method


@th.no_grad()
def generate_eval_set(model, method, eval_graphs, cfg, device, tmds, batch_size, seed):
    """One full generation pass over the eval set at the given seed (mirrors Trainer.evaluate)."""
    th.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        th.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    pred_perm = rng.permutation(np.arange(len(eval_graphs)))
    target_size = np.array([len(g) for g in eval_graphs])[pred_perm]
    nrc_all = np.array([
        g.degree[g.graph["root"]] if "root" in g.graph else 2 for g in eval_graphs
    ])[pred_perm]
    tmds_perm = tmds[pred_perm] if tmds is not None else None

    batches = [target_size[i:i + batch_size] for i in range(0, len(target_size), batch_size)]
    nrc_batches = [nrc_all[i:i + batch_size] for i in range(0, len(nrc_all), batch_size)]

    pred_graphs, cursor = [], 0
    for batch, nrc_batch in zip(batches, nrc_batches):
        tmd_batch = None
        if tmds_perm is not None:
            tmd_batch = th.from_numpy(tmds_perm[cursor:cursor + len(batch)]).to(device)
        pg = method.sample_graphs(
            target_size=th.tensor(batch, device=device),
            model=model, tmd=tmd_batch,
            num_root_children=th.tensor(nrc_batch, device=device),
        )
        pred_graphs += pg
        cursor += len(batch)

    inv = np.empty_like(pred_perm)
    inv[pred_perm] = np.arange(len(pred_perm))
    pred_graphs = [pred_graphs[i] for i in inv]

    psf = cfg.dataset.get("pos_scale_factor", None) if hasattr(cfg.dataset, "get") else getattr(cfg.dataset, "pos_scale_factor", None)
    if psf is not None:
        for G in pred_graphs:
            for n in G.nodes():
                G.nodes[n]["pos"] = G.nodes[n]["pos"] * float(psf)
    for G in pred_graphs:
        if "root" not in G.graph or G.graph.get("root") not in G.nodes:
            G.graph["root"] = 0 if G.number_of_nodes() > 0 else None
    if device == "cuda":
        th.cuda.empty_cache()
    return pred_graphs


def eval_metrics(pred_graphs, eval_graphs, cfg, uhat_np, gt_cache, embed_fn):
    return compute_distribution_metrics(
        pred_graphs, eval_graphs, uhat=uhat_np,
        ged_enabled=False,  # tree-edit distance not analysed; skip for speed
        enable_ks=True, enable_morphometrics=True, enable_light_joint=True,
        gt_cache=gt_cache, embed_fn=embed_fn,
        dc_k=getattr(cfg.validation, "dc_nearest_k", 5),
        tmd_pca_ncomp=getattr(cfg.validation, "tmd_pca_ncomp", 32),
    )


def load_ckpt(model, path, device):
    state = th.load(path, map_location=device)
    if "model" not in state:
        raise KeyError(f"'model' key not in checkpoint {path}; keys={list(state)[:8]}")
    model.load_state_dict(state["model"])
    model.eval()


def summarize(runs, label):
    """runs: list of dist dicts. Print per-metric mean/std/CV."""
    print(f"\n{'='*92}\n{label}  (N={len(runs)} runs)\n{'='*92}")
    print(f"{'metric':<30}{'mean':>11}{'std(seed)':>11}{'CV%':>8}{'min':>10}{'max':>10}{'vs σ_obs':>11}")
    out = {}
    for k in HEADLINE:
        vals = np.array([r.get(k, np.nan) for r in runs], float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        mu, sd = vals.mean(), vals.std()
        cv = 100 * sd / abs(mu) if mu else np.nan
        so = LATE_SIGMA_OBS.get(k)
        frac = (sd / so) if so else None
        verdict = (f"{frac:.2f}×" if frac is not None else "")
        out[k] = dict(mean=float(mu), std=float(sd), cv=float(cv), min=float(vals.min()), max=float(vals.max()))
        print(f"{k:<30}{mu:>11.4g}{sd:>11.4g}{cv:>8.1f}{vals.min():>10.4g}{vals.max():>10.4g}{verdict:>11}")
    print("\n  'vs σ_obs' = sigma_seed / late-phase sigma_obs (from logged run).")
    print("  If << 1, the per-eval sampling noise is far smaller than the checkpoint-to-checkpoint")
    print("  swings => the swings are weight-wander (no EMA), NOT sampling noise.  [watertight]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="folder containing step_*.pt")
    ap.add_argument("--eval-dir", required=True, help="GT eval SWC dir (e.g. .../val_extended)")
    ap.add_argument("--config-name", default="neuron_dataset_run_3")
    ap.add_argument("--config-dir", default=str(REPO / "config"))
    ap.add_argument("--seedvar-steps", type=int, nargs="+", default=[9000, 30000, 60000])
    ap.add_argument("--n-seeds", type=int, default=16)
    ap.add_argument("--wander-steps", type=int, nargs="*", default=[],
                    help="optional: checkpoints to re-eval once each (one fresh seed) for sigma_total")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="seed_variance_results.pkl")
    args = ap.parse_args()

    th.set_float32_matmul_precision("high")
    with initialize_config_dir(version_base="1.3", config_dir=args.config_dir):
        cfg = compose(config_name=args.config_name)
    bs = args.batch_size or getattr(cfg.validation, "batch_size", None) or cfg.training.batch_size

    print(f"device={args.device}  config={args.config_name}  batch_size={bs}")
    print("loading eval graphs ...")
    eval_graphs = load_swc_graphs_from_dir(args.eval_dir)
    for G in eval_graphs:  # mirror main.py _ensure_root
        if G.graph.get("root", None) is None or G.graph["root"] not in G.nodes:
            G.graph["root"] = next(iter(G.nodes))
    print(f"  {len(eval_graphs)} eval graphs")

    model, method = build_model_diffusion_method(cfg)
    model = model.to(args.device); method = method.to(args.device)
    uhat_np = model.uhat.detach().cpu().numpy().reshape(-1) if getattr(model, "uhat", None) is not None else np.array([0., 0., 1.])

    # conditioning TMDs only if the model uses them (unconditional run -> 0 -> None)
    tmds = None
    if getattr(model, "tmd_hidden_dim", 0) > 0:
        print("precomputing conditioning TMDs (compute_tmd_mixed) ...")
        tmds = np.stack([compute_tmd_mixed(g) for g in eval_graphs], axis=0)

    # GT cache + embed fn (fixed across all runs)
    print("building GT cache (compute_tmd_embedding over eval set) ...")
    tmd_bins = getattr(cfg.validation, "tmd_eval_bins", 16)
    filtration = getattr(cfg.validation, "tmd_eval_filtration", "radial_root")
    embed_fn = lambda G: compute_tmd_embedding(G, filtration=filtration, n_bins=tmd_bins)
    gt_cache = build_gt_cache(eval_graphs, uhat=tuple(uhat_np.astype(float).tolist()),
                              embed_fn=embed_fn, tmd_pca_ncomp=getattr(cfg.validation, "tmd_pca_ncomp", 32))

    results = {"config": args.config_name, "n_eval": len(eval_graphs), "exp_a": {}, "exp_b": {}}

    def ckpt_path(step):
        return str(Path(args.ckpt_dir) / f"step_{step}.pt")

    # ---------- EXP-A: seed variance at frozen checkpoints ----------
    for step in args.seedvar_steps:
        p = ckpt_path(step)
        if not Path(p).exists():
            print(f"[skip] {p} not found"); continue
        print(f"\n### EXP-A checkpoint step {step}: {args.n_seeds} fresh generations ###")
        load_ckpt(model, p, args.device)
        runs = []
        for i in range(args.n_seeds):
            t0 = time.time()
            seed = 10_000 + i
            pg = generate_eval_set(model, method, eval_graphs, cfg, args.device, tmds, bs, seed)
            dist = eval_metrics(pg, eval_graphs, cfg, uhat_np, gt_cache, embed_fn)
            runs.append(dist)
            print(f"  seed {seed}: w1_pooled={dist.get('w1_pooled_mean_normalized', float('nan')):.4f} "
                  f"node_w1={dist.get('node_count_w1', float('nan')):.2f} ({time.time()-t0:.0f}s)")
        results["exp_a"][step] = {"runs": runs, "summary": summarize(runs, f"EXP-A seed variance @ step {step}")}
        with open(args.out, "wb") as f:  # checkpoint progress to disk
            pickle.dump(results, f)

    # ---------- EXP-B (optional): one fresh seed per checkpoint ----------
    if args.wander_steps:
        print(f"\n### EXP-B: one fresh seed per checkpoint over {len(args.wander_steps)} steps ###")
        runs_b = []
        for j, step in enumerate(args.wander_steps):
            p = ckpt_path(step)
            if not Path(p).exists():
                print(f"[skip] {p} not found"); continue
            load_ckpt(model, p, args.device)
            pg = generate_eval_set(model, method, eval_graphs, cfg, args.device, tmds, bs, seed=777 + j)
            dist = eval_metrics(pg, eval_graphs, cfg, uhat_np, gt_cache, embed_fn)
            runs_b.append({"step": step, **dist})
            print(f"  step {step}: w1_pooled={dist.get('w1_pooled_mean_normalized', float('nan')):.4f}")
        results["exp_b"] = {"runs": runs_b, "summary": summarize([r for r in runs_b], "EXP-B sigma_total (fresh seed per ckpt)")}
        with open(args.out, "wb") as f:
            pickle.dump(results, f)

    with open(args.out, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved -> {args.out}")
    print("Send me this pkl and I'll fold sigma_seed (and sigma_wander, if EXP-B) into the report.")


if __name__ == "__main__":
    main()
