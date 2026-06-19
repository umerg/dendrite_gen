"""Teacher-forced distribution-level evaluation suite.

At each GT reduction level (teacher-forced = the partial tree is GT), run the FULL flow
sampler to produce the next-step new-leaf offsets + expansion decisions, then pool the
produced *local* morphometrics across all trees/levels and compare to the GT pools using
the SAME W1/KS distances the free-running validation uses. Swept across checkpoints, this
yields val curves that (a) correspond directly to the training task and (b) live in the
same units as the free-running `*_w1`, so the teacher-forced ↔ free-running (exposure) gap
reads off directly.

Design: `get_loss` already builds the correct teacher-forced inputs (GT positions, local
frames, GT child-identity one-hot, pre_geom). We intercept its `self.diffusion(...)` call
and route those same inputs into `diffusion.sample` (full ODE) instead of the training
forward — reusing the entire input construction. Only the new leaves start from noise; their
parents/context are GT, so this is teacher-forced.

This module is post-hoc (a CLI sweeps `step_*.pt`) and is NOT wired into `run_validation`;
`evaluate_teacher_forced(method, model, batches, uhat)` is a pure function so it can be later.
"""
from __future__ import annotations

import math
import numpy as np
import torch as th
from scipy.stats import rankdata

from validation.dist_metrics import _w1, _ks  # identical units to the free-running metrics
from graph_generation.method.helpers import (
    decode_parent_indices,
    select_training_leaf_indices,
)

_AXES = ("fwd", "side", "axial")


# --------------------------------------------------------------------------- helpers
def compute_node_depths(parent_idx: th.Tensor) -> np.ndarray:
    """Depth-from-root for a (batched) forest; parent_idx 0-based global, <0 for roots."""
    p = parent_idx.detach().cpu().numpy().astype(np.int64)
    N = p.shape[0]
    depth = np.zeros(N, np.int64)
    for _ in range(N):
        new = np.where(p < 0, 0, depth[np.clip(p, 0, N - 1)] + 1)
        if np.array_equal(new, depth):
            break
        depth = new
    return depth


def _sibling_angles(off_global: np.ndarray, parent_ids: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    """Pairwise sibling-branch angles (degrees) between offsets sharing a parent.

    Mirrors validation.structural_metrics.bifurcation_angle_values' pairwise formula.
    """
    angles: list[float] = []
    parent_ids = np.asarray(parent_ids)
    for p in np.unique(parent_ids):
        idx = np.where(parent_ids == p)[0]
        if idx.size < 2:
            continue
        V = off_global[idx]
        n = np.linalg.norm(V, axis=1)
        for i in range(idx.size):
            for j in range(i + 1, idx.size):
                d = float(n[i] * n[j])
                if d <= eps:
                    continue
                cos = float(np.clip(V[i] @ V[j] / d, -1.0, 1.0))
                angles.append(math.degrees(math.acos(cos)))
    return np.asarray(angles, dtype=np.float64)


def _to_global(C_local: np.ndarray, fwd: np.ndarray, side: np.ndarray, uhat: np.ndarray) -> np.ndarray:
    """local_to_global in numpy: C[:,0]*fwd + C[:,1]*side + C[:,2]*uhat."""
    return C_local[:, 0:1] * fwd + C_local[:, 1:2] * side + C_local[:, 2:3] * uhat[None, :]


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via the rank-based (Mann-Whitney U) statistic; nan if one class empty."""
    labels = labels.astype(bool)
    npos = int(labels.sum()); nneg = int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


# --------------------------------------------------------------------------- metrics
def compute_tf_distribution_metrics(
    cs_local: np.ndarray,  # [L,3] sampled offsets (local frame)
    c0_local: np.ndarray,  # [L,3] GT offsets (local frame)
    fwd: np.ndarray,       # [L,3] local forward basis
    side: np.ndarray,      # [L,3] local sideways basis
    uhat: np.ndarray,      # [3]
    leaf_parent: np.ndarray,  # [L] global parent id (for sibling grouping)
) -> dict:
    """Teacher-forced (sampled) vs GT distribution distances for the local morphometrics."""
    out: dict[str, float] = {}
    if cs_local.shape[0] == 0:
        return out

    bl_s = np.linalg.norm(cs_local, axis=1)
    bl_g = np.linalg.norm(c0_local, axis=1)
    out["branch_length_w1"] = _w1(bl_s, bl_g)
    out["branch_length_ks"] = _ks(bl_s, bl_g)

    # decomposed per-axis offset: signed component + magnitude
    for a, name in enumerate(_AXES):
        out[f"{name}_signed_w1"] = _w1(cs_local[:, a], c0_local[:, a])
        out[f"{name}_signed_ks"] = _ks(cs_local[:, a], c0_local[:, a])
        out[f"{name}_mag_w1"] = _w1(np.abs(cs_local[:, a]), np.abs(c0_local[:, a]))

    # turning angle psi = angle between offset and forward axis = acos(C_fwd / |C|)
    with np.errstate(invalid="ignore", divide="ignore"):
        psi_s = np.degrees(np.arccos(np.clip(cs_local[:, 0] / np.where(bl_s > 0, bl_s, np.nan), -1, 1)))
        psi_g = np.degrees(np.arccos(np.clip(c0_local[:, 0] / np.where(bl_g > 0, bl_g, np.nan), -1, 1)))
        af_s = np.abs(cs_local[:, 2]) / np.where(bl_s > 0, bl_s, np.nan)
        af_g = np.abs(c0_local[:, 2]) / np.where(bl_g > 0, bl_g, np.nan)
    out["turning_angle_w1"] = _w1(psi_s, psi_g)
    out["turning_angle_ks"] = _ks(psi_s, psi_g)
    out["axial_frac_w1"] = _w1(af_s, af_g)

    # bifurcation angle: needs global offsets, grouped by parent
    gs = _to_global(cs_local, fwd, side, uhat)
    gg = _to_global(c0_local, fwd, side, uhat)
    ba_s = _sibling_angles(gs, leaf_parent)
    ba_g = _sibling_angles(gg, leaf_parent)
    out["bifurcation_angle_w1"] = _w1(ba_s, ba_g)
    out["bifurcation_angle_ks"] = _ks(ba_s, ba_g)
    return out


def compute_tf_expansion_metrics(e_samp: np.ndarray, leaf_expansion: np.ndarray) -> dict:
    """Per-step expansion-decision classification: sampled e>0 vs GT expand.

    NOTE: `get_loss` passes leaf_expansion already decremented to {0,1} (expand=1),
    so the GT "expand" label is `> 0.5` (matches flow's e_0 = 2*leaf_expansion-1 > 0).
    """
    out: dict[str, float] = {}
    if e_samp.shape[0] == 0:
        return out
    pred = e_samp.reshape(-1) > 0.0
    true = leaf_expansion.reshape(-1) > 0.5
    out["n"] = float(true.size)
    out["base_rate"] = float(true.mean())
    out["acc"] = float((pred == true).mean())
    tp = float((pred & true).sum()); fp = float((pred & ~true).sum()); fn = float((~pred & true).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    out["precision"] = prec
    out["recall"] = rec
    if math.isfinite(prec) and math.isfinite(rec) and (prec + rec) > 0:
        out["f1"] = 2 * prec * rec / (prec + rec)
    out["auc"] = _auc(e_samp.reshape(-1).astype(np.float64), true)
    return out


def _metrics_from_pools(cap: dict, level_min: int = 30) -> dict:
    """Assemble overall + per-reduction-level metric blocks from pooled captured arrays."""
    res = {
        "n_leaves": int(cap["cs"].shape[0]),
        "dist": compute_tf_distribution_metrics(
            cap["cs"], cap["c0"], cap["fwd"], cap["side"], cap["uhat"], cap["lp"]),
        "exp": compute_tf_expansion_metrics(cap["es"], cap["lexp"]),
        "by_level": {},
    }
    levels = cap["level"]
    for lv in np.unique(levels):
        m = levels == lv
        if int(m.sum()) < level_min:
            continue
        res["by_level"][int(lv)] = {
            "n": int(m.sum()),
            "dist": compute_tf_distribution_metrics(
                cap["cs"][m], cap["c0"][m], cap["fwd"][m], cap["side"][m], cap["uhat"], cap["lp"][m]),
            "exp": compute_tf_expansion_metrics(cap["es"][m], cap["lexp"][m]),
        }
    return res


# --------------------------------------------------------------------------- runner
def evaluate_teacher_forced(method, model, batches, uhat, device="cpu", level_min: int = 30) -> dict:
    """Run teacher-forced sampling over GT reduction `batches`; return the metric suite.

    Pure given a built (method, model). Installs a temporary hook routing the training
    forward into `diffusion.sample` with GT context, captures per-leaf sampled offsets +
    expansion + GT targets + reduction level, then computes the metrics.
    """
    stash = {}
    cur = {}
    orig_fwd = method.diffusion.forward
    orig_sample = method.diffusion.sample

    def hook(*a, **kw):
        C_samp, e_samp = orig_sample(
            node_feats=kw.get("node_feats"), edge_index=kw["edge_index"], batch=kw["batch"],
            edge_attr=kw["edge_attr"], P_0=kw["P_0"], parent_idx=kw["parent_idx"],
            leaf_idx=kw["leaf_idx_train"], leaf_parent_idx=kw["leaf_parent_idx"],
            model=kw["model"], tmd=kw.get("tmd"),
            local_forward=kw["local_forward"], local_sideways=kw["local_sideways"],
            uhat=kw["uhat"], pre_geom_p0=kw["pre_geom_p0"],
        )
        lit = kw["leaf_idx_train"]
        leaf_graph = kw["batch"][lit]                     # [L] graph-in-batch per leaf
        levels = cur["red_level"][leaf_graph]
        # C_0, leaf_expansion, local_forward, local_sideways are already leaf-aligned ([L,...])
        stash.update(
            cs=C_samp.detach().cpu().numpy(),
            es=e_samp.detach().cpu().numpy(),
            c0=kw["C_0"].detach().cpu().numpy(),
            lexp=kw["leaf_expansion"].detach().cpu().numpy().reshape(-1),
            lp=kw["leaf_parent_idx"].detach().cpu().numpy(),
            fwd=kw["local_forward"].detach().cpu().numpy(),
            side=kw["local_sideways"].detach().cpu().numpy(),
            level=levels.detach().cpu().numpy(),
        )
        z = kw["P_0"].new_zeros(())
        return z, z, {}

    method.diffusion.forward = hook
    pools = {k: [] for k in ("cs", "es", "c0", "lexp", "lp", "fwd", "side", "level")}
    try:
        with th.no_grad():
            for batch in batches:
                batch = batch.to(device)
                num_graphs = int(batch.batch.max().item()) + 1
                rl = batch.reduction_level
                rl = rl.view(-1) if rl.dim() > 0 else rl.view(1)
                if rl.numel() != num_graphs:  # robustness fallback
                    rl = th.full((num_graphs,), int(rl.reshape(-1)[0].item()), device=batch.batch.device)
                cur["red_level"] = rl
                stash.clear()
                method.get_loss(batch, model)
                if "cs" not in stash:
                    continue
                for k in pools:
                    pools[k].append(stash[k])
    finally:
        method.diffusion.forward = orig_fwd

    if not pools["cs"]:
        return {"n_leaves": 0, "dist": {}, "exp": {}, "by_level": {}}
    cap = {k: np.concatenate(v) for k, v in pools.items()}
    cap["uhat"] = np.asarray(uhat, dtype=np.float64).reshape(3)
    return _metrics_from_pools(cap, level_min=level_min)


# --------------------------------------------------------------------------- CLI sweep
def _build_eval_batches(cfg, eval_dir, n_graphs, batch_size):
    """Deterministic fixed GT reduction batches (full coverage) from the eval trees."""
    import graph_generation as gg
    from torch_geometric.data import Batch
    from utils.data_loading import load_swc_graphs_from_dir, nx_graph_to_adj_pos
    from graph_generation.data.reduction_dataset import PrecomputedRedDataset

    psf = float(getattr(cfg.dataset, "pos_scale_factor", 1.0) or 1.0)
    graphs = load_swc_graphs_from_dir(eval_dir)
    for G in graphs:
        if G.graph.get("root") is None or G.graph["root"] not in G.nodes:
            G.graph["root"] = next(iter(G.nodes))
    graphs = graphs[:n_graphs]
    adjs, poses = [], []
    for G in graphs:
        A, P, _ = nx_graph_to_adj_pos(G)
        adjs.append(A); poses.append(P / psf)
    R = cfg.reduction
    fk = dict(mode=R.mode, cherry_p=R.cherry_p, ensure_progress=R.ensure_progress,
              root=getattr(R, "root", None), contract_root=getattr(R, "contract_root", None))
    if hasattr(R, "weighted_reduction"):
        fk["weighted_reduction"] = R.weighted_reduction
    rf = (gg.depth_reduction.DepthReductionFactory(**fk) if R.type == "depth"
          else gg.reduction.ReductionFactory(**fk))
    ds = PrecomputedRedDataset(adjs, poses, rf, tmds=None)
    samples = ds.samples
    return [Batch.from_data_list(samples[i:i + batch_size]) for i in range(0, len(samples), batch_size)], len(graphs)


def main():
    import argparse, pickle, time
    from pathlib import Path
    from hydra import initialize_config_dir, compose
    import sys
    REPO = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO / "data_analysis"))
    from seed_variance_probe import build_model_diffusion_method, load_ckpt

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--config-name", default="neuron_dataset_run_3")
    ap.add_argument("--config-dir", default=str(REPO / "config"))
    ap.add_argument("--steps", nargs="+", default=["all"], help="checkpoint steps, or 'all'")
    ap.add_argument("--n-graphs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="tf_dist.pkl")
    args = ap.parse_args()

    th.set_float32_matmul_precision("high")
    with initialize_config_dir(version_base="1.3", config_dir=args.config_dir):
        cfg = compose(config_name=args.config_name)
    uhat = np.asarray(getattr(cfg.model, "so2_axis", [0., 1., 0.]), dtype=float).reshape(3)

    print(f"device={args.device}  building fixed GT reduction batches ...")
    batches, n_used = _build_eval_batches(cfg, args.eval_dir, args.n_graphs, args.batch_size)
    print(f"  {n_used} eval graphs -> {len(batches)} batches")

    model, method = build_model_diffusion_method(cfg)
    model = model.to(args.device); method = method.to(args.device)

    ckdir = Path(args.ckpt_dir)
    if args.steps == ["all"]:
        steps = sorted(int(p.stem.split("_")[1]) for p in ckdir.glob("step_*.pt"))
    else:
        steps = [int(s) for s in args.steps]

    results = {"uhat": uhat.tolist(), "n_graphs": n_used, "by_step": {}}
    for step in steps:
        ck = ckdir / f"step_{step}.pt"
        if not ck.exists():
            print(f"[skip] {ck} not found"); continue
        t0 = time.time()
        load_ckpt(model, str(ck), args.device)
        res = evaluate_teacher_forced(method, model, batches, uhat, device=args.device)
        results["by_step"][step] = res
        d = res["dist"]
        print(f"  step {step}: tf branch_len_w1={d.get('branch_length_w1', float('nan')):.3f} "
              f"bif_angle_w1={d.get('bifurcation_angle_w1', float('nan')):.3f} "
              f"exp_acc={res['exp'].get('acc', float('nan')):.3f} "
              f"(n={res['n_leaves']}, {time.time()-t0:.0f}s)")
        with open(args.out, "wb") as f:
            pickle.dump(results, f)
    with open(args.out, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved -> {args.out}. Send me this pkl and I'll build the TF-vs-free-running curves.")


if __name__ == "__main__":
    main()
