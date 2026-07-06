#!/usr/bin/env python
"""
Teacher-forced vs free-running offset probe.

Resolves the open question: why does teacher-forced training MSE stay normal while
FREE-RUNNING (sampling) offsets inflate at the early-interior stage (depth-2)?

At a loaded checkpoint it measures, per checkpoint:
  (A) TEACHER-FORCED predicted offset scale  — runs the real `method.get_loss` on GT
      depth-reduction batches (parents/context at GT positions) and captures the model's
      predicted clean offset C_pred vs the GT target C_0 (non-invasive: wraps diffusion.forward
      + monkeypatches compute_flow_diagnostics). Reports median |C_pred|/|C_0| overall, for
      root-children, for interior nodes, and by partial-tree depth.
  (B) FREE-RUNNING realized offset scale — runs the real `method.sample_graphs` and measures
      each generated edge length by the child's depth-from-root, as a ratio to GT-by-depth.

Interpretation:
  - If TEACHER-FORCED ratio ≈ 1 at all depths (incl. interior) but FREE-RUNNING inflates at
    depth-2  => exposure bias: the head predicts correctly given GT context; the inflation is
    a free-running/compounding effect the loss never sees.
  - If TEACHER-FORCED ALSO inflates  => the offset head itself mis-scaled (weight-level).

Run on the cluster (checkpoints + data local). Example:
    conda run -n NEURO2 python data_analysis/teacher_forcing_probe.py \
        --ckpt-dir /path/to/run/checkpoints \
        --eval-dir /scratch/guptau/neurons_final/val_extended \
        --config-name neuron_dataset_run_3 \
        --steps 8000 9000 16000 60000 --n-graphs 400 --out tf_vs_fr.pkl
Then send me tf_vs_fr.pkl.
"""
import argparse, sys, time, pickle
from pathlib import Path
import numpy as np
import torch as th
import networkx as nx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "data_analysis"))
import graph_generation as gg
from hydra import initialize_config_dir, compose
from torch_geometric.data import Batch
from utils.data_loading import load_swc_graphs_from_dir, nx_graph_to_adj_pos
from graph_generation.data.reduction_dataset import PrecomputedRedDataset, RandRedDataset
import seed_variance_probe as P  # reuse build_model_diffusion_method + generate_eval_set


def node_depths_from_parent(parent_idx: th.Tensor) -> th.Tensor:
    """Depth-from-root for a (possibly batched) forest; parent_idx[n] = global parent or <0 for roots."""
    p = parent_idx.cpu().numpy().astype(np.int64)
    N = p.shape[0]; depth = np.zeros(N, np.int64)
    for _ in range(N):  # converges in <= max depth iters
        new = np.where(p < 0, 0, depth[np.clip(p, 0, N - 1)] + 1)
        if np.array_equal(new, depth):
            break
        depth = new
    return th.from_numpy(depth)


def teacher_forced(method, model, loader, device, n_batches, seed=0):
    """Run the real training forward on GT reduction batches; collect per-leaf
    |C_pred|, |C_0|, is_root_child, depth, t via `method.teacher_forced_diagnostics`
    (flow-family diffusion). No monkeypatching — the diagnostics arrays are exposed by the
    forward's `return_diag_arrays` flag."""
    rows = {"cp": [], "c0": [], "root": [], "depth": [], "t": []}
    th.manual_seed(seed)
    it = iter(loader)
    for b in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        batch = batch.to(device)
        d = method.teacher_forced_diagnostics(batch, model)
        if d["cp"] is None or d["cp"].numel() == 0:
            continue
        cp = d["cp"].norm(dim=-1).cpu().numpy()
        c0 = d["c0"].norm(dim=-1).cpu().numpy()
        root = d["root"].cpu().numpy().astype(bool).reshape(-1)
        t = d["t"].cpu().numpy().reshape(-1)
        depth_all = node_depths_from_parent(d["parent_idx"])
        dep = depth_all[d["leaf_idx_train"].cpu()].numpy()
        n = min(len(cp), len(c0), len(root), len(dep), len(t))
        rows["cp"].append(cp[:n]); rows["c0"].append(c0[:n])
        rows["root"].append(root[:n]); rows["depth"].append(dep[:n]); rows["t"].append(t[:n])
    return {k: (np.concatenate(v) if v else np.array([])) for k, v in rows.items()}


def edge_len_by_depth(graphs):
    """(child_depth, edge_length) over all edges, depth from graph['root']."""
    dep, ln = [], []
    for G in graphs:
        r = G.graph.get("root", next(iter(G.nodes)))
        d = nx.single_source_shortest_path_length(G, r)
        pos = {n: np.asarray(G.nodes[n]["pos"], float) for n in G.nodes}
        for u, v in G.edges():
            dep.append(max(d.get(u, 0), d.get(v, 0)))
            ln.append(float(np.linalg.norm(pos[u] - pos[v])))
    return np.array(dep), np.array(ln)


def ratio_by_depth(gen_dep, gen_len, gt_dep, gt_len, maxd=10, min_n=20):
    out = {}
    for dd in range(1, maxd + 1):
        gm = gen_dep == dd; tm = gt_dep == dd
        if gm.sum() >= min_n and tm.sum() >= min_n:
            out[dd] = float(np.median(gen_len[gm]) / np.median(gt_len[tm]))
    return out


def build_prefix_state(graphs, D, psf, device):
    """Batched generation state seeded from the GT subtree of depth<=D (GT topology + positions).
    Frontier = prefix leaves; they expand iff they have pruned GT children. Returns (state, target_size, nrc)."""
    from torch_geometric.typing import SparseTensor
    pos, bt, par1b, rows, cols, lidx, lexp = [], [], [], [], [], [], []
    tsz, nrc = [], []
    off = 0
    for gi, G in enumerate(graphs):
        r = G.graph.get("root", next(iter(G.nodes)))
        dep = nx.single_source_shortest_path_length(G, r)
        keep = sorted([n for n in G.nodes if dep[n] <= D], key=lambda n: (dep[n], n))
        gid = {n: off + i for i, n in enumerate(keep)}  # root gets smallest id in block
        for n in keep:
            pos.append(np.asarray(G.nodes[n]["pos"], float) / psf); bt.append(gi)
            if n == r:
                par1b.append(0)
            else:
                p = next(nb for nb in G.neighbors(n) if dep.get(nb, 1 << 30) == dep[n] - 1)
                par1b.append(gid[p] + 1)
        for n in keep:  # child edges within prefix (symmetric)
            for nb in G.neighbors(n):
                if nb in gid and dep[nb] == dep[n] + 1:
                    rows += [gid[n], gid[nb]]; cols += [gid[nb], gid[n]]
        for n in keep:  # frontier = no kept children
            kept_ch = [nb for nb in G.neighbors(n) if nb in gid and dep[nb] == dep[n] + 1]
            if not kept_ch:
                gt_ch = [nb for nb in G.neighbors(n) if dep.get(nb, -1) == dep[n] + 1]
                lidx.append(gid[n]); lexp.append(2 if gt_ch else 1)
        tsz.append(len(G)); nrc.append(int(G.degree[r])); off += len(keep)
    N = off
    L = lambda a, dt: th.tensor(a, dtype=dt, device=device)
    adj = SparseTensor(row=L(rows, th.long), col=L(cols, th.long),
                       value=th.ones(len(rows), device=device), sparse_sizes=(N, N))
    leaf_idx = L(lidx, th.long)
    leaf_mask = th.zeros(N, dtype=th.bool, device=device); leaf_mask[leaf_idx] = True
    state = dict(adj=adj, pos=th.tensor(np.array(pos), dtype=th.float, device=device),
                 batch=L(bt, th.long), parent_idx_1b=L(par1b, th.long),
                 leaf_idx=leaf_idx, leaf_expansion=L(lexp, th.long), leaf_mask=leaf_mask)
    return state, L(tsz, th.long), L(nrc, th.long)


def rollout_from_prefix(method, model, graphs, D, psf, device, tmd=None):
    """Scenario C: seed with GT depth<=D prefix, then run the real expand() rollout from there."""
    st, target_size, nrc = build_prefix_state(graphs, D, psf, device)
    adj, pos = st["adj"], st["pos"]; batch = st["batch"]
    parent_idx_1b, leaf_idx = st["parent_idx_1b"], st["leaf_idx"]
    leaf_expansion, leaf_mask = st["leaf_expansion"], st["leaf_mask"]
    eff = min(int(target_size.max().item()), method.max_tree_size); max_steps = eff * 2
    step, terminated = 0, False
    while not terminated and step < max_steps and leaf_idx.numel() > 0:
        (adj, pos, leaf_idx, leaf_expansion, parent_idx_1b, batch, leaf_mask, terminated) = method.expand(
            adj, batch, target_size, model, pos=pos, leaf_idx=leaf_idx,
            leaf_expansion=leaf_expansion, parent_idx_1b=parent_idx_1b, leaf_mask=leaf_mask,
            tmd=tmd, step=step + D, num_root_children=nrc)
        step += 1
    row, col, _ = adj.coo()
    out = []
    for g in range(int(batch.max().item()) + 1):
        node_ids = (batch == g).nonzero(as_tuple=False).flatten()
        lm = {int(n): i for i, n in enumerate(node_ids.tolist())}
        G = nx.Graph()
        for n in node_ids.tolist():
            G.add_node(lm[n], pos=pos[n].detach().cpu().numpy() * psf)
        for r, c in zip(row.tolist(), col.tolist()):
            if r in lm and c in lm and lm[r] <= lm[c]:
                G.add_edge(lm[r], lm[c])
        G.graph["root"] = 0  # prefix root has smallest global id -> local 0
        out.append(G)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--eval-dir", required=True, help="GT SWC dir (e.g. .../val_extended)")
    ap.add_argument("--config-name", default="neuron_dataset_run_3")
    ap.add_argument("--config-dir", default=str(REPO / "config"))
    ap.add_argument("--steps", type=int, nargs="+", default=[8000, 9000, 16000, 60000])
    ap.add_argument("--n-graphs", type=int, default=400, help="GT subset size for both passes")
    ap.add_argument("--tf-batches", type=int, default=8, help="# reduction batches for teacher-forced")
    ap.add_argument("--prefix-depth", type=int, default=2, help="scenario C: GT prefix depth D (rollout from D+1)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="tf_vs_fr.pkl")
    args = ap.parse_args()

    th.set_float32_matmul_precision("high")
    with initialize_config_dir(version_base="1.3", config_dir=args.config_dir):
        cfg = compose(config_name=args.config_name)
    bs = args.batch_size or getattr(cfg.validation, "batch_size", None) or cfg.training.batch_size
    psf = float(getattr(cfg.dataset, "pos_scale_factor", 1.0) or 1.0)

    print(f"device={args.device} config={args.config_name} psf={psf}")
    graphs = load_swc_graphs_from_dir(args.eval_dir)
    for G in graphs:
        if G.graph.get("root") is None or G.graph["root"] not in G.nodes:
            G.graph["root"] = next(iter(G.nodes))
    sub = graphs[: args.n_graphs]
    print(f"{len(graphs)} GT graphs; using {len(sub)} for the probe")

    # GT offset-by-depth reference (µm)
    gt_dep, gt_len = edge_len_by_depth(sub)

    # Build the GT depth-reduction dataset (teacher-forced training data), scaled like training
    R = cfg.reduction
    fk = dict(mode=R.mode, cherry_p=R.cherry_p, ensure_progress=R.ensure_progress,
              root=getattr(R, "root", None), contract_root=getattr(R, "contract_root", None))
    if hasattr(R, "weighted_reduction"):
        fk["weighted_reduction"] = R.weighted_reduction
    red_factory = (gg.depth_reduction.DepthReductionFactory(**fk)
                   if R.type == "depth" else gg.reduction.ReductionFactory(**fk))
    adjs, poses = [], []
    for G in sub:
        A, Pp, _ = nx_graph_to_adj_pos(G); adjs.append(A); poses.append(Pp / psf)
    ds = PrecomputedRedDataset(adjs, poses, red_factory, tmds=None)
    loader = th.utils.data.DataLoader(ds, batch_size=bs, collate_fn=Batch.from_data_list)

    model, method = P.build_model_diffusion_method(cfg)
    model = model.to(args.device); method = method.to(args.device)

    results = {"psf": psf, "gt_offset_um_by_depth": {}, "by_step": {}}
    for dd in range(1, 11):
        tm = gt_dep == dd
        if tm.sum() >= 20:
            results["gt_offset_um_by_depth"][dd] = float(np.median(gt_len[tm]))

    for step in args.steps:
        ck = Path(args.ckpt_dir) / f"step_{step}.pt"
        if not ck.exists():
            print(f"[skip] {ck} not found"); continue
        print(f"\n### step {step} ###")
        P.load_ckpt(model, str(ck), args.device)

        # (A) teacher-forced
        t0 = time.time()
        tf = teacher_forced(method, model, loader, args.device, args.tf_batches)
        tf_overall = float(np.median(tf["cp"]) / np.median(tf["c0"])) if tf["c0"].size else float("nan")
        rmask = tf["root"]; imask = ~tf["root"]
        tf_root = float(np.median(tf["cp"][rmask]) / np.median(tf["c0"][rmask])) if rmask.any() else float("nan")
        tf_int = float(np.median(tf["cp"][imask]) / np.median(tf["c0"][imask])) if imask.any() else float("nan")
        tf_bydepth = {}
        for dd in range(1, 11):
            m = tf["depth"] == dd
            if m.sum() >= 20:
                tf_bydepth[int(dd)] = float(np.median(tf["cp"][m]) / np.median(tf["c0"][m]))
        print(f"  [A teacher-forced]  overall |Cpred|/|C0| = {tf_overall:.3f}  "
              f"root-child={tf_root:.3f}  interior={tf_int:.3f}  ({time.time()-t0:.0f}s, n={tf['c0'].size})")
        print(f"      by partial-tree depth: " + "  ".join(f"d{d}={r:.2f}" for d, r in sorted(tf_bydepth.items())))

        # (B) free-running
        t1 = time.time()
        pg = P.generate_eval_set(model, method, sub, cfg, args.device, None, bs, seed=10000)
        g_dep, g_len = edge_len_by_depth(pg)
        fr = ratio_by_depth(g_dep, g_len, gt_dep, gt_len)
        print(f"  [B free-running]    offset/GT by depth: " +
              "  ".join(f"d{d}={r:.2f}" for d, r in sorted(fr.items())) + f"  ({time.time()-t1:.0f}s)")

        # (C) GT-prefix rollout: GT depth<=D given, model rolls out from D+1
        t2 = time.time()
        Dc = args.prefix_depth
        pgc = rollout_from_prefix(method, model, sub, Dc, psf, args.device)
        gc_dep, gc_len = edge_len_by_depth(pgc)
        frc = ratio_by_depth(gc_dep, gc_len, gt_dep, gt_len)
        print(f"  [C GT-prefix<=d{Dc}]  offset/GT by depth: " +
              "  ".join(f"d{d}={r:.2f}" for d, r in sorted(frc.items())) + f"  ({time.time()-t2:.0f}s)")

        results["by_step"][step] = {
            "tf_overall": tf_overall, "tf_root": tf_root, "tf_interior": tf_int,
            "tf_by_partial_depth": tf_bydepth, "fr_offset_ratio_by_depth": fr,
            "prefix_depth": Dc, "fr_prefix_offset_ratio_by_depth": frc,
            "tf_n_leaves": int(tf["c0"].size),
        }
        with open(args.out, "wb") as f:
            pickle.dump(results, f)

    with open(args.out, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved -> {args.out}")
    print("Verdict per step: teacher-forced ratios ≈1 at all depths + free-running inflated at d2 "
          "=> exposure bias. Teacher-forced ALSO inflated => head mis-scaled. Send me the pkl.")


if __name__ == "__main__":
    main()
