"""Offline SO(2) degeneracy / target-outlier scan of the neuron dataset.

Purpose (Hypothesis H2): the flow model's training loss spikes hard and transiently
(then recovers over ~20-30k steps), corrupting the backbone. There is no gradient
clipping, so a rare batch with a large gradient knocks the weights into a bad region.
The flow loss is data-prediction MSE (bounded across t), and the SO(2) geometry
features (cosψ, cosθ, unit frames) are bounded by construction and computed under
`th.no_grad()` — so they cannot blow up the gradient directly. The remaining
data-side lever is the *regression target itself*: C_0 = global_to_local(child-parent
offset). A neuron with an extreme branch offset gives a large ||C_0||, hence a large
MSE residual and a large (unclipped) gradient on the model parameters.

This script runs the REAL training pipeline (load SWC -> scale by pos_scale ->
depth reduction -> precompute_full_geometry -> global_to_local), exactly as
`Expansion.get_loss` does, via a capture stand-in for the diffusion module (same
trick as tests/analyse_c0_distribution.py). For every training leaf / node it records:

  - ||C_0||                  : local-frame target offset magnitude (HEADLINE H2 signal)
  - nin, nout                : perpendicular norms of v_in / v_out (near-axial branches)
  - root child min perp-dist : how close a root's children sit to the uhat axis
  - |cosψ|, |cosθ|           : branch-angle node features (near 1 => near-collinear)
  - non-finite counts        : any NaN/Inf in C_0 / frames / pre_geom (should be 0)
  - ||P_0|| coord magnitude  : raw input position scale fed to the EGNN

It prints tail fractions + percentiles, saves histograms, and writes a CSV of the
worst offenders (by ||C_0|| and by smallest nin/nout) with their source SWC file so
they can be eyeballed. No training run required.

Usage:
    conda run -n NEURO2 python data_analysis/so2_degeneracy_scan.py \
        --data-dir /Users/umer/Documents/neurons_final/train --num-graphs 2000

A heavy ||C_0|| tail (max far beyond the prior scale [0.74, 0.61, 0.83]) or any
non-finite value CONFIRMS H2 as a contributor. A clean distribution largely rules it
out, leaving H1 (unclipped ordinary loss variance) as the sole driver.
"""
import argparse
import csv
import heapq
import random
import sys
from pathlib import Path

import numpy as np
import torch as th
from torch.nn import Module
from torch_geometric.data import Batch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path when running the script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import graph_generation as gg
from graph_generation.method.helpers import _compute_tree_directions
from utils.data_loading import load_swc_graph, nx_graph_to_adj_pos

AXES = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}
# Per-axis prior std of C_0 in scaled space (from config/diffusion/flow25.yaml).
PRIOR_STD_POS = (0.74, 0.61, 0.83)


class CaptureDiffusion(Module):
    """Records geometry / target diagnostics from the real get_loss pipeline.

    `Expansion.get_loss` calls this in place of the flow module, passing the same
    kwargs the real diffusion forward receives. We never run the model — we just
    read C_0, pre_geom_p0 and the raw positions, then return zero loss.
    """

    cond_dim = 2

    def __init__(self, uhat_np, topk=20):
        super().__init__()
        self.uhat_np = np.asarray(uhat_np, dtype=np.float64)
        self.topk = topk
        # Flat accumulators (numpy arrays appended per batch).
        self.c0_norms = []
        self.nin = []
        self.nout = []
        self.root_min_perp = []
        self.abs_cospsi = []
        self.abs_costheta = []
        self.p0_absmax = []
        # Counters.
        self.n_leaves = 0
        self.n_nodes = 0
        self.n_roots = 0
        self.nonfinite = {"C_0": 0, "local_forward": 0, "local_sideways": 0, "pre_geom": 0}
        # Worst-offender heaps: (key, record). For ||C_0|| we keep the largest;
        # for nin/nout we keep the smallest (negate the key).
        self._worst_c0 = []      # max-heap on ||C_0|| via min-heap of (val, ...)
        self._worst_nin = []     # smallest nin via max-heap of (-val, ...)
        # Set by the driver before each graph's batches are processed.
        self.current_tag = "?"
        self.current_red = 0

    # ---- helpers -------------------------------------------------------
    def _push_worst(self, heap, key, record, largest):
        sign = 1.0 if largest else -1.0
        item = (sign * key, record)
        if len(heap) < self.topk:
            heapq.heappush(heap, item)
        else:
            heapq.heappushpop(heap, item)

    def forward(self, *, C_0, leaf_expansion, **kw):
        P_0 = kw["P_0"]
        parent_idx = kw["parent_idx"]
        uhat = kw["uhat"]
        leaf_idx_train = kw["leaf_idx_train"]
        pre_geom = kw.get("pre_geom_p0", None)
        local_forward = kw.get("local_forward", None)
        local_sideways = kw.get("local_sideways", None)

        # --- non-finite checks (these should never trigger) ---
        if C_0.numel() and not th.isfinite(C_0).all():
            self.nonfinite["C_0"] += int((~th.isfinite(C_0)).any(dim=-1).sum().item())
        if local_forward is not None and local_forward.numel() and not th.isfinite(local_forward).all():
            self.nonfinite["local_forward"] += int((~th.isfinite(local_forward)).any(dim=-1).sum().item())
        if local_sideways is not None and local_sideways.numel() and not th.isfinite(local_sideways).all():
            self.nonfinite["local_sideways"] += int((~th.isfinite(local_sideways)).any(dim=-1).sum().item())
        if pre_geom is not None:
            for v in pre_geom.values():
                if th.is_tensor(v) and v.numel() and v.is_floating_point() and not th.isfinite(v).all():
                    self.nonfinite["pre_geom"] += 1
                    break

        # --- ||C_0|| (the headline target-outlier signal) ---
        if C_0.numel():
            c0n = C_0.detach().float().norm(dim=-1).cpu().numpy()
            self.c0_norms.append(c0n)
            self.n_leaves += c0n.size
            # Worst leaves by ||C_0||, tagged with their parent->child offset.
            order = np.argsort(c0n)[::-1][: self.topk]
            for j in order:
                self._push_worst(
                    self._worst_c0,
                    float(c0n[j]),
                    {
                        "file": self.current_tag,
                        "red": self.current_red,
                        "c0_norm": float(c0n[j]),
                        "C_0": [round(float(x), 4) for x in C_0[j].tolist()],
                    },
                    largest=True,
                )

        # --- branch-angle node features for the training leaves ---
        if pre_geom is not None and leaf_idx_train.numel():
            li = leaf_idx_train
            for name, store in (("cospsi_node", self.abs_cospsi), ("cos_theta_node", self.abs_costheta)):
                t = pre_geom.get(name, None)
                if th.is_tensor(t) and t.numel():
                    store.append(t[li].detach().float().abs().view(-1).cpu().numpy())

        # --- v_in / v_out perpendicular norms via the real helper ---
        dirs = _compute_tree_directions(P_0.detach(), parent_idx, uhat)
        has_parent = dirs["has_parent"]
        if has_parent.any():
            sel = has_parent.nonzero(as_tuple=False).flatten()
            nin = dirs["nin"].view(-1)[sel].detach().float().cpu().numpy()
            nout = dirs["nout"].view(-1)[sel].detach().float().cpu().numpy()
            self.nin.append(nin)
            self.nout.append(nout)
            self.n_nodes += nin.size
            order = np.argsort(nin)[: self.topk]
            for j in order:
                gidx = int(sel[j].item())
                self._push_worst(
                    self._worst_nin,
                    float(nin[j]),
                    {
                        "file": self.current_tag,
                        "red": self.current_red,
                        "nin": float(nin[j]),
                        "nout": float(nout[j]),
                    },
                    largest=False,
                )

        # --- per-root minimum child perp-distance (root-frame degeneracy) ---
        uhat_np = uhat.detach().cpu().numpy().reshape(3).astype(np.float64)
        P0_np = P_0.detach().cpu().numpy().astype(np.float64)
        par_np = parent_idx.detach().cpu().numpy()
        roots = np.where(par_np < 0)[0]
        for r in roots:
            kids = np.where(par_np == r)[0]
            if kids.size == 0:
                continue
            off = P0_np[kids] - P0_np[r]
            perp = off - (off @ uhat_np)[:, None] * uhat_np[None, :]
            pd = np.linalg.norm(perp, axis=1)
            self.root_min_perp.append(float(pd.min()))
            self.n_roots += 1

        # --- raw input position scale ---
        if P_0.numel():
            self.p0_absmax.append(P_0.detach().float().abs().amax(dim=-1).cpu().numpy())

        z = C_0.new_zeros(())
        return z, z


def _flat(chunks):
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)


def _tail_below(x, thresholds):
    return {t: float((x < t).mean()) if x.size else 0.0 for t in thresholds}


def _tail_above(x, thresholds):
    return {t: float((x > t).mean()) if x.size else 0.0 for t in thresholds}


def _pct(x, ps):
    return {p: float(np.percentile(x, p)) if x.size else float("nan") for p in ps}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("/Users/umer/Documents/neurons_final/train"))
    ap.add_argument("--num-graphs", type=int, default=2000, help="-1 = all")
    ap.add_argument("--reductions-per-graph", type=int, default=1)
    ap.add_argument("--pos-scale", type=float, default=45.1)
    ap.add_argument("--axis", choices=list(AXES), default="y")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()

    uhat = AXES[args.axis]
    files = sorted(f for f in args.data_dir.iterdir() if f.suffix == ".swc")
    random.Random(args.seed).shuffle(files)
    if args.num_graphs >= 0:
        files = files[: args.num_graphs]
    print(f"Scanning {len(files)} SWC graphs from {args.data_dir} "
          f"(axis={args.axis}, pos_scale={args.pos_scale}, reductions/graph={args.reductions_per_graph})")

    # Model never runs (capture short-circuits), but get_loss needs an instance.
    model = gg.model.SO2_EGNN_Network(
        n_layers=2, feats_dim=4, pos_dim=3, m_dim=16, edge_attr_dim=1, so2_axis=uhat,
    )
    cap = CaptureDiffusion(uhat_np=uhat, topk=args.topk)
    method = gg.method.Expansion(diffusion=cap)

    n_graphs_ok = n_skip = 0
    for gi, f in enumerate(files):
        try:
            G = load_swc_graph(f)
            A, P, _ = nx_graph_to_adj_pos(G)
        except Exception as exc:  # noqa: BLE001
            n_skip += 1
            if n_skip <= 5:
                print(f"  skip {f.name}: {exc}")
            continue
        P = P / args.pos_scale
        cap.current_tag = f.name
        for r in range(args.reductions_per_graph):
            cap.current_red = r
            th.manual_seed(args.seed + r)
            np.random.seed(args.seed + r)
            red_factory = gg.depth_reduction.DepthReductionFactory(
                mode="stochastic", cherry_p=1.0, ensure_progress=True, root=0, contract_root=False,
            )
            ds = gg.data.PrecomputedRedDataset(adjs=[A], poses=[P], red_factory=red_factory, tmds=None)
            samples = ds.samples
            for i in range(0, len(samples), args.bs):
                batch = Batch.from_data_list(samples[i:i + args.bs])
                try:
                    method.get_loss(batch, model)
                except Exception as exc:  # noqa: BLE001
                    n_skip += 1
                    if n_skip <= 5:
                        print(f"  batch skip ({f.name} r{r}): {exc}")
        n_graphs_ok += 1
        if n_graphs_ok % 250 == 0:
            print(f"  ...{n_graphs_ok}/{len(files)} graphs, {cap.n_leaves} leaves so far")

    c0 = _flat(cap.c0_norms)
    nin = _flat(cap.nin)
    nout = _flat(cap.nout)
    rmp = np.asarray(cap.root_min_perp, dtype=np.float64)
    cospsi = _flat(cap.abs_cospsi)
    costheta = _flat(cap.abs_costheta)
    p0max = _flat(cap.p0_absmax)

    if c0.size == 0:
        print("No C_0 captured — aborting.")
        sys.exit(1)

    prior_norm = float(np.linalg.norm(PRIOR_STD_POS))  # ~1.28; typical ||C_0|| scale
    line = "=" * 78
    print("\n" + line)
    print(f"SO(2) DEGENERACY / TARGET-OUTLIER SCAN — {n_graphs_ok} graphs, "
          f"{cap.n_leaves} training leaves, {cap.n_nodes} nodes, {cap.n_roots} roots")
    print(line)

    print("\n[HEADLINE] ||C_0|| local-frame target offset magnitude")
    print(f"  reference prior scale ||PRIOR_STD_POS|| = {prior_norm:.3f}  (axes {PRIOR_STD_POS})")
    print(f"  mean={c0.mean():.3f}  median={np.median(c0):.3f}  std={c0.std():.3f}")
    pc = _pct(c0, [50, 90, 95, 99, 99.9, 100])
    print("  percentiles: " + "  ".join(f"p{p}={v:.3f}" for p, v in pc.items()))
    for mult in (3, 5, 10):
        print(f"  frac ||C_0|| > {mult}x prior ({mult*prior_norm:.2f}): {float((c0 > mult*prior_norm).mean()):.4%}")

    print("\n[geometry] v_in / v_out perpendicular norms (near-axial branches)")
    print(f"  nin : median={np.median(nin):.4f}  min={nin.min():.2e}  "
          f"frac<1e-2={_tail_below(nin,[1e-2])[1e-2]:.4%}  frac<1e-4={_tail_below(nin,[1e-4])[1e-4]:.4%}  "
          f"frac<1e-6={_tail_below(nin,[1e-6])[1e-6]:.4%}")
    print(f"  nout: median={np.median(nout):.4f}  min={nout.min():.2e}  "
          f"frac<1e-2={_tail_below(nout,[1e-2])[1e-2]:.4%}  frac<1e-4={_tail_below(nout,[1e-4])[1e-4]:.4%}  "
          f"frac<1e-6={_tail_below(nout,[1e-6])[1e-6]:.4%}")
    if rmp.size:
        print(f"  root child min perp-dist: median={np.median(rmp):.4f}  min={rmp.min():.2e}  "
              f"frac<1e-2={float((rmp<1e-2).mean()):.4%}  frac<1e-4={float((rmp<1e-4).mean()):.4%}")

    if cospsi.size:
        print("\n[geometry] branch-angle node features at training leaves")
        print(f"  |cosψ|   : frac>0.99={_tail_above(cospsi,[0.99])[0.99]:.4%}  "
              f"frac>0.999={_tail_above(cospsi,[0.999])[0.999]:.4%}  max={cospsi.max():.6f}")
        print(f"  |cosθ|   : frac>0.99={_tail_above(costheta,[0.99])[0.99]:.4%}  "
              f"frac>0.999={_tail_above(costheta,[0.999])[0.999]:.4%}  max={costheta.max():.6f}")

    print("\n[input scale] ||P_0||_inf per node (raw EGNN position input)")
    print("  percentiles: " + "  ".join(f"p{p}={v:.2f}" for p, v in _pct(p0max, [50, 95, 99, 100]).items()))

    print("\n[finiteness] non-finite occurrences (should all be 0):")
    print("  " + "  ".join(f"{k}={v}" for k, v in cap.nonfinite.items()))

    # ---- worst offenders -> CSV ----
    worst_c0 = sorted(cap._worst_c0, key=lambda it: -it[0])
    worst_nin = sorted(cap._worst_nin, key=lambda it: it[0])  # most-negative -> smallest nin
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "so2_degeneracy_worst.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "file", "reduction", "metric", "value", "detail"])
        for _, rec in worst_c0:
            w.writerow(["large_C0", rec["file"], rec["red"], "c0_norm", f"{rec['c0_norm']:.4f}", rec["C_0"]])
        for _, rec in worst_nin:
            w.writerow(["small_nin", rec["file"], rec["red"], "nin", f"{rec['nin']:.3e}", f"nout={rec['nout']:.3e}"])
    print(f"\nWorst offenders written to {csv_path}")
    print("  Top-5 largest ||C_0||:")
    for _, rec in worst_c0[:5]:
        print(f"    {rec['c0_norm']:.3f}  {rec['file']} (red {rec['red']})  C_0={rec['C_0']}")

    # ---- histograms ----
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    axs[0, 0].hist(c0, bins=120, color="darkorange", alpha=0.85)
    axs[0, 0].axvline(prior_norm, color="r", ls="--", lw=1, label=f"prior {prior_norm:.2f}")
    axs[0, 0].set_yscale("log"); axs[0, 0].set_title("||C_0|| (target offset)"); axs[0, 0].legend(fontsize=8)
    axs[0, 1].hist(np.log10(nin + 1e-12), bins=120, color="steelblue", alpha=0.85)
    axs[0, 1].set_title("log10(nin) — v_in perp norm")
    axs[0, 2].hist(np.log10(nout + 1e-12), bins=120, color="steelblue", alpha=0.85)
    axs[0, 2].set_title("log10(nout) — v_out perp norm")
    if cospsi.size:
        axs[1, 0].hist(cospsi, bins=120, color="seagreen", alpha=0.85)
        axs[1, 0].set_yscale("log"); axs[1, 0].set_title("|cosψ| at leaves")
        axs[1, 1].hist(costheta, bins=120, color="seagreen", alpha=0.85)
        axs[1, 1].set_yscale("log"); axs[1, 1].set_title("|cosθ| at leaves")
    if rmp.size:
        axs[1, 2].hist(np.log10(rmp + 1e-12), bins=80, color="purple", alpha=0.85)
        axs[1, 2].set_title("log10(root child min perp-dist)")
    fig.suptitle(f"SO(2) degeneracy scan — {n_graphs_ok} neurons, scale 1/{args.pos_scale}, axis {args.axis}")
    fig.tight_layout()
    png_path = args.out_dir / "so2_degeneracy_scan.png"
    fig.savefig(png_path, dpi=120)
    print(f"Saved histograms to {png_path}")
    print(line)


if __name__ == "__main__":
    main()
