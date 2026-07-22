"""Empirical distribution of the local-frame child offset C_0 (and expansion label e)
that a flow-matching prior must match.

Runs the REAL training pipeline (load SWC -> scale by pos_scale_factor -> depth
reduction -> precompute_full_geometry -> global_to_local) on a random subset of neurons,
capturing C_0 via a stand-in "capture" diffusion that records the exact tensor
Expansion.get_loss passes as C_0. Components are ordered (forward, sideways, axial=z).

Outputs per-axis mean/std/skew/kurtosis, covariance/correlation, offset-norm stats, the
expansion-label balance, and saves marginal histograms + a 3D scatter to a PNG.
"""
import argparse
import os
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
from utils.data_loading import load_swc_graph, nx_graph_to_adj_pos

N_GRAPHS = int(os.environ.get("N_GRAPHS", "400"))
BS = 16
SEED = 0

# SO(2) symmetry axis: neurons use y, trees use z. The axial (3rd) local-frame
# component is the offset projected onto this axis.
_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Empirical local-frame C_0 distribution a flow-matching prior must match."
    )
    ap.add_argument(
        "--data-dir", type=Path,
        default=Path("/Users/umer/Documents/neurons_final/train"),
        help="Directory of training SWC files (default: neuron train set).",
    )
    ap.add_argument(
        "--pos-scale", type=float, default=45.1,
        help="pos_scale_factor: positions are divided by this before geometry "
             "(default 45.1, neurons). Use 1.0 to inspect raw, unscaled C_0.",
    )
    ap.add_argument(
        "--axis", choices=("x", "y", "z"), default="y",
        help="SO(2) symmetry axis: 'y' for neurons (default), 'z' for trees.",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Output PNG path (default: c0_distribution_<axis>.png at repo root).",
    )
    ap.add_argument(
        "--prior-std-pos", type=float, nargs=3, default=[0.74, 0.61, 0.83],
        metavar=("FWD", "SIDE", "AXIAL"),
        help="Current per-axis prior std (forward, sideways, axial) to diagnose against "
             "(default: neuron flow config [0.74, 0.61, 0.83]).",
    )
    return ap.parse_args()


class CaptureDiffusion(Module):
    """Records C_0 and the raw expansion label; returns zero loss (model never runs)."""
    cond_dim = 2

    def __init__(self):
        super().__init__()
        self.C = []
        self.E = []

    def forward(self, *, C_0, leaf_expansion, **kw):
        if C_0.numel():
            self.C.append(C_0.detach().float().cpu())
            self.E.append(leaf_expansion.detach().float().cpu().view(-1))
        z = C_0.new_zeros(())
        return z, z


def _load_graphs(data_dir):
    files = [f for f in sorted(data_dir.iterdir())
             if f.is_file() and f.name.endswith(".swc") and not f.name.startswith("._")]
    rng = random.Random(SEED)
    rng.shuffle(files)
    files = files[:N_GRAPHS]
    graphs = []
    for f in files:
        try:
            graphs.append(load_swc_graph(f))
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {f.name}: {exc}")
    print(f"Loaded {len(graphs)} graphs (requested {N_GRAPHS}).")
    return graphs


def _moments(x):
    mu = x.mean(0)
    sd = x.std(0)
    z = (x - mu) / (sd + 1e-12)
    skew = (z ** 3).mean(0)
    kurt = (z ** 4).mean(0) - 3.0
    return mu, sd, skew, kurt


def main():
    args = parse_args()
    pos_scale = args.pos_scale
    so2_axis = _AXES[args.axis]
    axial_label = f"axial({args.axis}=uhat)"
    out = args.out or (Path(__file__).resolve().parent.parent / f"c0_distribution_{args.axis}.png")

    graphs = _load_graphs(args.data_dir)
    adjs, poses = [], []
    for G in graphs:
        A, P, _ = nx_graph_to_adj_pos(G)
        poses.append(P / pos_scale)
        adjs.append(A)

    red_factory = gg.depth_reduction.DepthReductionFactory(
        mode="stochastic", cherry_p=1.0, ensure_progress=True, root=0, contract_root=False,
    )
    ds = gg.data.PrecomputedRedDataset(adjs=adjs, poses=poses, red_factory=red_factory, tmds=None)

    # The local frame depends on uhat: neurons use the y-axis, trees the z-axis.
    model = gg.model.SO2_EGNN_Network(
        n_layers=2, feats_dim=4, pos_dim=3, m_dim=16, edge_attr_dim=1,
        so2_axis=so2_axis,
    )
    cap = CaptureDiffusion()
    method = gg.method.Expansion(diffusion=cap)

    samples = ds.samples
    n_skip = 0
    for i in range(0, len(samples), BS):
        batch = Batch.from_data_list(samples[i:i + BS])
        try:
            method.get_loss(batch, model)
        except Exception as exc:  # noqa: BLE001
            n_skip += 1
            if n_skip <= 3:
                print(f"  batch {i//BS} skipped: {exc}")
    if not cap.C:
        print("No C_0 captured — aborting.")
        sys.exit(1)

    C = th.cat(cap.C, 0).numpy()          # [M, 3] (forward, sideways, axial)
    E = th.cat(cap.E, 0).numpy()          # [M]
    M = C.shape[0]
    axes = ["forward", "sideways", axial_label]

    mu, sd, skew, kurt = _moments(C)
    norms = np.linalg.norm(C, axis=1)
    cov = np.cov(C.T)
    corr = np.corrcoef(C.T)

    print("\n" + "=" * 70)
    print(f"C_0 distribution over {M} leaf offsets from {len(graphs)} graphs")
    print(f"(positions scaled by 1/{pos_scale}; depth reduction; local frame; uhat={args.axis})")
    print("=" * 70)
    print(f"{'axis':<12}{'mean':>10}{'std':>10}{'skew':>10}{'kurtosis':>10}")
    for k, ax in enumerate(axes):
        print(f"{ax:<12}{mu[k]:>10.4f}{sd[k]:>10.4f}{skew[k]:>10.4f}{kurt[k]:>10.4f}")
    print(f"\noffset norm |C|:  mean={norms.mean():.4f}  std={norms.std():.4f}  "
          f"median={np.median(norms):.4f}  p95={np.percentile(norms,95):.4f}  max={norms.max():.4f}")
    print(f"\nanisotropy std ratio (max/min axis std): {sd.max()/ (sd.min()+1e-12):.2f}")
    print("\ncovariance matrix:\n", np.array2string(cov, precision=4))
    print("\ncorrelation matrix:\n", np.array2string(corr, precision=3))
    frac_pos = float((E > 0.5).mean())
    print(f"\nexpansion label e: fraction 'expand' (label 1) = {frac_pos:.4f}  (n={E.size})")
    print("\nReference: isotropic prior N(0, prior_std^2). For overlap, prior_std should")
    print(f"  ~ match per-axis std; current best single scalar ~= {sd.mean():.3f} (mean of axis stds).")
    print("=" * 70)

    # ---- prior-matching diagnosis (mean + std) ----
    # The flow prior is N(0, diag(prior_std_pos^2)) -> centered on the PARENT (offset mean 0).
    # The data offsets are NOT mean-zero (children grow outward), so the prior is mis-centered.
    # A flow whose prior matches BOTH data moments has the shortest/straightest transport.
    prior_std = np.asarray(args.prior_std_pos, dtype=float)
    mu_np = np.asarray(mu, dtype=float)
    sd_np = np.asarray(sd, dtype=float)
    print("\n" + "=" * 70)
    print("PRIOR-MATCHING DIAGNOSIS  (prior is N(mean=0, std=prior_std_pos))")
    print("=" * 70)
    print(f"{'axis':<12}{'data_mean':>10}{'data_std':>10}{'prior_std':>10}"
          f"{'mean/std':>10}{'shift_um':>10}")
    for k, ax in enumerate(axes):
        mean_over_std = mu_np[k] / (sd_np[k] + 1e-12)
        shift_um = mu_np[k] * pos_scale
        print(f"{ax:<12}{mu_np[k]:>10.4f}{sd_np[k]:>10.4f}{prior_std[k]:>10.4f}"
              f"{mean_over_std:>10.3f}{shift_um:>10.2f}")
    std_err = np.abs(sd_np - prior_std) / (sd_np + 1e-12)
    print(f"\nstd match: per-axis |Δ|/std = {np.array2string(std_err, precision=3)} "
          f"(prior_std_pos {'OK' if std_err.max() < 0.1 else 'NEEDS RETUNE'})")
    print("mean mismatch: prior is centered on the parent (0), data is not. The flow must")
    print("  transport this DC offset on every sample; on the worst axis that is the")
    print("  systematic push the sampler can overshoot (-> over-production).")
    print(f"\nRECOMMENDED prior_mean_pos (match data 1st moment): "
          f"[{mu_np[0]:.3f}, {mu_np[1]:.3f}, {mu_np[2]:.3f}]")
    print("  (set the noise draw to prior_mean_pos + randn*prior_std_pos in BOTH train & sample;")
    print("   equivalently regress the residual C_0 - prior_mean_pos. Shortens/straightens paths.)")
    print("=" * 70)

    # ---- plots ----
    fig = plt.figure(figsize=(16, 4))
    for k, ax in enumerate(axes):
        a = fig.add_subplot(1, 4, k + 1)
        a.hist(C[:, k], bins=120, density=True, alpha=0.7, color="steelblue")
        xs = np.linspace(C[:, k].min(), C[:, k].max(), 200)
        # overlay unit gaussian and data-fit gaussian
        a.plot(xs, np.exp(-0.5 * xs ** 2) / np.sqrt(2 * np.pi), "r--", lw=1, label="N(0,1)")
        a.plot(xs, np.exp(-0.5 * ((xs - mu[k]) / sd[k]) ** 2) / (sd[k] * np.sqrt(2 * np.pi)),
               "g-", lw=1.2, label=f"N({mu[k]:.2f},{sd[k]:.2f}²)")
        a.axvline(0, color="k", lw=0.6)
        a.axvline(mu[k], color="g", lw=0.8, ls=":")
        a.set_title(f"{ax}\nmean={mu[k]:.3f} std={sd[k]:.3f} skew={skew[k]:.2f}")
        a.legend(fontsize=7)
    a4 = fig.add_subplot(1, 4, 4)
    a4.hist(norms, bins=120, density=True, color="darkorange", alpha=0.8)
    a4.set_title(f"|C| offset norm\nmean={norms.mean():.3f}")
    a4.axvline(np.sqrt(3), color="r", ls="--", lw=1, label="E|N(0,1)³|≈√3")
    a4.legend(fontsize=7)
    fig.suptitle(f"Local-frame child offset C_0 — {M} offsets, {len(graphs)} graphs (scale 1/{pos_scale}, uhat={args.axis})")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nSaved plot to {out}")


if __name__ == "__main__":
    main()
