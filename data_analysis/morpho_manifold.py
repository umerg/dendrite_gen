"""
Probe whether the per-neuron embeddings of the GT dataset lie on a meaningful,
low-dimensional, learnable manifold. Works for two embeddings:

  * morpho : the 16-d hand-engineered morphometric vector (assemble_morpho_vector)
  * tmd    : the 256-d Euclidean-from-root TMD persistence image (compute_tmd_embedding)

Three questions, three measurements:
  1. Intrinsic dimension  -- PCA spectrum / effective rank (linear) + TwoNN and
     MLE estimators (nonlinear). Low vs the ambient dim => a manifold exists.
  2. Beyond marginals     -- compare intrinsic dim against a FEATURE-SHUFFLED null
     (each column permuted independently: kills joint structure, keeps marginals).
     real << null => genuine cross-feature correlation, not just narrow marginals.
  3. Reconstructability    -- PCA reconstruction error vs #components (the linear
     lower bound on how well a learned latent could reconstruct the data).

GT only -- no generated samples needed.

Usage:
    conda run -n NEURO2 python data_analysis/morpho_manifold.py [SWC_DIR] [N_SAMPLE] [morpho|tmd]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_loading import load_swc_graph  # noqa: E402
from utils.tmd import compute_tmd_embedding  # noqa: E402
from validation.dist_metrics import (  # noqa: E402
    assemble_morpho_vector,
    MORPHO_KEYS,
    _sholl_radii_from_graphs,
)


def load_sample(swc_dir: str, n_sample: int, seed: int = 0) -> list[nx.Graph]:
    files = [p for p in sorted(Path(swc_dir).iterdir())
             if p.is_file() and p.name.endswith(".swc") and not p.name.startswith("._")]
    rng = np.random.default_rng(seed)
    if n_sample and len(files) > n_sample:
        files = [files[i] for i in rng.choice(len(files), size=n_sample, replace=False)]
    graphs = []
    for p in files:
        try:
            G = load_swc_graph(p)
        except Exception:
            continue
        if G.number_of_nodes() == 0:
            continue
        H = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        if H.graph.get("root") not in H.nodes:
            H.graph["root"] = next(iter(H.nodes))
        graphs.append(H)
    return graphs


def build_vectors(graphs, embedding: str, n_bins: int = 16):
    """Return (X [n,D], feature_names, node_counts aligned with rows)."""
    if embedding == "morpho":
        radii = _sholl_radii_from_graphs(graphs, 32)
        X = np.stack([assemble_morpho_vector(G, uhat=(0, 0, 1), radii=radii) for G in graphs], axis=0)
        return X, list(MORPHO_KEYS), np.array([g.number_of_nodes() for g in graphs])
    rows, ncounts = [], []
    for G in graphs:
        try:
            e = compute_tmd_embedding(G, n_bins=n_bins)
        except Exception:
            continue
        if e.size and np.all(np.isfinite(e)):
            rows.append(e)
            ncounts.append(G.number_of_nodes())
    X = np.stack(rows, axis=0)
    return X, [f"pi{i}" for i in range(X.shape[1])], np.array(ncounts)


def preprocess(X: np.ndarray):
    """Impute nan (col median), drop (near-)constant columns, z-score the rest."""
    X = X.astype(np.float64).copy()
    if np.isnan(X).any():
        col_med = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_med, inds[1])
    std = X.std(axis=0)
    keep = std > 1e-8
    Xk = X[:, keep]
    Xz = (Xk - Xk.mean(axis=0)) / Xk.std(axis=0)
    return Xz, int(keep.sum()), int((~keep).sum())


def twonn_dimension(X: np.ndarray, discard_frac: float = 0.1) -> float:
    tree = cKDTree(X)
    d, _ = tree.query(X, k=3)
    r1, r2 = d[:, 1], d[:, 2]
    mask = r1 > 1e-12
    mu = np.sort(r2[mask] / r1[mask])
    N = mu.size
    keep = max(int(N * (1 - discard_frac)), 2)
    mu = mu[:keep]
    Femp = np.arange(1, keep + 1) / N
    x, y = np.log(mu), -np.log(1.0 - Femp)
    return float(np.sum(x * y) / np.sum(x * x))


def mle_dimension(X: np.ndarray, k: int = 10) -> float:
    tree = cKDTree(X)
    d, _ = tree.query(X, k=k + 1)
    d = np.maximum(d[:, 1:], 1e-12)
    logd = np.log(d)
    inv = np.mean(logd[:, -1][:, None] - logd[:, :-1], axis=1)
    inv = inv[inv > 1e-12]
    return float(1.0 / np.mean(inv))


def pca_spectrum(Xz: np.ndarray):
    mean = Xz.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xz - mean, full_matrices=False)
    var = (S ** 2) / max(len(Xz) - 1, 1)
    evr = var / var.sum()
    eff_rank = float((S.sum() ** 2) / (S ** 2).sum())
    return np.cumsum(evr), eff_rank, Vt, mean, S


def n_comps_for(cumevr: np.ndarray, frac: float) -> int:
    return int(np.searchsorted(cumevr, frac) + 1)


def analyze(Xz: np.ndarray, label: str) -> dict:
    n, D = Xz.shape
    cumevr, eff_rank, Vt, pmean, S = pca_spectrum(Xz)
    id_twonn = twonn_dimension(Xz)
    id_mle = mle_dimension(Xz, k=10)
    rng = np.random.default_rng(0)
    Xsh = np.column_stack([rng.permutation(Xz[:, j]) for j in range(D)])
    id_twonn_sh = twonn_dimension(Xsh)
    _, eff_rank_sh, _, _, _ = pca_spectrum(Xsh)

    res = {
        "n": n, "ambient": D,
        "eff_rank": eff_rank, "eff_rank_null": eff_rank_sh,
        "id_twonn": id_twonn, "id_twonn_null": id_twonn_sh, "id_mle": id_mle,
        "ncomp90": n_comps_for(cumevr, 0.90), "ncomp95": n_comps_for(cumevr, 0.95),
        "cumevr": cumevr, "Vt": Vt, "pmean": pmean,
    }
    print(f"\n===== {label}  (n={n}, ambient D={D}) =====")
    print(f"  effective rank        : {eff_rank:.1f}  (null {eff_rank_sh:.1f})")
    print(f"  intrinsic dim TwoNN   : {id_twonn:.1f}  (null {id_twonn_sh:.1f})")
    print(f"  intrinsic dim MLE     : {id_mle:.1f}")
    print(f"  #comp for 90% / 95%   : {res['ncomp90']} / {res['ncomp95']}")
    print("  PCA reconstruction RMSE vs k (standardized units):")
    Xc = Xz - pmean
    for k in (1, 2, 3, 5, 8, 16):
        if k > Vt.shape[0]:
            break
        recon = (Xc @ Vt[:k].T) @ Vt[:k] + pmean
        rmse = float(np.sqrt(np.mean((Xz - recon) ** 2)))
        print(f"    k={k:2d}: RMSE={rmse:.3f}  var={cumevr[k-1]*100:5.1f}%")
    return res


def make_plots(Xz, res, color, label, tag, out_dir):
    Vt, pmean, cumevr = res["Vt"], res["pmean"], res["cumevr"]
    proj = (Xz - pmean) @ Vt[:2].T
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(np.arange(1, len(cumevr) + 1), cumevr * 100, "o-", ms=3)
    axes[0].axhline(90, ls="--", c="gray", lw=0.8)
    axes[0].set_xlabel("# PCA components"); axes[0].set_ylabel("cumulative var (%)")
    axes[0].set_title(f"{label}: PCA spectrum")
    sc = axes[1].scatter(proj[:, 0], proj[:, 1], c=np.log10(color + 1), s=6, alpha=0.5, cmap="viridis")
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
    axes[1].set_title(f"{label}: PCA (color = log10 #nodes)")
    fig.colorbar(sc, ax=axes[1], shrink=0.8)
    fig.tight_layout(); fig.savefig(out_dir / f"{tag}_pca.png", dpi=130); plt.close(fig)
    try:
        from sklearn.manifold import TSNE
        ts = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(Xz)
        fig, ax = plt.subplots(figsize=(5.2, 4))
        ax.scatter(ts[:, 0], ts[:, 1], c=np.log10(color + 1), s=6, alpha=0.5, cmap="viridis")
        ax.set_title(f"{label}: t-SNE (color = log10 #nodes)"); ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(); fig.savefig(out_dir / f"{tag}_tsne.png", dpi=130); plt.close(fig)
    except Exception as e:
        print(f"(t-SNE skipped: {e})")


def main():
    swc_dir = sys.argv[1] if len(sys.argv) > 1 else "/Users/umer/Documents/neurons_final/train"
    n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    embedding = sys.argv[3] if len(sys.argv) > 3 else "morpho"
    out_dir = Path(__file__).resolve().parent / "morpho_manifold_out"
    out_dir.mkdir(exist_ok=True)

    graphs = load_sample(swc_dir, n_sample)
    split = Path(swc_dir).name
    print(f"Loaded {len(graphs)} neurons from {swc_dir}  | embedding={embedding}")

    X, names, ncounts = build_vectors(graphs, embedding)
    Xz, n_active, n_const = preprocess(X)
    print(f"  raw D={X.shape[1]}  active D={n_active}  (dropped {n_const} constant)")

    label = f"{split}/{embedding}"
    res = analyze(Xz, label)
    make_plots(Xz, res, ncounts[: len(Xz)] if embedding == "tmd" else ncounts,
               label, f"{split}_{embedding}", out_dir)
    print(f"\nPlots -> {out_dir}/{split}_{embedding}_*.png")


if __name__ == "__main__":
    main()
