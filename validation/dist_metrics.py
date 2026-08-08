"""
Distribution-level validation metrics for generated vs ground-truth trees.

The generator only conditions on TMD, so we do NOT expect a generated tree to
match a specific GT tree node-for-node. Instead we compare the *distribution* of
summary statistics pooled over the generated set against the same statistics
pooled over the GT validation set. This module is the in-loop (training-time)
monitor and emits clean point-estimate scalars:

  - Per-feature marginals: Wasserstein-1 (and KS for continuous features) over a
    battery of morphometrics (branch length, bifurcation angle, TMD bar lengths,
    path/radial distance to root, contraction, branch order, plus per-tree size,
    extent, Strahler, partition asymmetry and Sholl summaries).
  - Joint distribution: MMD + Density/Coverage on two per-tree embeddings -- a
    standardized morphometric vector and a Euclidean-from-root TMD persistence
    image. These catch broken correlations between features that marginals miss.
  - Topology: average tree-edit distance over index-paired trees.

Statistical rigor that complicates reading at a glance (bootstrap CIs, permutation
significance, FDR) lives in the offline paper protocol -- see docs/EVAL_PAPER_PROTOCOL.md.

The one invisible robustness rule kept here: the MMD kernel bandwidth and the
morphometric standardization / TMD PCA are FIT ON THE GT SET and reused, so the
MMD trajectory is comparable across training steps. The trainer caches these on the
fixed GT set (``build_gt_cache``) and passes them in; if no cache is supplied they
are rebuilt from ``gt_graphs`` here (deterministic, used by tests).

Returned dict is flat ``{str: float}`` so ``Trainer.log`` auto-logs every entry to
wandb under ``validation/ema_<beta>/dist/<key>``.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import networkx as nx
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.spatial.distance import pdist

from validation.structural_metrics import (
    branch_length_values,
    bifurcation_angle_values,
    branch_order_values,
    contraction_ratio_values,
    graph_edit_distance_topology,
    partition_asymmetry,
    path_length_to_root_values,
    radial_distance_to_root_values,
    sholl_summary,
    strahler_number,
    _root_tree,
    _pos_to_xyz,
)
from utils.dist_helper import (
    density_coverage,
    median_heuristic_bandwidth,
    mmd2_unbiased,
)

try:
    from utils.tmd import compute_tmd_barcode_diagram, compute_tmd_embedding
except ModuleNotFoundError:  # pragma: no cover - fallback when utils already on path
    from tmd import compute_tmd_barcode_diagram, compute_tmd_embedding  # type: ignore


# Skip tree-edit-distance for pairs above this node count (zss TED is superlinear
# and has no real timeout), and cap how many pairs we evaluate per validation.
GED_MAX_NODES = 200
GED_MAX_PAIRS = 64

# Default Sholl shell count and TMD-image grid (must match compute_tmd_embedding).
SHOLL_N_SHELLS = 32
TMD_N_BINS = 16

# Fixed-order morphometric feature vector used for the joint MMD/Density-Coverage.
#
# v2 (9-D). The v1 16-D vector was rank-deficient: `node_count`/`leaf_count`/
# `bifurcation_count` correlated at r = 1.000 (they are the same number on strictly binary
# trees), `axial_extent`/`total_extent` at 0.999, `mean_path_to_root`/`mean_radial_to_root`
# at 0.996. Because the RBF kernel runs on raw z-scores -- which equalise per-feature
# variance but NOT correlation -- each such block contributed ~m x to the squared distance,
# so size and reach dominated while the shape features got 1 x each. Pruning to the
# independent core measurably improves defect detection; see docs/VALIDATION_METRICS_AUDIT.md.
#
# Two deliberate, non-redundancy choices:
#   - `node_count` is EXCLUDED so mmd/density/coverage_morpho are a shape-only comparison
#     against baselines that are handed the node count. Size is still reported separately as
#     the `node_count_w1` marginal.
#   - `max_branch_order` replaces `strahler`. See _max_branch_order for what it measures.
MORPHO_KEYS = (
    "axial_extent",
    "radial_span",
    "max_branch_order",
    "partition_asymmetry",
    "mean_branch_length",
    "mean_bifurcation_angle",
    "mean_radial_to_root",
    "mean_contraction",
    "sholl_critical_radius",
)

# Bump whenever MORPHO_KEYS changes. mmd_morpho / density_morpho / coverage_morpho are NOT
# comparable across versions -- runs either side of a bump must not share a plot axis.
MORPHO_VERSION = 2

# Discrete (integer-valued, heavily-tied) features: report W1 only, never KS in-loop.
_DISCRETE_POOLED = {"branch_order"}
_DISCRETE_PERTREE = {"node_count", "leaf_count", "bifurcation_count", "strahler"}


def _root_of(G: nx.Graph) -> int | None:
    root = G.graph.get("root")
    if root is None or root not in G.nodes:
        return None
    return int(root)


def _w1(gen_vals: np.ndarray, gt_vals: np.ndarray) -> float:
    """Wasserstein-1 between two pooled value arrays; nan if either is empty."""
    gen_vals = np.asarray(gen_vals, dtype=np.float64)
    gt_vals = np.asarray(gt_vals, dtype=np.float64)
    gen_vals = gen_vals[np.isfinite(gen_vals)]
    gt_vals = gt_vals[np.isfinite(gt_vals)]
    if gen_vals.size == 0 or gt_vals.size == 0:
        return float("nan")
    return float(wasserstein_distance(gen_vals, gt_vals))


def _ks(gen_vals: np.ndarray, gt_vals: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic (sup|F-G|) between two arrays; nan if either empty."""
    gen_vals = np.asarray(gen_vals, dtype=np.float64)
    gt_vals = np.asarray(gt_vals, dtype=np.float64)
    gen_vals = gen_vals[np.isfinite(gen_vals)]
    gt_vals = gt_vals[np.isfinite(gt_vals)]
    if gen_vals.size == 0 or gt_vals.size == 0:
        return float("nan")
    return float(ks_2samp(gen_vals, gt_vals).statistic)


def _safe_mean(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _nan_frac(mat: np.ndarray) -> float:
    """Fraction of non-finite entries in a (N, d) morpho matrix; nan if empty."""
    mat = np.asarray(mat, dtype=np.float64)
    if mat.size == 0:
        return float("nan")
    return float(np.mean(~np.isfinite(mat)))


def _degenerate_frac(graphs: list[nx.Graph], uhat) -> float:
    """Fraction of trees that are structurally or spatially degenerate.

    Degenerate = no bifurcation at all (a path or a single node, so partition asymmetry and
    bifurcation angle are undefined) OR zero spatial extent (all nodes coincident, so every
    geometric feature collapses). These are exactly the cases whose nan features get imputed
    to the GT mean by ``standardize_vectors`` and would otherwise be invisible.
    """
    if not graphs:
        return float("nan")
    bad = 0
    for G in graphs:
        root = _root_of(G)
        if root is None or G.number_of_nodes() < 2:
            bad += 1
            continue
        _parent, children = _root_tree(G, root)
        if not any(len(ch) >= 2 for ch in children.values()):
            bad += 1
            continue
        ext = _size_extent(G, uhat)
        if not (float(ext.get("total_extent", 0.0) or 0.0) > 0.0):
            bad += 1
    return float(bad / len(graphs))


def _fit_whiten(Z: np.ndarray, *, eps: float = 1e-3):
    """Fit a ZCA whitening matrix on the standardized GT morpho matrix.

    Equalises the *directions* of the feature space, not just the per-axis variances that
    z-scoring already handles -- correlated feature groups otherwise contribute ~m x to the
    squared distance in the RBF kernel.

    ``eps`` is a hard requirement, not a tunable: on a rank-deficient vector, whitening
    amplifies null directions by 1/sqrt(eps). Measured on the old 16-D vector (which had four
    exactly-zero eigenvalues), eps=1e-6 produced MMD = 0.35 against a floor of 0.001 -- a ~300x
    artifact. Pruning MORPHO_KEYS to its independent core is what makes this safe.
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[0] < 2 or Z.shape[1] == 0:
        return None
    cov = np.cov(Z.T)
    cov = np.atleast_2d(cov)
    lam, V = np.linalg.eigh(cov)
    return V @ np.diag(1.0 / np.sqrt(np.maximum(lam, eps))) @ V.T


def _apply_whiten(Z: np.ndarray, W) -> np.ndarray:
    if W is None:
        return Z
    Z = np.asarray(Z, dtype=np.float64)
    return Z @ W if Z.size else Z


# --- per-graph statistic extractors --------------------------------------------------


def _branch_lengths(G: nx.Graph) -> np.ndarray:
    return branch_length_values(G)


def _bifurcation_angles(G: nx.Graph) -> np.ndarray:
    root = _root_of(G)
    if root is None:
        return np.zeros((0,), dtype=np.float64)
    try:
        return bifurcation_angle_values(G, root=root)
    except ValueError:
        return np.zeros((0,), dtype=np.float64)


def _max_branch_order(G: nx.Graph) -> float:
    """Deepest branch order in the tree (per-tree scalar); nan on a degenerate graph.

    NOTE ON WHAT THIS ACTUALLY MEASURES. ``branch_order_values`` increments only at non-root
    nodes of undirected degree >= 3. Our datasets are strictly binary away from the root with
    ZERO degree-2 non-root nodes, so every non-root internal node has degree 3 and increments
    -- i.e. on this data ``branch_order == hop_depth - 1`` exactly (verified 300/300 graphs on
    trees d10, trees d20 and neurons), and this scalar is simply TREE DEPTH - 1. It is kept
    under the "branch order" name because that is what the function computes in general; on a
    graph with degree-2 chain nodes the two would diverge.

    Caveat for the depth-capped tree sets: `trees_genus_d{10,15,20}` are trimmed to a depth
    cap in preprocessing, so this feature is heavily concentrated there (97% of d10 trees fall
    in {9,10,11}) and partly measures "did the generator reach the dataset's depth ceiling"
    rather than free morphological depth. It is well spread on neurons (19 distinct values).
    """
    vals = branch_order_values(G)
    return float(vals.max()) if vals.size else float("nan")


def _tmd_bar_lengths(
    G: nx.Graph, *, filtration: str = "radial_root", uhat=(0.0, 0.0, 1.0)
) -> np.ndarray:
    """|death - birth| for each persistence interval (raw scale, no per-graph norm).

    ``filtration`` must be threaded from ``cfg.validation.tmd_eval_filtration`` so this shares
    one source of truth with the joint TMD embedding. It previously took
    ``compute_tmd_barcode_diagram``'s default (``"path"``) while the joint block used
    ``radial_root``, so ``tmd_barlen_w1`` was silently in different units from every other TMD
    metric. ``uhat`` matters only for the axis-dependent filtrations (``height``, ``rho``).
    """
    root = _root_of(G)
    if root is None:
        return np.zeros((0,), dtype=np.float64)
    try:
        _barcode, diagram = compute_tmd_barcode_diagram(
            G, filtration=filtration, normalize_mode="none", uhat=uhat
        )
    except Exception:
        return np.zeros((0,), dtype=np.float64)
    pairs = np.asarray(diagram.as_pairs(), dtype=np.float64).reshape(-1, 2)
    if pairs.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.abs(pairs[:, 1] - pairs[:, 0])


def _size_extent(G: nx.Graph, uhat) -> dict[str, float]:
    """
    Per-tree size/extent stats decomposed in the model's SO(2) symmetry frame.

    The generator is equivariant to rotation about ``uhat``, so extents are measured
    relative to that axis (not world x/y/z) to be both semantically correct and
    invariant to azimuthal rotation about uhat:
      - axial_extent : extent along uhat (max - min of pos . uhat) -- the "height"
      - radial_span  : planar diameter perpendicular to uhat (max pairwise distance
                       of nodes projected onto the plane orthogonal to uhat)
      - total_extent : 3D diameter (max pairwise distance) -- overall reach

    Returns a nan-filled dict on degenerate (empty/unrooted) trees.
    """
    out = {
        "node_count": float(G.number_of_nodes()),
        "leaf_count": float("nan"),
        "bifurcation_count": float("nan"),
        "axial_extent": float("nan"),
        "radial_span": float("nan"),
        "total_extent": float("nan"),
    }
    root = _root_of(G)
    if root is not None:
        _parent, children = _root_tree(G, root)
        out["leaf_count"] = float(sum(1 for ch in children.values() if len(ch) == 0))
        out["bifurcation_count"] = float(sum(1 for ch in children.values() if len(ch) >= 2))
    n = G.number_of_nodes()
    if n > 0:
        u = np.asarray(uhat, dtype=np.float64).reshape(3)
        u = u / (np.linalg.norm(u) + 1e-12)
        pts = np.stack([_pos_to_xyz(G.nodes[k].get("pos", np.zeros(3))) for k in G.nodes()], axis=0)
        s = pts @ u  # axial coordinate along uhat
        out["axial_extent"] = float(s.max() - s.min())
        if n >= 2:
            pts_perp = pts - np.outer(s, u)  # components in the plane orthogonal to uhat
            out["radial_span"] = float(pdist(pts_perp).max())  # planar diameter
            out["total_extent"] = float(pdist(pts).max())      # 3D diameter
        else:
            out["radial_span"] = 0.0
            out["total_extent"] = 0.0
    return out


def _per_tree_scalars(G: nx.Graph, uhat, *, radii=None) -> dict[str, float]:
    """Merge size/extent with the new per-tree scalar morphometrics."""
    out = _size_extent(G, uhat)
    out["strahler"] = strahler_number(G)
    out["partition_asymmetry"] = partition_asymmetry(G)
    out.update(sholl_summary(G, radii=radii, n_shells=SHOLL_N_SHELLS))
    return out


# --- morphometric vector + standardization + PCA (for joint metrics) -----------------


def assemble_morpho_vector(G: nx.Graph, *, uhat, radii=None) -> np.ndarray:
    """Fixed-order per-tree morphometric vector (see MORPHO_KEYS). May contain nan."""
    ext = _size_extent(G, uhat)
    vals = {
        "node_count": ext["node_count"],
        "leaf_count": ext["leaf_count"],
        "bifurcation_count": ext["bifurcation_count"],
        "axial_extent": ext["axial_extent"],
        "radial_span": ext["radial_span"],
        "total_extent": ext["total_extent"],
        "strahler": strahler_number(G),
        "max_branch_order": _max_branch_order(G),
        "partition_asymmetry": partition_asymmetry(G),
        "mean_branch_length": _safe_mean(branch_length_values(G)),
        "mean_bifurcation_angle": _safe_mean(_bifurcation_angles(G)),
        "mean_path_to_root": _safe_mean(path_length_to_root_values(G)),
        "mean_radial_to_root": _safe_mean(radial_distance_to_root_values(G)),
        "mean_contraction": _safe_mean(contraction_ratio_values(G)),
    }
    sh = sholl_summary(G, radii=radii, n_shells=SHOLL_N_SHELLS)
    vals.update(sh)
    return np.asarray([vals[k] for k in MORPHO_KEYS], dtype=np.float64)


def standardize_vectors(vecs: np.ndarray, *, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """z-score by GT mean/std, then impute non-finite entries to 0 (the GT mean)."""
    vecs = np.asarray(vecs, dtype=np.float64)
    if vecs.size == 0:
        return vecs.reshape(0, len(mean))
    z = (vecs - mean) / (std + eps)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_pca(X: np.ndarray, ncomp: int | None):
    """Fit a top-``ncomp`` PCA (centered SVD). Returns None when reduction is a no-op."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape if X.ndim == 2 else (0, 0)
    if not ncomp or n < 2 or ncomp >= min(n, d):
        return None
    mean = X.mean(axis=0)
    _U, _S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    return {"mean": mean, "components": Vt[:ncomp]}


def _apply_pca(X: np.ndarray, pca) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if pca is None:
        return X
    return (X - pca["mean"]) @ pca["components"].T


def _effective_rank(X: np.ndarray) -> float:
    """Participation-ratio effective rank of a (centered) embedding matrix."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        return float("nan")
    s = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    s = s[s > 0]
    if s.size == 0:
        return float("nan")
    return float((s.sum() ** 2) / (s ** 2).sum())


def _sholl_radii_from_graphs(graphs: Iterable[nx.Graph], n_shells: int) -> np.ndarray | None:
    """Shared Sholl shell radii: evenly spaced over (0, max radial extent across GT].

    NOT used by the three per-tree Sholl summaries any more -- they use per-tree shells (see
    the note in ``compute_distribution_metrics``). Kept, and still cached by ``build_gt_cache``,
    because a mean-Sholl-profile PLOT (gen vs GT on one x-axis) does need a common grid.
    """
    max_r = 0.0
    for G in graphs:
        vals = radial_distance_to_root_values(G)
        if vals.size:
            max_r = max(max_r, float(vals.max()))
    if max_r <= 0.0:
        return None
    return np.linspace(0.0, max_r, int(n_shells) + 1, dtype=np.float64)[1:]


def _embed_matrix(graphs: Iterable[nx.Graph], embed_fn: Callable[[nx.Graph], np.ndarray]) -> np.ndarray:
    """Stack per-tree TMD embeddings, skipping graphs whose embedding fails."""
    rows = []
    for G in graphs:
        try:
            e = np.asarray(embed_fn(G), dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if e.size and np.all(np.isfinite(e)):
            rows.append(e)
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    return np.stack(rows, axis=0)


def build_gt_cache(
    gt_graphs: list[nx.Graph],
    *,
    uhat=(0.0, 0.0, 1.0),
    embed_fn: Callable[[nx.Graph], np.ndarray] | None = None,
    tmd_pca_ncomp: int | None = 32,
    sholl_n_shells: int = SHOLL_N_SHELLS,
    morpho_whiten: bool = False,
) -> dict:
    """
    Precompute the GT-derived objects the joint metrics need, ONCE on the fixed GT
    set: Sholl radii, morphometric mean/std + standardized GT vectors + bandwidth,
    and the TMD persistence-image PCA + reduced GT embeddings + bandwidth.

    Reusing these across training steps keeps the MMD trajectory comparable.

    ``morpho_whiten`` additionally fits a ZCA transform on the standardized GT morpho matrix
    (see ``_fit_whiten``). Off by default; the pruned MORPHO_KEYS already removes most of the
    correlation distortion whitening would fix.
    """
    if embed_fn is None:
        embed_fn = compute_tmd_embedding

    sholl_radii = _sholl_radii_from_graphs(gt_graphs, sholl_n_shells)

    morpho = np.stack(
        [assemble_morpho_vector(G, uhat=uhat, radii=None) for G in gt_graphs], axis=0
    ) if gt_graphs else np.zeros((0, len(MORPHO_KEYS)), dtype=np.float64)
    morpho_mean = np.nanmean(morpho, axis=0) if morpho.shape[0] else np.zeros(len(MORPHO_KEYS))
    morpho_std = np.nanstd(morpho, axis=0) if morpho.shape[0] else np.ones(len(MORPHO_KEYS))
    morpho_mean = np.nan_to_num(morpho_mean, nan=0.0)
    morpho_std = np.nan_to_num(morpho_std, nan=1.0)
    # Guard (near-)zero-variance features: a constant GT feature would otherwise
    # turn any deviation into a huge z-score and dominate the MMD kernel.
    morpho_std = np.where(morpho_std < 1e-8, 1.0, morpho_std)
    morpho_z = standardize_vectors(morpho, mean=morpho_mean, std=morpho_std)
    # Sanity record: this should be 0.0 on any healthy dataset. A non-zero value means some
    # GT trees are themselves degenerate, which would invalidate reading gen-side
    # `morpho_nan_frac` as a pure generator-failure signal.
    morpho_gt_nan_frac = _nan_frac(morpho)
    whiten = _fit_whiten(morpho_z) if morpho_whiten else None
    morpho_z = _apply_whiten(morpho_z, whiten)
    morpho_sigma = median_heuristic_bandwidth(morpho_z) if morpho_z.shape[0] > 1 else 1.0

    tmd_raw = _embed_matrix(gt_graphs, embed_fn)
    pca = _fit_pca(tmd_raw, tmd_pca_ncomp)
    tmd_reduced = _apply_pca(tmd_raw, pca) if tmd_raw.shape[0] else tmd_raw
    tmd_sigma = median_heuristic_bandwidth(tmd_reduced) if tmd_reduced.shape[0] > 1 else 1.0
    tmd_eff_rank = _effective_rank(tmd_raw)

    return {
        "uhat": tuple(np.asarray(uhat, dtype=np.float64).reshape(3).tolist()),
        "sholl_radii": sholl_radii,
        "morpho_mean": morpho_mean,
        "morpho_std": morpho_std,
        "morpho_z": morpho_z,
        "morpho_sigma": morpho_sigma,
        "morpho_whiten": whiten,
        "morpho_gt_nan_frac": morpho_gt_nan_frac,
        "morpho_version": MORPHO_VERSION,
        "embed_fn": embed_fn,
        "pca": pca,
        "tmd_reduced": tmd_reduced,
        "tmd_sigma": tmd_sigma,
        "tmd_eff_rank": tmd_eff_rank,
    }


def joint_metrics_from_vectors(
    gen_vecs: np.ndarray,
    gt_vecs: np.ndarray,
    *,
    prefix: str,
    sigma: float,
    k: int,
) -> dict[str, float]:
    """MMD + Density/Coverage between two already-transformed embedding matrices."""
    gen_vecs = np.asarray(gen_vecs, dtype=np.float64)
    gt_vecs = np.asarray(gt_vecs, dtype=np.float64)
    out = {
        f"mmd_{prefix}": float("nan"),
        f"density_{prefix}": float("nan"),
        f"coverage_{prefix}": float("nan"),
    }
    if gen_vecs.shape[0] < 2 or gt_vecs.shape[0] < 2:
        return out
    out[f"mmd_{prefix}"] = mmd2_unbiased(gen_vecs, gt_vecs, sigma)
    dens, cov = density_coverage(gen_vecs, gt_vecs, k=k)
    out[f"density_{prefix}"] = dens
    out[f"coverage_{prefix}"] = cov
    return out


# --- main entry point -----------------------------------------------------------------


def compute_distribution_metrics(
    gen_graphs: list[nx.Graph],
    gt_graphs: list[nx.Graph],
    *,
    uhat=(0.0, 0.0, 1.0),  # SO(2) symmetry axis; pass model.uhat at the call site
    ged_enabled: bool = True,
    ged_timeout: float | None = 5.0,  # kept for config compatibility; zss has no real timeout
    enable_ks: bool = True,
    enable_morphometrics: bool = True,
    enable_light_joint: bool = True,
    gt_cache: dict | None = None,
    embed_fn: Callable[[nx.Graph], np.ndarray] | None = None,
    tmd_pca_ncomp: int | None = 32,
    dc_k: int = 5,
    sholl_n_shells: int = SHOLL_N_SHELLS,
    tmd_filtration: str = "radial_root",  # must match cfg.validation.tmd_eval_filtration
    morpho_whiten: bool = False,
) -> dict[str, float]:
    """
    Compare distributions of summary statistics between generated and GT trees.

    Returns a flat dict of float scalars (point estimates). Keys with insufficient
    data are nan. If ``enable_light_joint`` and ``gt_cache`` is None, the GT-derived
    objects are rebuilt from ``gt_graphs`` (deterministic; convenient for tests).
    """
    metrics: dict[str, float] = {}

    # Sholl summaries use PER-TREE shells (radii=None -> 32 shells over each tree's own
    # support), never the shared GT grid. Crossing counts are inherently scale-invariant, but
    # a fixed grid breaks that by undersampling small trees: on one tree scaled x1/x1.5/x3 the
    # shared grid gave sholl_peak 7/13/13 where per-tree shells give 14/14/14. That made a
    # small generated tree and a large GT tree OF IDENTICAL SHAPE score differently -- a W1
    # signal that was pure grid artifact. Nothing here compares profiles pointwise, so the
    # shared grid bought nothing. See docs/VALIDATION_METRICS_AUDIT.md.

    def _pool(graphs: Iterable[nx.Graph], fn) -> np.ndarray:
        arrs = [np.asarray(fn(G), dtype=np.float64).reshape(-1) for G in graphs]
        arrs = [a for a in arrs if a.size > 0]
        return np.concatenate(arrs) if arrs else np.zeros((0,), dtype=np.float64)

    pooled_norms: list[float] = []
    pertree_norms: list[float] = []

    # Pooled (per-element) statistics: W1 (+ KS for continuous), and contribute to the
    # normalized aggregate (W1 divided by the GT spread so units don't dominate).
    pooled_features = [
        ("branch_length", _branch_lengths),
        ("bifurcation_angle", _bifurcation_angles),
        # Closure, not a bare reference: the filtration and axis have to reach it.
        ("tmd_barlen", lambda G: _tmd_bar_lengths(G, filtration=tmd_filtration, uhat=uhat)),
    ]
    if enable_morphometrics:
        pooled_features += [
            ("path_to_root", path_length_to_root_values),
            ("radial_to_root", radial_distance_to_root_values),
            ("contraction", contraction_ratio_values),
            ("branch_order", branch_order_values),
        ]
    for name, fn in pooled_features:
        gen_pool = _pool(gen_graphs, fn)
        gt_pool = _pool(gt_graphs, fn)
        w1 = _w1(gen_pool, gt_pool)
        metrics[f"{name}_w1"] = w1
        if enable_ks and name not in _DISCRETE_POOLED:
            metrics[f"{name}_ks"] = _ks(gen_pool, gt_pool)
        scale = float(np.nanstd(gt_pool)) if gt_pool.size else float("nan")
        if np.isfinite(w1) and np.isfinite(scale) and scale > 1e-12:
            pooled_norms.append(w1 / scale)

    # Per-tree scalar statistics (one value per tree -> distribution over trees).
    gen_ext = [_per_tree_scalars(G, uhat, radii=None) for G in gen_graphs]
    gt_ext = [_per_tree_scalars(G, uhat, radii=None) for G in gt_graphs]
    pertree_keys = ["node_count", "leaf_count", "bifurcation_count", "axial_extent", "radial_span", "total_extent"]
    if enable_morphometrics:
        pertree_keys += ["strahler", "partition_asymmetry", "sholl_peak", "sholl_critical_radius", "sholl_auc"]
    for key in pertree_keys:
        gen_vals = np.array([d[key] for d in gen_ext], dtype=np.float64) if gen_ext else np.zeros((0,))
        gt_vals = np.array([d[key] for d in gt_ext], dtype=np.float64) if gt_ext else np.zeros((0,))
        w1 = _w1(gen_vals, gt_vals)
        metrics[f"{key}_w1"] = w1
        if enable_ks and key not in _DISCRETE_PERTREE:
            metrics[f"{key}_ks"] = _ks(gen_vals, gt_vals)
        scale = float(np.nanstd(gt_vals[np.isfinite(gt_vals)])) if gt_vals.size else float("nan")
        if np.isfinite(w1) and np.isfinite(scale) and scale > 1e-12:
            pertree_norms.append(w1 / scale)

    if pooled_norms:
        metrics["w1_pooled_mean_normalized"] = float(np.mean(pooled_norms))
    if pertree_norms:
        metrics["w1_pertree_mean_normalized"] = float(np.mean(pertree_norms))

    # Degeneracy disclosure. `standardize_vectors` imputes non-finite morpho entries to 0 =
    # the GT mean, so a generated tree with zero forks is scored as having PERFECTLY AVERAGE
    # branching asymmetry and fork angle -- the imputation neutralises exactly the dimensions
    # that would have flagged it. We keep that imputation (it keeps mmd_morpho interpretable)
    # and surface the degeneracy on its own line instead. On real GT the nan-fraction is 0.000
    # for every feature on both datasets, so a non-zero value here is always a generator
    # failure, never a data property.
    if gen_graphs:
        metrics["gen_degenerate_frac"] = _degenerate_frac(gen_graphs, uhat)

    # Joint-distribution metrics on two cheap per-tree embeddings (morpho vector +
    # Euclidean-from-root TMD persistence image). Standardization/PCA/bandwidth come
    # from the GT-fit cache so the MMD is comparable across steps.
    if enable_light_joint:
        if gt_cache is None:
            gt_cache = build_gt_cache(
                gt_graphs,
                uhat=uhat,
                embed_fn=embed_fn,
                tmd_pca_ncomp=tmd_pca_ncomp,
                sholl_n_shells=sholl_n_shells,
                morpho_whiten=morpho_whiten,
            )
        ef = embed_fn or gt_cache.get("embed_fn") or compute_tmd_embedding
        k = int(dc_k)

        # Morphometric-vector joint
        gen_morpho = (
            np.stack([assemble_morpho_vector(G, uhat=uhat, radii=None) for G in gen_graphs], axis=0)
            if gen_graphs else np.zeros((0, len(MORPHO_KEYS)))
        )
        metrics["morpho_nan_frac"] = _nan_frac(gen_morpho)
        gen_morpho_z = standardize_vectors(gen_morpho, mean=gt_cache["morpho_mean"], std=gt_cache["morpho_std"])
        gen_morpho_z = _apply_whiten(gen_morpho_z, gt_cache.get("morpho_whiten"))
        metrics.update(
            joint_metrics_from_vectors(
                gen_morpho_z, gt_cache["morpho_z"], prefix="morpho", sigma=gt_cache["morpho_sigma"], k=k
            )
        )

        # TMD persistence-image joint (PCA-reduced)
        gen_tmd_raw = _embed_matrix(gen_graphs, ef)
        gen_tmd = _apply_pca(gen_tmd_raw, gt_cache["pca"]) if gen_tmd_raw.shape[0] else gen_tmd_raw
        metrics.update(
            joint_metrics_from_vectors(
                gen_tmd, gt_cache["tmd_reduced"], prefix="tmd", sigma=gt_cache["tmd_sigma"], k=k
            )
        )
        metrics["mmd_bandwidth_morpho"] = float(gt_cache["morpho_sigma"])
        metrics["mmd_bandwidth_tmd"] = float(gt_cache["tmd_sigma"])
        metrics["tmd_eff_rank"] = float(gt_cache.get("tmd_eff_rank", float("nan")))

    # Average tree-edit distance over index-paired trees (both target the same node
    # count, so pairing by index is valid). Capped by node count and pair count.
    if ged_enabled:
        ged_vals: list[float] = []
        skipped_size = 0
        skipped_error = 0
        n_pairs = min(len(gen_graphs), len(gt_graphs))
        evaluated = 0
        for i in range(n_pairs):
            if evaluated >= GED_MAX_PAIRS:
                break
            g, h = gen_graphs[i], gt_graphs[i]
            if _root_of(g) is None or _root_of(h) is None:
                skipped_error += 1
                continue
            if g.number_of_nodes() > GED_MAX_NODES or h.number_of_nodes() > GED_MAX_NODES:
                skipped_size += 1
                continue
            try:
                d = graph_edit_distance_topology(g, h, timeout=ged_timeout)
            except Exception:
                skipped_error += 1
                continue
            if d is not None:
                ged_vals.append(float(d))
                evaluated += 1
        metrics["tree_edit_dist_mean"] = float(np.mean(ged_vals)) if ged_vals else float("nan")
        considered = max(n_pairs, 1)
        metrics["tree_edit_skipped_frac"] = float((skipped_size + skipped_error) / considered)
        metrics["tree_edit_n_pairs"] = float(len(ged_vals))

    return metrics
