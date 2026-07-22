###############################################################################
# RBF-kernel MMD and Density/Coverage for fixed-length feature vectors.
#
# A proper Gaussian RBF kernel, an *unbiased* MMD^2 estimator, and the
# Naeem et al. (2020) Density & Coverage, operating on (N, d) matrices
# (morphometric vectors, persistence images). Used by validation/dist_metrics.py.
###############################################################################
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist


def gaussian_rbf(x, y, sigma=1.0):
    """RBF kernel exp(-||x-y||^2 / (2 sigma^2)) for two equal-length vectors."""
    diff = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return float(np.exp(-float(diff @ diff) / (2.0 * sigma * sigma)))


def median_heuristic_bandwidth(X, *, max_n=512, seed=0):
    """
    Median-heuristic RBF bandwidth: median of pairwise Euclidean distances over X
    (subsampled to ``max_n`` rows for cost). Floored at 1e-8 so sigma is never 0.

    Compute this ONCE on the fixed reference (gt) set and reuse it across training
    steps — a per-step bandwidth would make the MMD trajectory non-comparable.
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n < 2:
        return 1.0
    if n > max_n:
        rng = np.random.default_rng(seed)
        X = X[rng.choice(n, size=max_n, replace=False)]
    d = pdist(X, metric="euclidean")
    d = d[d > 0]
    if d.size == 0:
        return 1.0
    return float(max(np.median(d), 1e-8))


def mmd2_unbiased(X, Y, sigma):
    """
    Unbiased estimate of squared MMD between row-sets X (n,d) and Y (m,d) under a
    Gaussian RBF kernel of bandwidth ``sigma``.

    Unbiased = excludes the diagonal self-terms; the estimate CAN be slightly
    negative when the two distributions match — that is expected and the value is
    returned unclipped (clipping to 0 would reintroduce bias and break comparison
    against the real-vs-real floor). nan if either set has < 2 rows.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        return float("nan")
    denom = 2.0 * sigma * sigma
    Kxx = np.exp(-cdist(X, X, metric="sqeuclidean") / denom)
    Kyy = np.exp(-cdist(Y, Y, metric="sqeuclidean") / denom)
    Kxy = np.exp(-cdist(X, Y, metric="sqeuclidean") / denom)
    sum_xx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    sum_yy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    sum_xy = Kxy.mean()
    return float(sum_xx + sum_yy - 2.0 * sum_xy)


def density_coverage(gen_emb, gt_emb, *, k=5):
    """
    Naeem et al. (2020) Density and Coverage between generated and real embeddings.

    The real manifold is the union of k-th-NN hyperspheres around each real point.
      - density  = (1/(k*M)) sum over fake points of how many real spheres contain it
      - coverage = fraction of real points whose sphere contains >= 1 fake point

    Density separates fidelity (are fakes realistic?), Coverage separates diversity
    (do fakes cover the real variety / no mode collapse?). k is guarded to N-1.
    Returns (density, coverage); (nan, nan) if sets are too small.
    """
    real = np.asarray(gt_emb, dtype=np.float64)
    fake = np.asarray(gen_emb, dtype=np.float64)
    N, M = len(real), len(fake)
    if N < 2 or M < 1:
        return float("nan"), float("nan")
    k = int(min(k, N - 1))
    if k < 1:
        return float("nan"), float("nan")
    real_tree = cKDTree(real)
    # k-th NN distance among real points (query k+1 to skip the self-match at dist 0)
    knn_dists, _ = real_tree.query(real, k=k + 1)
    radii = np.asarray(knn_dists)[:, -1]
    fake_tree = cKDTree(fake)
    # For each real point, the fake indices falling inside its sphere of radius r_i.
    within = fake_tree.query_ball_point(real, radii)
    total = sum(len(c) for c in within)
    density = float(total / (k * M))
    coverage = float(np.mean([len(c) > 0 for c in within]))
    return density, coverage
