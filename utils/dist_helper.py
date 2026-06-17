###############################################################################
#
# Adapted from https://github.com/lrjconan/GRAN/ which in turn is adapted from https://github.com/JiaxuanYou/graph-generation
#
###############################################################################
# import pyemd
import numpy as np
import networkx as nx
import concurrent.futures
from functools import partial
from scipy.linalg import toeplitz


def emd(x, y, distance_scaling=1.0):
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(np.float64)
    distance_mat = d_mat / distance_scaling

    # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    emd = pyemd.emd(x, y, distance_mat)
    return emd



def l2(x, y):
    dist = np.linalg.norm(x - y, 2)
    return dist


def emd(x, y, sigma=1.0, distance_scaling=1.0):
    ''' EMD
        Args:
            x, y: 1D pmf of two distributions with the same support
            sigma: standard deviation
    '''
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(np.float64)
    distance_mat = d_mat / distance_scaling

    # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    return np.abs(pyemd.emd(x, y, distance_mat))


def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    ''' Gaussian kernel with squared distance in exponential term replaced by EMD
        Args:
            x, y: 1D pmf of two distributions with the same support
            sigma: standard deviation
    '''
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(np.float64)
    distance_mat = d_mat / distance_scaling

    # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    emd = pyemd.emd(x, y, distance_mat)
    return np.exp(-emd * emd / (2 * sigma * sigma))


def gaussian(x, y, sigma=1.0):  
    support_size = max(len(x), len(y))
    # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    dist = np.linalg.norm(x - y, 2)
    return np.exp(-dist * dist / (2 * sigma * sigma))


def gaussian_tv(x, y, sigma=1.0):  
    support_size = max(len(x), len(y))
    # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    dist = np.abs(x - y).sum() / 2.0
    return np.exp(-dist * dist / (2 * sigma * sigma))


def kernel_parallel_unpacked(x, samples2, kernel):
    d = 0
    for s2 in samples2:
        d += kernel(x, s2)
    return d


def kernel_parallel_worker(t):
    return kernel_parallel_unpacked(*t)


def disc(samples1, samples2, kernel, is_parallel=True, *args, **kwargs):
    ''' Discrepancy between 2 samples '''
    d = 0

    if not is_parallel:
        for s1 in samples1:
            for s2 in samples2:
                d += kernel(s1, s2, *args, **kwargs)
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for dist in executor.map(kernel_parallel_worker, [
                    (s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1
            ]):
                d += dist
    if len(samples1) * len(samples2) > 0:
        d /= len(samples1) * len(samples2)
    else:
        d = 1e+6
    return d


def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    ''' MMD between two samples '''
    # normalize histograms into pmf  
    if is_hist:
        samples1 = [s1 / (np.sum(s1) + 1e-6) for s1 in samples1]
        samples2 = [s2 / (np.sum(s2) + 1e-6) for s2 in samples2]
    return disc(samples1, samples1, kernel, *args, **kwargs) + \
                    disc(samples2, samples2, kernel, *args, **kwargs) - \
                    2 * disc(samples1, samples2, kernel, *args, **kwargs)


def compute_emd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    ''' EMD between average of two samples '''
    # normalize histograms into pmf
    if is_hist:
        samples1 = [np.mean(samples1)]
        samples2 = [np.mean(samples2)]
    return disc(samples1, samples2, kernel, *args,
                            **kwargs), [samples1[0], samples2[0]]


###############################################################################
# RBF-kernel MMD and Density/Coverage for fixed-length feature vectors.
#
# The histogram kernels above (gaussian_emd, gaussian_tv, ...) treat their inputs
# as 1D distributions over bins and are unsuitable for arbitrary per-tree feature
# vectors (morphometric vectors, persistence images). The helpers below operate on
# (N, d) matrices: a proper Gaussian RBF kernel, an *unbiased* MMD^2 estimator, and
# the Naeem et al. (2020) Density & Coverage. Used by validation/dist_metrics.py.
###############################################################################

from scipy.spatial import cKDTree  # noqa: E402
from scipy.spatial.distance import cdist, pdist  # noqa: E402


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
