"""Unit tests for the RBF-MMD / Density-Coverage estimators in utils.dist_helper."""

import numpy as np

from utils.dist_helper import (
    gaussian_rbf,
    median_heuristic_bandwidth,
    mmd2_unbiased,
    density_coverage,
)


def test_mmd2_unbiased_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    Y = rng.normal(size=(200, 8))
    sigma = median_heuristic_bandwidth(np.vstack([X, Y]))
    val = mmd2_unbiased(X, Y, sigma)
    # Unbiased estimate can be slightly negative; should be close to 0 for same dist.
    assert abs(val) < 0.05, val


def test_mmd2_unbiased_increases_with_shift():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 8))
    Y_close = rng.normal(loc=0.2, size=(200, 8))
    Y_far = rng.normal(loc=2.0, size=(200, 8))
    sigma = median_heuristic_bandwidth(X)
    assert mmd2_unbiased(X, Y_far, sigma) > mmd2_unbiased(X, Y_close, sigma)


def test_mmd2_unbiased_not_clipped_can_be_negative():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 4))
    # comparing a set to itself -> unbiased estimate is negative (diagonal excluded).
    sigma = median_heuristic_bandwidth(X)
    assert mmd2_unbiased(X, X, sigma) < 0.0


def test_median_heuristic_positive_and_subsamples():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(5000, 6))
    sigma = median_heuristic_bandwidth(X, max_n=256)
    assert sigma > 0.0


def test_gaussian_rbf_basic():
    assert abs(gaussian_rbf([0.0, 0.0], [0.0, 0.0], sigma=1.0) - 1.0) < 1e-12
    assert gaussian_rbf([0.0], [10.0], sigma=1.0) < 1e-6


def test_density_coverage_perfect_overlap():
    rng = np.random.default_rng(4)
    real = rng.normal(size=(200, 5))
    fake = real.copy()
    dens, cov = density_coverage(fake, real, k=5)
    assert cov > 0.95
    assert dens > 0.5


def test_density_coverage_disjoint_sets():
    rng = np.random.default_rng(5)
    real = rng.normal(size=(200, 5))
    fake = rng.normal(loc=50.0, size=(200, 5))
    dens, cov = density_coverage(fake, real, k=5)
    assert cov < 0.05
    assert dens < 0.05


def test_density_coverage_guards_small_k():
    rng = np.random.default_rng(6)
    real = rng.normal(size=(3, 4))
    fake = rng.normal(size=(3, 4))
    dens, cov = density_coverage(fake, real, k=5)  # k auto-clamped to N-1
    assert np.isfinite(dens) and np.isfinite(cov)
