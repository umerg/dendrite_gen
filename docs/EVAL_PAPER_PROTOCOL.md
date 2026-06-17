# Evaluation: paper protocol (deferred / offline)

This document specifies the **one-shot, offline** evaluation to run on the *final*
checkpoint for the paper. It is intentionally separate from the in-loop training
monitor (`validation/dist_metrics.py`, logged every validation step), which reports
clean **point-estimate scalars** only.

Everything here adds statistical rigor that would clutter a training dashboard:
confidence intervals, significance tests, FDR, and the heavier generative-set checks
(memorization, multi-seed variance). **None of it is built yet** — this is the plan
of record for when we write up results. Run it once, against the held-out **test
set**, not the validation set.

Status: **NOT IMPLEMENTED** (Phase 2). The in-loop monitor is Phase 1 and is shipped.

---

## What already exists and is reused

The in-loop monitor already provides the building blocks the offline protocol wraps:

- Marginals: `compute_distribution_metrics` (W1 + KS per feature) in
  `validation/dist_metrics.py`.
- Per-tree morphometric vector + standardization: `assemble_morpho_vector`,
  `standardize_vectors`, `build_gt_cache`.
- Joint embeddings: standardized 16-d morphometric vector and the PCA-reduced
  Euclidean-from-root TMD persistence image (`utils.tmd.compute_tmd_embedding`,
  filtration `radial_root`).
- Kernels/estimators in `utils/dist_helper.py`: `gaussian_rbf`,
  `median_heuristic_bandwidth`, `mmd2_unbiased` (unbiased, unclipped),
  `density_coverage`.

The offline protocol = these point estimators + (1) uncertainty, (2) significance,
(3) generative-set checks, evaluated once with paper-grade settings.

---

## Sample sizes

Validation/test N ≈ **400 (trees)**, **2000 (neurons)**. Large enough that point
estimates are stable, so the **dominant statistical concern is over-powered
significance**: at N=2000 even biologically trivial gaps give p≈0. Therefore the
protocol **leads with effect sizes (excess-over-floor) and their CIs**; p-values are
supporting, not the headline.

---

## 1. Uncertainty — tree-level bootstrap CIs

Generation is a single deterministic draw, so bootstrap captures **finite-eval-set
sampling only, NOT generation stochasticity** (that is §3 multi-seed). State this
limitation wherever a CI appears.

- **Clustered (tree-level) bootstrap.** Elements within a tree are correlated, so
  resample **whole trees** (paired gen/gt indices to preserve size-matching), re-pool
  elements, recompute the metric. Element-level resampling would badly understate
  variance.
- `B = 2000` replicates (final); percentile CIs at 90/95%. Seed =
  `f(global_seed, key)` for reproducibility.
- Emit per metric `K`: `K`, `K_ci_lo`, `K_ci_hi`, `K_ci_level`, `K_boot_B`, `K_n_eff`.
- Suggested helper: `tree_block_bootstrap(stat_fn, gen_items, gt_items, *, B, ci_level, seed)`.

## 2. Excess-over-floor (the headline effect size)

The real-vs-real floor (train-subset vs test, matched to N) carries the same
finite-sample upward bias as the gen-vs-test metric, so report:

- `excess = metric_gen − metric_floor` (additive; keep sign — MMD² can be negative).
- CI via a **joint per-replicate difference** bootstrap (resample trees once per
  replicate, compute gen-metric and floor-metric on the same resample, take the
  difference) — captures the bias correlation → tighter, honest intervals.
- Emit `K_excess_over_floor`, `K_excess_ci_lo`, `K_excess_ci_hi`, `K_at_floor`
  (bool: excess CI contains 0). "At floor" = indistinguishable from two real samples
  at this N.
- Suggested helper: `excess_over_floor_bootstrap(stat_fn, gen, gt, floor_a, floor_b, *, B, seed)`.

## 3. Significance (effect-size-led)

- **Omnibus MMD permutation test** on the morphometric embedding: pool gen+gt, relabel
  into original-size groups, recompute MMD², `P = 5000`; `p = (1 + #{perm ≥ obs}) / (P + 1)`.
  **Hold the kernel bandwidth FIXED across all permutations** (use the cached gt
  bandwidth) or the test is invalid.
- **Per-feature W1/KS:** tree-level **permutation** p-values (permute whole-tree
  labels, re-pool). **Never** use `ks_2samp` asymptotic p-values for discrete/tied
  features (Strahler, counts, branch order).
- At these N, report p-values as a binary "distinguishable: yes/no" alongside the
  effect size; do not lead with them.
- **Multiple comparisons:** if making per-feature significance claims, control FDR
  with **Benjamini–Hochberg** (`*_pval_bh`) at q=0.05; metrics are correlated so a
  permutation max-statistic or BH (not Bonferroni) is appropriate. If only reporting
  effect sizes with CIs, FDR is unnecessary — phrase as "within/above floor band."

## 4. Generative-set checks (run on the TEST set)

- **Memorization.** For each generated tree, NN distance to the nearest **train**
  tree in the PCA-TMD (and/or standardized morphometric) embedding, vs the train→test
  NN-distance baseline. Emit `mem_gen_nn_train_{mean,median,p05}`,
  `mem_train_nn_test_{mean,median}`, and `mem_ratio` = gen→train median / train→test
  median. `mem_ratio < 1` flags copying. Cap/cache the train reference subset.
- **Multi-seed variance.** Add a `seed` parameter to `Trainer.evaluate` (when set:
  `np.random.default_rng(seed)` for the prediction permutation **and**
  `th.manual_seed(seed)` / `th.cuda.manual_seed_all(seed)` before the diffusion
  sampling loop; `seed=None` preserves current behavior). Generate with K seeds,
  recompute the metric dict per seed, emit `*_mean` / `*_std` (nan-robust). This is
  the generation-stochasticity component the Phase-1 CIs explicitly exclude. K small
  (e.g. 3–5); it multiplies generation cost by K.

## 5. Cost controls

- The O(N²) MMD Gram per bootstrap/permutation replicate dominates at N=2000. Cap the
  per-replicate sample to ~1000 (random subsample + average over a few draws);
  Density/Coverage uses the full N cheaply via `cKDTree`.
- Reuse the cached gt bandwidth / PCA / standardization (`build_gt_cache`); never
  refit per replicate.

## 6. What to report vs only monitor

**Paper (final checkpoint):**
- Excess morphometric-MMD² over floor + permutation p across a {0.5σ, σ, 2σ}
  bandwidth band.
- Coverage (and Density) vs floor with CIs, on the PCA-reduced embeddings.
- Pooled per-feature W1/KS as excess-over-floor with CIs, aggregated into the two
  normalized headline numbers; per-feature supplement table with BH-FDR if claiming
  significance.
- Memorization and multi-seed variance.
- A limitation paragraph: in-loop CIs are eval-set-only; single-seed in-loop; seed
  variance quantified here.

**Monitor-only (in-loop, already shipped):** full per-feature W1/KS trajectories,
raw signed MMD², raw Density; checkpoint selection on `headline_excess_mmd_morpho`.

---

## Suggested new keys (offline)

```
K_ci_lo, K_ci_hi, K_ci_level, K_boot_B, K_n_eff
K_excess_over_floor, K_excess_ci_lo, K_excess_ci_hi, K_at_floor
mmd_morpho_pval, mmd_morpho_perm_P, *_pval, *_pval_bh
mem_gen_nn_train_mean/median/p05, mem_train_nn_test_mean/median, mem_ratio
*_mean, *_std            # multi-seed
```
