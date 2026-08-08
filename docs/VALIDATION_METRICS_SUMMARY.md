# Validation metrics — current state

**What the training loop computes, what reaches the dashboard, and what each number is worth.**

- **As of:** 2026-08-08 · `MORPHO_VERSION 2`
- **Scope:** everything under `validation/` that runs inside `Trainer.run_validation`.
- **Companion:** `docs/VALIDATION_METRICS_AUDIT.md` is the *analysis* — why the suite looks like
  this, what was measured, what is still wrong. This file is the *reference* — what it does now.
  Where they disagree, this file is current; the audit is partly `MORPHO_VERSION 1`.

---

## 1. What runs

Per validation, per EMA beta:

| Block | Keys | When | Cost |
|---|---|---|---|
| `dist` — marginals + joint | **31 computed**, 19 reported at `standard` | every validation | **1.24 s** at N=200 |
| `floor` — real-vs-real reference | same key set | **off** (`enable_floor: False`) | one cached call |
| `tmd_cond` — matched pairwise | 18 | every 5th validation, needs `enable_pairwise_metrics` | 0.17 s / 64 pairs |
| `teacher_forced` | ~40 | **off** (`eval_mode: rollout`) | expensive (full ODE per level) |
| `class_<name>` | same as `dist` | needs `per_cell_class` + class conditioning | ~1 s per class |

`enable_ks: True` would add 13 KS twins (44 keys); `ged_enabled: True` adds 3 tree-edit keys and
costs **~0.35 s/pair** — more than the entire rest of the suite. Both are off, correctly.

## 2. Sign conventions — read these first

| Metric family | Worse means |
|---|---|
| `*_w1`, `mmd_*` | **larger** |
| `density_*`, `coverage_*` | **smaller** |
| `gen_degenerate_frac`, `morpho_nan_frac` | **larger** (0.0 is healthy) |

`mmd_*` is an **unbiased** estimator, so it is legitimately slightly negative when the two sets
match. Do not clip it and do not read a negative value as an error.

## 3. The dashboard at `standard` (19 keys)

### Joint — "does the population look right, including cross-feature structure?"

| Key | Plain meaning |
|---|---|
| `mmd_morpho` | **The single best monitor.** Overall distance between generated and real morphology, including correlations that every marginal misses. |
| `density_morpho` | *Fidelity.* Do generated trees land where real trees actually live? |
| `coverage_morpho` | *Diversity.* Does every region of real tree-space have a generated neighbour? Low = mode collapse. |
| `mmd_tmd` | Branching topology weighted by spatial reach (persistence image). Scale-free by construction. |

### Marginals — one distribution each

| Key | Plain meaning |
|---|---|
| `radial_span_w1` | Lateral spread perpendicular to `uhat` — canopy width. |
| `axial_extent_w1` | Extent along `uhat` — height / apical reach. |
| `branch_length_w1` | Length of a typical segment between branch points. |
| `bifurcation_angle_w1` | How wide a fork opens (degrees). |
| `radial_to_root_w1` | Straight-line distance from soma/base to each node — where the mass sits. |
| `contraction_w1` | Per leaf: straight-line ÷ along-cable distance from root. Tortuosity. |
| `branch_order_w1` | **On our data this is exactly the node-depth distribution** (§6). |
| `partition_asymmetry_w1` | Van Pelt asymmetry: 0 = every fork splits evenly, 1 = caterpillar. |
| `node_count_w1` | Tree size. |
| `sholl_critical_radius_w1` | Where the arbor is densest, as a fraction of the tree's own reach. |
| `tmd_barlen_w1` | Persistence bar lengths — how far sub-branches reach before merging. |

### Aggregates and health

| Key | Plain meaning |
|---|---|
| `w1_pooled_mean_normalized` | Mean of per-element W1s ÷ GT spread. Blunt but scale-free. |
| `w1_pertree_mean_normalized` | Same, over per-tree scalars. |
| `gen_degenerate_frac` | Fraction of generated trees with no bifurcation or zero extent. **Should be 0.0.** |
| `morpho_nan_frac` | Fraction of non-finite morpho entries. **Should be 0.0.** |

> **Why the two health metrics exist.** `standardize_vectors` imputes non-finite morpho features
> to the GT *mean*, so a generated tree with zero forks scores as having perfectly average
> branching asymmetry and fork angle — the imputation neutralises exactly the dimensions that
> would flag it. The imputation was kept (it keeps `mmd_morpho` interpretable); these two keys
> are the disclosure. On real GT both are 0.000 for every feature on both datasets, so **any
> non-zero value is a generator failure, never a data property.**

### Dropped from the dashboard (still computed, still in `step_*.pkl`)

`sholl_peak_w1`, `strahler_w1` (detect nothing — max z 0.1/0.9 and 0.0/0.9), `leaf_count_w1`,
`bifurcation_count_w1` (arithmetically identical to `node_count` on binary trees),
`total_extent_w1`, `path_to_root_w1`, `sholl_auc_w1` (duplicates of retained keys),
`density_tmd`, `coverage_tmd` (never leave 0.95–0.99 under any defect), `mmd_bandwidth_*`,
`tmd_eff_rank` (per-run constants → now pushed once to wandb *config*).

Set `validation.metric_report_level: full` to see them all. **The tier is a pure logging filter**
— `compute_distribution_metrics` does not take it as an argument, so it is structurally
impossible for it to change a computed number.

---

## 4. How to read a number: floors and detection power

**Every metric has a large real-vs-real floor.** Two disjoint halves of the *same real dataset*
differ by this much. A run near the floor is at ceiling performance, not failing.

Measured on the current code: 6 random disjoint splits, n per side as shown.
`z = (defect − floor_mean) / floor_sd`. **|z| ≳ 3 is a real signal; |z| ≲ 1 is noise.**

### Trees (`trees_genus_d10/val`, n = 168/side)

| metric | floor (mean ± sd) | scale ×1.1 | squash ×0.75 | jitter .15 | prune 15% |
|---|---|---|---|---|---|
| `mmd_morpho` | −0.0004 ± 0.0013 | 2.5 | **11.3** | 3.5 | **14.3** |
| `coverage_morpho` | 0.9673 ± 0.0141 | −0.8 | −2.5 | −1.1 | **−4.2** |
| `density_morpho` | 1.0438 ± 0.0647 | −0.9 | −1.0 | −0.5 | −1.6 |
| `radial_span_w1` | 0.1825 ± 0.0375 | **5.1** | **17.1** | **7.8** | 0.2 |
| `branch_length_w1` | 0.0200 ± 0.0067 | 2.3 | **6.7** | 0.5 | −0.1 |
| `w1_pooled_mean_normalized` | 0.0571 ± 0.0090 | 2.3 | **3.3** | **3.0** | 0.9 |
| `branch_order_w1` | 0.0771 ± 0.0458 | 0.0 | 0.0 | 0.0 | **3.3** |
| `axial_extent_w1` | 0.5967 ± 0.1685 | **3.1** | 0.0 | 2.1 | 0.3 |
| `radial_to_root_w1` | 0.6007 ± 0.1320 | 2.0 | −0.1 | 2.8 | −0.1 |
| `contraction_w1` | 0.0100 ± 0.0056 | 0.0 | 2.9 | 0.5 | 0.0 |
| `w1_pertree_mean_normalized` | 0.1268 ± 0.0257 | 1.4 | 1.3 | 1.3 | 0.1 |
| `bifurcation_angle_w1` | 1.0893 ± 0.4463 | 0.0 | 1.5 | 0.7 | 0.1 |
| `node_count_w1` | **6.96 ± 2.63** | 0.0 | 0.0 | 0.0 | 0.0 |
| `tmd_barlen_w1` | 0.0347 ± 0.0146 | 0.9 | 0.8 | 0.5 | −0.1 |
| `partition_asymmetry_w1` | 0.0113 ± 0.0033 | 0.0 | 0.0 | 0.0 | 0.7 |
| `sholl_critical_radius_w1` | 0.0304 ± 0.0106 | 0.0 | −0.1 | 0.4 | −0.2 |
| `mmd_tmd` | 0.0001 ± 0.0016 | 0.0 | 0.3 | 0.6 | 0.1 |

### Neurons (`neurons_conditional_full/val`, n = 200/side)

| metric | floor (mean ± sd) | scale ×1.1 | squash ×0.75 | jitter .15 | prune 15% |
|---|---|---|---|---|---|
| `mmd_morpho` | −0.0003 ± 0.0017 | **12.8** | **38.3** | 3.8 | **19.9** |
| `coverage_morpho` | 0.9600 ± 0.0147 | −2.4 | **−16.5** | −1.3 | **−5.9** |
| `density_morpho` | 0.9600 ± 0.0629 | −0.1 | **−7.0** | 0.0 | −2.9 |
| `radial_span_w1` | 8.944 ± 1.891 | **11.1** | **31.7** | **5.1** | 2.2 |
| `tmd_barlen_w1` | 1.816 ± 0.511 | **11.1** | **20.2** | 2.3 | 2.0 |
| `branch_length_w1` | 0.9927 ± 0.4528 | **8.7** | **14.2** | 1.6 | 0.3 |
| `radial_to_root_w1` | 3.685 ± 1.173 | **6.9** | **7.0** | 1.1 | **4.4** |
| `axial_extent_w1` | 14.81 ± 4.81 | **6.3** | 0.0 | 1.3 | **6.3** |
| `w1_pooled_mean_normalized` | 0.0373 ± 0.0119 | **4.9** | **6.7** | 1.7 | **3.2** |
| `w1_pertree_mean_normalized` | 0.1293 ± 0.0317 | 2.9 | **4.7** | 0.9 | **4.5** |
| `partition_asymmetry_w1` | 0.0119 ± 0.0018 | 0.0 | 0.0 | 0.0 | **12.8** |
| `mmd_tmd` | 0.0025 ± 0.0062 | 0.0 | **4.9** | 0.6 | 2.8 |
| `branch_order_w1` | 0.1319 ± 0.0648 | 0.0 | 0.0 | 0.0 | **3.6** |
| `bifurcation_angle_w1` | 0.6044 ± 0.1728 | 0.0 | 1.0 | 1.5 | **3.5** |
| `sholl_critical_radius_w1` | 0.0114 ± 0.0062 | 0.0 | **3.1** | 0.3 | 2.5 |
| `node_count_w1` | **3.37 ± 1.04** | 0.0 | 0.0 | 0.0 | 2.2 |
| `contraction_w1` | 0.0037 ± 0.0021 | 0.0 | 0.8 | **3.1** | 0.4 |

### What this table says in practice

1. **`mmd_morpho` is the metric to watch.** It is the only one clearly above floor for *every*
   defect class on both datasets.
2. **Geometry vs topology split cleanly.** `radial_span_w1` / `branch_length_w1` own the
   geometric defects and are blind to pruning; `branch_order_w1` and `partition_asymmetry_w1`
   are **exactly 0.0** on every geometric defect and are the only things that see pruning. Watch
   one from each group.
3. **`node_count_w1` is orthogonal to geometry** — bit-identical across every geometric
   transform. It only moves on size/topology, which is what it is for.
4. **W1 is a distance, so it is not monotone in "how wrong" along any single axis.** A single
   metric improving is not progress. That is what the joint metrics are for.
5. **Detection threshold is roughly a 10% global defect** at these sample sizes. Below that, the
   suite cannot separate defect from sampling noise.

---

## 5. The morphometric vector (`MORPHO_KEYS` v2, 9-D)

The per-tree feature vector behind `mmd_morpho` / `density_morpho` / `coverage_morpho`.

```
axial_extent          radial_span            max_branch_order
partition_asymmetry   mean_branch_length     mean_bifurcation_angle
mean_radial_to_root   mean_contraction       sholl_critical_radius
```

Pipeline: z-score by GT mean/std → (optional ZCA whitening) → Gaussian RBF MMD with a
median-heuristic bandwidth. **All GT-derived quantities (mean, std, bandwidth, TMD PCA) are fit
once on the fixed eval set and reused every step** — that is what makes the trajectory
comparable across checkpoints, and it is the thing this class of metric most often gets wrong.

**`node_count` is deliberately absent.** Not for redundancy — it is excluded so `mmd_morpho` is a
*shape-only* comparison against baselines that are handed the node count. Size is still reported
as `node_count_w1`.

Seven v1 features were dropped as redundant (`leaf_count`/`bifurcation_count` at r = 1.000 with
`node_count`; `total_extent` at 0.999 with `axial_extent`; `mean_path_to_root` at 0.996 with
`mean_radial_to_root`; `sholl_peak`/`sholl_auc` as size proxies) and `strahler` was replaced by
`max_branch_order`. Measured payoff — `mmd_morpho` detection z:

| defect | v1 (16-D) | v2 (9-D) |
|---|---|---|
| trees: prune 15% leaves | **0.2** | **14.3** |
| trees: perpendicular squash | 5.2 | **11.3** |
| neurons: perpendicular squash | 21.6 | **38.3** |
| neurons: prune 30% leaves | 40.3 | **65.1** |

`MORPHO_VERSION` is stamped into the GT cache. **v1 and v2 `mmd_morpho` are not comparable** — do
not put runs from either side of the change on one axis.

---

## 6. `branch_order` is tree depth — the one naming trap

`branch_order_values` increments at every non-root node of undirected degree ≥ 3. Our datasets
are strictly binary away from the root with **zero degree-2 non-root nodes**, so every non-root
internal node increments:

> `branch_order == hop_depth − 1`, **exactly**, verified 300/300 graphs on trees d10, trees d20
> and neurons. Pinned by `test_branch_order_equals_hop_depth_on_binary_tree`.

So `branch_order_w1` is the **node-depth distribution** and `max_branch_order` (in the vector) is
**tree depth − 1**. The general name is kept because the function computes the general quantity;
on a graph that retained its degree-2 chain nodes the two would diverge.

**Caveat on the depth-capped tree sets.** `trees_genus_d{10,15,20}` are trimmed to a depth cap in
preprocessing, so the feature is concentrated there — **97% of d10 trees fall in {9,10,11}**
(7 distinct values), 85% of d20 in {19,20,21}; neurons are well spread (19 distinct). On d10/d20
it therefore partly measures "did the generator reach the dataset's ceiling" rather than free
morphological depth. (Values exceed the nominal cap because binarization *inserts* nodes after
the trim.)

---

## 7. Matched pairwise (`tmd_cond`) — is the conditioning actually used?

Only meaningful with `tmd_hidden_dim > 0`. `gen[i]` was conditioned on `gt[i]` and the rollout
keeps them index-aligned, so each pair is compared directly. The distributional suite **cannot**
answer this: a model that ignores its conditioning entirely can still match every population
marginal.

| Key | Meaning |
|---|---|
| `pd_wasserstein_{path,radial_root}_{mean,median}` | Did the generated tree realise the *specific* barcode it was handed? The core number. |
| `pd_nan_frac_{filtration}` | Fraction of pairs whose diagram could not be built. |
| `height_absdiff_*`, `span_xy_absdiff_*` | Per-pair \|Δ\| of axial extent / perpendicular diameter. |
| `bbox_diag_absdiff_*` | Per-pair \|Δ\| of the 3D bounding-box diagonal. ⚠️ **not rotation-invariant** about `uhat` — it penalises a symmetry the model is licensed to exercise. Prefer the other two. |
| `branch_length_w1_pairwise_*`, `bifurcation_angle_w1_pairwise_*` | Per-tree analogues of the pooled marginals. |
| `n_pairs`, `n_pairs_skipped` | Sample size and structural-failure count. |

PD distances use `minmax` normalization and are therefore **scale-free**; the `*_absdiff` keys
carry the scale. Do not read `pd_wasserstein_*` alone as "the tree matches".

---

## 8. Known blind spots

1. **The suite cannot see a flipped tree.** Reflect every generated tree along `uhat` and **0 of
   31 metrics change** — re-verified on `MORPHO_VERSION 2`; the v2 prune neither helped nor hurt
   here, because every geometric feature is still sign-blind (extents are ranges, distances are
   unsigned, `max_branch_order`/`partition_asymmetry` are topological, and the `radial_root` TMD
   image is invariant to *any* isometry). A perfect pyramidal neuron pointing the wrong way
   scores perfectly. This is why `docs/APICAL_AXIAL_ERROR_MODE.md` needed a dedicated offline
   probe. **Fix (not yet done):** signed axial features in `MORPHO_KEYS`.

   For contrast, on the same test: rotation *about* `uhat` changes 1 of 31 (`density_morpho`, by
   float jitter only) — correct, that is the SO(2) equivariance the model is licensed to use;
   a 40° off-axis tilt correctly changes 6 of 31.
2. **`enable_floor: False` in every config**, so the floors in §4 are not on the dashboard and
   `headline_excess_mmd_morpho` is never computed. Turning it on costs one cached call.
3. **`tmd_eval_filtration: radial_root` is inside `model.tmd_filtrations`**, so on TMD-conditioned
   runs `mmd_tmd` partly scores the model echoing its own input. Do not cite it as independent
   evidence there.
4. **Root degree is given** (`num_root_children` is copied from each GT graph and hard-enforced),
   so any credit for matching soma/base fan-out is unearned. `partition_asymmetry` and depth are
   partly informed by it. Target node count is **not** enforced — `use_size_ratio: False`, so
   `node_count_w1` is a genuine measure of learned stopping.
5. **`max_tree_size: 400`** caps generation, but neuron GT reaches 537 nodes — the upper tail is
   structurally unreachable.
6. **`mmd_tmd` is weak on trees** (max z 0.6 across all defects) and `density_tmd`/`coverage_tmd`
   are inert everywhere. The TMD joint block earns its cost on neurons, not on trees.

---

## 9. Config reference

```yaml
validation:
  enable_dist_metrics: True     # the marginal + joint suite
  enable_morphometrics: True    # path/radial/contraction/order/strahler/asymmetry/sholl
  enable_light_joint: True      # morpho + TMD MMD/Density/Coverage
  enable_ks: False              # KS twins alongside W1 (+13 keys)
  ged_enabled: False            # tree-edit distance — ~0.35 s/pair, keep off
  enable_floor: False           # real-vs-real reference lines  <-- worth turning ON
  metric_report_level: standard # headline | standard | full   (logging filter only)
  morpho_whiten: False          # ZCA on the morpho vector     (untested on v2)
  tmd_eval_filtration: radial_root   # single source of truth for tmd_barlen AND mmd_tmd
  tmd_eval_bins: 16
  tmd_pca_ncomp: 64
  dc_nearest_k: 5
  eval_mode: rollout            # rollout | teacher_forced | both
  enable_pairwise_metrics: True # tmd_cond block
  tmd_cond_every: 5
```

wandb key layout:

```
validation/ema_<beta>/dist/<key>
validation/ema_<beta>/floor/<key>          (enable_floor)
validation/ema_<beta>/tmd_cond/<key>       (every 5th validation)
validation/ema_<beta>/class_<name>/<key>   (per_cell_class)
validation/ema_<beta>/teacher_forced/...   (eval_mode)
```

Per-run constants (`mmd_bandwidth_morpho`, `mmd_bandwidth_tmd`, `tmd_eff_rank`,
`morpho_gt_nan_frac`, `morpho_version`) go to **wandb config**, not the time series.

---

## 10. Changelog

**2026-08-08 — `MORPHO_VERSION 2`.** Breaks comparability of `mmd_morpho` / `density_morpho` /
`coverage_morpho` and of `tmd_barlen_w1`, `sholl_peak_w1`, `sholl_critical_radius_w1`,
`sholl_auc_w1` with earlier runs.

- `MORPHO_KEYS` 16-D → 9-D; `strahler` → `max_branch_order`; `node_count` excluded.
- `sholl_critical_radius` divides by the tree's **own** radial extent (was the shell grid's outer
  edge — which made it a size feature); Sholl summaries now use **per-tree shells** (the shared
  grid was undersampling small trees, making `sholl_peak` scale-dependent: 7/13/13 vs 14/14/14
  on one tree scaled ×1/×1.5/×3).
- `tmd_barlen` follows `tmd_eval_filtration` (was hardcoded to `path` while the joint block used
  `radial_root`). **See the note below — this cost tree-side sensitivity.**
- `uhat` threaded into `_eval_embed_fn` (latent wrong-axis bug for `height`/`rho` filtrations).
- Added `gen_degenerate_frac`, `morpho_nan_frac`, `morpho_gt_nan_frac`.
- Added `metric_report_level` (logging filter) and `morpho_whiten` (default off).
- Integer leaves now reach wandb (previously dropped, losing teacher-forced `n_leaves`).

> ### ⚠️ Open regression from this change: `tmd_barlen_w1` on trees
>
> Switching `tmd_barlen` from `path` to `radial_root` unified it with the joint TMD block, but
> measurably **weakened it on trees**. Same defects, same splits, same n — only the filtration
> differs:
>
> | | scale ×1.1 | squash ×0.75 | jitter .15 |
> |---|---|---|---|
> | trees, `path` (old) | 2.7 | **7.5** | 0.6 |
> | trees, `radial_root` (now) | 0.9 | **0.8** | 0.5 |
> | neurons, `path` (old) | 12.3 | 21.1 | 2.5 |
> | neurons, `radial_root` (now) | 11.1 | 20.2 | 2.3 |
>
> Neurons are unaffected; trees lost roughly an order of magnitude. `radial_span_w1` still covers
> those defects on trees (z 5.1 / 17.1), so nothing is *undetected* — but `tmd_barlen_w1` is now
> close to inert there. Options: accept it (consistency over one redundant marginal), or give
> `tmd_barlen` its own `path` filtration and document the unit mismatch explicitly. Not yet
> decided.
