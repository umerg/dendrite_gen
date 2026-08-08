# Audit of the in-loop validation metrics (`validation/`)

**Date:** 2026-08-08 · **Scope:** every metric that runs inside `Trainer.run_validation`
during training, as configured by the current `parity_*` / `neuron.yaml` configs.
**Status:** code audit + empirical calibration on `trees_genus_d10/val` (337 real trees).

This document answers four questions for each live metric:

1. **What does it say, in plain terms?** (and why it matters for a tree/neuron)
2. **How is it actually computed?** (the code, not the docstring)
3. **What does it assume?** (the invisible preconditions)
4. **How does it fail?** (silent degeneracies, blind spots, mis-calibration)

Then it audits how the marginals are combined into the **morphometric vector** and the two
**joint** metrics, gives an empirical calibration table (real-vs-real floors and detection
power), and ends with a ranked list of confirmed defects and recommended fixes.

Companion documents: `docs/MORPHO_MANIFOLD_ANALYSIS.md` (structure of the GT embeddings),
`docs/EVAL_PAPER_PROTOCOL.md` (the offline/statistical protocol this deliberately omits),
`docs/APICAL_AXIAL_ERROR_MODE.md` (an error mode this suite provably cannot see — §7.1).

---

> ## ⚠️ Status: partly superseded (2026-08-08)
>
> **This document describes `MORPHO_VERSION 1`** — the 16-D morphometric vector, shared Sholl
> shells, and the full 29-key dashboard. Most of it still stands, but the following findings
> have since been **implemented**, so the corresponding sections describe code that no longer
> exists. Each affected section carries an inline note.
>
> | Finding | Section | Change shipped |
> |---|---|---|
> | 3 — nan hides degeneracy | §4.3, §7.2 | `gen_degenerate_frac` + `morpho_nan_frac` now emitted; imputation itself unchanged (disclose-only) |
> | 4 — morpho vector redundancy | §4.2 | **`MORPHO_KEYS` v2: 16-D → 9-D.** See "MORPHO_KEYS v2" below |
> | 5 — `density_tmd`/`coverage_tmd` dead | §6.2 | demoted to the `full` reporting tier |
> | 6 — eval filtration inside conditioning set | §6.3 | unchanged; `tmd_barlen` now at least *matches* the joint block's filtration |
> | 8 — `uhat` not threaded | §6.4 | fixed; `_eval_embed_fn` takes the model axis |
> | 9 — `sholl_critical_radius` wrong divisor | §3.2 | fixed (own extent), **and** shells are now per-tree |
> | 10 — integer leaves dropped from wandb | §1.2 | fixed |
> | 11–14 — dashboard noise | §3.3, §6.5 | `validation.metric_report_level: headline\|standard\|full` |
>
> **Still open, unchanged:** finding **1** (§7.1 — the suite is blind to a reflection along
> `uhat`; no polarity feature yet) and finding **2** (§8.1 — `enable_floor: False` in every
> config, so `headline_excess_mmd_morpho` is still never computed).
>
> **Also shipped but NOT evaluated on v2:** `validation.morpho_whiten` (default **off**) fits a
> ZCA transform on the standardized GT morpho matrix. On the 11-D intermediate it roughly
> doubled shape sensitivity (trees perp-squash z 9.2 → 17.3), but that was measured *before*
> the final 9-D composition and it has had no 6-split evaluation since. Treat it as an
> untested A/B lever, not a recommendation. Its eigenvalue floor (`eps = 1e-3`) is load-bearing:
> on the rank-deficient v1 vector, ZCA at `eps = 1e-6` amplified the four null directions into
> an MMD of 0.35 against a floor of 0.001 — a ~300× artifact. Pruning first is what makes
> whitening safe at all.
>
> ### MORPHO_KEYS v2 (9-D, shipped)
>
> ```
> axial_extent, radial_span, max_branch_order, partition_asymmetry, mean_branch_length,
> mean_bifurcation_angle, mean_radial_to_root, mean_contraction, sholl_critical_radius
> ```
>
> Dropped: `node_count`, `leaf_count`, `bifurcation_count`, `total_extent`,
> `mean_path_to_root`, `sholl_peak`, `sholl_auc`, `strahler`. Added: `max_branch_order`.
>
> `node_count` was removed **not** for redundancy but so `mmd_morpho` is a shape-only
> comparison against baselines that are handed the node count; it is still reported as the
> `node_count_w1` marginal. Measured effect of the whole change on `mmd_morpho` detection
> power (z vs a 6-split real-vs-real floor), reproduced against the shipped code:
>
> | defect | v1 (16-D) | **v2 (9-D)** |
> |---|---|---|
> | trees: perpendicular squash ×0.75 | 5.2 | **11.3** |
> | trees: prune 15% deepest leaves | **0.2** | **14.3** |
> | trees: prune 30% deepest leaves | 0.9 | **37.5** |
> | neurons: perpendicular squash ×0.75 | 21.6 | **38.3** |
> | neurons: prune 30% deepest leaves | 40.3 | **65.1** |
>
> The trees-pruning row is the headline: `mmd_morpho` went from **completely unable** to see
> topological pruning on trees to detecting it decisively.
>
> ### `max_branch_order` is tree depth — know this before reading it
>
> `branch_order_values` increments only at non-root nodes of undirected degree ≥ 3. Our
> datasets are strictly binary away from the root with **zero degree-2 non-root nodes**, so
> every non-root internal node increments: `branch_order == hop_depth − 1` **exactly**
> (verified 300/300 graphs on trees d10, trees d20 and neurons; pinned by
> `test_branch_order_equals_hop_depth_on_binary_tree`). So `max_branch_order` ≡ tree depth − 1,
> and the pooled `branch_order_w1` ≡ the **node-depth distribution**.
>
> It replaced `strahler` rather than joining it — keeping both was measurably *worse* (trees
> Σz 56.3 vs 69.0) because they correlate (+0.48 / +0.31) and the redundant one acts as a
> dilution dimension. The vector now decomposes topology into **depth** (`max_branch_order`)
> and **balance** (`partition_asymmetry`) instead of carrying one blended hierarchy measure.
>
> **Caveat on the depth-capped tree sets.** `trees_genus_d{10,15,20}` are trimmed to a depth
> cap in preprocessing, so this feature is concentrated there — **97% of d10 trees fall in
> {9,10,11}** (7 distinct values), 85% of d20 in {19,20,21}. It is well spread on neurons (19
> distinct). On d10/d20 it therefore partly measures "did the generator reach the dataset's
> depth ceiling" rather than free morphological depth. (Values exceed the nominal cap because
> binarization *inserts* nodes after the depth trim.)

---

## 1. What actually runs

`validation/` has nine files. Only four are on the training path:

| File | On the live path? | Entry point |
|---|---|---|
| `dist_metrics.py` | **Yes, every validation** | `compute_distribution_metrics` |
| `structural_metrics.py` | **Yes** (all the primitives) | via `dist_metrics` |
| `geometric_metric.py` | **Yes**, but only 3 of 4 functions | via `tmd_conditional_eval` |
| `tmd_conditional_eval.py` | **Yes**, every 5th validation, opt-in | `compute_conditional_pairwise_metrics` |
| `teacher_forced_eval.py` | Only if `eval_mode ∈ {teacher_forced, both}` — **off in every current config** | `evaluate_teacher_forced` |
| `plot.py`, `plot_sequence.py`, `3D_plot.py`, `chamfer.py` | No (plots / offline) | — |

`precision_recall_f1_radius` in `geometric_metric.py` is **dead on the live path** — nothing
imports it from the trainer.

### 1.1 Config gating (as set in `parity_trees_d10_tmd.yaml` / `parity_neurons_tmd.yaml` / `neuron.yaml`)

```
eval_mode: rollout            -> free-running generation only; NO teacher-forced eval
enable_dist_metrics: True     -> the whole marginal + joint suite
enable_morphometrics: True    -> adds path/radial/contraction/order/strahler/asymmetry/sholl
enable_light_joint: True      -> morpho + TMD MMD / Density / Coverage
enable_ks: False              -> W1 only, no KS anywhere
ged_enabled: False            -> tree-edit distance OFF   (see §3.4 — it costs ~0.35 s/pair)
enable_floor: False           -> real-vs-real reference OFF (see §8.1 — this is the big one)
tmd_eval_filtration: radial_root ; tmd_eval_bins: 16 ; tmd_pca_ncomp: 64 ; dc_nearest_k: 5
enable_pairwise_metrics: True ; tmd_cond_every: 5   (parity configs only)
per_cell_class: False
```

So the **live metric set is exactly 29 scalars** per EMA beta: 18 W1 marginals (7 pooled + 11
per-tree), 2 normalized W1 aggregates, 6 joint scalars, and 3 per-run constants — plus, every
5th validation, 18 matched-pair scalars under `tmd_cond`. Nothing else. (Verified by counting
the returned dict: 29 keys with `enable_ks: False`, `ged_enabled: False`, `enable_floor: False`.)

### 1.2 Where the numbers land

`Trainer.log` (`graph_generation/training.py:1004`) flattens the nested results dict into
wandb keys:

```
validation/ema_<beta>/dist/<key>            <- compute_distribution_metrics
validation/ema_<beta>/floor/<key>           <- real-vs-real floor  (only if enable_floor)
validation/ema_<beta>/tmd_cond/<key>        <- matched pairwise
validation/ema_<beta>/teacher_forced/...    <- TF eval (if enabled)
validation/ema_<beta>/timing/<key>
```

> **Defect (minor, confirmed).** `_collect_log` (`training.py:1026`) logs a leaf only if
> `isinstance(value, float)`. Python `int` is not `float`, so **integer leaves are silently
> dropped from wandb**. On the TF path this loses `n_leaves` and `min_depth` — i.e. the sample
> size behind every TF number never reaches the dashboard. `dist_metrics` and
> `tmd_conditional_eval` are unaffected (they wrap everything in `float()`).

---

## 2. The generation contract (what is *given* vs what is *earned*)

Before reading any metric you have to know what the sampler was handed. Getting this wrong is
the fastest way to over- or under-credit the model.

| Quantity | Given to the sampler? | Consequence for metrics |
|---|---|---|
| **Target node count** | *Soft only.* `target_size` is passed but `use_size_ratio: False` in `config/method/expansion.yaml`, so the model never sees it as a feature. It only sets the outer loop bound (`max_steps = 2·max(target_size)`) and gates the root spawn. `remaining_capacity` is computed at `expansion.py:222` and `:531` and **never used**. | `node_count_w1`, `leaf_count_w1`, `bifurcation_count_w1` are **genuine** metrics of learned stopping. Good. |
| **Root degree** *k* | **Hard-enforced.** `num_root_children` is copied from each GT graph and the root spawns exactly *k* children (`expansion.py:243-248`). | Root branching is **free**. Any credit for matching soma/base fan-out is unearned. `strahler` and `partition_asymmetry` are partially informed by it. |
| **TMD vector** | Given, when `tmd_hidden_dim > 0` (`path`,`height`,`radial_root` × 16²). | The eval TMD embedding uses `radial_root` — **which is inside the conditioning set**. `mmd_tmd` therefore partly scores reproduction of the model's own input. See §6.3. |
| **Cell class / genus** | Given when `class_hidden_dim > 0`. | Fine; per-class metrics are gated separately. |
| **Hard size cap** | `max_tree_size: 400`. | Neuron GT reaches 537 nodes (`docs/NEURON_DATASET_STATS.md`), so the generator **cannot** produce the upper tail. `node_count_w1` has a structural floor on that dataset. |

---

## 3. The marginal metrics, one by one

All marginals are **Wasserstein-1** (`scipy.stats.wasserstein_distance`) between a pooled
generated array and a pooled GT array. Two pooling regimes:

- **Pooled (per-element)**: every branch / node / bar from every tree goes into one big array.
  Large trees contribute proportionally more elements.
- **Per-tree (one scalar per tree)**: the distribution *over trees*.

`_w1` (`dist_metrics.py:108`) **drops non-finite entries on both sides** and returns `nan` if
either side ends up empty. This is the single most important shared behaviour — see §7.2.

### 3.1 Pooled features

| Key | Plain-English meaning | Why it matters | Computed by |
|---|---|---|---|
| `branch_length_w1` | How long is a typical segment between two branch points? | The fundamental length scale of the arbor. Wrong here = wrong metabolic cost, wrong cable properties, visibly wrong drawing. | `branch_length_values` — Euclidean length of every edge. |
| `bifurcation_angle_w1` | How wide does a branch fork open? (degrees) | The classic dendritic signature; species/type-discriminative, and set by growth-cone mechanics. Too narrow = "broom", too wide = "starburst". | `bifurcation_angle_values` — all **pairwise** angles between (child − parent) vectors at every node with ≥2 children. |
| `tmd_barlen_w1` | How long-lived are the topological branches, in the persistence sense? | A bar's length = how far a sub-branch extends before it merges into a longer one. Long bars = a few dominant arms; many short bars = a bushy skirt. | `_tmd_bar_lengths` → `compute_tmd_barcode_diagram(G, normalize_mode="none")` |
| `path_to_root_w1` | Distance from soma/base to each node **along the cable**. | Sets signal attenuation and conduction delay in neurons; sets hydraulic path length in trees. | `path_length_to_root_values` (Dijkstra, Euclidean edge weights) |
| `radial_to_root_w1` | Straight-line distance from soma/base to each node. | Where the arbor's mass sits in space — the "reach" profile. | `radial_distance_to_root_values` |
| `contraction_w1` | For each leaf: straight-line / along-cable distance from the root. ∈ (0,1]. | 1 = the branch shoots straight out; 0.5 = it meanders twice as far as it needed to. Tortuosity, i.e. how "wiggly" growth was. | `contraction_ratio_values` |
| `branch_order_w1` | How many forks lie between the root and each node — **on our data, exactly the node-depth distribution** (see below). | The topological depth profile. Measured: it is the **only** metric in the entire suite that detects leaf pruning on trees (z = 3.3 at 15%, vs ≤ 0.7 for everything else and 0.2 for `mmd_morpho` v1), and it scores exactly 0.0 on every geometric defect — a perfectly orthogonal topology probe. | `branch_order_values` |

**Assumptions and failure modes in this block:**

- **`bifurcation_angle` weights multifurcations quadratically.** A node with *k* children
  contributes *k(k−1)/2* angles. A trifurcating soma contributes 3 samples, a bifurcation 1.
  Datasets are binarized away from the root, but the **root keeps its true degree** (up to 23
  for neurons), so root fan-out dominates this pool disproportionately — and root degree is
  *given* to the sampler (§2). Consider excluding the root or capping its contribution.
- **`tmd_barlen` silently uses a different filtration from every other TMD metric.**
  `_tmd_bar_lengths` calls `compute_tmd_barcode_diagram(G, normalize_mode="none")` and takes
  the **default `filtration="path"`** — it ignores `cfg.validation.tmd_eval_filtration`
  (`radial_root`). So `tmd_barlen_w1` is in raw *path-length* units while `mmd_tmd` is in
  normalized *radial* space. Verified: 65 bars, range [0.027, 4.479] raw units. The name gives
  no hint. Not wrong, but a documentation/naming trap.

  > **✅ SHIPPED.** `_tmd_bar_lengths` now takes `filtration` and `uhat`, threaded from
  > `cfg.validation.tmd_eval_filtration` via a closure in `pooled_features` and a new
  > `tmd_filtration` argument to `compute_distribution_metrics`. One source of truth, so the
  > pooled and joint TMD metrics cannot drift apart again. Units changed from raw path-length
  > to raw radial distance; `tmd_barlen_w1` is not comparable across that boundary. Pinned by
  > `test_tmd_barlen_follows_the_configured_filtration` — the behaviour previously had **no**
  > test coverage at all.
- **`tmd_barlen` is wrapped in a bare `except: return empty`** (`dist_metrics.py:159-161`). If
  every generated graph fails `assert_rooted_tree_graph`, this reports `nan` with no counter.
- **`contraction` is not L-Measure contraction.** L-Measure defines Contraction *per branch
  segment* — chord ÷ arc-length of each unbranched cable run between two branch points, typically
  ~0.85–0.95 in reconstructed neurons. That quantity is **identically 1.0 on our data**: degree-2
  chains are collapsed in preprocessing, so every branch is a single straight edge, and
  `_edge_length` (`structural_metrics.py:210-213`) is the Euclidean chord between its endpoints —
  chord ÷ arc = 1 exactly, for every branch of every graph, real or generated. It would have zero
  variance and report W1 ≡ 0 no matter what the model did. So the code measures something else
  (`structural_metrics.py:246-274`): per **leaf**, `radial(root→leaf) / path(root→leaf)` — the
  straightness of the *whole route* from root to tip, accumulated over many segments and forks.
  Measured on the 337 GT val trees: **0.852 ± 0.091**, range [0.468, 0.993]. Three consequences:

  - **Not comparable to published numbers.** 0.85 here and 0.85 in an L-Measure table are
    different quantities, and the discrepancy is not even a fixed offset — route straightness
    falls with tree depth, segment contraction does not.
  - **Mostly redundant.** Numerator and denominator are already separate metrics in this same
    block. Regressing per-tree `mean_contraction` on `(mean_path_to_root, mean_radial_to_root)`
    over the 337 GT trees gives **R² = 0.774** (Pearson r = +0.687 and +0.625 respectively) — yet
    it consumes a full unit of kernel weight in the morpho vector (§4.2).
  - **It cannot localise.** A route average cannot distinguish a kinked trunk from a meandering
    twig, and it samples tips only.

  What it *does* buy is the ~23% of variance its parents cannot give: the ratio is
  **scale-invariant**, where `path_to_root` and `radial_to_root` both scale with the tree. On this
  corpus it is only weakly confounded with size and depth (r = −0.151 with node count, −0.163 with
  max branch order). **Recommended fix: rename** to `root_tip_straightness` /
  `tortuosity_root_to_tip`, and never table it beside literature contraction values. Note also
  that per-segment tortuosity is *unrecoverable* from these corpora — degree-2 collapse destroyed
  it in the **training data** — so "does the generator produce realistic cable tortuosity?" is out
  of scope for this representation rather than a gap in the metric suite.
- **`branch_order` treats the root as never-branching.** Confirmed empirically: with a root of
  degree 2 *or* degree 3, all root children get order 0; the increment happens only at non-root
  nodes of undirected degree ≥3. Self-consistent, but it means the metric is **blind to root
  multifurcation**, and offset by one from the common "root children are order 1" convention.
- **On our data, `branch_order` IS node depth — not a distinct quantity.** The degree-≥3 rule
  only skips a node when it has exactly one child, and our datasets are strictly binary away
  from the root with **zero degree-2 non-root nodes**, so every non-root internal node
  increments. Verified: `branch_order == hop_depth − 1` for **300/300 graphs** on trees d10,
  trees d20 and neurons; pinned by `test_branch_order_equals_hop_depth_on_binary_tree`. Read
  `branch_order_w1` as "the node-depth distribution" and it will not mislead you. The general
  name is retained because the function computes the general quantity — on a graph that *did*
  keep its degree-2 chain nodes the two would diverge.

  This is also why `max_branch_order` (the per-tree max) is in `MORPHO_KEYS` v2 in place of
  `strahler`: it is the depth axis, and it is what closed the tree-topology blind spot. Its
  one caveat is that on the depth-capped tree sets it is concentrated near the cap (97% of d10
  trees in {9,10,11}), so there it partly measures "did the generator reach the ceiling".
- **Pooling is size-weighted.** A 286-node tree contributes 4× the branch-length samples of a
  69-node tree. If the generator's size distribution is skewed, these pooled marginals inherit
  the skew even when per-tree geometry is perfect. This is a *coupling*, not a bug — but it
  means `branch_length_w1` is not independent of `node_count_w1`.

### 3.2 Per-tree scalar features

| Key | Plain-English meaning | Why it matters |
|---|---|---|
| `node_count_w1` / `leaf_count_w1` / `bifurcation_count_w1` | How big is the tree; how many tips; how many forks. | Total complexity / wiring budget. **Perfectly collinear on binary trees** (leaves = bifurcations + 1, nodes = 2·leaves − 1). Empirically r = **1.000**. |
| `axial_extent_w1` | Extent along the model's symmetry axis `uhat` (max − min of `pos·uhat`). | The "height" of a tree; the apical reach of a pyramidal neuron. |
| `radial_span_w1` | Max pairwise distance **in the plane ⟂ `uhat`**. | The lateral spread / canopy width. Rotation-invariant about `uhat` by construction — correct for an SO(2)-equivariant model. |
| `total_extent_w1` | 3D diameter (max pairwise distance). | Overall reach. |
| `strahler_w1` | Horton–Strahler order of the whole tree. | A single integer for hierarchical complexity. Grows ~log₂(leaves) for balanced trees; stays low for path-like ones. Standard in both river networks and neuroanatomy. |
| `partition_asymmetry_w1` | Van Pelt asymmetry: mean over forks of \|r−s\|/(r+s−2) on subtree leaf counts. ∈[0,1]. | **The** topological shape number. 0 = every fork splits evenly (a balanced bush); 1 = every fork sheds a single tip off a main trunk (a caterpillar). Distinguishes real dendrites (~0.4–0.6) from both balanced and degenerate generators. |
| `sholl_peak_w1` | Max number of branches crossing any concentric shell around the root. | Peak arbor density — the classic Sholl readout used across neuroscience. |
| `sholl_critical_radius_w1` | Radius at which that peak occurs (÷ shell max). | *Where* the arbor is densest — proximal-heavy vs distal-heavy. |
| `sholl_auc_w1` | Area under the Sholl profile. | Total "amount of arbor", integrated over radius. |

**Assumptions and failure modes in this block:**

- **`_size_extent` costs O(N²) memory** (`scipy.spatial.distance.pdist` on all node positions,
  twice, at `dist_metrics.py:204-205`). At N ≤ 537 (neurons) / 286 (trees) this is trivially
  fine; a dataset with 10k-node trees would blow up. The raw QSM trees are 10,496 nodes
  on average before skeletonisation — do not point this at unreduced data.
- **Sholl shells are fit on GT and shared** (`_sholl_radii_from_graphs`, 32 shells over
  (0, max GT radial extent]). This is the right call for comparability, but it **changes what
  `sholl_critical_radius` means**, and the docstring is now wrong. It says "normalised by max
  radial extent"; with shared radii the divisor is the *GT set's* max, not the tree's own.
  Measured on one real tree scaled ×1 / ×1.5 / ×3:

  | | shared shells (what runs) | own shells (what the docstring describes) |
  |---|---|---|
  | ×1.0 | peak 7, crit_r **0.094**, auc 10.0 | peak 14, crit_r 0.656, auc 13.8 |
  | ×1.5 | peak 13, crit_r **0.156**, auc 18.8 | peak 14, crit_r 0.656, auc 20.6 |
  | ×3.0 | peak 13, crit_r **0.313**, auc 39.6 | peak 14, crit_r 0.656, auc 41.3 |

  So the live `sholl_critical_radius` is effectively **another size feature** (it tracks
  absolute scale, correlation 0.90 with `mean_radial_to_root`), not a shape feature. And a
  generated tree larger than the GT max has its outer arbor fall **outside all shells** —
  truncated, silently.

  The mechanism is one line — `structural_metrics.py:435`, `rmax = float(r.max())`: the max of the
  radii array that was *passed in*, not of the tree. With `radii=None` that is the tree's own
  extent (a genuine fraction, scale-invariant); with shared GT radii it is a **constant**, and
  dividing by a constant is not normalisation — `crit_r` becomes the absolute peak radius in fixed
  units. The shared-shell column above is `3/32`, `5/32`, `10/32`, which exposes a second
  consequence: **`crit_r` can only take 32 distinct values**, multiples of 0.03125 (measured shell
  spacing on d10: 0.8008 units over (0, 25.627]). It is a *coarsely quantised* size proxy, so W1
  on it is stair-stepped.

  The truncation is **asymmetric between the two sides being compared**: the grid is fit on GT, so
  no GT tree is ever truncated — only generated trees, and only those exceeding the largest GT
  extent. Note the direction of the resulting bias: the excess outer arbor *disappears* rather
  than producing a profile mismatch, so `sholl_auc` / `sholl_peak` land closer to GT than the truth
  warrants. The failure flatters the model, which is the dangerous direction.

  **Fix (one line, and better than either column of the table).** Shells and divisor are
  independent choices. Keep the shells shared — that genuinely is right for comparability — and
  divide by the *tree's own* max radial extent:

  ```python
  own_rmax = float(radial_distance_to_root_values(G, root=root).max())
  out["sholl_critical_radius"] = float(r[peak_idx] / own_rmax) if own_rmax > 0 else float("nan")
  ```

  Profiles stay directly comparable, `crit_r` becomes scale-invariant again and decorrelates from
  `mean_radial_to_root`, it leaves the 32-value lattice, and the docstring at `:422` becomes true
  instead of needing a rewrite. Separately, log `n_edges_beyond_last_shell` to turn the silent
  truncation into an observable.

  > **✅ SHIPPED (2026-08-08), and it went one step further.** The divisor fix above is in
  > `sholl_summary` verbatim (with an empty-array guard — `radial_distance_to_root_values`
  > returns `[]` for `N < 2`, so a bare `.max()` raises). **In addition, the live calls now pass
  > `radii=None`**, i.e. per-tree shells, because the shared grid turned out to be generating
  > false signal on its own: crossing counts are inherently scale-invariant, but a fixed grid
  > breaks that by undersampling small trees. On one tree scaled ×1/×1.5/×3 the shared grid gives
  > `sholl_peak` 7/13/13 where per-tree shells give **14/14/14** — so a small generated tree and a
  > large GT tree *of identical shape* scored differently. Nothing in the codebase compares
  > profiles pointwise, so the shared grid was buying no comparability in exchange. Per-tree
  > shells also make the truncation-of-oversized-generated-trees bug (and the
  > `n_edges_beyond_last_shell` observable it needed) structurally impossible.
  > `_sholl_radii_from_graphs` and the `sholl_radii` cache entry are **kept, unused**, for a
  > future mean-Sholl-profile plot, which is the one thing that does need a common grid.
  >
  > Measured payoff beyond the intended one: the fix cut `sholl_critical_radius`'s redundancy on
  > trees from **R² 0.858 → 0.567**, promoting it from "mostly explained by the other features"
  > to genuinely independent — which is what earned it a place in the 9-D vector. Pre-fix it took
  > only **4 distinct values across 400 neurons**; post-fix it is continuous.
- **Sholl crossing counts are endpoint-based, not geometric.** `sholl_intersection_profile`
  counts an edge as crossing radius *r* iff `min(d_u,d_v) < r ≤ max(d_u,d_v)`. A straight segment
  whose closest approach to the root is nearer than both endpoints pierces every sphere in
  `(d_min, min(d_u,d_v)]` **twice** and is counted zero times. This ought to matter *more* here
  than in the literature, because degree-2 collapse turns each branch into one long chord spanning
  many original cable steps, and long chords can dip inward in a way short steps cannot. Measured
  on the 337 GT val trees:

  | | count |
  |---|---|
  | edges whose segment passes closer to the root than both endpoints | 935 / 28,061 = **3.3%** |
  | shell crossings counted | 9,686 |
  | shell crossings geometrically missed | **16** (0.16% of true crossings) |

  So the approximation is real in principle and near-free in practice — the inward dips are shallow
  relative to the 0.80-unit shell spacing and rarely span a boundary. Consistent between gen and
  GT, so the comparison is fair; it is just not literature-exact Sholl. The residual risk is that
  the error rate is **data-dependent, not a fixed offset**: a generator producing inward-curling
  arbors would raise its own miss rate above GT's 0.16%, and the metric would silently
  under-report exactly that error mode.
- **`sholl_peak` is integer-valued but is *not* in `_DISCRETE_PERTREE`** (which contains only
  `node_count, leaf_count, bifurcation_count, strahler`). The two KS gates are
  `dist_metrics.py:454` (`_DISCRETE_POOLED = {"branch_order"}`) and `:471` (`_DISCRETE_PERTREE`).
  Every *other* integer-valued feature in the suite is listed, including the pooled one — which
  shows this class of problem was being tracked. `sholl_peak` is a max over integer crossing counts
  and is simply missing, so it reads as an oversight rather than a judgement call: a safe one-token
  fix (add `"sholl_peak"` to `_DISCRETE_PERTREE`).

  Why the ties matter: the two-sample KS null distribution assumes continuous data. With heavy ties
  the ECDFs advance in large jumps, the statistic's null distribution is stochastically smaller,
  and the test turns **conservative** — non-significant results for differences that are real.
  Verified: 20 trees → only 10 distinct peak levels. Currently inert (`enable_ks: False`
  everywhere), but it fails quietly rather than loudly when flipped.

  `sholl_critical_radius` is now confined to a 32-value lattice too (see the shared-shells bullet
  above), so it belongs in the discrete set as well — *unless* the divisor fix lands, which restores
  it to a continuous quantity and makes the question moot. Another reason to prefer the fix over
  documenting the current behaviour.

  > **✅ Partly resolved.** The divisor fix landed, so `sholl_critical_radius` is continuous again
  > and the lattice question is moot as predicted. `sholl_peak` is **still missing from
  > `_DISCRETE_PERTREE`** — the one-token fix was not taken, because `sholl_peak` was demoted to
  > the `full` reporting tier (it detects nothing: max z **0.1 / 0.6** across every defect class)
  > and `enable_ks: False` everywhere, so the trap is now two switches deep rather than one. It
  > remains a genuine latent oversight worth closing.
- **`strahler` and `partition_asymmetry` are partly *given*** via the enforced root degree.

### 3.3 The two normalized aggregates

```python
pooled_norms.append(w1 / np.nanstd(gt_pool))     # dist_metrics.py:456-458
pertree_norms.append(w1 / np.nanstd(gt_vals))    # dist_metrics.py:473-475
metrics["w1_pooled_mean_normalized"]  = mean(pooled_norms)
metrics["w1_pertree_mean_normalized"] = mean(pertree_norms)
```

**What it says:** "on average, how many GT standard deviations apart are the generated and
real distributions, across my feature battery?" This is the right idea — it makes µm-scale and
degree-scale and count-scale features commensurable so one can be averaged with another.

**Failure modes:**

1. **Unweighted mean over a redundant battery.** `w1_pertree_mean_normalized` averages 11
   features of which `node_count`, `leaf_count`, `bifurcation_count` are the *same number*
   (r = 1.000) and `axial_extent`/`total_extent` are r = 0.999. So "size" enters the average
   with weight 3/11 and "reach" with weight ~3/11, while `partition_asymmetry` — arguably the
   most informative single topological number — gets 1/11.
2. **Silently variable membership.** A feature whose GT std is ≤1e-12, or whose W1 is `nan`,
   is dropped from the mean *without changing the key*. The aggregate can be an average over
   11 features at one step and 9 at the next, and nothing in the log says so.
3. **Not a metric in the mathematical sense** — it is a mean of ratios and has no
   distributional interpretation. Fine as a monitor, not reportable as a headline.

### 3.4 Tree-edit distance (`ged_enabled`, currently **off**)

`graph_edit_distance_topology` uses `zss` (ordered TED) with unit insert/delete and zero
substitution cost, canonicalising child order by recursive subtree signature to approximate
*unordered* TED.

- **Pairing is by index**, justified because `gen[i]` targets `gt[i]`'s size. With
  `use_size_ratio: False` that justification is weaker than the comment claims — the sizes are
  only softly matched, so a size mismatch shows up as edit distance and is double-counted
  against `node_count_w1`.
- **Cost, measured:** ~22 s for the 64-pair cap at N=100 (≈0.35 s/pair) vs **1.8 s for the
  entire rest of the suite** at N=300. Keeping it off is correct.
- **`ged_timeout` is a no-op.** Documented in the code, but the config still advertises
  `ged_timeout: 5.0` as if it did something.
- **Bug (minor):** `tree_edit_skipped_frac` divides by `n_pairs` = `min(len(gen), len(gt))`,
  not by the number of pairs actually *considered* before the `GED_MAX_PAIRS` break
  (`dist_metrics.py:548-549`). With 337 eval graphs and a 64-pair cap, the reported skipped
  fraction is understated by ~5×.
- **Recursion.** `_build_zss_tree`'s `_signature`/`_build` are recursive; safe at
  `GED_MAX_NODES = 200`, would break on a deep path tree beyond ~1000 nodes.

---

## 4. The morphometric vector (`assemble_morpho_vector`) — the core of the joint metrics

This is the part the question asked about most directly, so it gets the most space.

### 4.1 Construction

> **⚠️ This section describes `MORPHO_VERSION 1`.** The live vector is the 9-D v2 listed in the
> status banner at the top. The v1 analysis below is kept because it is *why* v2 exists, and
> because §8's floor and sensitivity tables were measured on v1.

A fixed-order 16-vector per tree (`MORPHO_KEYS`, `dist_metrics.py:77-94`) — **v1, superseded**:

```
[0] node_count            [8]  mean_branch_length
[1] leaf_count            [9]  mean_bifurcation_angle
[2] bifurcation_count     [10] mean_path_to_root
[3] axial_extent          [11] mean_radial_to_root
[4] radial_span           [12] mean_contraction
[5] total_extent          [13] sholl_peak
[6] strahler              [14] sholl_critical_radius
[7] partition_asymmetry   [15] sholl_auc
```

Then (`build_gt_cache`, `dist_metrics.py:313`):

1. GT mean/std computed with `nanmean`/`nanstd`; std < 1e-8 → forced to 1.
2. `standardize_vectors`: `z = (v − µ_GT)/(σ_GT + 1e-8)`, then `nan_to_num(..., nan=0.0)`.
3. `morpho_sigma = median_heuristic_bandwidth(morpho_z)` — median pairwise distance over the
   GT set, subsampled to 512 rows, computed **once** and reused across all training steps.
4. MMD² (unbiased, Gaussian RBF) + Density/Coverage (Naeem et al. 2020, k = 5).

**The good decisions here, stated plainly:** fitting µ/σ and σ_kernel on the *fixed GT set* and
reusing them is exactly right — a per-step bandwidth would make the MMD trajectory
non-comparable across checkpoints, which is the single most common way this class of metric is
gotten wrong. The near-zero-variance guard is right. Returning the unbiased MMD *unclipped*
(so it can go slightly negative when the sets match) is right, and is what makes comparison
against a real-vs-real floor meaningful. The extents are decomposed in the model's own SO(2)
frame rather than world x/y/z, which is both semantically correct and azimuth-invariant.

### 4.2 The redundancy problem (measured)

On 200 real `trees_genus_d10/val` trees, the standardized 16-d vector has:

```
effective rank (participation ratio) = 7.32 / 16
correlation eigenvalues = [6.58 5.22 1.14 0.86 0.67 0.43 0.41 0.24 0.17 0.14 0.09 0.06 0 0 0 0]
top-1 / top-3 share of variance = 0.41 / 0.81
```

**Four eigenvalues are exactly zero** — the representation is at most rank 12, and behaves like
rank ~7. The |r| > 0.8 pairs:

| pair | r |
|---|---|
| node_count ↔ leaf_count | **+1.000** |
| node_count ↔ bifurcation_count | **+1.000** |
| leaf_count ↔ bifurcation_count | **+1.000** |
| node/leaf/bifurcation_count ↔ strahler | +0.805 |
| axial_extent ↔ total_extent | **+0.999** |
| mean_path_to_root ↔ mean_radial_to_root | **+0.996** |
| axial_extent / total_extent ↔ mean_path_to_root / mean_radial_to_root | +0.90 |
| mean_path_to_root / mean_radial_to_root ↔ sholl_critical_radius | +0.90 |

(`docs/MORPHO_MANIFOLD_ANALYSIS.md` reports the same structure on neurons: effective rank 8.3,
counts at r = 1.00. This is a property of the representation, not of one dataset.)

Two further redundancies are by *construction* rather than merely empirical, and both are cheaper
to fix at the source than to let whitening absorb: `mean_contraction` is a ratio of two features
already in the vector (R² = 0.774 on `mean_path_to_root` + `mean_radial_to_root`, §3.1), and
`sholl_critical_radius` was turned into a scale feature by the shared-shell divisor (§3.2).

**Why this matters for the metric, not just for the manifold.** The MMD kernel is
`exp(−‖z_a − z_b‖²/2σ²)` on the **raw z-scored** vector — z-scoring equalises per-feature
*variance* but does nothing about *correlation*. A group of *m* near-identical features
contributes ~*m*× to the squared distance. Concretely the kernel's implicit weighting is
roughly:

- **size** (node/leaf/bifurcation count, + strahler at 0.8) ≈ **3.6 units of weight**
- **reach** (axial, total, path, radial, sholl_critical_radius) ≈ **4–5 units**
- **shape** (`partition_asymmetry`, `mean_bifurcation_angle`, `mean_contraction`) ≈ **1 unit each**

So `mmd_morpho` is, to first order, a metric on *(size, reach)* with topological shape as a
rounding error — despite topology being the thing that distinguishes a dendrite from a bush.
This is consistent with the measured sensitivity table in §8.2: `mmd_morpho` moves 42× above
its floor for a pure 25% rescale, but `partition_asymmetry_w1` barely moves for a 30% leaf
prune.

**Fix (recommended, cheap):** whiten instead of z-score. Fit ZCA/PCA-whitening on the GT
morpho matrix in `build_gt_cache` and apply the same transform to gen. That equalises the
*directions*, not just the axes, and makes the kernel a genuine Mahalanobis kernel. If a full
whitening is unattractive (it makes individual dimensions uninterpretable), the minimal version
is to **drop the exact duplicates**: `leaf_count`, `bifurcation_count`, `total_extent`,
`mean_path_to_root` carry no information the retained features lack, at r ≥ 0.996. That alone
takes the vector from 16 nominal / 7.3 effective to 12 nominal / ~7 effective, with the weight
distortion roughly halved.

### 4.3 The nan-imputation trap (confirmed, and this one is dangerous)

`standardize_vectors` maps non-finite entries to **0 in z-space = exactly the GT mean**.
Measured on synthetic degenerate graphs against the real GT cache:

| Degenerate generated graph | features that go nan | what they become after standardization |
|---|---|---|
| Path graph (no bifurcation at all) | `partition_asymmetry`, `mean_bifurcation_angle` | **the GT mean** (z = 0.0) |
| Single node | 9 of 16 features | **the GT mean** (z = 0.0) |
| All nodes collapsed to the origin | `mean_bifurcation_angle`, `mean_contraction`, all 3 Sholl | **the GT mean** (z = 0.0) |

A generated tree with *zero forks* is scored as having **perfectly average branching
asymmetry and a perfectly average fork angle**. The imputation converts "catastrophically
degenerate" into "unremarkably typical" in exactly the dimensions that would have caught it.

The marginal path has the mirror-image problem: `_w1` **drops** non-finite entries, so if 40%
of generated trees have undefined `partition_asymmetry`, `partition_asymmetry_w1` is computed
on the surviving 60% and looks fine. **No nan-fraction is logged anywhere in
`dist_metrics.py`** — unlike `tmd_conditional_eval.py`, which does log `pd_nan_frac_*` and
`n_pairs_skipped`.

**Fix:** emit `morpho_nan_frac_<key>` (or at minimum a single `morpho_nan_frac_any`) and an
`n_gen_degenerate` counter. Consider imputing to a *penalising* value (e.g. the GT 99th
percentile of the deviation) rather than to the mean, or excluding degenerate trees from the
joint metrics and reporting their count as its own metric.

### 4.4 Heavy tails are not addressed

Measured skew on the 16 features: `node_count` / `leaf_count` / `bifurcation_count` +1.25,
`sholl_peak` +1.29, `mean_branch_length` +1.35, `mean_contraction` −1.04. Z-scoring does not
symmetrise. The median-heuristic bandwidth is set by the bulk, so tail mismatches are
compressed.

*However* — I checked whether this actually saturates the kernel, and it does **not**:

```
morpho_z     sigma=5.070  d/sigma p05=0.48 p50=1.00 p95=1.77  kernel p50=0.607  frac k<0.01: 0.000
tmd_reduced  sigma=0.871  d/sigma p05=0.37 p50=1.00 p95=2.18  kernel p50=0.607  frac k<0.01: 0.000
```

No pair falls below k = 0.01 in either space. The median heuristic is doing its job. A
`log1p` transform on the count/extent features would still improve tail sensitivity, but this
is an enhancement, not a defect.

### 4.5 MMD and Density/Coverage — correctness check

I verified `utils/dist_helper.py` against the reference definitions:

- **`mmd2_unbiased`** — correct. Diagonal excluded from both `Kxx` and `Kyy` with the proper
  `n(n−1)` / `m(m−1)` denominators; `Kxy` uses the full mean. Returns unclipped, as documented.
- **`density_coverage`** — correct implementation of Naeem et al. (2020). Radii are the k-th
  NN distance among *real* points (queried at k+1 to skip the self-match);
  `density = Σᵢ|{fakes in B(realᵢ, rᵢ)}| / (k·M)`; `coverage = mean(has ≥1 fake)`. `k` is
  guarded to `N−1`.
- **`median_heuristic_bandwidth`** — correct, deterministic (seeded subsample), floored at 1e-8.

**One interpretation trap, measured.** With `gen == gt` exactly, `density_morpho = 1.148`, not
1.0 — because each real point's sphere contains its own duplicate plus its k neighbours, giving
(k+1)/k inflation. And for two *disjoint halves of the real data*, density is **0.962** and
coverage **0.952**. So "density ≈ 1.0" is not the target; the target is whatever the real-vs-real
floor says. **Which is why `enable_floor: False` is the most consequential config choice in the
whole suite** (§8.1).

---

## 5. The joint metrics — `mmd_morpho` / `density_morpho` / `coverage_morpho`

**What they say in plain terms:**

- **MMD** — "if I pick a random generated tree and a random real tree, how differently do they
  *feel* overall?" It compares whole distributions including cross-feature structure, so it
  catches broken correlations that every marginal misses: e.g. a model that produces the right
  distribution of sizes and the right distribution of heights but pairs *small* trees with
  *tall* heights scores 0 on both marginals and non-zero on MMD.
- **Density** — *fidelity*. "Do my generated trees land where real trees actually live?"
  Low density = the model is emitting plausible-looking-but-off-manifold shapes.
- **Coverage** — *diversity*. "Does every region of real tree-space have at least one
  generated neighbour?" Low coverage = mode collapse; the model found three good tree types and
  is producing only those.

Density and Coverage split what a single MMD number conflates, which is exactly why both are
logged. That is a good design.

**Failure modes:**

- **The GT set is used twice** — to fit µ/σ/σ_kernel *and* as the reference sample in MMD. This
  biases MMD slightly downward for GT-like inputs. Unavoidable, and correctly handled by
  comparing to the real-vs-real floor — which is off.
- **Density/Coverage degrade in high dimension.** On the 16-d morpho space they behave well
  (§8.2). On the 64-d PCA-reduced TMD space they barely move at all (§6.2).
- **k = 5 with N = 337** — the manifold estimate is coarse. Fine as a monitor; do not report
  the absolute value.
- **Sample size is bounded by the eval set.** One generated tree per eval graph, so |X| = |Y| =
  |validation_graphs|. MMD's variance scales as 1/N; at N ≈ 337 the step-to-step noise is
  visible (the identity/floor gap in §8.1 is ~0.0006, i.e. the same order as the wiggle).

---

## 6. The TMD joint block — `mmd_tmd` / `density_tmd` / `coverage_tmd`

### 6.1 Construction

`compute_tmd_embedding(G, filtration="radial_root", n_bins=16)` → 256-d persistence image →
PCA to 64 components **fit on GT** → MMD + D/C with a GT-fit bandwidth.

The persistence image is built from the paper-style TMD barcode on the **critical tree**
(degree-2 chains contracted), with `weighting="persistence"` and `sigma=0.05` on a 16×16 grid
over `[0,1]²`. I checked the Gaussian sampling: σ/h = 0.75 gives a Poisson-summation ripple
of ~3×10⁻⁵ — **no aliasing problem**, the grid is adequate.

**What it says in plain terms:** the barcode records, for each sub-branch, how far from the
root it starts and how far it reaches before merging into a longer branch. Long bars = a few
dominant arms. Many short bars near the origin = a dense proximal skirt. The persistence image
turns that into a fixed-size picture you can average and compare. It is the single most
information-dense summary of *branching topology weighted by spatial reach* that exists for
neurons (Kanari et al.), and it is genuinely complementary to the morpho vector.

### 6.2 The problem: it barely moves (measured)

`compute_tmd_mixed` always applies `normalize_filtration_values(mode="minmax")` — the
filtration is rescaled to [0,1] **per graph**. Verified directly:

```
||e(G) − e(3·G)||        = 0.000e+00      (||e|| = 1.787)
||e(G) − e(Rz(40°)·G)||  = 0.000e+00
```

The eval TMD embedding is **exactly invariant to uniform scaling and to any rigid motion**. It
is a pure *shape* descriptor. That is defensible on its own, but combined with the PCA-on-GT
reduction it leaves very little dynamic range. From the defect sweep (§8.2):

| defect | `mmd_morpho` | `mmd_tmd` | `density_tmd` | `coverage_tmd` |
|---|---|---|---|---|
| real-vs-real floor | 0.0006 | 0.0003 | 1.0095 | 0.9762 |
| global scale ×1.25 | **0.0251** (42× floor) | 0.0003 (1× floor) | 1.0095 | 0.9762 |
| perpendicular squash ×0.5 | **0.0358** | 0.0042 (14×) | 1.1095 | 0.9702 |
| branch jitter sd = 0.3 | **0.0135** | −0.0014 (**below floor**) | 0.9631 | 0.9702 |
| prune 30% deepest leaves | **0.0107** | 0.0050 (17×) | 1.0179 | 0.9702 |

`coverage_tmd` **never leaves the 0.95–0.99 band** under any defect I could construct —
including deleting 30% of every tree's deepest leaves. In 64 dimensions with N = 168, the k-NN
hyperspheres are so large that coverage saturates. `density_tmd` is nearly as flat.

**Verdict:** `mmd_tmd` carries a real but weak signal (it is the *only* metric that responds
specifically to topology-with-reach). `density_tmd` and `coverage_tmd` are, on current dataset
sizes, effectively **constants dressed as metrics** and are actively misleading on a dashboard.

### 6.3 Two structural concerns

1. **PCA is fit on GT only.** Any generator artefact living in a direction the GT set does not
   span is projected away and is invisible to `mmd_tmd`. `_fit_pca` also does not whiten, so
   the leading GT components dominate the distance.
2. **`radial_root` is inside the model's conditioning set.** `model.tmd_filtrations =
   [path, height, radial_root]`, and `tmd_eval_filtration: radial_root`. The
   `compute_tmd_embedding` docstring explicitly claims the opposite — *"by using a filtration
   outside the model's conditioning set, avoids merely scoring reproduction of the conditioning
   input"* — but with the current config that claim is **false**. On TMD-conditioned runs,
   `mmd_tmd` partly measures how well the model echoes its own input. Either change
   `tmd_eval_filtration` to something not in `model.tmd_filtrations`, or fix the docstring and
   stop treating `mmd_tmd` as independent evidence on conditioned runs.

### 6.4 Latent bug: `uhat` is never threaded into the eval embedding

```python
def _eval_embed_fn(self):                                    # training.py:488
    return lambda G: compute_tmd_embedding(G, filtration=filtration, n_bins=tmd_bins)
```

`compute_tmd_embedding`'s `uhat` defaults to `(0,0,1)`. For `radial_root` and `path` this is
harmless (both are axis-agnostic — verified, ‖Δ‖ = 0.0000). But for `height` or `rho` the
embedding **would silently use the z-axis** even on neurons, where `so2_axis = [0,1,0]`.
Measured:

```
filtration=height  ||e(uhat=z) − e(uhat=y)|| = 1.4326
filtration=rho     ||e(uhat=z) − e(uhat=y)|| = 1.1659
```

Currently latent (no config sets those), but it is a one-line trap waiting for whoever tries
`tmd_eval_filtration: height`. `tmd_conditional_eval` does thread `uhat` correctly, which makes
the inconsistency easy to miss. **Fix:** pass `uhat=self.model_uhat` in `_eval_embed_fn`, or
raise if the filtration is axis-dependent and `so2_axis ≠ z`.

### 6.5 `tmd_eff_rank` and `mmd_bandwidth_*` are constants

All three are read straight out of the GT cache and re-logged unchanged at every validation.
They are useful once as sanity checks; as wandb time series they are three flat lines.

---

## 7. Cross-cutting assumptions

### 7.1 The suite is blind to a flipped tree — confirmed, and this is the headline finding

I transformed the entire generated set and recomputed all 29 live metrics against an untouched
GT cache:

| Global transform applied to every generated tree | metrics that changed |
|---|---|
| **Reflection along `uhat`** (every tree upside-down) | **0 of 29** |
| Rotation 40° about `uhat` | 1 of 29 (`density_morpho` 1.1427 → 1.1280 — float jitter only) |
| Tilt 40° off `uhat` | 7 of 29 (`axial_extent_w1` 0 → 3.00, `mmd_morpho` −0.006 → 0.235, …) |

Rotation-invariance about `uhat` is **correct and intended** — the model is SO(2)-equivariant,
so any azimuth is equally valid. Off-axis tilt is correctly detected. But **reflection along
`uhat` changes nothing**, because every geometric feature in the suite is sign-blind:

- `axial_extent` = `s.max() − s.min()` — a range, so ±s are identical
- `radial_span`, `total_extent` — pairwise distances
- `path_to_root`, `radial_to_root`, all Sholl — unsigned distances from the root
- `bifurcation_angle`, `contraction` — unsigned
- `strahler`, `partition_asymmetry`, `branch_order` — purely topological
- the `radial_root` TMD image — invariant to *any* isometry, reflections included

A generator that produces an anatomically perfect pyramidal neuron **pointing the wrong way**
scores an identical, perfect result on every live metric. This connects directly to
`docs/APICAL_AXIAL_ERROR_MODE.md` (multiple weak axial arms instead of one dominant apical) —
that error mode was found by a dedicated offline probe precisely because the in-loop suite
cannot see polarity.

**Fix (cheap and high value):** add signed axial features. Concretely:
`signed_axial_reach = max(s) − s_root` and `min(s) − s_root` as separate per-tree scalars, plus
`axial_skew = mean((s − s_root)³)/std³` or simply the *fraction of nodes with s > s_root*. Any
of these breaks the reflection symmetry, costs microseconds, and slots straight into
`MORPHO_KEYS`. This is the single highest-value change in this document.

### 7.2 nan is silence, not signal

Consolidating §3 and §4.3: three different nan policies coexist, none of them logged.

| Path | Policy | Consequence |
|---|---|---|
| `_w1` / `_ks` | **drop** non-finite on both sides | Sample sizes shrink invisibly; a set where half the trees are degenerate reports a clean W1 on the good half. |
| `standardize_vectors` | **impute to GT mean** | Degenerate trees are scored as perfectly typical in exactly the dimensions that would flag them. |
| `_embed_matrix` | **drop the row entirely** on exception or non-finite | A generated tree that fails `assert_rooted_tree_graph` vanishes from `mmd_tmd` with no counter. |
| `_tmd_bar_lengths`, `_bifurcation_angles` | bare `except` → empty array | Total failure across the set reports `nan`, not an error. |

I did check how often the `_embed_matrix` drop actually fires: on 40 synthetic path graphs +
110 real trees it kept **150/150** — the TMD embedding does not raise for degenerate-but-valid
trees, it returns an all-zero image, which *is* correctly penalised by MMD. So this path is
low-probability. The imputation and W1-drop paths are the real risks.

### 7.3 `G.graph["root"] = 0` on generated graphs

`training.py:788-790` assigns root index 0 to every generated graph, justified by "roots get the
smallest global indices". `Expansion.sample_graphs` never sets `G.graph["root"]` (verified — no
assignment anywhere in `graph_generation/method/expansion.py`), so this fallback **always
fires**: it is not a fallback, it is the sole definition of the root on the generated side.
Every root-anchored metric in the suite — bifurcation angles, path/
radial to root, contraction, branch order, Strahler, asymmetry, all Sholl, the entire TMD block
— depends on this being true. It is asserted nowhere. Given the known ordinal/apical
sensitivities in this codebase, a one-line `assert` (e.g. that node 0 has no parent in the
generated adjacency, or that it is the unique degree-`k` node matching `num_root_children`)
would be cheap insurance.

### 7.4 Caches keyed by `id(list)`

`_gt_cache_for`, `_floor_for`, `_tf_batches_for`, `_tmd_cond_gt_cache_for` all key on
`id(eval_graphs)`. The cache dict does not hold a reference to the list, so if
`self.validation_graphs` is ever rebound and the old list is collected, a new list can reuse the
same `id` and silently receive a stale cache. Currently safe (the list lives for the whole run),
but it is a latent footgun; keying on a content hash or on the attribute name would be safer.

### 7.5 Redundant computation (measured)

Instrumenting one `compute_distribution_metrics(20 gen, 20 gt)` call — 40 graph-visits:

| primitive | calls | per graph |
|---|---|---|
| `_root_tree` | 300 | **7.5×** |
| `path_length_to_root_values` (Dijkstra) | 60 | 1.5× |
| `radial_distance_to_root_values` | 60 | 1.5× |
| `contraction_ratio_values` (another Dijkstra) | 60 | 1.5× |
| `sholl_intersection_profile` | 60 | 1.5× |
| `branch_length_values` | 60 | 1.5× |

The 1.5× factor is gen-side duplication between the marginal pass (`_per_tree_scalars`) and the
joint pass (`assemble_morpho_vector`), which recompute the identical quantities. `_root_tree` is
rebuilt 7.5× per graph. Also, `path_length_to_root_values` uses `nx.single_source_dijkstra` on a
*tree*, where a single BFS accumulation would do.

At the current scale this is irrelevant — the whole suite is **1.8 s at N = 300**. Fixing it is
a clarity win, not a performance need. Worth doing only if the eval set grows 10×.

---

## 8. Empirical calibration

> **⚠️ Measured on `MORPHO_VERSION 1`** (the 16-D vector, shared Sholl shells). The floors in
> §8.1 for the *marginals* are unaffected by the v2 change — `node_count_w1`,
> `branch_length_w1`, `bifurcation_angle_w1` etc. are computed identically. The
> `mmd_morpho`/`density_morpho`/`coverage_morpho` rows and the §8.2 `mmd_morpho` column are
> **v1 numbers**; the v2 equivalents are in the status banner at the top. `sholl_*` rows also
> shifted with the per-tree-shell change.

All numbers below: 168 real `trees_genus_d10/val` trees as GT, a **disjoint** 168 real trees as
the "generated" set, GT cache fit on the first half. Reproduce with the script in §11.

### 8.1 The floor is large, metric-specific, and currently not logged

| metric | `gen == gt` (identity) | **real vs real** (disjoint halves) |
|---|---|---|
| `mmd_morpho` | −0.0049 | **+0.0006** |
| `density_morpho` | 1.1476 | **0.9619** |
| `coverage_morpho` | 1.0000 | **0.9524** |
| `mmd_tmd` | −0.0051 | **+0.0003** |
| `density_tmd` / `coverage_tmd` | 1.1476 / 1.0000 | **1.0095 / 0.9762** |
| `w1_pooled_mean_normalized` | 0.0000 | **0.0651** |
| `w1_pertree_mean_normalized` | 0.0000 | **0.1470** |
| `node_count_w1` | 0.0 | **10.73 nodes** |
| `branch_length_w1` | 0.0 | **0.0199** |
| `bifurcation_angle_w1` | 0.0 | **1.56°** |
| `partition_asymmetry_w1` | 0.0 | **0.0094** |

Read `node_count_w1 = 10.73` carefully: **two disjoint halves of the same real dataset differ
by 10.7 nodes in W1.** A run reporting `node_count_w1 = 12` is essentially at the floor. Without
the floor line on the dashboard, that number reads as a serious failure. Every metric in this
suite has this property to some degree, and none of the floors are currently visible because
**`enable_floor: False` in every config**.

The machinery already exists and is correct (`_floor_for`, `training.py:508`, using a
size-matched train subset against the eval set with the same GT cache). It costs one extra
`compute_distribution_metrics` call, **cached for the whole run** — measured at 1.8 s once, at
N = 300. Turning it on is nearly free and roughly doubles the interpretability of the dashboard.
It also restores `headline_excess_mmd_morpho`, which is currently never computed despite being
described in the code as the checkpoint-selection signal (and which, note, **nothing in the
trainer actually consumes for checkpoint selection** — the only other reference is a test).

### 8.2 Detection power: which metric catches which defect

Each row is the disjoint real half with one controlled defect applied, so the honest baseline is
the **real-vs-real** row, not zero. Bold = clearly above floor.

| defect | mmd_morpho | dens_morpho | cov_morpho | mmd_tmd | w1_pooled | w1_pertree | node_ct_w1 | branch_len_w1 | bif_angle_w1 | part_asym_w1 |
|---|---|---|---|---|---|---|---|---|---|---|
| **real-vs-real (floor)** | 0.0006 | 0.962 | 0.952 | 0.0003 | 0.065 | 0.147 | 10.73 | 0.0199 | 1.560 | 0.0094 |
| scale ×1.02 | −0.0001 | 0.962 | 0.970 | 0.0003 | 0.061 | 0.139 | 10.73 | 0.0254 | 1.560 | 0.0094 |
| scale ×1.05 | 0.0016 | 0.936 | 0.976 | 0.0003 | 0.059 | 0.155 | 10.73 | **0.0361** | 1.560 | 0.0094 |
| scale ×1.10 | **0.0040** | 0.927 | 0.964 | 0.0003 | 0.066 | 0.176 | 10.73 | **0.0575** | 1.560 | 0.0094 |
| scale ×1.25 | **0.0251** | **0.855** | 0.935 | 0.0003 | **0.138** | **0.285** | 10.73 | **0.1238** | 1.560 | 0.0094 |
| perp squash ×0.9 | 0.0023 | 0.921 | 0.946 | 0.0009 | 0.080 | 0.161 | 10.73 | 0.0277 | **1.887** | 0.0094 |
| perp squash ×0.75 | **0.0097** | 0.868 | 0.935 | 0.0022 | **0.109** | 0.189 | 10.73 | **0.0545** | **2.587** | 0.0094 |
| perp squash ×0.5 | **0.0358** | **0.643** | **0.726** | **0.0042** | **0.165** | **0.246** | 10.73 | **0.0993** | **5.214** | 0.0094 |
| branch jitter sd 0.05 | 0.0000 | 0.988 | 0.970 | 0.0002 | 0.064 | 0.138 | 10.73 | 0.0212 | 1.587 | 0.0094 |
| branch jitter sd 0.15 | 0.0008 | 0.973 | 0.982 | −0.0004 | 0.060 | 0.141 | 10.73 | 0.0279 | 1.273 | 0.0094 |
| branch jitter sd 0.30 | **0.0135** | **0.794** | 0.917 | −0.0014 | 0.088 | 0.192 | 10.73 | **0.0508** | 1.339 | 0.0094 |
| prune 5% deepest leaves | 0.0017 | 0.946 | 0.952 | 0.0003 | 0.071 | 0.162 | **12.06** | 0.0217 | 1.534 | 0.0116 |
| prune 15% deepest leaves | **0.0042** | 0.898 | 0.970 | 0.0013 | 0.087 | 0.191 | **15.40** | 0.0249 | 1.424 | 0.0135 |
| prune 30% deepest leaves | **0.0107** | 0.905 | 0.941 | **0.0050** | **0.112** | **0.244** | **21.07** | 0.0330 | 1.139 | 0.0106 |

**What this table establishes:**

1. **`mmd_morpho` is the best single monitor.** It is the only metric that rises clearly above
   floor for *every* defect class, and it has by far the widest dynamic range (42× floor for a
   25% rescale).
2. **Detection threshold is roughly a 10% global defect.** Scale ×1.05 is at the edge; ×1.10 is
   clearly detected. Below that, the suite cannot distinguish defect from sampling noise at
   N ≈ 170. This is the honest sensitivity claim for the in-loop dashboard.
3. **`bifurcation_angle_w1` is the specialist for anisotropy** (1.56 → 5.21 under perpendicular
   squash, 3.3×) and is almost blind to isotropic scale — a useful, orthogonal signal.
4. **`node_count_w1` is exactly orthogonal to geometry.** It is bit-identical (10.7262) across
   *every* geometric transform and responds only to pruning. Good separation of concerns.
5. **`partition_asymmetry_w1` is nearly frozen.** 0.0094 → 0.0135 for a 15% leaf prune. It is
   correctly geometry-blind, but its topological sensitivity is weak enough that it will not
   catch subtle branching-pattern errors on its own.
6. **W1 is a distance, so it is not monotone in defect size along any single axis.**
   `bifurcation_angle_w1` **improves** under branch jitter (1.560 → 1.273 at sd 0.15) — noise
   happened to push the distribution toward the reference. **Never read a single-metric
   improvement as progress**; that is what the joint metrics are for.
7. **The TMD joint block underperforms its cost.** `mmd_tmd` clears floor only for squash and
   pruning; `density_tmd`/`coverage_tmd` clear it essentially never.

---

## 9. The matched pairwise block (`tmd_cond`, every 5th validation)

This is the metric that answers *"is the conditioning actually being used?"* — the
distributional suite cannot, because a model that ignores its TMD input entirely can still match
every population marginal. Because `_evaluate_rollout` reorders predictions back to GT order,
`gen[i]` was conditioned on `gt[i]`, so per-pair comparison is valid.

| Key | What it says |
|---|---|
| `pd_wasserstein_{path,radial_root}_{mean,median}` | Did the generated tree realise the *specific* barcode it was handed? The core conditioning-fidelity number. |
| `pd_nan_frac_{filtration}` | Fraction of pairs where a diagram could not be built. **The only nan-rate reported anywhere in the live suite.** |
| `height_absdiff_{mean,median}` | Per-pair \|Δ\| of extent along `uhat`. |
| `span_xy_absdiff_{mean,median}` | Per-pair \|Δ\| of perpendicular diameter. |
| `bbox_diag_absdiff_{mean,median}` | Per-pair \|Δ\| of 3D bounding-box diagonal. |
| `branch_length_w1_pairwise_{mean,median}` | Per-tree analog of the pooled branch-length W1. |
| `bifurcation_angle_w1_pairwise_{mean,median}` | Per-tree analog of the pooled angle W1. |
| `n_pairs`, `n_pairs_skipped` | Sample size and structural-failure count. |

**This module is the best-engineered file in `validation/`** — it caches the fixed GT side,
threads `uhat` correctly, never raises on a bad pair, degrades gracefully when `persim` is
missing, reports both mean and median (so one catastrophic pair does not swamp the number), and
**logs its own nan rate**. The rest of the suite should adopt these conventions.

**Defects:**

1. **`bbox_diag_length` is not SO(2)-invariant.** It is an *axis-aligned* world-frame bounding
   box (`geometric_metric.py:95-102`). Under a rotation about `uhat` — which the model is
   explicitly equivariant to, so the rotated tree is *equally correct* — the x/y extents change
   and so does the diagonal. `bbox_diag_absdiff_*` therefore penalises a symmetry the model is
   licensed to exercise. `height_z_range` and `span_xy_diameter` are correctly invariant.
   **Fix:** replace with the rotation-invariant 3D diameter (`pdist(pts).max()`, i.e. the same
   quantity as `total_extent`), or drop the key.
2. **`_pd_distance` on an empty-vs-nonempty pair.** Both-empty returns 0.0; a lone non-empty
   diagram is left to `persim` to match against the diagonal, with any exception falling back to
   `nan`. So a generated tree with *no* topology can score `nan` (dropped by the nan-aware
   aggregation) rather than a large distance — the §7.2 pattern again, though here at least
   `pd_nan_frac_*` records it.
3. **`normalize_mode: minmax`** means the PD distances are, like `mmd_tmd`, scale-free. The
   geometric `*_absdiff` keys cover scale, so the block as a whole is complete — but do not read
   `pd_wasserstein_*` alone as "the tree matches".
4. **`_tmd_cond_due` is `self.step`-based**, correctly justified in the docstring (`evaluate` is
   called once per beta, so a counter would misfire). Correct as written.

---

## 10. Teacher-forced eval (`eval_mode`, currently **off**)

Not on the live path in any current config, but it is the closest metric to the training
objective and worth understanding.

At every GT reduction level, the full ODE sampler produces the next-step leaf offsets and
expansion decisions *from GT context*, and the produced local morphometrics are compared to GT
with the same `_w1`/`_ks` used by the free-running path.

| Block | Keys | Meaning |
|---|---|---|
| `dist` | `branch_length_w1`, `{fwd,side,axial}_{signed,mag}_w1`, `turning_angle_w1`, `axial_frac_w1`, `bifurcation_angle_w1` | Distributional quality of a *single* expansion step, decomposed in the local frame. |
| `dist` | `*_mean_samp` / `*_mean_gt` | **Directional** means. W1 is symmetric and cannot tell over- from under-production; these can. Good design. |
| `pos_mse` | `fwd/side/axial/total/dist_mean/dist_median` | True per-node reconstruction error — each sampled leaf against *its own* GT target. The closest analog of the training `pos_loss`. |
| `exp` | `n`, `base_rate`, `acc`, `precision`, `recall`, `f1`, `auc` | Expansion-decision classification. `auc` is the threshold-free one to trust. |

**Assumptions:**

- **Units differ from the free-running metrics** — TF runs in the training coordinate space
  (parent-relative offsets divided by `pos_scale_factor`, e.g. 0.45 for trees). Documented at
  the top of the file. The two families **must not be subtracted**; only same-family curves are
  comparable.
- **Conditioning is deliberately not threaded** into TF batches: `build_reduction_batches_from_graphs`
  passes `tmds=None`, and `_assemble_diffusion_inputs` fails fast if the model expects
  conditioning. So **TF eval cannot run on a TMD- or class-conditioned model** — which is every
  current `parity_*` config. That is a correct fail-fast, but it means TF is unavailable exactly
  where it would be most useful.
- `exp` metrics use the *sampled* decision (`e_samp > 0`), not an argmax, so `acc`/`precision`/
  `recall` include sampling stochasticity. `auc` is the robust one.
- `n_leaves` and `min_depth` are ints and therefore **never reach wandb** (§1.2).

---

## 11. Ranked findings and recommended fixes

> **Status as of 2026-08-08.** ✅ = shipped, ⬜ = still open. Findings 4, 5, 8, 9, 10, 12(partly),
> 14 shipped; **1 and 2 — the two highest-ranked — are still open.** Details in the status
> banner at the top of this document.

### Must fix

| # | Finding | Where | Fix |
|---|---|---|---|
| ⬜ 1 | **The whole live suite is blind to a reflection along `uhat`** — 0 of 29 metrics change when every generated tree is flipped. Directly related to the known apical/axial error mode. | §7.1 | Add signed axial features to `MORPHO_KEYS` (`max(s)−s_root`, `min(s)−s_root`, or fraction of nodes above the root). Microseconds of cost. |
| ⬜ 2 | **`enable_floor: False`** — every metric has a large, metric-specific real-vs-real floor (`node_count_w1` = **10.7 nodes**; `density_morpho` = 0.962, not 1.0) and none of them are visible. `headline_excess_mmd_morpho` is consequently never computed. | §8.1 | Set `enable_floor: True`. It is one cached call, 1.8 s once per run. |
| ✅ 3 | **nan → GT-mean imputation hides degeneracy.** A generated tree with zero forks is scored as having perfectly average asymmetry and fork angle. No nan rate is logged anywhere in `dist_metrics.py`. | §4.3, §7.2 | Log `morpho_nan_frac_*` / `n_gen_degenerate`; consider a penalising imputation or exclusion-with-count. Copy the `pd_nan_frac_*` convention from `tmd_conditional_eval`. |

### Should fix

| # | Finding | Where | Fix |
|---|---|---|---|
| ✅ 4 | **Morpho vector is rank-12-at-most, effective rank 7.3/16.** Three features are pairwise r = 1.000; two more pairs at r ≥ 0.996. The RBF kernel therefore weights *size* ≈3.6× and *reach* ≈4–5× against *shape* at 1× each. | §4.2 | Whiten (ZCA on the GT morpho matrix) in `build_gt_cache`; or minimally drop `leaf_count`, `bifurcation_count`, `total_extent`, `mean_path_to_root`. |
| ✅ 5 | **`density_tmd` / `coverage_tmd` are effectively constants** — coverage never leaves 0.95–0.99 under any constructed defect, including a 30% leaf prune. 64-d k-NN spheres saturate at N = 168. | §6.2 | Drop them, or reduce the TMD PCA to ~8–16 components for the D/C computation specifically (keep 64 for MMD). |
| ⬜ 6 | **`tmd_eval_filtration: radial_root` is inside `model.tmd_filtrations`**, contradicting the `compute_tmd_embedding` docstring's explicit independence claim. | §6.3 | Change the eval filtration to one outside the conditioning set, or correct the docstring and stop citing `mmd_tmd` as independent evidence on conditioned runs. |
| ⬜ 7 | **`bbox_diag_absdiff_*` is not rotation-invariant about `uhat`**, penalising a symmetry the model is licensed to exercise. | §9 | Replace with the 3D diameter (`pdist(pts).max()`), or drop the key. |
| ✅ 8 | **`_eval_embed_fn` never passes `uhat`** — latent silent-wrong-axis bug for `height`/`rho` filtrations (‖Δ‖ = 1.43 / 1.17 measured). | §6.4 | Thread `uhat` through, or raise on an axis-dependent filtration with `so2_axis ≠ z`. |
| ✅ 9 | **`sholl_critical_radius` docstring is wrong on the live path** and the feature is effectively a size feature (r = 0.90 with `mean_radial_to_root`), not a shape feature. Oversized generated trees are silently truncated by the GT-fit shells. | §3.2 | Either normalise by the tree's own max radial extent (restoring the documented meaning), or rename and stop counting it as a shape feature. Log an out-of-range fraction. |
| ✅ 10 | **Integer leaves are dropped by `_collect_log`**, losing TF `n_leaves` / `min_depth`. | §1.2 | Widen the isinstance check to `(int, float)` excluding `bool`. |

### Nice to have

| # | Finding | Where |
|---|---|---|
| ⬜ 11 | `tree_edit_skipped_frac` denominator is the full pair count, not the count considered before the 64-pair cap — understated ~5× at N = 337. GED is off, so this is dormant. | §3.4 |
| ⬜ 12 | `sholl_peak` is integer-valued but not in `_DISCRETE_PERTREE` — a KS trap if `enable_ks` is ever enabled. | §3.2 |
| ⬜ 13 | `w1_*_mean_normalized` averages a redundant battery with silently variable membership. Log the member count alongside. | §3.3 |
| ✅ 14 | `mmd_bandwidth_morpho`, `mmd_bandwidth_tmd`, `tmd_eff_rank` are per-run constants logged as time series. **Shipped**: pushed once to wandb *config* at first GT-cache build, alongside the new `morpho_gt_nan_frac` and `morpho_version`. | §6.5 |
| ⬜ 15 | `_root_tree` is rebuilt **7.5× per graph**; Dijkstra is used where a BFS would do; gen-side morphometrics are computed twice. Irrelevant at 1.8 s / N=300 — revisit only if the eval set grows 10×. | §7.5 |
| ⬜ 16 | `G.graph["root"] = 0` on generated graphs is assumed, never asserted, and every root-anchored metric depends on it. | §7.3 |
| ⬜ 17 | Caches keyed by `id(list)`; safe today, latent stale-cache risk. | §7.4 |
| ⬜ 18 | `ged_timeout` is advertised in config but is a documented no-op. | §3.4 |
| ⬜ 19 | Heavy-tailed count/extent features would benefit from `log1p` before z-scoring — though the kernel is verified **not** saturated (0.000 of pairs below k = 0.01), so this is an enhancement, not a defect. | §4.4 |

### What is correct and should not be changed

- Fitting µ/σ, the PCA basis, the MMD bandwidth, and the Sholl shells **once on the fixed GT
  set** and reusing them across steps. This is what makes the MMD trajectory comparable across
  checkpoints, and it is the thing most implementations get wrong.
- Returning the unbiased MMD² **unclipped** so it can go slightly negative (measured −0.0049 at
  identity). Clipping would reintroduce bias and break floor comparison.
- Measuring extents in the model's **own SO(2) frame** rather than world x/y/z.
- The near-zero-variance guard on `morpho_std`.
- `mmd2_unbiased`, `density_coverage`, and `median_heuristic_bandwidth` are all faithful to
  their reference definitions — I checked each against the published estimator.
- The persistence-image grid is adequately sampled (σ/h = 0.75 → ~3×10⁻⁵ ripple; no aliasing).
- Keeping GED off (0.35 s/pair vs 1.8 s for the entire rest of the suite).
- The entire design of `tmd_conditional_eval.py` — caching, `uhat` threading, mean+median,
  self-reported nan rates, never raising on a bad pair.
- `_tmd_cond_due` being `self.step`-based rather than counter-based.

---

## 12. Reproducing the empirical results

The four audit scripts used here live in this session's scratchpad; the checks are short enough
to restate. All run against `/Users/umer/Documents/trees_genus_d10/val` (337 real trees) in the
`NEURO2` environment.

- **§4.2 redundancy / §4.4 tails** — build `assemble_morpho_vector` over 200 graphs, then
  `_effective_rank`, `np.corrcoef`, and per-feature skew.
- **§7.1 invariance** — apply `diag(1,1,−1)`, `Rz(40°)`, and a 40° off-axis tilt to every
  generated tree; recompute `compute_distribution_metrics` against a fixed `build_gt_cache`;
  diff the metric dicts.
- **§4.3 nan** — pass a path graph, a single node, and an all-at-origin balanced tree through
  `assemble_morpho_vector` → `standardize_vectors` and inspect which dims land at z = 0.
- **§8.1/§8.2 floor and power** — split the val set into two disjoint halves; use half A as GT
  (and for the cache), half B as "generated"; then re-run half B under each controlled defect
  (global scale, perpendicular squash, per-branch offset jitter, deepest-leaf pruning).
- **§4.4 saturation** — `pdist` on `morpho_z` / `tmd_reduced`, divide by the cached sigma,
  report the kernel-value quantiles and the fraction below 0.01.
- **§7.5 redundancy** — monkeypatch counters onto the `structural_metrics` primitives, reload
  `dist_metrics`, run one `compute_distribution_metrics`, read the counters.

`docs/EVAL_PAPER_PROTOCOL.md` covers the statistical machinery this in-loop suite deliberately
omits (bootstrap CIs, permutation significance, FDR). Nothing in this audit changes that
division of labour: the in-loop suite should stay cheap point estimates. It should just stop
being blind to polarity, stop hiding degeneracy, and start showing its floor.
