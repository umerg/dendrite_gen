# Interpretability study — Methodology, decisions and assumptions

> **What this document is.** The authoritative description of *how* the interpretability
> experiments on the dendrite generator are run: what each experiment measures, the exact
> definitions (with math where it matters), every design decision forced by the architecture,
> the assumptions each conclusion rests on, what has been verified and how, and how the
> implementation deviates from the original plan.
>
> **What it is not.** Results live in `docs/INTERP_PROBES_REPORT.md` (pilot, provisional).
> The original plan is `~/.claude/plans/i-think-let-s-plan-piped-river.md`; §13 lists every
> deviation from it.
>
> **Code.** Branch `interp`, package `interp/` (commit `acbb776` + later). File references are
> relative to the repo root.
>
> **Status legend.** ✅ verified (test or direct measurement, named) · 🟡 argued from the code or
> mathematics but not empirically checked · ⛔ not built / not run yet.

---

## Contents

1. [The model as a probe sees it](#1-the-model-as-a-probe-sees-it)
2. [From a neuron to a table of activations](#2-from-a-neuron-to-a-table-of-activations)
3. [Targets: what each node becomes](#3-targets-what-each-node-becomes)
4. [Observables and baselines: what the model can see](#4-observables-and-baselines-what-the-model-can-see)
5. [Linear probes](#5-linear-probes)
6. [Δ over observables — the main statistic](#6-δ-over-observables--the-main-statistic)
7. [Controls](#7-controls)
8. [Uncertainty](#8-uncertainty)
9. [The experiments](#9-the-experiments)
10. [Graph LayerNorm and interventions (steering, erasure, ablation)](#10-graph-layernorm-and-interventions)
11. [Global-attention ablation](#11-global-attention-ablation)
12. [Verification ledger](#12-verification-ledger)
13. [Deviations from the original plan](#13-deviations-from-the-original-plan)
14. [Known limitations and open issues](#14-known-limitations-and-open-issues)
15. [Code map](#15-code-map)

---

## 1. The model as a probe sees it

Everything downstream depends on a handful of architectural facts. Each is stated with where it
comes from.

### 1.1 Generation order: depth layer by depth layer

The parity configs use `reduction.type = depth`, `mode = deterministic`, `contract_root = False`
(`config/parity_neurons_{uncond,tmd}.yaml`). Let a neuron have hop depths d(u) (soma = 0) and
maximum depth D. One reduction step removes every leaf at the current maximum depth (all
deepest "cherries" are contracted; the soma is never contracted). By induction:

> **Level ℓ (ℓ = 0 is the full neuron) contains exactly the original nodes with d(u) ≤ D − ℓ**,
> in their original order, at their original positions. There are exactly D levels
> (ℓ = 0 … D−1); the last one is the soma plus its children.

*Why:* in a rooted tree every node at the maximum depth is a leaf whose parent's children are all
at depth ≤ max depth, so the deterministic step removes exactly the deepest layer; nothing is
merged, so surviving rows are copies of original rows.

Generation runs this sequence backwards: at the step that produces level ℓ, the model places the
**frontier** — the nodes with d(u) = v where **v = D − ℓ** is the *visible depth* — given every
node with d < v. All shallower leaves are finished tips.

Consequences used everywhere:

* **Every node at every level has a known future**: its descendants in the real neuron.
* **Levels remaining** after this step = ℓ; final depth D = v + ℓ. The model never receives ℓ or
  D (`use_size_ratio: False`; the only size information is what is visible).

✅ *Verified* (`tests/test_interp_reduction_ids.py`, 50 val neurons): the composed reducer
`survivor_mask` equals the closed form `nonzero(depth ≤ D−ℓ)` at every level; positions are
exact copies; the frontier equals the depth-v layer; the branch label equals
"has children in the real neuron"; the apical flag sits on a soma child that heads the apical
subtree; D equals the number of levels.

### 1.2 What is noised: the flow-matching input

The live diffusion is velocity flow matching (`graph_generation/diffusion/flow_v.py`). For each
frontier node f with true parent-relative offset C₀ (local frame) and branch label
e₀ = 2·label − 1 ∈ {−1, +1}:

$$
C_t = (1-t)\,\varepsilon_C + t\,C_0,\qquad e_t = (1-t)\,\varepsilon_e + t\,e_0,
\qquad \varepsilon \sim \mathcal N(0, \operatorname{diag}(\sigma^2)).
$$

The frontier node is placed at P_t = P_parent + C_t (converted to the global frame), and
`e_t` and `t` are appended to every node's input features. Non-frontier ("placed") nodes are at
their true positions and carry e = 0. The network outputs a velocity v̂ ≈ C₀ − ε.

Two consequences:

1. **At t = 0 the frontier inputs are pure noise** — both the frontier positions (and every edge
   feature computed from them) and `e_t`. Nothing about a frontier node's own future can be read
   off its input; it must be computed from the placed tree. At t → 1 they approach the truth.
2. The model's own implied clean branch prediction is
   $$\hat e_1 = e_t + (1-t)\,\hat v_e,$$
   since along the linear path x₁ = x_t + (1−t)(x₁ − x₀). (For a data-prediction model the output
   itself is ê₁; `extract.py` branches on the model class.) ✅ matches the library's own
   reconstruction in `flow_v.forward`.

**Decision — all analyses run at t = 0** (§13): it is the only flow time at which "the network
knows X" cannot be explained by X being partly present in the noised frontier input. t = 0.6/0.9
are kept as an optional t-sweep.

### 1.3 The network: residual stream, messages, normalisation, global tokens

`SO2_EGNN_Network` (`graph_generation/model/egnn_so2.py`), parity configs: 12 blocks, 256-d node
features h, no input projection.

* **Block i (message passing):**
  $$m_{ij} = \phi_e\big([h_i, h_j, a_{ij}]\big),\quad M_i = \textstyle\sum_{j\in\mathcal N(i)} m_{ij},\quad
  h_i \leftarrow h_i + \phi_h\big([\mathrm{LN}_g(h)_i,\, M_i]\big).$$
  Messages use the **raw** features; the graph LayerNorm (below) is applied only to the node's own
  features on their way into φ_h. The residual stream h is never normalised.
  Edges are bidirectional (parent↔child), so one block moves information one hop.
* **Edge features a_ij** are SO(2)-invariant scalars: ρ (distance perpendicular to the axis û),
  du (axial component), branch-angle cosines/sines, and an edge-type embedding. Coordinates are
  not part of h (`update_coors = False`). 🟡 Hence h is invariant under rotation about û by
  construction, and probes target invariant quantities only (lengths, angles, |axial|, counts).
* **Graph LayerNorm** — PyG 2.6.1 `LayerNorm(mode="graph")`, verified by reading the installed
  source ✅. For graph g with N_g nodes and F = 256 channels it uses **one scalar mean and one
  scalar variance pooled over all nodes and all channels**:
  $$\mu_g = \frac{1}{N_g F}\sum_{i\in g}\sum_{c} h_{ic},\qquad
  \sigma_g^2 = \frac{1}{N_g F}\sum_{i\in g}\sum_c (h_{ic}-\mu_g)^2,\qquad
  \mathrm{LN}_g(h)_i = \gamma\odot\frac{h_i-\mu_g}{\sqrt{\sigma_g^2+\epsilon}}+\beta .$$
  Every node's normalised input therefore depends on **every other node of the same graph**. This
  is the root of the "off-target" issues in §10.
* **Global attention** at blocks 1, 3, 5, 7, 9, 11 (an ISAB, `GlobalLinearAttention_Sparse`),
  *before* the EGNN of that block:
  $$I = q + \mathrm{Attn}_1(\mathrm{LN}(q),\, \mathrm{LN}(h)),\qquad
  h \leftarrow h + \mathrm{Attn}_2(\mathrm{LN}(h),\, I),\qquad h \leftarrow h + \mathrm{FF}(\mathrm{LN}(h)),$$
  where q are **8 learned token vectors, re-tiled from the same parameter at every global block**
  (the updated tokens are discarded by the network). So the tokens are a fresh per-graph
  read-out at each global block, not a memory carried across layers. The only path from global
  information back to nodes is Attn₂.
* **Conditioning (TMD arm):** `tmd_mlp(TMD)` (128-d) is concatenated to every node's input.
  Every node therefore receives a summary of the *final* tree's persistence image.

### 1.4 Receptive-field arithmetic

With 12 message-passing blocks and one hop per block, information can travel at most 12 hops by
message passing alone (🟡 standard; exact up to the fact that edge features at a frontier edge
depend on both endpoints). The 6 global blocks give an additional all-to-all route. Two nodes in
**different** primary dendrites are connected only through the soma, so their hop distance is
d(a) + d(b) (§3.3). Dataset: max depth mean 8.2, p95 13 (`docs/NEURON_DATASET_STATS.md`); only
~1.3 % of nodes are deeper than 12 (✅ measured on val/test in this study). So:
*own-lineage* information is almost always within message-passing reach, while *cross-dendrite*
information often is not — which is why P6 is framed around cross-dendrite hop distance.

---

## 2. From a neuron to a table of activations

### 2.1 Rows

One **row** = one (neuron, level ℓ, node u). Each row carries the activation h_u at every
recorded **site**:

| Site | Definition |
|---|---|
| `in` | input features (positions excluded; includes the TMD embedding in the TMD arm) |
| `L0…L11` | features after block i |
| `A1…A11` | features right after the global-attention step of block i |
| `G1…G11` | the 8 global tokens after block i's ISAB, [8 × 256] per item |
| `pool_<site>` | mean of a site over all nodes of the item (graph-level) |

Row groups (`interp/extract.py`): **dense** = frontier rows + soma-child rows, stored at every
site; **sparse** = dense + a fixed 20 % sample of the other placed rows, stored at `in` and odd
`L` sites. Graph-level arrays are stored for every item. Rationale: frontier rows are the subject
of P2/P6b; soma children are needed for apical identity; all-row storage at every site would be
~15 GB per arm per t (✅ measured row counts: val 951k node-rows / 150k frontier rows; test 641k /
96k).

### 2.2 Teacher forcing at fixed t

Rows are produced by the **real training forward pass** (`Expansion.teacher_forced_diagnostics`)
on the true partial tree, with the flow time pinned to t for every graph
(`hooks.fixed_flow_time`). Weights: **EMA** (`model_ema_0.999`, strict load), model in eval mode
(dropout off; the extraction refuses a model in train mode). ✅ EMA and raw-weight losses agree
on 621 val items (0.6035 vs 0.6034 position, 1.264 vs 1.266 expansion); random init gives ~9e3.

### 2.3 Keyed noise and batch independence

The forward draws ε for all frontier nodes of a batch at once, so by default a node's noise would
depend on which other items share its batch. We intercept exactly those two draws and give every
frontier node its own stream seeded by (seed, neuron, level, node)
(`hooks.keyed_flow_noise`). t is deliberately not in the key, so a t-sweep compares the same ε at
different t. Any unexpected draw (e.g. σ_min > 0) raises.

✅ Re-extracting with different batch sizes gives bit-identical inputs (`e_t`, `in`) and
activations equal to ~1e-7 at L0 (float rounding in padded batched attention). In a random-init
network that rounding is amplified ~10× per block; the trained network is much better conditioned.
Graph LayerNorm and attention are per-graph, so no information crosses items in a batch.

### 2.4 Storage precision

Activations are stored as float16. ✅ Trained-network activations peak at |h| ≈ 346 (well within
range) on a 64-item batch; the random-init network overflows float16 in deep layers and must be
extracted with `--dtype float32`. Every write counts non-finite values per site; the store
refuses to return an overflowed site.

### 2.5 Splits

Probes are fit and scored by **grouped cross-validation within a split** (§5.3). Val
(2 529 neurons) is the primary analysis; test (1 167) is analysed independently as a replication.
A train subset can be extracted (`--splits train --max-neurons N`) for a memorisation contrast
(⛔ not run).

---

## 3. Targets: what each node becomes

All targets are computed on the real neuron (`interp/targets.py`), indexed by original node id.

### 3.1 Final-tree targets (per node u)

| Target | Definition |
|---|---|
| `desc_final` | number of descendants of u |
| `height_final` | longest downward hop path from u (0 for a tip) |
| `strahler_final` | Horton–Strahler order (tip = 1; +1 when the two largest child orders tie) — same rule as `validation/structural_metrics.py` ✅ |
| `subtree_pathlen` | total edge length in u's subtree |
| `parent_dist`, `branch_angle` | length of u's incoming edge; angle between parent's incoming edge and u's edge |
| `owner` | the soma child whose subtree contains u (u's primary dendrite) |
| `is_apical_subtree` | owner is the apical soma child |
| `axial_coord` | (p_u − p_soma)·û |

**Apical** = the soma child whose subtree reaches furthest along −û (the training flag,
`root_child_order = axial_extent`; it is also ordinal slot 0 in the model input).

### 3.2 Visible / hidden decomposition (per level)

At visible depth v: `vis_desc(u)` = descendants with depth ≤ v; `hidden_desc = desc_final − vis_desc`;
`vis_height = min(height_final, v − d(u))`; `hidden_height = height_final − vis_height`;
`label2 = height_final ≥ 1` (the one-step branch label for frontier rows);
`y_h2 = height_final ≥ 2` (will grow at least two more generations).

Note the cap: a frontier node at level ℓ has height_final ≤ ℓ (the tree ends ℓ levels later). This
is why P2 has a *strict* variant (§9.2).

### 3.3 Cross-dendrite targets (P6b), and why they are defined this way

For each primary dendrite s at visible depth v, its **current reach** is

$$R_s = \max_{u \in s,\; d(u) < v} |(p_u - p_{\text{soma}})\cdot\hat u| .$$

*Only placed nodes (d < v) count* — the frontier's positions are noise in the model input at
t = 0, so "currently visible reach" must not include them (this was a fix, §13). Dendrites with no
placed node yet have no reach.

| Target (per frontier node n in a **non-apical** dendrite) | Definition |
|---|---|
| `apical_vis_reach` | R_apical / max_s R_s |
| `apical_is_tallest_vis` | apical has the largest R_s |
| `other_arm_tall_vis` | some dendrite other than n's own has R_s ≥ 0.6·max R |
| `own_rank` | rank of n's dendrite by R_s |
| `apical_final_reach` | the apical's reach on the *finished* neuron / tallest finished reach (a future quantity) |

**Hop distance through the soma (dAB).** Let a* be the apical's farthest-reaching placed node.
Because n and a* are in different primary dendrites, the unique tree path between them goes
through the soma, so

$$\mathrm{dAB}(n) = d(n) + d(a^*).$$

For rows in the apical's own dendrite dAB is undefined (NaN) — they are excluded from P6b anyway.

*Why dAB is the right binning variable:* by §1.4, information computed **at** a* reaches n by
message passing only if dAB ≤ 12. Caveat (🟡): the reach of the apical is not localised at a*
alone — partial proxies (e.g. the apical's depth-1…k geometry) exist nearer the soma. So
dAB > 12 is a *sufficient* condition for "the full information cannot arrive by message passing",
not a necessary one. This is why the claim is phrased as "signal present at 9–12 hops and its
fate under global-attention ablation", not as a sharp receptive-field cut.

*Why |axial| (absolute) reach* while the apical is defined along −û: this matches the
"≥ 2 tall arms" failure-mode metric in `docs/APICAL_AXIAL_ERROR_MODE.md` (reach = max |(p−p_root)·û|,
tall if ≥ 0.6 × tallest). Consequence: a basal arm reaching far along +û can be "the tallest".
Kept for consistency with the failure-mode metric (§14).

---

## 4. Observables and baselines: what the model can see

**Principle.** A baseline contains everything the model can see at that step *without learning*:
counts, structure and the clean geometry of placed nodes, plus anything it is given as input
(conditioning, its own branch input `e_t`, its own output ê₁). It never contains the level ℓ, the
final depth D, anything from the hidden future, or the frontier's clean positions. If the
activation beats this baseline, the network has computed something that is not simply visible.

### 4.1 Row-level observables (`interp/baselines.py`)

| Set | Columns | Used by |
|---|---|---|
| `OBS_COMMON` | depth, slot (ordinal), is_root_child, t, n_visible, n_frontier, own-arm visible size / reach / height, n_arms, tallest_vis_reach | — |
| `OBS_FRONTIER` | `OBS_COMMON` + e_t, ê₁, parent's edge length, parent's branch angle, number of frontier siblings | P2 primary |
| `OBS_CONTEXT` | `OBS_COMMON` + own edge length, own branch angle, own axial coordinate, visible descendants, visible height, visible children | P2 secondary |
| `OBS_LOCAL` | depth, slot, t, e_t, ê₁, parent edge/angle, frontier siblings, own-arm size/reach/height — **nothing about other dendrites** | P6b |

All reach / axial quantities use placed nodes only (§3.3). Sizes and heights count frontier nodes
too (their existence and depth are known; only their positions are noise).

### 4.2 The 12-hop neighbourhood profile (`khop_profile`)

For node u in the visible tree with hop-distance matrix H, for k = 1…12:

$$n_k(u) = |\{w: H_{uw}=k\}|,\quad f_k(u) = |\{w: H_{uw}=k,\ w \text{ frontier}\}|,\quad
a_k(u) = \operatorname{mean}_{w: H_{uw}=k,\ w \text{ placed}} |\text{axial}_w| .$$

36 features summarising the structure within message-passing reach. It is the strongest
*non-learned* baseline for anything local (it beats every layer of the trained network on depth,
§9.1). Limitation: it summarises structure and axial geometry only, not ρ or angles of neighbours.

### 4.3 Conditioning in the baseline (TMD arm)

The TMD arm receives `tmd_mlp(TMD)` at every node. "Beyond observables" claims for that arm
therefore add the **first 64 principal components** of the 512-d conditioning vector to every
baseline (`common.tmd_obs`; PCA fit on the split's neurons — unsupervised, no labels involved).
Residual information beyond 64 PCs is still input information and could slightly inflate Δ for
this arm (§14); the unconditional arm has no such confound.

### 4.4 Graph-level baselines (P3, P5)

Per item: visible depth, sizes, frontier size/fraction, the model's predicted continuation rate
over the frontier (mean of 1[ê₁ > 0]), per-arm size/reach/height summaries, the 9 morphometrics of
the **placed** partial tree (`validation/dist_metrics.assemble_morpho_vector`), and optionally the
placed partial tree's own TMD (16 PCs). `frac_visible = v/D` is deliberately excluded (it leaks D).

---

## 5. Linear probes

### 5.1 Why linear

A linear read-out is what the next layer, or a steering vector, can use directly. A flexible
probe can compute the answer itself and conflates "the information is laid out" with "the
information is recoverable by a strong learner" (Alain & Bengio 2016; Hewitt & Liang 2019).

### 5.2 Fitting

* **Pre-processing (per training split):** each feature is winsorised at the training split's
  0.5 / 99.5 percentiles (skipped when that range collapses, so rare features survive), then
  z-scored with training statistics. Rationale: activations can be heavy-tailed; a handful of
  extreme rows otherwise dominate the fit (✅ observed: R² down to −40 on random-init features
  before this).
* **Neuron weights:** every row of neuron g gets weight w = 1/n_g (rescaled to mean 1), so each
  neuron counts equally regardless of size.
* **Ridge (continuous targets)** with an unpenalised intercept, solved by eigendecomposition for
  all α at once:
  $$\hat\beta_\alpha = \arg\min_\beta \sum_i w_i\big(y_i - \bar y_w - (x_i-\bar x_w)^\top\beta\big)^2 + \alpha\|\beta\|^2,$$
  with weighted means x̄_w, ȳ_w. ✅ `test_weighted_ridge_has_intercept`.
* **Logistic (binary / multiclass targets)**, L2-penalised, sample-weighted.
* **Hyper-parameters** (α ∈ 10^[−2,5], 15 values; C ∈ 10^[−4,1], 6 values) are chosen by
  **grouped inner cross-validation inside each training fold** (held-out weighted MSE for ridge,
  held-out score for logistic). GCV was abandoned because it chose badly on heavy-tailed
  features. If no inner split can be scored, C falls back to 1 with a warning.

### 5.3 Cross-validation: grouped by neuron

Rows of the same neuron are strongly correlated (and share all neuron-level targets). Every split
— outer 5 folds, inner splits for hyper-parameters and baselines — is a `GroupKFold` by neuron,
randomised by seed. A neuron is never in both the fitting and the scoring part. Every probe
returns **out-of-fold (OOF) predictions**: each row is predicted by a probe that never saw its
neuron. All scores are computed on OOF predictions.

### 5.4 Metrics

R² (regression; 0 = no better than the mean), AUROC (binary; 0.5 = chance), balanced accuracy
(multiclass), Spearman ρ (for scoring a pooled fit inside a subgroup — §9.6). All weighted by
neuron weights except Spearman.

---

## 6. Δ over observables — the main statistic

### 6.1 Definition (cross-fitted stacking)

For each outer fold (train T, test S):

1. **Baseline feature ô.** Several observables models are tried — gradient boosting at three
   complexities and a linear model — and the one with the best inner-CV score on T is chosen.
   Its inner-OOF predictions on T and its full-T-fit predictions on S give a stacking feature ô
   (the prediction itself for regression; the logit for binary; log-probabilities for multiclass,
   aligned to a global class list so folds missing a class do not misalign columns).
2. **Baseline score.** A second-stage linear probe on **[ô] alone**, fit on T, predicts S.
3. **Full score.** The same second-stage probe on **[ô, x]** (x = the activation), fit on T,
   predicts S.

$$\Delta = \text{score}\big(\text{OOF of }[\hat o, x]\big) - \text{score}\big(\text{OOF of }[\hat o]\big).$$

### 6.2 Why each piece is there

* **Non-linear baseline:** x must add information beyond a *non-linear* summary of everything
  visible — otherwise Δ would reward the network for computing simple non-linear functions of
  visible quantities.
* **CV-selected baseline complexity:** many targets are neuron-constant (e.g. cell class, apical
  final reach). A free gradient-boosting model memorises neurons from row-level features and then
  fails on held-out neurons, which makes Δ look large for *any* x (✅ observed in the pilot: a
  random-init network "gained" +1.3 R² this way). Inner-CV selection picks a regularised or
  linear model when the free one overfits. ✅ `test_incremental_not_inflated_by_neuron_memorisation`.
* **Baseline score from the same second-stage probe on [ô]:** the second stage can recalibrate ô
  (rescale, re-threshold). If the baseline score were the raw ô, that recalibration would be
  credited to x. ✅ `test_delta_does_not_credit_recalibration_multiclass` (Δ for an all-zero x
  is < 0.01). AUROC is invariant to monotone recalibration, so binary results were never affected.
* **ô effectively unpenalised:** after standardisation the ô columns are multiplied by s = 30 and
  never winsorised. A ridge penalty α on the scaled coefficient corresponds to α/s² = α/900 on the
  original scale, so the stacker can always reproduce the baseline; Δ < 0 then only reflects
  genuine overfitting to x. ✅ `test_binary_incremental_no_penalty_artifact`.
* **Cross-fitting:** ô on training rows is itself out-of-fold, so the stacker never sees a
  baseline prediction that was fit on the row it is learning from.

### 6.3 Interpretation rule

Δ > 0 with a CI excluding 0 means the activation carries **linearly readable** information about
the target that is not recoverable from the observables. It does not mean the network *uses* that
information (§9.8, §10), and it is a lower bound on information present non-linearly.

---

## 7. Controls

| Control | What it rules out | Status |
|---|---|---|
| **Random-initialised network** (same architecture, data, pipeline; seed 0; float32) | structure that message passing produces without learning | ✅ used in the pilot; one seed so far |
| **Shuffle within neuron** (permute labels among one neuron's rows) | probes exploiting neuron identity rather than node-level information | ✅ (~0.52–0.55 AUROC in the pilot) |
| **Hewitt–Liang control task** (random label per (depth, slot) type) | probe memorisation of types | opt-in only: depth and slot are inputs here, so the control is trivially solvable (0.92–0.98) and does not measure selectivity |
| **Strict P2 variant** (levels remaining added to the baseline) | look-ahead that is only "how much tree is left" | ✅ built |
| **Train-split memorisation contrast** | look-ahead that is memorised training trees | ⛔ not run |

---

## 8. Uncertainty

**Neuron bootstrap.** Given OOF predictions, resample neurons with replacement B times (500 in
the cluster job), recompute the score, take the 2.5 / 97.5 percentiles. **Paired bootstrap** for
Δ: the same resampled neurons for both [ô] and [ô, x] predictions.

Limitations (🟡): the bootstrap holds the OOF predictions fixed, so it ignores refitting
variance and is somewhat narrow; scores are not corrected for multiple comparisons across sites
and targets (the plan's Holm correction is not implemented). Conclusions should rest on patterns
across layers and controls (monotone growth with depth, random-init ≈ 0), not on single cells.

---

## 9. The experiments

Each card: **question → rows → targets → inputs → baseline → metric → controls → interpretation
→ assumptions → status.** All default to t = 0.

### 9.1 P1 — local structure (sanity / calibration) · `analyses/p1_local.py`

* **Question.** Is local geometry and structure linearly encoded, and how does it build up across
  layers? Does the network keep an explicit depth coordinate?
* **Rows / targets.** Placed rows: depth, number of visible children (internal nodes only),
  own edge length, branch angle. Frontier rows: depth, own edge length. All rows: is-frontier.
* **Inputs.** Every site (frontier rows); sparse sites (placed rows, which include the 20 % sample).
* **Baselines.** Raw input (= the probe at site `in`) and the 12-hop profile, with the same probe.
* **Depth diagonal.** For each true depth d: the AUROC with which the depth probe's OOF score
  separates depth d from d+1. AUROC is used because ridge shrinks predictions towards the mean
  depth, which makes exact-integer accuracy meaningless for deep nodes (a pilot design error).
* **Interpretation.** is-frontier at `in` ≈ 1 is a pipeline check (it is an input flag).
  A diagonal pattern (depth decodable only up to ~layer index) would indicate message-passing-only
  propagation of depth.
* **Assumption.** Depth is a meaningful variable for the network. 🟡 The pilot suggests it is not
  (neither network separates d from d+1 for d ≥ 6), so P1 is a calibration, not a routing test.
* **Status.** ✅ built; pilot run at t = 0.6 (re-run at t = 0 pending).

### 9.2 P2 — look-ahead · `analyses/p2_lookahead.py`

* **Question.** The loss asks only whether a frontier node branches *next* step. Does the
  representation encode how far *beyond* that its subtree will grow?
* **Rows.** Frontier rows with label2 = 1 (will branch next step) — the population for which "one
  step" and "the future" differ.
* **Targets.** y_h2 (binary), log(1+desc_final), height_final, strahler_final,
  log(1+subtree_pathlen). Secondary: placed rows with hidden descendants →
  log(1+hidden_desc), hidden_height.
* **Inputs.** Every site.
* **Baseline.** `OBS_FRONTIER` + 12-hop profile (+ 64 TMD PCs for the TMD arm); secondary uses
  `OBS_CONTEXT`. Includes e_t and the model's own ê₁, so Δ is beyond the model's own one-step
  decision and confidence.
* **Strict variant.** Same, with **levels remaining ℓ** added to the baseline. Because
  height_final ≤ ℓ for a frontier node, the default Δ mixes (a) node-specific look-ahead with (b)
  the global "how much tree is left" signal measured by P5. The model never sees ℓ, so this
  baseline is deliberately stronger than anything it observes; Δ_strict > 0 is the defensible
  claim of *node-specific* look-ahead.
* **Metric.** Plain probe score per site + Δ (and Δ_strict) with paired-bootstrap CIs.
* **Controls.** Random-init; shuffle within neuron.
* **Steering directions** (written per site to `directions_p2_*.npz`, for S-growth):
  $$v_{\text{grow}} = \frac{\sum_s n_s(\mu^+_s-\mu^-_s)}{\sum_s n_s},\quad n_s = \min(n_s^+, n_s^-),$$
  the depth-stratified difference of mean activations between y_h2 = 1 and y_h2 = 0 among label2
  rows (mass-mean; Marks & Tegmark 2023 found mass-mean directions more causally effective than
  probe weights). Likewise v_label (label2 = 1 vs 0 over all frontier rows) and
  $$v_\perp = v_{\text{grow}} - \Pi_{\{v_{\text{label}},\,v_{\text{depth}}\}}\,v_{\text{grow}},$$
  where v_depth is the least-squares depth read-out direction. v_⊥ is the cleanest test of
  look-ahead beyond "branch next" and depth. Stratifying by depth stops the direction absorbing
  the depth difference between the groups (✅ `test_mass_mean_stratification_removes_confound`).
* **Interpretation.** Δ growing with depth in the trained net and ≈ 0 in the random net ⇒ the
  network computes forward-looking subtree information. Presence only (§9.8).
* **Assumptions.** (i) t = 0 removes all frontier-side leakage (§1.2). (ii) The observables are a
  complete enough description of the visible tree that what remains is "computed" — 🟡 the 12-hop
  profile does not include neighbour ρ / angles, so some local geometric information is outside
  the baseline. (iii) TMD arm: 64 PCs capture the conditioning (§14).
* **Status.** ✅ built; pilot (pre-audit, t = 0) run; strict variant not yet run.

### 9.3 P3 — latent discovery · `analyses/p3_latent.py`

* **Question.** From a *partial* tree, does the network's global state already encode what kind of
  neuron is being built — cell class and the final TMD signature?
* **Rows.** One per (neuron, visible-depth bin): bins v = 1,2,3,4,5,6,7–8,9+; within a wide bin the
  shallowest item of the neuron.
* **Inputs.** `G{i}` (8 tokens flattened, 2 048-d) and `pool_L{i}` (node-mean).
* **Targets.** Cell class (7-way; classes with < 10 neurons in a bin dropped; neurons without a
  class dropped). Final-TMD PCs 1–5 (PCA on the split's TMDs) — **unconditional arm only**: for
  the TMD arm they are model input, so probing them only measures pass-through.
* **Baseline.** Graph observables + placed-partial-tree morphometrics + placed-partial-tree TMD
  (16 PCs) (+ 64 conditioning-TMD PCs for the TMD arm).
* **Interpretation.** Δ > 0 at small v: the final type/signature is *linearly available* in the
  representation earlier than our summary of the partial tree makes it. It does **not** mean the
  network knows more than the partial tree determines.
* **Status.** ✅ built; ⛔ not run on trained weights.

### 9.4 P5 — how much tree is left · `analyses/p5_stop.py`

* **Question.** With no size input, the only stop signal is the model's own branch head. Does it
  represent how many generations remain?
* **Rows.** One per reduction item. **Inputs.** `G{i}`, `pool_L{i}`.
* **Targets.** levels remaining ℓ (regression), final depth D (regression), is-last-step
  (ℓ = 0, binary).
* **Baseline.** Graph observables + placed-partial-tree morphometrics (+ 64 TMD PCs, TMD arm);
  also reported: visible-depth-only.
* **Note.** ℓ = D − v and v is an observable, so the ℓ and D analyses differ only in target
  variance scale; read them together.
* **Status.** ✅ built; pilot run at t = 0.6 (pre-audit).

### 9.5 P6a — own-lineage apical identity · `analyses/p6_apical.py`

* **Question.** Does a node know it is in the apical dendrite? The apical flag enters the input
  only at the slot-0 soma child.
* **Rows.** Every stored row with depth ≥ 2 (soma children excluded — they carry the flag).
* **Target.** is_apical_subtree. **Inputs.** Sparse sites.
* **Baseline.** depth + 12-hop profile. The apical subtree is geometrically distinctive (it runs
  along −û), so geometry alone predicts it well (pilot AUROC ≈ 0.95); Δ isolates what the network
  adds beyond geometry.
* **Metric.** AUROC overall and per depth; Δ per depth with paired CI. Per-depth bins require
  ≥ 30 rows, ≥ 10 neurons and ≥ 10 positives and ≥ 10 negatives each spread over ≥ 5 neurons.
* **Status.** ✅ built; pilot run at t = 0.6.

### 9.6 P6b — cross-dendrite awareness

* **Question.** Does a frontier node in a basal dendrite carry information about the *apical*
  (or other dendrites)? Relevant to the "multiple weak apicals" failure mode
  (`docs/APICAL_AXIAL_ERROR_MODE.md`).
* **Rows.** Frontier rows in non-apical dendrites with defined dAB. **Targets.** §3.3.
* **Baseline.** `OBS_LOCAL` + 12-hop profile (+ 64 TMD PCs) — nothing summarising other dendrites,
  so cross-dendrite information must come from the network (or from the 12-hop profile when the
  apical is within 12 hops).
* **Binning.** dAB ∈ {2–4, 5–8, 9–12, 13–16, 17+}. One probe is fit on all rows (pooled), then
  scored inside each bin.
* **Metric inside bins.** For continuous targets **Spearman ρ**, not R²: a pooled fit has one
  intercept, and within a bin of restricted target range that intercept bias drives R² negative
  even when the within-bin ordering is good (✅ seen in the pilot: baseline R² −0.4 to −1.4 in
  deep bins, with the random-init network showing the same spurious "gain"). Binary targets use
  AUROC (rank-based, unaffected). Same per-bin data requirements as P6a.
* **Interpretation.** Δ > 0 in the 9–12 and 13+ bins, absent in the random net, and removed by the
  global-attention ablation (P6c) ⇒ cross-dendrite information routed through the global tokens.
* **Assumptions.** dAB caveat in §3.3; reach uses |axial| (§3.3, §14).
* **Status.** ✅ built; pilot run at t = 0.6 (pre-audit; two targets were artefacts, §14).

### 9.7 P6c — does the information travel through the global tokens?

* **Design.** Re-extract a subset (500 test neurons) with the tokens→nodes read ablated in every
  global block (§11), match rows to the intact extraction on (neuron, level, node), and compare
  per dAB bin:
  * **fixed probe** — fit on intact activations, scored on ablated ones: has the information
    disappeared from where the read-out expects it?
  * **refit** — new probe on ablated activations: is it gone entirely?
  * **position error ratio** (teacher-forced, ablated / intact): does the model *need* the global
    read for placing nodes at that hop distance?
* **Prediction if global attention is the route.** Cross-dendrite information at dAB > 12 vanishes
  under ablation; at ≤ 12 it partly survives via message passing.
* **Status.** ✅ built and tested end-to-end on random init; ⛔ not run on trained weights.
  The rollout part of P6c (applying probes to the model's own generations, "1 tall arm" vs "≥ 2"),
  from the plan, is ⛔ not built.

### 9.8 Planned, not built

* **P4 — amnesic conditioning test (TMD arm).** Erase the TMD subspace at a site with LEACE and
  measure the loss in conditioning fidelity. `probe.leace_fit/apply` exist and are unit-tested;
  the analysis and its graph-LayerNorm checks (§10.3) are ⛔ not built.
* **S-growth / S-apical — steering.** Add α·v (e.g. v_grow, v_⊥, or an apical direction) to
  selected nodes at one site during generation and measure the generated neurons. Requires a
  rollout driver with per-node keyed noise so steered and unsteered rollouts are paired.
  `hooks.SiteEditor` exists; everything else is ⛔ not built. Graph-LayerNorm consequences: §10.2.

---

## 10. Graph LayerNorm and interventions

This section answers "how do we handle the PyG norm" precisely, and states what has and has not
been verified.

### 10.1 What the norm does to an edit

Let an intervention change node features h → h' = h + δ for nodes in a set S of graph g (zero
elsewhere). With the graph-mode norm of §1.3, the pooled statistics change:

$$\mu_g' = \mu_g + \frac{1}{N_g F}\sum_{i\in S}\mathbf 1^\top\delta_i,\qquad
\sigma_g'^2 = \frac{1}{N_g F}\sum_{i\in g}\|h_i'-\mu_g'\mathbf 1\|^2 .$$

So for **every** node j ∉ S, its normalised input to φ_h in the next block changes,

$$\mathrm{LN}_g(h')_j - \mathrm{LN}_g(h)_j \;=\; \gamma\odot\Big(\frac{h_j-\mu_g'}{\sigma_g'} - \frac{h_j-\mu_g}{\sigma_g}\Big)\neq 0,$$

even though h_j itself is unchanged. The size of this off-target shift scales with |S|/N_g and with
‖δ‖ relative to σ_g. Two further routes spread an edit: messages from S to its neighbours (φ_e reads
raw features), and the global tokens (Attn₁ reads every node). Only the first is specific to graph
LayerNorm; all three are real.

### 10.2 Steering (⛔ not built — design decisions)

* Off-target effects are **expected, not a bug**. They will be **measured**, not assumed away:
  every steering run reports changes in unsteered dendrites alongside the steered one.
* Calibrate α in units of the direction's natural scale (‖μ⁺ − μ⁻‖), and report the induced
  |Δμ_g| / σ_g, so dose–response curves can be read against the normalisation shift.
* Controls that share the normalisation side-effect: a random direction of equal norm and an
  "on-manifold" random mass-mean direction. If the steering effect exceeds these at matched norm,
  the effect is attributable to the direction, not to the norm shift.
* Not verified yet: nothing in the steering pipeline has been run.

### 10.3 LEACE erasure (⛔ not built — design decisions and the open question)

LEACE (Belrose et al. 2023) gives, for a site x and concept Z, the least-squares-optimal affine
eraser

$$r(x) = x - P\,(x - \bar x),\qquad P = W^{+}\,\Pi_{\operatorname{col}(W\Sigma_{xz})}\,W,\quad W = \Sigma_{xx}^{-1/2},$$

after which **no linear function of r(x) has non-zero covariance with Z** over the fitting
distribution of rows. ✅ Implementation unit-tested on synthetic data (linear R² for Z drops from
> 0.9 to < 0.02; the edit has rank = dim Z).

Decisions forced by the architecture:

1. **Erase every node of the graph, not a subset.** Erasing a subset creates exactly the
   |S|/N_g normalisation shift of §10.1, which would confound "concept removed" with "statistics
   perturbed".
2. **Fit the eraser inside the CV fold** when measuring erasure efficacy; fitting on all rows and
   then probing is in-sample. Use one-hot Z for categorical concepts.
3. **The open question — can graph LayerNorm re-expose the erased concept?** LEACE's guarantee is
   about *rows*. The norm then computes per-graph statistics. For a graph-level concept z_g
   (e.g. TMD, constant within a graph):
   * the graph mean of erased rows is linear in r(x); zero row-level covariance implies
     Σ_g N_g (r̄_g − r̄)(z_g − z̄) = 0, i.e. zero covariance **when graphs are weighted by size**,
     but not necessarily for an unweighted average over graphs;
   * σ_g'² is a *quadratic* function of the rows and is not constrained by LEACE at all.

   So after erasure, μ_g' and σ_g' — which every node's next-layer input depends on — can still carry
   information about z_g. **This has not been verified.** The planned check for P4: after erasure,
   probe z from (a) the erased rows (should be ≈ chance linearly), (b) the per-graph (μ_g', σ_g')
   pair, (c) the next layer's activations; report all three. If (b) is informative, the erasure
   result must be stated as "linear row-level erasure", not "concept removed".

### 10.4 Global-attention ablation

The attention block has its own graph-mode norms. The ablation replaces the Attn₂ output, which
is added to the residual h; that modified h then passes through the same block's feed-forward
norm (h ← h + FF(LN_g(h))) and every later norm. Because the ablation is applied to **all nodes of
every graph at once**, the normalisation statistics change as a consequence of the ablation itself,
not as a side-effect of editing a subset (contrast §10.1) — i.e. the whole-graph ablation is
self-consistent. Attn₁ (nodes → tokens) and the token stream are untouched. ✅ Verified: under ablation `in` and `L0` are bit-identical and `A1` onward
changes (`test_ablation_only_changes_downstream_of_first_global_block`).

---

## 11. Global-attention ablation

`hooks.AttnAblation` replaces the output of Attn₂ (tokens → nodes) in every global block. Hooks on
the attention submodules do not fire (the batched implementation calls the parent forward
directly), so the method is overridden per instance and restored on exit. Modes:

| Mode | Replacement | What it removes | Caveat |
|---|---|---|---|
| `zero` | 0 | the global read **and** Attn₂'s output bias | removes a constant offset too, so it is a stronger perturbation |
| `mean` | dataset-mean read per block | node- and graph-specific global information; keeps the average offset | mean estimated from 20 batches spread evenly over the extraction |
| `graphmean` | each node gets its graph's average read | node-specific routing; keeps graph-level broadcast | — |

**Primary mode: `mean`** — it removes routed information while keeping the network on its usual
operating point. `zero` is a robustness check.

---

## 12. Verification ledger

| Claim | How verified | Status |
|---|---|---|
| Level structure / node identity (§1.1) | composed reducer vs closed form, 50 val neurons | ✅ |
| Branch labels and apical flag match the real tree | same test | ✅ |
| Strahler / leaf counts / descendants match repo utilities | `validation/structural_metrics`, networkx, 10 neurons | ✅ |
| Recording does not change the forward; L11 = final node state; token shapes | `test_interp_hooks.py` | ✅ |
| Fixed-t patch gives t ≡ t₀ | same | ✅ |
| Keyed noise: identical inputs across batch packings; activations ~1e-7 at L0 | `test_extraction_is_independent_of_batch_packing` | ✅ |
| e_t/ê₁ defined only on the frontier (NaN elsewhere) | `test_frontier_flow_inputs` | ✅ |
| At t = 0.9, e_t sign agrees with the true label > 90 % | same | ✅ |
| Ablation affects only blocks ≥ 1 | `test_ablation_only_…` | ✅ |
| EMA weights load strictly and reproduce the raw-weight loss | direct measurement, 621 items | ✅ (not compared to the W&B-logged loss) |
| Trained activations within float16 range | one 64-item batch (max 346) + per-site overflow guard on every write | ✅ |
| Graph LayerNorm semantics (one μ, σ per graph over nodes × channels) | PyG 2.6.1 source | ✅ |
| Hidden features SO(2)-invariant | by construction (invariant inputs and edge features) | 🟡 not tested |
| Probe library: signal recovered, noise not; grouped folds; controls; bootstrap; mass-mean; LEACE | `test_interp_probe.py` (synthetic, known answers) | ✅ |
| Δ not inflated by baseline overfitting or recalibration; no penalty artefact | three dedicated tests | ✅ |
| Weighted ridge intercept | `test_weighted_ridge_has_intercept` | ✅ |
| All analyses run end-to-end | `test_interp_analyses.py` (random init, 40 neurons) | ✅ |
| Receptive field = 12 hops by message passing | architecture argument | 🟡 |
| dAB > 12 ⇒ information about a* cannot arrive by message passing | argument; partial proxies exist | 🟡 |
| Graph-LN off-target effects of steering | — | ⛔ |
| Whether graph LN re-exposes LEACE-erased concepts | — | ⛔ |
| Representations in the model's own rollouts match teacher-forced ones | — | ⛔ |

---

## 13. Deviations from the original plan

| Plan said | Implemented | Why |
|---|---|---|
| Teacher-forced grid t ∈ {0, 0.6, 0.9}; P2 at 0, P1/P5/P6 at 0.6 | **all analyses at t = 0**; extraction default t = 0 (0.6/0.9 optional) | at t = 0.6 the frontier — including the apical's newest nodes — is 60 % formed, mixing "inferred" with "seen" (§1.2) |
| Fit/select probes on val, report on test | grouped 5-fold CV **within** each split; val primary, test an independent replication | CV on grouped neurons is already an out-of-sample estimate; analysing test separately gives a replication rather than a single transfer number |
| k-hop baseline = SGC-style aggregation of raw inputs | 12-hop **neighbourhood profile** (counts, frontier counts, mean |axial|, §4.2) | directly interpretable and cheap; weaker on neighbour geometry (ρ, angles) — §14 |
| Observables = visible tree + model output | + **conditioning TMD (64 PCs)** for the TMD arm; reach/axial/morphometrics from **placed nodes only** | the TMD arm is given the final-tree summary; frontier positions are noise at t = 0 |
| Δ via GBM on observables | **CV-selected** observables model (3 GBMs + linear) and baseline = **same stacker on ô** | neuron memorisation inflated Δ; raw-ô baseline credited recalibration (§6.2) |
| Ridge α by GCV | grouped inner CV | GCV mis-selected on heavy-tailed features |
| Hewitt–Liang control task as a primary control | opt-in | trivially solvable here (keys are inputs) |
| — | **P2 strict variant** (+ levels remaining) | target cap height ≤ ℓ (§3.2) |
| P6b per-bin R² | **Spearman** per bin for continuous targets; minimum positives per bin | pooled-intercept artefact (§9.6) |
| Cross-dendrite reach over visible nodes | over **placed** nodes | §3.3 |
| P3 tokens PCA → 256 | flattened 8×256 | simpler; ridge/logistic handle 2 048 dims at these sizes |
| P3 TMD-PC targets for both arms | unconditional arm only | input pass-through in the TMD arm |
| Mass-mean stratified by depth × t | by depth (t is fixed) | single t |
| Random init, 2 seeds | 1 seed so far | cost; the sbatch accepts any seed |
| Holm correction within families | not implemented | §8 |
| P4 (LEACE amnesic), steering, rollout capture, GT-prefix rollouts, train memorisation contrast | not built | ordering: probes first |
| Store tiers A / B1 / B2 / C | dense / sparse row groups + graph arrays (same content) | implementation detail |

---

## 14. Known limitations and open issues

1. **Presence, not use.** Every current result is a probe. Causal claims need P6c, P4 and
   steering.
2. **Teacher forcing only.** The model sees true partial trees; its own rollouts may differ.
3. **TMD arm residual.** Conditioning information beyond 64 PCs is not in the baseline; the
   unconditional arm is the clean test of look-ahead.
4. **12-hop profile omits neighbour ρ / angles.** Some local geometric information sits outside
   the baseline; a stronger learned local baseline (e.g. a small GNN trained on visible inputs only)
   would tighten "computed vs visible".
5. **Reach uses |axial|** (consistent with the failure-mode metric); a basal arm reaching along +û
   can be "tallest". A signed variant would align with the apical definition.
6. **dAB is a heuristic** (§3.3).
7. **Bootstrap CIs are narrow** (fixed OOF predictions) and **uncorrected for multiplicity**.
8. **One random-init seed.**
9. **Small details:** `n_front_siblings` counts the row itself (consistent across rows);
   `matrix()` adds missing-value indicator columns based on the rows it is given (consistent within
   an analysis, not across); zero-ablation also removes Attn₂'s bias (§11).
10. **Pilot artefacts already excluded:** P6b `apical_final_reach` and 13+-hop bins in the pilot
    were baseline artefacts (random net showed the same gain); fixed by §9.6 and §6.

---

## 14b. Execution: checkpointing and parallelism

Every analysis is decomposed into independent **units** — one target × one layer (P6: one part ×
target × layer; P3: one visible-depth bin × target × layer). Units run in parallel worker
processes (`INTERP_JOBS`, default 8 in the cluster job), each limited to CPUs / jobs BLAS
threads. A finished unit is appended immediately to `<out>/<analysis>.partial.jsonl`, so a
killed or timed-out job is resumed by resubmitting the same command; finished units are read
back instead of recomputed. A checkpoint carries a signature (analysis, store, t, split, B,
options, engine version); if any of these change, the old file is moved to `*.stale<k>` and the
analysis starts fresh, so results from different settings are never mixed. The observables model
(Δ baseline) is fitted lazily — only if some unit of that target still has to run.

This changes **no result**: every unit runs exactly the same probe with the same seeds as the
serial code. ✅ `tests/test_interp_engine.py`: a parallel P5 run equals the serial run exactly; a
run resumed after deleting half the checkpoint equals the uninterrupted run; truncated last lines
(from a kill) are ignored and recomputed.

Two cost-only simplifications made at the same time: P1's "is-frontier" pipeline check is run
only at `in` and the last layer; and P1's separate "raw-input baseline" was removed because it
was the identical probe on identical data as layer `in` (read `in` as the raw-input baseline).
The P1 depth-diagonal now reuses the sweep's own out-of-fold predictions instead of refitting
the same ridge (identical numbers).

---

## 15. Code map

| File | Role |
|---|---|
| `interp/load.py` | config compose; model via `main.get_expansion_items`; EMA / raw / random weights; TMD vector |
| `interp/reduction_ids.py` | reduction sequences with node ids; root/contract assertions |
| `interp/targets.py` | §3 targets |
| `interp/hooks.py` | sites, recorder, input capture, attention ablation, site editor, fixed t, keyed noise |
| `interp/extract.py`, `interp/store.py` | extraction → store; overflow guard; no-overwrite |
| `interp/baselines.py` | §4 observables, 12-hop profile, graph table (cached, versioned) |
| `interp/probe.py` | §5–8 probes, Δ, controls, bootstrap, mass-mean, LEACE |
| `interp/analyses/common.py` | §14b unit engine: checkpoint, lazy baselines, parallel runner, per-site probe unit |
| `interp/analyses/p1…p6, report.py` | §9 experiments; text summary |
| `scripts/interp_extract.sbatch`, `scripts/interp_analyze.sbatch` | cluster jobs |
| `tests/test_interp_*.py` | §12 verification |
