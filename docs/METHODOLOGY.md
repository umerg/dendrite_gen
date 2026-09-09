# Methodology: Training & Sampling Flow Trace

> **Verified against `c7e469e` (2026-08-15), re-traced 2026-08-31.** Reference configuration is
> `config/parity_neurons_tmd.yaml` and its two siblings (`_uncond`, `_class`) — the neuron parity
> arms. Where trees differ from neurons the difference is called out inline.
>
> Line numbers live **only** in the [Code Anchor Table](#21-anchor-table) at the end; prose cites
> `file::function`. If you refactor, update that one table.

An end-to-end trace of how a tree is generated: from data construction through geometry
computation, flow-matching training and integration, to loss and position recovery. The organising
problem is that a root (soma) expands into **k children in a single step** with no prior sibling to
reference, so most of the machinery here exists to give those k children distinguishable identities
without breaking SO(2) equivariance. Covers the design invariants (SO(2) equivariance, locked
frames, ordinal features), the apical-anchored ordering, and edge cases for k=1, k=2, and k>2.

**The live stack** is `egnn_so2` + diffusion-wrapped `Expansion` + `flow_v.py`
(`VFlowMatchingModel`). `basic.py`, `edm.py`, and `flow.py` are dormant alternatives kept for
A/B comparison — if you are reading this to understand a run, it is almost certainly `flow_v`.

---

## Table of Contents

1. [Design Invariants and Key Decisions](#1-design-invariants)
2. [High-Level Overview: Training Path](#2-training-overview)
3. [High-Level Overview: Sampling Path](#3-sampling-overview)
4. [Phase 1: Data Pipeline — `num_root_children`, Apical Flag, Reduction](#4-phase-1-data-pipeline)
   - [1a: `num_root_children`](#1a-num-root-children)
   - [1b: The Apical Flag (`axial_extent` mode)](#1b-apical-flag)
5. [Phase 2: Shared Directions + Geometric Ordering — `_compute_tree_directions` + `compute_geo_order`](#5-phase-2-geometric-ordering)
   - [2a: Root Child Ordering via `_order_root_children_by_uhat`](#2a-so2-invariant-ordering)
   - [2b: Shared v_in and Reference Frame](#2b-shared-frame)
   - [2c: Ordinal Feature Assignment](#2c-ordinal-feature)
   - [2d: Binary Interior Children](#2d-binary-interior)
6. [Phase 3: Local Frames (Training) — `compute_local_bases`](#6-phase-3-training-frames)
7. [Phase 4: Full Geometry Precomputation — `precompute_full_geometry`](#7-phase-4-precompute)
8. [Phase 5: Training Feature Assembly — `get_loss()`](#8-phase-5-training-features)
   - [5a: One-Hot Ordinal Feature and the Dimension Budget](#5a-geo-ordinal-feature)
   - [5b: Local-Frame Target Conversion](#5b-local-frame-targets)
9. [Phase 6: Flow-Matching Forward (Training) — `VFlowMatchingModel.forward()`](#9-phase-6-flow-training)
   - [6a: The OT Path and the Anisotropic Prior](#6a-noise-sampling)
   - [6b: Local-to-Global Conversion for P_t](#6b-local-to-global-training)
   - [6c: Geometry Patching — Root Children Frames Locked](#6c-geometry-patching)
   - [6d: Model Call, Velocity Loss, Diagnostics](#6d-model-call-loss)
10. [Phase 7: Spawn Logic (Sampling) — `expand()`](#10-phase-7-spawn-logic)
    - [7a: Root Spawn Count from `num_root_children`](#7a-root-spawn-count)
    - [7b: Child Materialisation with Ordinal Tracking](#7b-child-materialisation)
    - [7c: Spawn-Order Ordinals for New Leaves](#7c-spawn-order-ordinals)
    - [7d: The Ordinal Freeze (`axial_extent` mode)](#7d-ordinal-freeze)
11. [Phase 8: Local Frames (Sampling) — `compute_local_bases_for_leaves`](#11-phase-8-sampling-frames)
    - [8a: Shared Random Frame for Root Children](#8a-shared-random-frame)
    - [8b: Legacy Fallback (No Ordinal Info)](#8b-legacy-fallback)
    - [8c: Precompute Geometry + Build Feature](#8c-precompute-sampling)
12. [Phase 9: Velocity Integration (Sampling) — `VFlowMatchingModel.sample()`](#12-phase-9-flow-sampling)
    - [9a: Prior Initialisation at t=0](#9a-noise-init)
    - [9b: Precomputed Geometry + Per-Step Patching](#9b-precompute-and-patch)
    - [9c: Local-to-Global Conversion at Each Step](#9c-local-to-global-sampling)
    - [9d: Geometry Patching vs Internal Computation](#9d-geometry-patching)
13. [Phase 10: Position Recovery](#13-phase-10-position-recovery)
14. [Conditioning: TMD and Cell Type](#14-conditioning)
15. [Training vs Sampling Consistency Analysis](#15-training-vs-sampling)
16. [Edge Cases: k=1, k=2, k>2](#16-edge-cases)
17. [Tensor Shape Reference](#17-tensor-shapes)
18. [Function Call Graph](#18-call-graph)
19. [Operational Envelope](#19-operational-envelope)
20. [Function Architecture and Liveness](#20-function-architecture)
21. [Code Anchor Table](#21-anchor-table)

---

## 1. Design Invariants and Key Decisions {#1-design-invariants}

### The SO(2) axis is dataset-specific

Everything below is stated relative to `uhat`, the SO(2) axis, set by `cfg.model.so2_axis`:

| Dataset | `so2_axis` | Note |
|---|---|---|
| **Neurons** (`neurons_conditional_full`) | `[0, 1, 0]` — **y** | Apical dendrites point **−y**; "lowest uhat component" means "most −y" |
| Trees (`trees_genus_d*`, `trees`) | `[0, 0, 1]` — z | The original convention |

The neuron axis is the one that matters for reading the ordering rule: the anchor child is the one
reaching furthest in **−uhat**, which for neurons is the downward-growing apical trunk.

### SO(2) Equivariance — The Central Constraint

The entire pipeline preserves SO(2) equivariance around `uhat`. This means:

- **Neither ordering nor frames use global axes** (`global_inplane_basis`) for root children. Global axes are used **only** as a degenerate fallback (e.g., `v_in` parallel to `uhat` for non-root nodes) and for sampling frame generation (random angle needs a basis).
- **Child ordering is SO(2)-invariant**: child_0 is the root child whose *subtree* reaches deepest along −uhat. A subtree's extent along `uhat` is unchanged by rotation **around** `uhat`, so the label is invariant — see below.
- **The `forward` direction comes from geometry itself**: the direction from root to child_0, projected onto the plane perpendicular to `uhat`.
- Rotating the entire tree around `uhat` must produce identical ordinals, identical delta-theta values, and identical predictions (up to the same rotation).

### Ordering → Relative Frame

| Layer | Purpose | Uses Global Axes? | Enters Model? |
|-------|---------|-------------------|---------------|
| **Geometric ordering** | Label which child is child_0 (deepest −uhat subtree) | **No** — flag computed from `pos · uhat` only (SO(2)-invariant) | No — only determines index assignment |
| **Shared frame** | Define `forward`/`sideways` basis for position prediction (shared by all root children) | No — uses root→child_0 direction | Yes — this is the local frame for the flow |

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Spawn strategy** | All k children at once in one `expand()` step | Simplest; root has no prior children to reference |
| **k source** | From GT/metadata (`num_root_children` field) | No prediction head needed; k is a structural property |
| **Child_0 selection** | The root child whose **subtree** reaches deepest along −uhat (the apical). Computed once on the finest tree by the dataset, carried as `is_apical_root_child` | SO(2)-invariant; picks the true apical **96.7%** of the time vs 62.2% for the superseded first-edge rule (`docs/APICAL_AXIAL_ERROR_MODE.md` §5) |
| **Remaining children** | Ordered clockwise relative to child_0's perp direction | SO(2)-equivariant (relative to geometric reference, not global axis) |
| **Feature encoding** | `MAX_CHILDREN`-wide one-hot (**23**) where child i lights up bit i (NOT `i/(k-1)` scalar, NOT angular `θ/2π`) | One-hot gives maximal separation between children regardless of k; does not leak GT angular information into features |
| **Reference direction** | Root → child_0 in perp plane | SO(2)-equivariant (co-rotates with positions) |
| **Frame stability** | Locked to P_0 positions throughout the flow path | Per-child frames must not drift with the interpolant |
| **Sampling frames** | One random `forward` per root, shared by ALL children | SO(2)-equivariant — absolute orientation is free; model learns angular placement from one-hot child identity |
| **Root ordinals at sampling** | Assigned once at spawn and **frozen** for the rollout (`axial_extent` mode) | Keeps slot 0 = apical fixed across steps, matching the training flag — see [§7d](#7d-ordinal-freeze) |
| **Global axes policy** | NEVER for root ordering/frames; ONLY for non-root degenerate fallbacks and sampling basis | Prevents SO(2) violation at the root level |
| **Backward compat** | Not needed | Full retrain acceptable |

### Why the apical anchor, and why it is still SO(2)-invariant

`MAX_CHILDREN`-wide one-hot is the **only** per-child identity signal the model receives — all k
root children share one frame, so ordinal *is* identity. That makes the accuracy of "which slot is
the long trunk" decisive. The superseded rule picked child_0 by the `uhat` component of the child's
**first edge**, which agreed with the true apical only ~62% of the time; the model was trained on a
blurry signal, never committed a slot to the dominant −uhat trunk, and at generation smeared
apical-ness across siblings (measured: 35.8% of generated neurons had ≥2 tall arms vs 19.7% in GT).
Switching the criterion to **subtree axial extent** raises label fidelity to ~97%.

Invariance survives the change because the criterion is `min(pos · uhat)` over a subtree, and
`pos · uhat` is unaffected by rotation about `uhat`. Only the azimuthal *angle* of `fwd0` rotates,
and that is exactly the SO(2) freedom the design intends (at sampling it is a freely-drawn random
value anyway).

> **Footnote — the superseded `first_edge` mode.** `cfg.model.root_child_order` still accepts
> `"first_edge"`, and it remains the *code* default in `main.py` for backward compatibility, but no
> current config uses it. It picks child_0 by the lowest `uhat` component of the child's first edge
> (tiebreak: largest perp-plane distance) and threads no apical flag. A checkpoint must be sampled
> in the mode it was trained in — `main.py` validates the value and `expand()` branches on
> `model.root_child_order`. Everything else in this document describes `axial_extent`.

### `global_inplane_basis` Usage Policy

`global_inplane_basis` is **never used for root child ordering or root child frames** — only as a genuine degenerate fallback, and as the basis for the random sampling frame (where the angle drawn on top of it is uniform, so the choice of basis carries no information). Its usages are:

| Location | Status | Reason |
|----------|--------|--------|
| `_compute_tree_directions` (v_in_unit degenerate) | **KEPT** | True degenerate fallback when `nin ≤ eps` (v_in parallel to uhat or zero) |
| `compute_local_bases` standalone (forward ≈ 0) | **KEPT** | True degenerate fallback when called without `_directions` |
| `compute_local_bases_for_leaves` (sampling basis) | **KEPT** | Random rotation basis for sampling (equivariant in distribution since angle is random) |
| `compute_geo_angle_for_new_leaves` (non-root binary) | test-only | Degenerate fallback for non-root binary parents with no grandparent; the function itself is no longer on the live path |
| `compute_geo_lr_mask`, `compute_root_child_angles`, `compute_geo_lr_mask_f2` | **DELETED** | Absorbed into `_compute_tree_directions` + `compute_geo_order` |

### Ordinal Feature Table (23-wide One-Hot)

`MAX_CHILDREN = 23` is module-level in `expansion.py` and is the **max primary-dendrite count
observed in the corpus** — chosen so no neuron is filtered out by soma degree
(`neurons_conditional_full`: 26,469 kept, vs 26,445 under the old cap of 16). Training and
sampling read the same module constant, so the two one-hot widths cannot desync.

| k | Child indices | One-hot encoding (MAX_CHILDREN=23) |
|---|---------------|-------------------------------------|
| 1 | [0] | bit 0 |
| 2 | [0, 1] | bits 0, 1 |
| 3 | [0, 1, 2] | bits 0, 1, 2 |
| k | [0, ..., k-1] | bit i for child i (k ≤ 23) |

Child index is **absolute, not normalized by k**: the 3rd child (rank 2) of a k=5 root and of a k=9
root both encode as bit 2. Binary interior children: left = bit 0, right = bit 1. Non-children
(root, internal non-leaf) get all-zeros via the `geo_ordinal == -1` sentinel. Ranks are clamped with
`clamp(0, MAX_CHILDREN - 1)`, so a hypothetical k > 23 would collide onto the last bit rather than
index out of bounds.

---

## 2. High-Level Overview: Training Path {#2-training-overview}

```
Dataset (PrecomputedRedDataset)
    |   ★ _compute_apical_flag() ONCE on the finest tree -> is_apical_root_child [N] bool
    |     re-sliced by survivor_mask at EVERY reduction level (stays aligned with pos)
    v
DataLoader (batch: num_root_children, is_apical_root_child, tmd, cell_class)
    |
    v
Expansion.get_loss(batch, model)
    |
    +-- decode_parent_indices(batch)              -> parent_idx [N], -1 for roots
    +-- build_directed_edge_index(parent_idx)     -> edge_index [2,E], edge_types [E]
    +-- select_training_leaf_indices(batch)        -> leaf_idx_train [L]
    +-- leaf_rel_targets(pos_gt, ...)             -> C_0 [L,3] (relative offsets in global frame)
    |
    +-- precompute_full_geometry(P_0, ..., apical_flag=batch.is_apical_root_child)
    |       |                                      -> pre_geom_p0 dict (ONCE on clean P_0)
    |       +-- _compute_tree_directions(apical_flag=...)
    |       |       |                              -> dirs (topology, v_in, v_out, projections, root ordering)
    |       |       +-- _order_root_children_by_uhat(child0_override=apical) per root  ★
    |       |       +-- v_in[root_children] = fwd0 (shared frame, NOT per-child rotated)
    |       |       +-- v_in[interior] = pos[parent] - pos[grandparent]
    |       |       +-- perp projections + normalization (ONCE for all nodes)
    |       |
    |       +-- compute_geo_order(_directions=dirs) -> geo_ordinal [N], geo_delta_theta [N]
    |       |       +-- Root children: integer rank from root_ordering (0=apical, 1..k-1 clockwise)
    |       |       +-- Binary interior: sinψ-based L/R → ordinal 0/1
    |       |
    |       +-- compute_branch_angles_parent_centric(_directions=dirs) -> (cospsi, sinpsi, cos_theta) [N,1]
    |       +-- edge SO(2) decomposition          -> rel_coors, r_perp, rho, du
    |       +-- assign_branch_angles_to_edges()
    |       +-- compute_local_bases(_directions=dirs) -> local_forward [N,3], local_sideways [N,3]
    |               +-- forward = dirs['v_in_unit'] (shared fwd0 for root children)
    |               +-- sideways = uhat × forward
    |
    +-- assemble node_feats [N, avail_feats_dim]
    |       (is_leaf, geo_onehot [23D] ★, new_leaf_flag, padding)   -- size_ratio OFF
    |
    +-- Convert C_0 to local frame: global_to_local(C_0, leaf_fwd, leaf_side, uhat) -> leaf_rel_pos [L,3]
    |
    +-- VFlowMatchingModel.forward(...)
            |
            +-- sample flow time t per graph (uniform | beta)
            +-- C_noise ~ N(0, diag(prior_std_pos^2))   ★ ANISOTROPIC in local frame
            +-- C_t = (1-t) * C_noise + t * C_0          (linear / OT path)
            +-- e_t = (1-t) * e_noise + t * e_0
            +-- local_to_global(C_t) -> C_t_global
            +-- P_t[leaf_idx] = P_0[parent] + C_t_global
            +-- patch_geometry_for_noised_leaves(pre_geom_p0, P_t, ...)  -> pre_geom
            |       ★ Root children: v_in LOCKED to P_0-based frame (NOT updated with P_t)
            +-- assemble x_in = [P_t | node_feats | e_feat | t]      ★ t, not log_sigma
            +-- model(x_in, ..., tmd=, cell_class=, pre_geom=) -> rel_pred, expansion_pred
            +-- v_target = C_0 - C_noise                 ★ VELOCITY, not the clean offset
            +-- loss = MSE(v_pred, v_target) + MSE(ev_pred, e_0 - e_noise)
            +-- diagnostics on implied clean C1 = C_t + (1-t) * v_pred
```

---

## 3. High-Level Overview: Sampling Path {#3-sampling-overview}

```
Expansion.sample_graphs(target_size, model, tmd, num_root_children, cell_class)
    |   ★ num_root_children comes from the GT eval graph's root degree -> k is TEACHER-FORCED
    |
    +-- Initialize: pos=[0,0,0] per graph, leaf_expansion=1
    |
    +-- while not terminated:
    |     Expansion.expand(adj, batch, target_size, model, ..., num_root_children=nrc)
    |       |
    |       +-- Spawn count determination:
    |       |     ★ Root leaves: spawn_counts = num_root_children[graph_idx] (k children)
    |       |       Non-root: spawn_counts from leaf_expansion (0 or 2)
    |       |       Capped: 0 for any graph already at max_tree_size
    |       |
    |       +-- Child materialisation:
    |       |     ★ Track ordinal_new[i] and sibling_count_new[i] per child
    |       |       geo_angle_new[i] = i (raw integer child index for new leaves)
    |       |
    |       +-- compute_local_bases_for_leaves(pos, parent_idx, ...,
    |       |       child_ordinal=ordinal_t, sibling_count=sib_count_long)
    |       |     |
    |       |     +-- Root children (degenerate: all at parent pos):
    |       |     |     ALL children get SAME random forward (shared frame, no rotation)
    |       |     +-- Non-root: grandparent→parent direction (standard)
    |       |
    |       +-- precompute_full_geometry(pos_new, ...) -> pre_geom_p0  (ONCE on P_0)
    |       |     NOTE: called WITHOUT apical_flag -- there is no GT at sampling
    |       |     geo_feat_all = geo_ordinal (internal nodes) + geo_angle_new (new leaves)
    |       |
    |       +-- ★ axial_extent ORDINAL FREEZE:
    |       |     root children's geo_ordinal is OVERWRITTEN with their spawn-slot rank
    |       |     (ascending node index within parent), discarding the geometric value.
    |       |     Slot 0 = apical stays fixed for the whole rollout. -> 23D one-hot
    |       |
    |       +-- Override pre_geom_p0['local_forward/sideways'][leaves] = leaf_fwd/side
    |       |     ★ Critical for root children at step 0: replaces deterministic fallback
    |       |       with random shared frame from compute_local_bases_for_leaves
    |       |
    |       +-- VFlowMatchingModel.sample(..., pre_geom_p0=pre_geom_p0)
    |       |     |
    |       |     +-- C ~ N(0, diag(prior_std_pos^2))     (prior at t=0, local frame)
    |       |     +-- grid = linspace(0, 1, num_steps+1)  (num_steps = 10)
    |       |     +-- for each step (t_cur -> t_next, dt):
    |       |     |     C_global = local_to_global(C, leaf_fwd, leaf_side, uhat)
    |       |     |     P_cur[leaf] = parent_pos + C_global
    |       |     |     pre_geom_t = patch_geometry_for_noised_leaves(pre_geom_p0, P_cur, ...)
    |       |     |     model(P_cur, ..., pre_geom=pre_geom_t) -> vel_C, vel_e
    |       |     |     Euler update: C = C + dt * vel_C   ★ no 1/(1-t), no DDIM
    |       |     +-- return C, e  (the state AT t = 1 == the final offset)
    |       |
    |       +-- Position update:
    |       |     rel_global = local_to_global(C, leaf_fwd, leaf_side, uhat)
    |       |     pos[leaf] = parent_pos + rel_global
    |       |
    |       +-- Expansion thresholding: leaf_expansion = (exp_pred > 0.0) + 1
    |             (targets live in [-1, +1], so the MAP threshold is 0.0)
    |
    +-- Unbatch into list[nx.Graph]
```

---

## 4. Phase 1: Data Pipeline — `num_root_children`, Apical Flag, Reduction {#4-phase-1-data-pipeline}

**Files**: `graph_generation/data/reduction_dataset.py`, `graph_generation/data/data.py`

Two per-graph facts are computed here and carried through every reduction level: **how many**
children the root has (`num_root_children`, which drives spawning) and **which one is the apical**
(`is_apical_root_child`, which drives ordinal 0).

### 1a: `num_root_children` {#1a-num-root-children}

When building a `ReducedGraphData` sample from a reduction sequence, `_build_reduced_graph_data`
computes how many children the root node has:

```python
root_idx = graph._state.root
num_root_children = int(np.sum(parent_idx == root_idx)) if root_idx is not None else 0
```

This counts all nodes whose parent is the root. For a binary tree with root having 2 children:
`num_root_children = 2`. For SWC neuron data where the soma (root) may have 3+ dendrite origins:
`num_root_children = k` (mean ≈ 7.7 in `neurons_conditional_full`, max 23).

**Storage**: passed to the `ReducedGraphData` constructor as a scalar int, converted to a `th.long`
tensor. Under PyG batching, scalar tensors concatenate into a 1D tensor of length `num_graphs`; no
`__inc__` offset is needed (it is not an index). After batching, `batch.num_root_children[g]` gives
the root branching factor of graph `g`.

### 1b: The Apical Flag (`axial_extent` mode) {#1b-apical-flag}

**Function**: `reduction_dataset.py::RandRedDataset._compute_apical_flag`

The apical label is a property of the **full tree**, not of any single reduction level — a soma
child's subtree extent is only visible before contraction. So it is computed **once on the finest
tree** and then carried down:

```python
axial = pos.astype(np.float64) @ self.uhat          # (n,) projection along uhat
best_child, best_reach = None, np.inf
for c in children:
    # deepest -uhat reach = MIN projection over c's whole subtree (DFS via children map)
    reach = axial[c]
    stack = list(state.children.get(c, []))
    while stack:
        v = stack.pop()
        if axial[v] < reach:
            reach = axial[v]
        stack.extend(state.children.get(v, []))
    if reach < best_reach:
        best_reach, best_child = reach, c
flag = np.zeros(graph.n, dtype=bool)
flag[best_child] = True
```

Returns `None` unless `root_child_order == "axial_extent"` and `uhat` is set — legacy `first_edge`
mode threads no flag at all, and every downstream `apical_flag` parameter is optional with a `None`
default, so the two modes share one code path.

**Carrying it through reduction.** In `get_random_reduction_sequence`, the flag is re-sliced by the
reducer's `survivor_mask` at every level, exactly parallel to `pos`:

```python
pos = pos[reduced_graph.survivor_mask]
if is_apical is not None:
    is_apical = is_apical[reduced_graph.survivor_mask]   # keep aligned with pos
```

This keeps the flag aligned with node indexing as indices are renumbered. **Exactly one `True`
survives per level**, because the flagged node is a root child and root children never contract
(`contract_root: False` in every config) — so the anchor is present and unique at every level the
model trains on.

**Storage**: `is_apical_root_child` on `ReducedGraphData` is a **per-node** BoolTensor `[N]` — a
mask, not a node index. It is therefore intentionally absent from the `__inc__` offset tuple: PyG
concatenates it per-node aligned with `pos`/`parent_idx_1b`, with no index offsetting (which would
corrupt a boolean).

**Consumption**: `get_loss` reads `batch.is_apical_root_child` and passes it to
`precompute_full_geometry(apical_flag=...)`. Sampling has no GT tree, so it never has this flag —
see [§7d](#7d-ordinal-freeze) for how the equivalent signal is maintained during a rollout.

---

## 5. Phase 2: Shared Directions + Geometric Ordering {#5-phase-2-geometric-ordering}

**Functions**: `helpers.py::_compute_tree_directions` and `helpers.py::compute_geo_order`

This phase is handled by two functions called in sequence from `precompute_full_geometry`:
1. **`_compute_tree_directions`**: Computes tree topology, root ordering, and ALL direction vectors (v_in, v_out, projections) **once**. Root children get a **shared** v_in = fwd0 (root→child_0 direction).
2. **`compute_geo_order`**: Assigns ordinal features for all children — root children from root ordering, binary interior from sinψ-based L/R.

### 2a: Root Child Ordering via `_order_root_children_by_uhat` {#2a-so2-invariant-ordering}

**Purpose**: Determine which child is child_0, and order the rest around it. Called inside
`_compute_tree_directions` once per root.

`_compute_tree_directions` first resolves the apical override, translating the per-node flag from
[§1b](#1b-apical-flag) into a **local index within this root's children**:

```python
child0_override = None
if apical_flag is not None:
    flagged = apical_flag[children].nonzero(as_tuple=False).flatten()
    if flagged.numel() > 0:
        child0_override = int(flagged[0].item())
sorted_idx, fwd0_unit, delta_angles = _order_root_children_by_uhat(
    offsets, uhat, eps=eps, child0_override=child0_override,
)
```

Then `_order_root_children_by_uhat` runs:

1. **Project onto uhat axis**: `uhat_components = (offsets · uhat)` → [k] scalar per child
2. **Project onto perp plane**: `offsets_perp = offsets - uhat_components * uhat` → [k, 3]
3. **Compute perp distances**: `perp_dist = ||offsets_perp||` → [k]
4. **Select child_0**: **`child0_override` if given** (the apical, `axial_extent` mode); otherwise the legacy first-edge rule — lowest uhat component, tiebreak largest perp distance
5. **Build reference direction**: `fwd0 = offsets_perp[child0]` → forward direction from root to child_0 in perp plane
6. **Sort remaining children**: clockwise by `atan2` relative to `fwd0`
7. **Compute delta_angles**: angle of each child relative to `fwd0` (child_0 gets 0.0)

A final swap guarantees `sorted_idx[0] == child0_local` even under float-precision ties, so the
override can never be silently displaced by the angular sort.

**Why SO(2)-invariant**: the apical flag depends only on `pos · uhat` over a subtree; uhat
components, perp distances, and angles relative to fwd0 are likewise all unchanged by rotation
around uhat.

**Degenerate branch** (`fwd0_norm <= eps`): child_0 is (near-)axial, so there is no usable perp
direction. Children are sorted by ascending uhat component instead, and a forced apical is
explicitly re-inserted at position 0 so the override survives. In practice this branch **never
fires** on real data — measured 0/2528 on the neuron val set, with min observed `fwd0_norm ≈ 0.003`,
some 3×10⁵ times `eps`.

> **Naming caveat.** The function is still called `_order_root_children_by_uhat`, which described the
> legacy rule. In `axial_extent` mode the uhat-component rule is bypassed entirely for child_0
> selection (it survives only in the tiebreak-free legacy branch and the degenerate sort). Read the
> name as "orders root children with respect to the uhat axis", not "selects child_0 by uhat component".

### 2b: Shared v_in and Reference Frame {#2b-shared-frame}

After root ordering, `_compute_tree_directions` sets the v_in for **all root children to the same fwd0**:

```python
# _compute_tree_directions
# Shared frame: ALL children get the same fwd0 as v_in
for c in children.tolist():
    v_in[c] = fwd0_unit
```

This means ALL root children of a given root share the same `forward` direction (after perp projection and normalization). The 23-wide one-hot ordinal feature is the only cue telling the model which child it is.

**`geo_delta_theta`** is still computed (for ordinal ordering and post-diffusion refinement) but is **NOT used for frame rotation**. All children share fwd0 as their frame.

### 2c: Ordinal Feature Assignment {#2c-ordinal-feature}

`compute_geo_order` reads root ordering from `_directions` and assigns integer ordinals:

```python
# compute_geo_order
for r, (sorted_idx, _fwd0, delta_angles, children) in root_ordering.items():
    k = children.numel()
    for rank, si in enumerate(sorted_idx.tolist()):
        c = children[si]
        geo_ordinal[c] = float(rank)  # integer child index (0, 1, ..., k-1)
        geo_delta_theta[c] = delta_angles[si]
```

| Rank | geo_ordinal | One-hot bit | Meaning (`axial_extent`) |
|------|-------------|-------------|--------------------------|
| 0 | 0.0 | bit 0 | **the apical** (deepest −uhat subtree) |
| 1 | 1.0 | bit 1 | first sibling clockwise from the apical |
| 2 | 2.0 | bit 2 | second sibling clockwise |
| ... | ... | ... | ... |
| k-1 | k-1.0 | bit k-1 | last sibling clockwise |

**Critical design choices**:
1. The ordinal encodes **rank in the geometric ordering**, NOT the actual angular position `θ/2π`. Using `θ/2π` would leak GT angular information into the feature.
2. The ordinal is an **absolute child index** — the 3rd child (rank=2) of k=5 and k=9 both get `geo_ordinal=2.0` → one-hot bit 2. This is NOT normalized by k.
3. Converted to a **23-wide one-hot vector** in feature assembly (`MAX_CHILDREN = 23`), giving maximal separation between any two children regardless of k.
4. Rank 0 carries semantic weight that ranks 1..k-1 do not: it is the apical anchor. Ranks 1..k-1 are a purely azimuthal ordering, and the azimuthal signal in the data is weak (basal fan gap CV 0.62 vs 0.80 for an i.i.d.-uniform null — basals repel mildly).

### 2d: Binary Interior Children {#2d-binary-interior}

For non-root parents with exactly 2 children, `compute_geo_order` computes L/R from sinψ (using pre-computed `v_in_unit` and `v_out_unit` from shared directions):

```python
# compute_geo_order
# Case 1: opposite-sign sines → left = sin > 0 → ordinal 0.0
# Case 2: same-sign or near-zero → left = larger atan2 angle → ordinal 0.0
geo_ordinal[case_nodes] = (~is_left).to(dtype)  # left→0.0, right→1.0
```

This is consistent with the one-hot scheme: left=index 0 (bit 0), right=index 1 (bit 1). The sinψ values come from the shared `_compute_tree_directions` output — no separate `compute_geo_lr_mask` call needed.

---

## 6. Phase 3: Local Frames (Training) — `compute_local_bases` {#6-phase-3-training-frames}

**Function**: `helpers.py::compute_local_bases`

This function computes per-node local coordinate frames (`forward`, `sideways`) used for SO(2)-equivariant position prediction.

When called with `_directions` from `_compute_tree_directions` (the normal training path), this function is trivial:

```python
# compute_local_bases, when _directions is supplied
if _directions is not None:
    forward = _directions['v_in_unit']
    sideways = uhat × forward
    return {'local_forward': forward, 'local_sideways': sideways}
```

All the heavy lifting — grandparent lookup, v_in computation, root children shared frame, perp projection, normalization, degenerate fallback — is done in `_compute_tree_directions`.

### How root children get their frame

All root children share `forward = fwd0` (root→child_0 direction, projected and normalized). This was set in `_compute_tree_directions`:

```python
# _compute_tree_directions
for c in children.tolist():
    v_in[c] = fwd0_unit  # SAME for all children of this root
```

The 23-wide one-hot ordinal feature is the only thing that differentiates children for the model. During training, each child's target offset `C_0 = global_to_local(pos[child] - pos[root], fwd0, side0, uhat)` naturally has different forward/sideways components because children are at different angular positions. The model learns to predict these different offsets based on the one-hot child identity.

### Interior nodes

For non-root nodes with grandparent: `v_in = pos[parent] - pos[grandparent]` (unchanged, computed in `_compute_tree_directions`).

---

## 7. Phase 4: Full Geometry Precomputation — `precompute_full_geometry` {#7-phase-4-precompute}

**Function**: `helpers.py::precompute_full_geometry`

Called once on clean P_0 positions during training. Orchestrates all geometry via the shared `_compute_tree_directions` base:

```python
def precompute_full_geometry(pos, parent_idx, edge_index, uhat, *, eps, tol, debug):
    # 1. Shared directions + root ordering (ONCE — eliminates all v_in redundancy)
    dirs = _compute_tree_directions(pos, parent_idx, uhat, eps=eps)

    # 2. Unified ordinal feature (root + binary interior)
    geo_ordinal, geo_delta_theta = compute_geo_order(
        pos, parent_idx, uhat, eps=eps, tol=tol, _directions=dirs,
    )

    # 3. Branch angles (reuses shared directions — no recomputation)
    cospsi_node, sinpsi_node, cos_theta_node, intermediates = \
        compute_branch_angles_parent_centric(
            pos, parent_idx, uhat, eps=eps, return_intermediates=True,
            _directions=dirs,
        )

    # 4. Edge SO(2) decomposition (unchanged)
    # 5. Edge angle assignment (unchanged)

    # 6. Local bases (trivial: forward = dirs['v_in_unit'], sideways = uhat × forward)
    local_bases = compute_local_bases(pos, parent_idx, uhat, eps=eps, _directions=dirs)

    return {
        'rel_coors', 'r_perp', 'rho', 'du',
        'cospsi_edge', 'sinpsi_edge', 'cos_theta_edge',
        'cospsi_node', 'sinpsi_node', 'cos_theta_node',
        'geo_ordinal', 'geo_delta_theta',
        'local_forward', 'local_sideways',
    }
```

**Key changes from previous version**:
- `geo_lr_mask` is **no longer in the return dict** (absorbed into `compute_geo_order`)
- `v_in`, `v_out`, `has_gp` **removed from return dict** — `patch_geometry_for_noised_leaves` now uses `local_forward` directly instead of raw `v_in`
- v_in, v_out, projections computed **once** in `_compute_tree_directions`, reused by all downstream
- `compute_local_bases` is trivial when `_directions` provided
- Root children v_in = fwd0 (shared), not v_out (old hack) or per-child rotated (previous version)

**Critical**: Computed **once** on P_0 and cached. `geo_ordinal`, `geo_delta_theta`, and local frames do NOT change during diffusion noising. Used by both training (`get_loss` → `DenoisingDiffusionModel.forward`) and sampling (`expand` → `DenoisingDiffusionModel.sample`).

---

## 8. Phase 5: Training Feature Assembly — `get_loss()` {#8-phase-5-training-features}

**Function**: `expansion.py::Expansion._assemble_diffusion_inputs` (called by `get_loss`)

### 5a: One-Hot Ordinal Feature and the Dimension Budget {#5a-geo-ordinal-feature}

```python
# _assemble_diffusion_inputs -- feature assembly (MAX_CHILDREN is module-level = 23)
geo_ordinal = pre_geom_p0['geo_ordinal'].to(device=pos_gt.device, dtype=pos_gt.dtype)
geo_idx = geo_ordinal.long().clamp(0, MAX_CHILDREN - 1)
geo_onehot = pos_gt.new_zeros((N_nodes, MAX_CHILDREN))
child_mask = geo_ordinal >= 0  # sentinel -1 → all-zeros (non-children)
geo_onehot[child_mask] = geo_onehot[child_mask].scatter_(1, geo_idx[child_mask].unsqueeze(-1), 1.0)
features.append(geo_onehot)
feats_used += MAX_CHILDREN
```

`MAX_CHILDREN` is **module-level** in `expansion.py`, read by both `_assemble_diffusion_inputs`
(training) and `expand` (sampling), so the two one-hot widths stay in lockstep by construction.

**Feature vector layout** (per node, `avail_feats_dim` slots):

| Slot(s) | Feature | Shape | Description |
|---------|---------|-------|-------------|
| 0 | `is_leaf` | [N, 1] | 1.0 if node is a leaf, 0.0 otherwise |
| 1–23 | `geo_onehot` | [N, 23] | One-hot child index: bit i = 1.0 for child i. All-zeros for non-children (sentinel -1). |
| 24 | `new_leaf_flag` | [N, 1] | 1.0 if node is a "new leaf from next level" |
| 25+ | padding | [N, ...] | zeros |

**`size_ratio` is not emitted.** `config/method/expansion.yaml` sets `use_size_ratio: False`, so the
`current_size / total_tree_size` slot the assembly *can* append is skipped in every current config.
The code path still exists behind that flag.

**Dimension budget**, `feats_dim = 256` and `cond_dim = 2` across all three neuron parity arms:

| Arm | `tmd_hidden_dim` | `class_hidden_dim` | `avail_feats_dim` | Used | Padding |
|-----|------------------|--------------------|-------------------|------|---------|
| `parity_neurons_uncond` | 0 | 0 | 254 | 25 | 229 |
| `parity_neurons_tmd` | 128 | 0 | 126 | 25 | 101 |
| `parity_neurons_class` | 0 | 16 | 238 | 25 | 213 |

`avail_feats_dim = feats_dim − cond_dim − tmd_hidden_dim − class_hidden_dim`, where `cond_dim = 2`
is a class attribute of the flow model (the `e_t` feature plus the flow-time feature). Whenever any
conditioning is on, a guard rejects `avail_feats_dim < MAX_CHILDREN + 4` up front, so a conditioner
can never silently squeeze out the ordinal one-hot.

**Why one-hot and not a scalar**: the ordinal was once a single scalar `i/(k-1)` in [0, 1]. For k=8
that gives only 0.143 separation between adjacent children — too weak to distinguish them, and
the *only* per-child signal that exists (all root children share one frame). The one-hot gives
maximal, orthogonal separation regardless of k.

### 5b: Local-Frame Target Conversion {#5b-local-frame-targets}

The model predicts parent-relative offsets in the **local frame**. Ground-truth targets must be converted:

```python
# _assemble_diffusion_inputs
leaf_rel_pos_global = leaf_rel_targets(pos_gt, leaf_idx_train, leaf_parent_idx)  # [L, 3]

# _assemble_diffusion_inputs -- local-frame conversion
leaf_fwd = local_fwd[leaf_idx_train]      # [L, 3] from precomputed frames
leaf_side = local_side[leaf_idx_train]     # [L, 3]
leaf_rel_pos = global_to_local(leaf_rel_pos_global, leaf_fwd, leaf_side, uhat)  # [L, 3]
```

**For a root child**: `leaf_fwd` is `fwd0` (shared by all root children). The global offset `pos[child] - pos[root]` is decomposed into `(forward_component, sideways_component, axial_component)` in this shared frame. Each child gets different components because they're at different angular positions — the one-hot ordinal feature tells the model which angular offset to predict.

---

## 9. Phase 6: Flow-Matching Forward (Training) — `VFlowMatchingModel.forward()` {#9-phase-6-flow-training}

**Function**: `flow_v.py::VFlowMatchingModel.forward` — config `config/diffusion/flow_v10.yaml`

> **This replaced the EDM-style diffusion path.** `basic.py::DenoisingDiffusionModel` (log-normal σ,
> `C_t = C_0 + σ·ε`, DDIM sampling, data-prediction) and `edm.py` are still in the tree and still
> selectable via `cfg.diffusion.name`, but no current config uses them. `flow.py::FlowMatchingModel`
> is the data-prediction flow variant, kept as the A/B partner to the velocity model below.

### 6a: The OT Path and the Anisotropic Prior {#6a-noise-sampling}

```python
t_graph = self._sample_time(num_graphs, device)   # uniform | beta, one per graph
t_leaf = t_graph[leaf_batch].view(-1, 1)

# Linear / OT path: noise at t=0, data at t=1.
C_noise = th.randn_like(C_0) * self._pos_scale(C_0.device, C_0.dtype)
e_noise = th.randn_like(e_0) * self.prior_std
C_t = (1.0 - t_leaf) * C_noise + t_leaf * C_0
e_t = (1.0 - t_leaf) * e_noise + t_leaf * e_0
if self.sigma_min > 0.0:                     # 0 in every current config = pure linear OT path
    C_t = C_t + self.sigma_min * th.randn_like(C_t)
```

The interpolant is a **straight line** from a prior sample at t=0 to the data at t=1, which is why
sampling needs only `num_steps: 10` rather than the ~64 an EDM schedule wanted.

**The prior is anisotropic.** `_pos_scale` returns `prior_std_pos` as a `[1, 3]` per-axis scale
rather than a scalar, and those three axes are the **local-frame axes** `(forward, sideways,
axial)` — not world axes:

```yaml
prior_std_pos: [0.74, 0.61, 0.83]   # neurons_conditional_full, pos_scale_factor 45.1 (axis y)
```

This matters twice over. First, it is **dataset-specific and must be re-pinned on any dataset
swap** — the values are measured against that dataset's `pos_scale_factor`
(`tests/analyse_c0_distribution.py` produces them; `config/diffusion/flow_v10.yaml` tabulates the
known ones). Every `parity_*` config re-declares it explicitly with `_self_` last in the defaults
list, precisely so a dataset swap cannot silently inherit the wrong prior. Second, trees are far
more axially anisotropic than neurons (ratio ≈ 3.4 at depth cap 10 vs ≈ 1.3 for neurons), so a
scalar prior would be badly mismatched on one of the two corpora whichever value you chose.

Setting `prior_std_pos: null` restores an isotropic scalar prior.

### 6b: Local-to-Global Conversion for P_t {#6b-local-to-global-training}

```python
P_t = P_0.clone()
if local_forward is not None and local_sideways is not None and uhat is not None:
    C_t_global = local_to_global(C_t, local_forward, local_sideways, uhat)
    P_t[leaf_idx_train] = P_0[leaf_parent_idx] + C_t_global
```

Where `local_to_global` reconstructs:
```
offset_global = C_t[:, 0:1] * forward + C_t[:, 1:2] * sideways + C_t[:, 2:3] * uhat
```

**For root child i**: its interpolated position is `pos[root] + local_to_global(C_t_i, fwd0, side0,
uhat)` — every root child of a given root uses the *same* `fwd0`/`side0`, and differs only through
its own `C_t_i`.

### 6c: Geometry Patching — Root Children Frames Locked {#6c-geometry-patching}

```python
if pre_geom_p0 is not None:
    with th.no_grad():
        pre_geom = patch_geometry_for_noised_leaves(
            pre_geom_p0, P_t, leaf_idx_train, parent_idx,
            edge_index, model.uhat,
        )
```

Inside `patch_geometry_for_noised_leaves` (`helpers.py::patch_geometry_for_noised_leaves`):

**What gets patched** (for noised leaf positions):
- `v_out_new` = `P_t[leaf] - P_t[parent[leaf]]` — the outgoing direction is recomputed from noised positions
- `cospsi_leaf`, `sinpsi_leaf`, `cos_theta_leaf` — branch angles recomputed using `v_in_unit` from `local_forward` and `v_out_unit` from noised positions
- Edge-level `rel_coors`, `r_perp`, `rho`, `du` — recomputed for affected edges

**What stays locked** (the critical invariant):
```python
# patch_geometry_for_noised_leaves
# v_in direction reused from P_0 via local_forward (already projected,
# normalized, and degenerate-fallbacked — locked to P_0 frame).
v_in_unit = pre_geom_p0['local_forward'][leaf_idx_train]  # (L, 3)
```

The branch angle reference direction `v_in_unit` is read directly from `pre_geom_p0['local_forward']` — the per-node forward basis computed at P_0 time. This is already projected onto the perp plane, normalized, and has degenerate fallback applied. It is **reused unchanged** even though leaf positions have been noised.

**Why `local_forward` instead of raw `v_in`**: Previously, `patch_geometry` stored raw `v_in` in `pre_geom_p0` and re-projected/normalized it internally. Since `local_forward` is derived from `v_in` (it IS the projected, normalized version with degenerate fallback), using it directly eliminates redundant computation and ensures the degenerate fallback (for root children) is always applied consistently.

**Why this matters**: If `v_in` were updated with noised `v_out`, the local frame would drift with each noise realization, breaking the correspondence between the frame the model sees and the frame the targets were computed in.

### 6d: Model Call, Velocity Loss, Diagnostics {#6d-model-call-loss}

```python
e_feat = P_0.new_zeros((N, 1)); e_feat[leaf_idx_train] = e_t
t_node = t_graph[batch].view(N, 1)                        # ← flow time, NOT log_sigma
node_feats_t = th.cat([node_feats, e_feat, t_node], dim=-1)
x_in = th.cat([P_t, node_feats_t], dim=-1)

out = model(x=x_in, edge_index=edge_index, batch=batch,
            edge_attr=edge_attr, parent_idx=parent_idx,
            tmd=tmd, cell_class=cell_class, pre_geom=pre_geom)

# Model output is the VELOCITY, not the clean offset.
v_pred  = out["rel_pred"][leaf_idx_train]        # [L, 3]
ev_pred = out["expansion_pred"][leaf_idx_train]  # [L, 1]

v_target_C = C_0 - C_noise      # constant along the linear path
v_target_e = e_0 - e_noise
pos_loss = F.mse_loss(v_pred,  v_target_C)
exp_loss = F.mse_loss(ev_pred, v_target_e)
```

The model receives interpolated positions `P_t` and the patched geometry (with locked root-child
frames), and regresses the **velocity of the path**, `v = x_data − x_noise`. Because the path is
linear, that velocity is *constant* along it — so if `v_pred` were exact, a single Euler step would
land exactly on `C_0` for any `num_steps`. The practical payoff is that sampling never divides by
`(1 − t)`, so the terminal amplification the data-prediction sampler suffers as t→1 is removed by
construction.

**Targets and the expansion head.** `e_0 = 2·leaf_expansion − 1` maps the {1, 2} expansion label
into {−1, +1}, which is why the sampling-side MAP threshold is `0.0` and not `0.5`.

**Diagnostics** (`diffusion/diagnostics.py::compute_flow_diagnostics`, no-grad) are deliberately
kept in **clean-offset space** rather than velocity space, so the numbers stay comparable against a
data-prediction run. The implied clean prediction is reconstructed and scored:

```python
C1_implied = C_t + (1.0 - t_leaf) * v_pred      # exact when sigma_min == 0 and v_pred is exact
is_root_child = parent_idx[leaf_parent_idx] < 0
diag = compute_flow_diagnostics(C_pred=C1_implied, C_0=C_0, ..., t_leaf=t_leaf,
                                is_root_child=is_root_child, prior_var=prior_var)
```

Results are stratified **by flow time and by root-child status** — the latter because root children
are the hard case this whole document is about, and their error is worth watching separately from
the interior nodes that dominate the average.

---

## 10. Phase 7: Spawn Logic (Sampling) — `expand()` {#10-phase-7-spawn-logic}

**Function**: `expansion.py::Expansion.expand`

### 7a: Root Spawn Count from `num_root_children` {#7a-root-spawn-count}

```python
# expand() -- spawn count determination
spawn_counts = (leaf_expansion == 2).long() * 2  # non-root: 0 or 2 children

is_root_leaf = parent_idx[leaf_idx] < 0
if is_root_leaf.any():
    if num_root_children is not None:
        # Spawn k children for each root, if capacity allows
        root_k = num_root_children[leaf_batch[is_root_leaf]]
        has_capacity = (target_size[leaf_batch[is_root_leaf]] > 1).long()
        root_spawn = root_k * has_capacity
        spawn_counts_final[is_root_leaf] = root_spawn
    else:
        # Legacy: spawn exactly 1 child
        root_should_spawn = (target_size[leaf_batch] > 1).long()
        spawn_counts_final = th.where(is_root_leaf, root_should_spawn, spawn_counts_final)
```

**Step-by-step**:
1. Identify root leaves: `parent_idx[leaf_idx] < 0` (root has parent_idx = -1)
2. Look up `num_root_children` for each root's graph
3. Gate by capacity: `has_capacity = target_size > 1` (at least 2 nodes total needed)
4. Set `spawn_counts_final` for root leaves to `k * has_capacity`

**Result**: Root spawns k children in one step. Non-root leaves spawn 0 or 2 based on `leaf_expansion`.

### 7b: Child Materialisation with Ordinal Tracking {#7b-child-materialisation}

```python
# expand() -- child materialisation loop
ordinal_new = []
sibling_count_new = []
running_child_index = 0

for leaf_global, sc in zip(leaf_idx.tolist(), spawn_counts_final.tolist()):
    if sc == 0:
        continue
    parent_pos = pos[leaf_global].unsqueeze(0)
    placeholder = parent_pos.expand(sc, -1).clone()  # all children start at parent pos
    new_positions.append(placeholder)

    for local_child in range(sc):
        global_child = base_N + running_child_index
        parent_child_edges.append((leaf_global, global_child))
        new_parents.append(leaf_global)
        new_batches.append(int(batch_reduced[leaf_global].item()))
        ordinal_new.append(local_child)          # ★ 0, 1, ..., k-1
        sibling_count_new.append(sc)             # ★ k for all siblings
        running_child_index += 1
```

**Key**: Each new child gets:
- Position = parent's position (placeholder, will be updated by diffusion)
- `ordinal_new[i]` = its index among siblings (0 to k-1)
- `sibling_count_new[i]` = total number of siblings including itself

### 7c: Spawn-Order Ordinals for New Leaves {#7c-spawn-order-ordinals}

```python
# expand() -- spawn-order ordinals
ordinal_t = th.tensor(ordinal_new, device=device, dtype=th.float)
geo_angle_new = ordinal_t  # raw integer child index: 0, 1, ..., k-1
```

This gives each new child its spawn-order ordinal as a raw integer index:
- k=1 child: 0
- k=2 children: 0, 1
- k=3 children: 0, 1, 2
- k=8 children: 0, 1, 2, 3, 4, 5, 6, 7

These integer indices are later converted to 23-wide one-hot vectors in feature assembly. New leaves are at placeholder positions, so no real geometry is available. For internal nodes (from previous steps), the ordinal feature comes from `precompute_full_geometry`'s `geo_ordinal` which is computed from real positions (see [§8c](#8c-precompute-sampling)) — **except** for root children, which are frozen instead (next).

### 7d: The Ordinal Freeze (`axial_extent` mode) {#7d-ordinal-freeze}

At training time, ordinal 0 means *the apical*, established from GT subtree extent. At sampling
there is no GT — `expand()` calls `precompute_full_geometry` **without** `apical_flag`, so its
root-child ordinals would fall back to the legacy first-edge rule and, worse, would be **recomputed
from the current positions at every expansion step**. A root child could drift into a different
ordinal mid-rollout, and slot 0 would stop meaning "apical".

The fix is to assign root-child ordinals once at spawn and freeze them:

```python
if getattr(model, "root_child_order", "first_edge") == "axial_extent":
    is_root_node = parent_idx_new_0b < 0
    is_root_child = (parent_idx_new_0b >= 0) & is_root_node[parent_idx_new_0b.clamp(min=0)]
    rc_idx = is_root_child.nonzero(as_tuple=False).flatten()
    if rc_idx.numel() > 0:
        # rank within each parent by ascending node index (= creation/spawn order)
        order = th.argsort(parent_idx_new_0b[rc_idx], stable=True)
        rc_sorted = rc_idx[order]
        _, counts = th.unique_consecutive(parent_idx_new_0b[rc_sorted], return_counts=True)
        ranks = th.cat([th.arange(int(c), device=device) for c in counts])
        geo_feat_all[rc_sorted] = ranks.to(geo_feat_all.dtype)
```

Root children are ranked by **ascending node index within their parent**, which is exactly their
creation order — and because `expand()` materialises them in a `for local_child in range(sc)` loop,
spawn slot *i* is node index `base_N + i`. So rank == spawn slot, at every step, forever. Slot 0 is
whichever child the model chose to place first, and the model was trained to make slot 0 the apical.

**Scope of the freeze**: root children only. Deeper and binary-interior nodes keep the per-step
geometric correction already sitting in `geo_feat_all` — their L/R identity *is* a function of their
realised positions, so recomputing it is correct.

This is the one place where training and sampling deliberately diverge; see the last row of
[§15](#15-training-vs-sampling). In legacy `first_edge` mode the block is skipped entirely and root
children get the recomputed geometric ordinal like everything else.

---

## 11. Phase 8: Per-Child Local Frames (Sampling) — `compute_local_bases_for_leaves` {#11-phase-8-sampling-frames}

**Function**: `helpers.py::compute_local_bases_for_leaves`

Called during `expand()` to compute local frames for newly spawned leaf nodes. This runs BEFORE `precompute_full_geometry`.

**Signature**:
```python
def compute_local_bases_for_leaves(
    pos, parent_idx, leaf_parent_idx, uhat, eps=1e-8,
    child_ordinal: th.Tensor | None = None,
    sibling_count: th.Tensor | None = None,
) -> tuple[th.Tensor, th.Tensor]:  # (leaf_fwd [L, 3], leaf_side [L, 3])
```

**Call site** in `expand()`:
```python
# expand() -- call site
leaf_fwd, leaf_side = compute_local_bases_for_leaves(
    pos_new, parent_idx_new_0b, leaf_parent_idx_next, model.uhat,
    child_ordinal=child_ordinal_t,
    sibling_count=sib_count_long,
)
```

### Standard non-root children

For children whose parent has a grandparent (non-root):
```python
# compute_local_bases_for_leaves -- non-root branch
gp = parent_idx[leaf_parent_idx]  # grandparent
v_in[sel] = pos[leaf_parent_idx[sel]] - pos[gp[sel]]  # grandparent → parent direction
```
This is unchanged — the incoming direction at the parent is inherited from the grandparent.

### Root children (degenerate case)

When k children are spawned at root, they are all at the parent (root) position initially. The parent (root) has no grandparent. This triggers the **degenerate** path where `v_in` is zero → `nin ≤ eps`.

```python
# compute_local_bases_for_leaves -- root-parent branch
no_gp = ~has_gp  # True for root children
if no_gp.any():
    sel = no_gp.nonzero(as_tuple=False).flatten()
    for s in sel.tolist():
        p = leaf_parent_idx[s].item()
        children = (parent_idx == p).nonzero(as_tuple=False).flatten()
        real_children = [c for c in children.tolist() if (pos[c] - pos[p]).norm() > eps]
        if real_children:
            v_in[s] = pos[real_children[0]] - pos[p]
        # else: v_in stays zero → degenerate fallback
```

Since all k children are at the parent position (placeholders), `real_children` is empty → `v_in` stays zero → `degenerate = True`.

### 8a: Shared Random Frame for Root Children {#8a-shared-random-frame}

When root children are spawned (all at parent position, degenerate), a **single** random forward direction is sampled per root and shared by ALL children:

```python
# compute_local_bases_for_leaves -- shared random frame
if child_ordinal is not None and sibling_count is not None:
    degen_sel = degenerate.nonzero(as_tuple=False).flatten()
    degen_parents = leaf_parent_idx[degen_sel]
    unique_parents, inv = torch.unique(degen_parents, return_inverse=True)

    e1_v, e2_v = global_inplane_basis(uhat, eps=eps)
    base_theta = torch.rand(unique_parents.numel(), device=device) * (2 * torch.pi)

    for j, up in enumerate(unique_parents.tolist()):
        group_mask = inv == j
        group_indices = degen_sel[group_mask]
        theta0 = base_theta[j]
        # Shared frame: all children get the same forward direction
        fwd = theta0.cos() * e1_v + theta0.sin() * e2_v
        forward[group_indices] = fwd.unsqueeze(0)
```

**For k=3 with random θ_0 = 0.7 rad**: ALL 3 children get `forward = cos(0.7)*e1 + sin(0.7)*e2`. The 23-wide one-hot ordinal (`[1,0,0,...,0]`, `[0,1,0,...,0]`, `[0,0,1,...,0]`) is the only cue telling the model which angular offset to predict for each child.

**SO(2) equivariance**: Since `θ_0` is random, the absolute orientation carries no information. The model learns angular placement from the one-hot child identity alone.

### 8b: Legacy Fallback (No Ordinal Info) {#8b-legacy-fallback}

```python
# compute_local_bases_for_leaves -- legacy fallback
else:
    # Legacy: random angle per degenerate leaf
    degen_sel = degenerate.nonzero(as_tuple=False).flatten()
    theta = torch.rand(degen_sel.numel(), device=device) * (2 * torch.pi)
    e1_v, e2_v = global_inplane_basis(uhat, eps=eps)
    forward[degen_sel] = (
        theta.cos().unsqueeze(-1) * e1_v.unsqueeze(0)
        + theta.sin().unsqueeze(-1) * e2_v.unsqueeze(0)
    )
```

Without ordinal info, each degenerate leaf gets an independent random frame.

### 8c: Precompute Geometry + Build Feature {#8c-precompute-sampling}

After computing local bases, `expand()` precomputes full geometry and builds the ordinal node feature:

```python
# expand() -- precompute + ordinal feature   (NOTE: no apical_flag -- there is no GT here)
with th.no_grad():
    pre_geom_p0 = precompute_full_geometry(
        pos_new, parent_idx_new_0b, edge_index, model.uhat,
    )

# geo_ordinal for internal nodes (real positions), spawn ordinals for new leaves
geo_feat_all = pre_geom_p0['geo_ordinal'].clamp(min=0.0).clone()
geo_feat_all[leaf_idx_next] = geo_angle_new  # raw child index (0, 1, ..., k-1)

# ... then the axial_extent ordinal freeze overwrites root children (see 7d) ...

# Feature assembly: convert to 23-wide one-hot (MAX_CHILDREN is module-level = 23)
geo_idx = geo_feat_all.long().clamp(0, MAX_CHILDREN - 1)
geo_onehot = pos_new.new_zeros((N, MAX_CHILDREN))
child_mask = pre_geom_p0['geo_ordinal'] >= 0
child_mask[leaf_idx_next] = True  # new leaves always get one-hot
geo_onehot[child_mask] = geo_onehot[child_mask].scatter_(1, geo_idx[child_mask].unsqueeze(-1), 1.0)
```

**Why override new leaves?** `precompute_full_geometry` computes `geo_ordinal` for ALL nodes, but new leaves are at placeholder positions (= parent pos), making their computed ordinals meaningless. The spawn-order indices (0, 1, 2 for k=3) are used instead.

**Why use `geo_ordinal` for internal nodes?** These nodes have finalized positions from prior expansion steps. `precompute_full_geometry` computes their ordinals from real geometry (sinψ-based L/R for binary interior). This replaces the old `geo_angle` state that was carried between expansion steps.

**But not for root children.** In `axial_extent` mode the freeze in [§7d](#7d-ordinal-freeze) runs
immediately after this block and overwrites every root child's entry with its spawn-slot rank. The
geometric root-child ordinals computed here are therefore discarded — which is intended, since
without `apical_flag` they would carry the legacy first-edge semantics the model was not trained on.

#### Critical: Override `local_forward`/`local_sideways` for Leaves

After precomputing geometry, `expand()` overrides the leaf entries in `pre_geom_p0` with the frames from `compute_local_bases_for_leaves`:

```python
# expand() -- leaf frame override
pre_geom_p0['local_forward'] = pre_geom_p0['local_forward'].clone()
pre_geom_p0['local_sideways'] = pre_geom_p0['local_sideways'].clone()
pre_geom_p0['local_forward'][leaf_idx_next] = leaf_fwd
pre_geom_p0['local_sideways'][leaf_idx_next] = leaf_side
```

**Why this is necessary**: `precompute_full_geometry` internally calls `compute_local_bases`, which handles degenerate root children (placeholder positions at step 0) with a **deterministic** `global_inplane_basis(e1)` fallback. But `compute_local_bases_for_leaves` handles the same case with a **random** shared frame (`base_theta = torch.rand(...)`). Without this override, `patch_geometry_for_noised_leaves` (which reads `pre_geom_p0['local_forward']` as its v_in reference) would use the deterministic `e1` direction, while the local↔global coordinate conversion in `diffusion.sample()` uses `leaf_fwd` (random direction). The model would see branch angles computed relative to one frame but produce predictions interpreted in a different frame — an inconsistency that corrupts the SO(2) geometric signal.

**For non-root steps (step > 0)**: Leaf positions are real (not placeholders), so both functions compute the same direction from grandparent→parent geometry. The override is a no-op.

**`pre_geom_p0`** is then passed to `diffusion.sample()` for efficient per-step geometry patching (see Phase 9).

---

## 12. Phase 9: Velocity Integration (Sampling) — `VFlowMatchingModel.sample()` {#12-phase-9-flow-sampling}

**Function**: `flow_v.py::VFlowMatchingModel.sample`

### 9a: Prior Initialisation at t=0 {#9a-noise-init}

```python
steps = max(int(self.num_steps), 1)                       # 10
grid = th.linspace(0.0, 1.0, steps=steps + 1, device=device)

# Initialise from the Gaussian prior at t=0.
C = th.randn((L, 3), device=device) * self._pos_scale(device, P_0.dtype)   # anisotropic
e = th.randn((L, 1), device=device) * self.prior_std
```

**Critical**: integration starts from a **prior sample**, drawn with the *same* per-axis
`prior_std_pos` used in training ([§6a](#6a-noise-sampling)) — matching the two ends of the path is
the whole point of the anisotropic prior. The placeholder positions (all children at parent pos) are
irrelevant; what matters is:
1. The local frame (`leaf_fwd`, `leaf_side`) defines how the prior maps to 3D space
2. All root children share the SAME random forward (from [§8a](#8a-shared-random-frame)), with independent draws

The time grid is a **uniform** `linspace(0, 1, num_steps + 1)` — there is no σ schedule to shape.

### 9b: Precomputed Geometry + Per-Step Patching {#9b-precompute-and-patch}

**Optimization**: Rather than recomputing ALL geometry from scratch at every integration step, `sample()` receives `pre_geom_p0` (precomputed on P_0 in `expand()`) and patches only leaf-affected quantities at each step:

```python
for step in range(steps):
    t_cur  = float(grid[step].item())
    t_next = float(grid[step + 1].item())
    dt = t_next - t_cur

    P_cur = P_0.clone()

    # Convert local-frame C to global positions
    if local_forward is not None and local_sideways is not None and uhat is not None:
        C_global = local_to_global(C, local_forward, local_sideways, uhat)
        P_cur[leaf_idx] = parent_pos + C_global

    e_feat = P_0.new_zeros((N, 1)); e_feat[leaf_idx] = e
    t_feat = P_0.new_full((N, 1), t_cur)                  # flow time, broadcast to all nodes
    x_in = th.cat([P_cur, node_feats, e_feat, t_feat], dim=-1)

    # Patch precomputed P_0 geometry for the current leaf positions
    pre_geom_t = None
    if pre_geom_p0 is not None:
        pre_geom_t = patch_geometry_for_noised_leaves(
            pre_geom_p0, P_cur, leaf_idx, parent_idx,
            edge_index, uhat,
        )

    out = model(x=x_in, edge_index=edge_index, batch=batch,
                edge_attr=edge_attr, parent_idx=parent_idx,
                pre_geom=pre_geom_t, **model_kwargs)

    # Model output IS the velocity: integrate it directly.
    vel_C = out["rel_pred"][leaf_idx]
    vel_e = out["expansion_pred"][leaf_idx]
    C = C + dt * vel_C          # explicit Euler -- no 1/(1-t), no DDIM
    e = e + dt * vel_e

# After integration C (and e) is the final offset at t = 1.
return C, e
```

**Note what is *absent*.** There is no `1/(1 - t)` reconstruction and no DDIM noise re-injection. The
data-prediction sampler had to divide by `(1 - t)` to back out a velocity from a clean-offset
prediction, which amplifies error as t→1; regressing the velocity directly removes that term by
construction. And because velocity is constant along a linear path, an exact `v_pred` would land on
`C_0` for *any* `num_steps` — which is why 10 steps suffice and why `num_steps` can be swept upward
without the U-turn the data-prediction variant shows.

**The returned value is the integrated state, not a prediction.** `sample()` returns `C` after the
final Euler step — the state at t=1 — whereas the diffusion sampler returned `C0_pred` from the last
model call. Downstream (`expand()`'s position reconstruction, the eval suite) is
parameterization-agnostic because both are "the final local-frame offset".

**This mirrors the training path**: training uses `precompute_full_geometry` on P_0 + `patch_geometry_for_noised_leaves` per sampled time. Sampling does the same — geometry is computed once on P_0, then only leaf-affected quantities are patched at each integration step.

### 9c: Local-to-Global Conversion at Each Step {#9c-local-to-global-sampling}

At each integration step, `C` (in local frame) is converted to global positions:

```python
C_global = local_to_global(C, local_forward, local_sideways, uhat)
P_cur[leaf_idx] = parent_pos + C_global
```

For root child i with shared random forward:
```
P_cur[child_i] = pos[root] + C[i,0]*forward + C[i,1]*sideways + C[i,2]*uhat
```

**The local frames are FIXED throughout all integration steps**. They were computed once in `compute_local_bases_for_leaves` and do not change. This is consistent with training, where frames are locked to P_0.

### 9d: Geometry Patching vs Internal Computation {#9d-geometry-patching}

When `pre_geom_p0` is provided (normal sampling path), `patch_geometry_for_noised_leaves` patches only:
- **Node-level**: Branch angles (cosψ, sinψ, cosθ) for leaves only, using `local_forward` as locked v_in reference
- **Edge-level**: `rel_coors`, `r_perp`, `rho`, `du` for edges touching leaves

Everything for internal nodes (positions fixed from prior expansion steps) is reused from `pre_geom_p0` without recomputation. The model skips its internal `_compute_static_so2_geometry()` when `pre_geom` is provided.

**Performance**: For a tree with N=100 nodes and L=10 new leaves, patching touches ~10 nodes and ~20 edges instead of recomputing all ~100 nodes and ~200 edges. Over 10 integration steps per expansion, across hundreds of expansions per graph, this compounds.

**Root children at step 0**: `pre_geom_p0['local_forward']` at leaf positions has been overridden with `leaf_fwd` from `compute_local_bases_for_leaves` (see [§8c](#8c-precompute-sampling)). This ensures `patch_geometry_for_noised_leaves` uses the same random shared frame as the local↔global conversion — the branch angles and the coordinate transform are always consistent.

---

## 13. Phase 10: Position Recovery {#13-phase-10-position-recovery}

**Function**: `expansion.py::Expansion.expand` (tail)

After `sample()` returns the integrated offset at t=1 (in the local frame):

```python
# expand() -- position update
rel_pred_global = local_to_global(rel_pred, leaf_fwd, leaf_side, model.uhat)
parent_pos_for_children = pos_new[leaf_parent_idx_next]
pos_new[leaf_idx_next] = parent_pos_for_children + rel_pred_global
```

**For root child i** (all share the same forward from [§8a](#8a-shared-random-frame)):
```
final_pos[child_i] = pos[root] + C[i,0]*forward + C[i,1]*sideways + C[i,2]*uhat
```

The one-hot ordinal feature told the model which angular offset to predict for each child. All children share the same frame, but predict different offsets based on their one-hot identity.

**No post-hoc ordinal refinement**: `compute_geo_angle_for_new_leaves()` was once called here to re-order children based on final positions. It is no longer called from anywhere on the live path (it survives only as a test fixture). Interior nodes get a fresh `geo_ordinal` from `precompute_full_geometry` at the START of the next expansion step; root children get the frozen spawn-slot rank of [§7d](#7d-ordinal-freeze). Either way the ordinal is derived at the top of each step, never carried forward as mutable state.

---

## 14. Conditioning: TMD and Cell Type {#14-conditioning}

**Function**: `egnn_so2.py::SO2_EGNN_Network.forward` — wiring in `main.py`

Two **orthogonal per-graph conditioners** exist. Each reserves a slice of `feats_dim`, is embedded
once per graph, then broadcast to that graph's nodes by the `batch` vector and concatenated
**after** the real node features (so the `LR_offset_head`'s fixed-index read is undisturbed). The
order is fixed: **tmd, then class**.

```python
reserved = self.tmd_hidden_dim + self.class_hidden_dim
if reserved > 0:
    cond_embs = []
    if self.tmd_hidden_dim > 0:
        cond_embs.append(self.tmd_mlp(tmd))                      # (B, tmd_hidden_dim)
    if self.class_hidden_dim > 0:
        onehot = F.one_hot(cell_class.long(), self.num_classes).to(x.dtype)
        cond_embs.append(self.class_lin(onehot))                 # (B, class_hidden_dim)
    coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]
    for emb in cond_embs:
        feats = torch.cat([feats, emb[batch]], dim=-1)           # broadcast per node
    x = torch.cat([coors, feats], dim=-1)
```

The incoming `feats` width is asserted to equal `feats_dim − reserved`, so a mismatch between what
`expansion.py` assembled and what the model reserved fails loudly rather than silently shifting
every feature channel.

### TMD (structure) conditioning

A persistence-image vector per graph through a 2-layer SiLU MLP. `tmd_hidden_dim > 0` turns it on.

- **Filtrations**: `[path, radial_root]` at `tmd_bins: 16`.
- **`tmd_in_dim` is DERIVED**, never hand-set: `tmd_conditioning_dim(filtrations, bins) =
  len(filtrations) · bins² = 2 · 16² = 512`. `main.py` raises if a stale `cfg.model.tmd_in_dim`
  disagrees, so the embedding width and the model's input projection cannot desync.
- **The axis-aware `height` filtration was dropped** (commit `cac169d`). It was the only
  `uhat`-dependent conditioning channel; `path` and `radial_root` are both rotation-invariant.
- Validation recomputes TMDs with the **same** filtrations, bins, and axis as training
  (`training.py::_evaluate_rollout`) — a mismatch here would silently condition on a different
  quantity than the model was trained on.

### Cell-type (class) conditioning

A categorical per-graph label — `num_classes: 7` kept pyramidal types
(`utils.data_loading.CELL_CLASS_NAMES`) — as `one_hot(7) → Linear(7, class_hidden_dim)`. That is
arithmetically an embedding lookup, but the one-hot stays explicit in the assembly. There is **no
classifier-free guidance**; the label is a plain conditioning input.

`class_hidden_dim > 0` also gates the per-class stratified validation metrics
(`validation.per_cell_class`).

### The three neuron parity arms

The model block is identical across all of them **except these two switches**, so any difference
between arms is attributable to the conditioning being varied:

| Config | `tmd_hidden_dim` | `class_hidden_dim` | Conditioning |
|---|---|---|---|
| `parity_neurons_uncond` | 0 | 0 | none |
| `parity_neurons_tmd` | 128 | 0 | TMD persistence image only |
| `parity_neurons_class` | 0 | 16 | cell-type label only |

Both conditioners are plumbed all the way through sampling: `Expansion.sample_graphs(tmd=,
cell_class=)` → `expand()` → `model_kwargs` → `flow_v.sample()` → `model(**model_kwargs)`.
`get_loss` raises if a conditioner is enabled but its batch field is missing.

---

## 15. Training vs Sampling Consistency Analysis {#15-training-vs-sampling}

| Aspect | Training | Sampling |
|--------|----------|----------|
| **Frame source** | `_compute_tree_directions` → `compute_local_bases` | `compute_local_bases_for_leaves` with random θ_0 |
| **All root children forward** | fwd0 (root→child_0 direction from GT, shared) | Random direction in perp plane (shared) |
| **Root child_0 identity** | The apical, from the GT `is_apical_root_child` flag | Spawn slot 0 — the model's own first-placed child |
| **Feature** | 23-wide one-hot from integer `geo_ordinal` (apical-anchored GT ordering) | 23-wide one-hot: frozen spawn-slot rank for root children, `geo_ordinal` for other internal nodes, raw spawn index for new leaves |
| **Root ordinal over time** ⚠️ | Recomputed from GT geometry at every reduction level (always the true apical) | **Frozen at spawn** and never recomputed — see [§7d](#7d-ordinal-freeze) |
| **Path** | `C_t = (1−t)·C_noise + t·C_0`, t sampled per graph | Integrate `C ← C + dt·v_pred` from t=0 to t=1 over 10 uniform steps |
| **Prior** | `C_noise ~ N(0, diag(prior_std_pos²))` in local frame | Same anisotropic prior, same `prior_std_pos` |
| **Model output** | Velocity `v`, scored against `C_0 − C_noise` | Velocity `v`, integrated directly |
| **2nd cond feature** | Flow time `t` (per-graph, broadcast) | Flow time `t_cur` of the current step |
| **Frame stability** | Locked to P_0 (via `patch_geometry_for_noised_leaves`) | Locked to P_0 (via `patch_geometry_for_noised_leaves`) |
| **Geometry computation** | Precomputed once (`precompute_full_geometry`), patched per sampled time | Precomputed once (`precompute_full_geometry`), patched per integration step |
| **Branch angle reference** | `local_forward` from P_0 (real positions) | `local_forward` from `compute_local_bases_for_leaves` (random shared frame at step 0, real positions at step > 0) — overridden into `pre_geom_p0` before integration |

**The one deliberate asymmetry** is the ⚠️ row. Training can afford to re-derive "which child is the
apical" at every level because it has the GT tree; sampling cannot, so it commits once and holds.
The two agree in *meaning* — slot 0 is the apical in both — but reach it by different routes.

### Why the shared frame works

**Training**: All root children share fwd0. Each child's target offset `C_0 = global_to_local(pos[child]-pos[root], fwd0, side0, uhat)` has different forward/sideways components because children are at different angular positions. The one-hot identity tells the model which offset to predict.

**Sampling**: All root children share a random forward. The model has learned from training that one-hot bit 0 means "predict offset in the fwd0 direction" and bit 2 means "predict offset at some learned angle from fwd0". The absolute orientation is irrelevant (SO(2)-equivariant).

**The consistency invariant holds because**: Both training and sampling use the **same shared-frame structure** (all root children share one forward direction), the **same one-hot ordinal feature scheme** (child i → bit i, regardless of k), the **same local_to_global / global_to_local conversions**, and now the **same precompute + patch geometry pattern**.

---

## 16. Edge Cases: k=1, k=2, k>2 {#16-edge-cases}

### k=1 (Single Root Child)

| Phase | Behavior |
|-------|----------|
| **Ordering** | Short-circuits before `_order_root_children_by_uhat`: `fwd0 = pos[child] − pos[root]`, perp-projected. `geo_ordinal = 0`, `geo_delta_theta = 0.0`. The apical flag is irrelevant — the only child is child_0 by default. |
| **Training frame** | `v_in = fwd0` (perp-projected) |
| **Sampling frame** | Random forward (single child, degenerate path) |
| **Feature** | one-hot bit 0 |
| **Spawn** | `spawn_counts_final = 1 * has_capacity` |

### k=2 (Binary Root)

| Phase | Behavior |
|-------|----------|
| **Ordering** | Apical (deepest −uhat subtree) = child_0; the other child follows. `geo_ordinal = [0, 1]` |
| **Training frame** | Both children share `fwd = fwd0` (root→apical direction) |
| **Sampling frame** | Both children share same random forward |
| **Feature** | one-hot bits 0 and 1 |
| **Spawn** | `spawn_counts_final = 2 * has_capacity` |

### k>2 (Multi-Child Root)

| Phase | Behavior |
|-------|----------|
| **Ordering** | Apical = child_0, remaining clockwise relative to child_0's perp direction. `geo_ordinal = [0, 1, 2, ..., k-1]` |
| **Training frame** | ALL children share `fwd = fwd0` (root→apical direction) |
| **Sampling frame** | ALL children share same random forward |
| **Feature** | 23-wide one-hot, bit i for child i — orthogonal separation regardless of k |
| **Spawn** | `spawn_counts_final = k * has_capacity` |

Neurons live here: mean root degree ≈ 7.7, max 23.

### Apical Edge Cases (`axial_extent` mode)

The flag forces **exactly one** child to ordinal 0, but the biology does not always cooperate. On
`neurons_conditional_full`, ~93% of neurons have exactly one primary apical trunk, ~4.7% have none,
and ~2.6% have two.

| Scenario | Handling |
|----------|----------|
| **No apical** (~5%) | The flag still fires — it lands on whichever basal has the deepest −uhat subtree. Ordinal 0 then means "deepest arm", not "apical". Class conditioning is the intended absorber for this variation, since apical presence is strongly class-dependent (4P 98.7%, 6P-IT 84.6%). |
| **Two apicals** (~2.6%) | Only one is flagged (the deeper); the second is ordered azimuthally like a basal. The one-flag model cannot represent this tail faithfully — a dedicated `is_apical` bit would (option 2 in `docs/APICAL_AXIAL_ERROR_MODE.md` §5), but is not implemented. |
| **Root has no children** | `_compute_apical_flag` returns `None`; no flag threaded. |

### Degenerate Cases

| Scenario | Handling |
|----------|----------|
| **Child on uhat axis** (offset ⊥ perp plane is zero) | `fwd0_norm ≤ eps` → sort by uhat ascending, `delta_angles = 0`, forced apical re-inserted at position 0. Measured 0/2528 occurrences on neuron val — this branch does not fire in practice. |
| **All children at root** (sampling, pre-integration) | Degenerate path → shared random forward for all children |
| **`v_in` parallel to uhat** | `global_inplane_basis` fallback in `_compute_tree_directions` |
| **`num_root_children` = None** | Legacy spawn: exactly 1 child per root |
| **k > MAX_CHILDREN** | `clamp(0, 22)` collides the overflow onto bit 22 rather than indexing out of bounds. Cannot occur on the current corpus (23 = observed max). |

---

## 17. Tensor Shape Reference {#17-tensor-shapes}

### Training Tensors

| Tensor | Shape | Type | Source |
|--------|-------|------|--------|
| `pos_gt` (P_0) | [N, 3] | float32 | Batch |
| `parent_idx` | [N] | long | Decoded from `parent_idx_1b` |
| `is_apical_root_child` | [N] | **bool** | Batch — per-node mask, not an index (absent from `__inc__`) |
| `geo_ordinal` | [N] | float32 | `compute_geo_order`; integer rank, −1 sentinel for non-children |
| `geo_delta_theta` | [N] | float32 | `_compute_tree_directions` / `compute_geo_order` |
| `local_forward` | [N, 3] | float32 | `compute_local_bases` |
| `local_sideways` | [N, 3] | float32 | `compute_local_bases` |
| `leaf_rel_pos` (C_0) | [L, 3] | float32 | `global_to_local(leaf_rel_pos_global)` |
| `leaf_fwd` | [L, 3] | float32 | `local_forward[leaf_idx_train]` |
| `leaf_side` | [L, 3] | float32 | `local_sideways[leaf_idx_train]` |
| `geo_onehot` | [N, 23] | float32 | 23-wide one-hot from `geo_ordinal` integer index (`MAX_CHILDREN = 23`) |
| `t_graph` | [G] | float32 | Flow time per graph, `_sample_time` (uniform \| beta) |
| `t_leaf` | [L, 1] | float32 | `t_graph[leaf_batch]` |
| `C_noise` | [L, 3] | float32 | `randn * prior_std_pos` — anisotropic in the local frame |
| `C_t` | [L, 3] | float32 | `(1−t)·C_noise + t·C_0` |
| `v_target_C` | [L, 3] | float32 | `C_0 − C_noise` — the regression target |
| `C1_implied` | [L, 3] | float32 | `C_t + (1−t)·v_pred`, diagnostics only |
| `num_root_children` | [G] | long | Batch (per-graph scalar) |
| `tmd` | [G, 512] | float32 | Batch; `len(filtrations) · bins²` |
| `cell_class` | [G] | long | Batch (per-graph scalar) |

### Sampling Tensors

| Tensor | Shape | Type | Source |
|--------|-------|------|--------|
| `pos_new` | [N', 3] | float32 | Accumulated, updated after each expansion |
| `geo_feat_all` | [N'] | float32 | Frozen spawn-slot rank for root children (`axial_extent`), `geo_ordinal` for other internal nodes, raw child index for new leaves |
| `geo_angle_new` | [K] | float32 | Raw child index (0, 1, ..., k-1) for new leaves only |
| `pre_geom_p0` | dict | — | Full geometry precomputed on P_0, passed to `sample()` for per-step patching |
| `ordinal_new` | [K] | long | 0 to k-1 per child group |
| `sibling_count_new` | [K] | long | k for each child in group |
| `child_ordinal_t` | [K] | long | Passed to `compute_local_bases_for_leaves` |
| `sib_count_long` | [K] | long | Passed to `compute_local_bases_for_leaves` |
| `leaf_fwd` | [K, 3] | float32 | Shared frame per root (random forward) |
| `leaf_side` | [K, 3] | float32 | Shared frame per root |
| `grid` | [11] | float32 | `linspace(0, 1, num_steps + 1)`, `num_steps = 10` |
| `C` | [K, 3] | float32 | Integrated offset in local frame; prior at t=0, final offset at t=1 |
| `vel_C` | [K, 3] | float32 | Predicted velocity at the current step |
| `nrc` | [G] | long | `num_root_children` per graph — from the GT eval graph's root degree |

---

## 18. Function Call Graph {#18-call-graph}

### Training Path
```
RandRedDataset._compute_apical_flag()                  (ONCE on the finest tree)
  └── per-level survivor_mask slicing -> batch.is_apical_root_child

Expansion.get_loss()
  └── Expansion._assemble_diffusion_inputs()
        ├── decode_parent_indices()
        ├── build_directed_edge_index()
        ├── select_training_leaf_indices()
        ├── leaf_rel_targets()
        ├── precompute_full_geometry(apical_flag=batch.is_apical_root_child)
        │     ├── _compute_tree_directions(apical_flag=...)   (ONCE: topology, ordering, v_in/v_out, projections)
        │     │     ├── _order_root_children_by_uhat(child0_override=apical) per root
        │     │     └── v_in[root_children] = fwd0 (shared frame)
        │     ├── compute_geo_order(_directions=dirs)         (ordinals: root from ordering, interior from sinψ)
        │     ├── compute_branch_angles_parent_centric(_directions=dirs)
        │     ├── assign_branch_angles_to_edges()
        │     ├── assign_parent_scalar_to_edges()
        │     └── compute_local_bases(_directions=dirs)       (trivial: forward = v_in_unit)
        ├── global_to_local()                                 (targets: global → local)
        └── [feature assembly: is_leaf | 23-wide one-hot | new_leaf_flag | padding]

  └── VFlowMatchingModel.forward()
        ├── _sample_time()                             (t per graph)
        ├── _pos_scale()                               (anisotropic prior_std_pos)
        ├── C_t = (1-t)*C_noise + t*C_0                (linear / OT path)
        ├── local_to_global()                          (C_t: local → global for P_t)
        ├── patch_geometry_for_noised_leaves()         (v_in_unit = local_forward, locked to P_0)
        ├── model.forward(tmd=, cell_class=, pre_geom=)
        ├── MSE(v_pred, C_0 - C_noise)                 (velocity loss, local frame)
        └── compute_flow_diagnostics(C1_implied, ...)  (stratified by t and root-child)
```

### Sampling Path
```
Expansion.sample_graphs(num_root_children=nrc, tmd=, cell_class=)
  └── while not terminated:
        Expansion.expand(num_root_children=nrc)
          ├── spawn count: k children for root (capped by max_tree_size)
          ├── ordinal tracking: ordinal_new, sib_count
          ├── geo_angle_new = i (raw child index) per new child
          ├── build_directed_edge_index()
          ├── compute_local_bases_for_leaves(
          │     child_ordinal=..., sibling_count=...)
          │     └── shared random frame: all root children get same θ_0
          ├── precompute_full_geometry()                  (ONCE on P_0; NO apical_flag)
          │     ├── _compute_tree_directions()
          │     ├── compute_geo_order()                   → geo_ordinal for internal nodes
          │     ├── compute_branch_angles_parent_centric()
          │     └── compute_local_bases()                 → local_forward for patch_geometry
          ├── override pre_geom_p0['local_forward/sideways'][leaves] = leaf_fwd/side
          │     └── critical for root children at step 0: replaces deterministic
          │       fallback with random shared frame from compute_local_bases_for_leaves
          ├── geo_feat_all = geo_ordinal (internal) + geo_angle_new (new leaves)
          ├── ★ axial_extent ORDINAL FREEZE: root children ← spawn-slot rank
          ├── [feature assembly: is_leaf | 23-wide one-hot | new_leaf_flag | padding]
          ├── VFlowMatchingModel.sample(pre_geom_p0=...)
          │     ├── C ~ N(0, diag(prior_std_pos^2))     (prior at t=0, local frame)
          │     └── for each of num_steps (=10) Euler steps:
          │           ├── local_to_global(C)             (local → global for P_cur)
          │           ├── patch_geometry_for_noised_leaves()  (CHEAP: leaves only)
          │           ├── model.forward(pre_geom=...)    (skips internal geometry)
          │           └── C = C + dt * vel_C             (no 1/(1-t), no DDIM)
          ├── local_to_global(C)                         (final: local → global)
          ├── pos update
          └── leaf_expansion = (exp_pred > 0.0) + 1;  forced to 1 at max_tree_size
```

---

## 19. Operational Envelope {#19-operational-envelope}

Mechanism that shapes every run but sits outside the geometry story.

**`max_tree_size` = 400** (`config/method/expansion.yaml`). Enforced in three places:
`sample_graphs` caps the loop at `max_steps = 2 · min(target_size.max(), max_tree_size)`; `expand()`
zeroes `spawn_counts_final` for any graph already at the cap; and after prediction it forces
`leaf_expansion = 1` (no expansion) for leaves in capped graphs. A graph that hits the cap stops
growing regardless of what the expansion head predicts.

**k is teacher-forced at sampling.** `training.py::_evaluate_rollout` takes
`num_root_children` straight from the GT eval graph:

```python
nrc_all = np.array([g.degree[g.graph["root"]] if "root" in g.graph else 2 for g in eval_graphs])
```

So generated neurons have **exactly** the GT root degree (measured: 7.74 vs 7.74). This is why the
apical failure mode is a **placement/identity** problem and never a count problem — worth
remembering before attributing any root-level metric gap to branching.

**Target sizes are likewise teacher-forced** from `len(g)` per eval graph, shuffled by `pred_perm`
to spread the size distribution across batches.

**RBF edge features** (`rbf_k`, default **0 = off**). When enabled, the per-edge scalar pair
`(rho, du)` is expanded through Gaussian RBF kernels into `2·rbf_k` channels with ranges given in
`pos_scale_factor`-normalized units (`rbf_rho_max`, `rbf_du_max`).

**Checkpoint-shape-critical knobs.** `edge_mlp_hidden: 1042` and `node_mlp_hidden: 512` are pinned
to legacy values in every parity config. They change `state_dict` shapes, so a checkpoint must be
loaded with the same values it was trained with. Same applies to `feats_dim`, `m_dim`,
`offset_head_hidden`, `tmd_hidden_dim`, `class_hidden_dim`, and `rbf_k`.

**`root_child_order` is checkpoint-critical too**, though not for shapes: sampling in a different
mode than the checkpoint trained in silently changes what ordinal 0 means. `main.py` validates the
value; nothing can validate it against the checkpoint.

> **Dead config key.** `use_global_fallback_frames: False` appears in every config and is read by
> **no code anywhere**. Removing it from configs would change nothing.

---

## 20. Function Architecture and Liveness {#20-function-architecture}

| Function | Status | Role |
|----------|--------|------|
| `_compute_tree_directions` | live | **Shared base**: tree topology, root ordering via `_order_root_children_by_uhat`, v_in/v_out, projections, normalization. Accepts `apical_flag`. Root children get shared fwd0 as v_in. |
| `_order_root_children_by_uhat` | live | Root child ordering: child_0 = `child0_override` (the apical) or the legacy lowest-uhat rule, then clockwise sort |
| `compute_geo_order` | live | Unified integer ordinal: root children get rank (0, 1, ..., k-1) from root_ordering, binary interior get 0/1 from sinψ L/R |
| `compute_branch_angles_parent_centric` | live | cosψ, sinψ, cosθ from shared directions (or standalone) |
| `compute_local_bases` | live | forward/sideways from shared v_in_unit (or standalone) |
| `compute_local_bases_for_leaves` | live | Sampling: shared random frame for root children |
| `precompute_full_geometry` | live | Orchestrates: `_compute_tree_directions` → `compute_geo_order` → branch angles → local bases |
| `patch_geometry_for_noised_leaves` | live | Patches leaf geometry per flow step; uses `local_forward` as locked v_in reference (both training and sampling) |
| `RandRedDataset._compute_apical_flag` | live | Marks the deepest-−uhat-subtree root child; `axial_extent` mode only |
| `Expansion.sample_graphs` | live | Rollout driver; `num_root_children`, `tmd`, `cell_class` params |
| `Expansion.expand` | live | k-child spawn; precompute; ordinal freeze; shared frame; one flow integration per step |
| `Expansion._assemble_diffusion_inputs` / `get_loss` | live | Feature assembly + loss entry point |
| `VFlowMatchingModel` (`flow_v.py`) | **live** | The denoiser in every current config |
| `FlowMatchingModel` (`flow.py`) | dormant | Data-prediction flow variant; the A/B partner to `flow_v` |
| `DenoisingDiffusionModel` (`basic.py`) | dormant | Original EDM-style diffusion; superseded, still selectable |
| `EDMDiffusionModel` (`edm.py`) | dormant | EDM variant; superseded, still selectable |
| `compute_geo_angle_for_new_leaves` | **test-only** | Was post-hoc ordinal refinement; now referenced only by `tests/test_so2_invariance.py` |
| `compute_geo_lr_for_new_leaves` | **dead** | Zero callers anywhere including tests; superseded by `compute_geo_order` |

### Deleted Functions
| Function | Reason |
|----------|--------|
| `compute_geo_lr_mask` | Absorbed into `compute_geo_order` |
| `compute_root_child_angles` | Absorbed into `compute_geo_order` |
| `compute_geo_lr_mask_f2` | Dead code (was never called) |

---

## 21. Code Anchor Table {#21-anchor-table}

**Verified `c7e469e`, 2026-08-31.** This is the only place in the document carrying line numbers —
prose cites `file::function`. Refresh this table alone after a refactor.

| Function / field | File | Line |
|---|---|---|
| `_compute_tree_directions` | `graph_generation/method/helpers.py` | 111 |
| `_order_root_children_by_uhat` | `graph_generation/method/helpers.py` | 248 |
| `global_to_local` / `local_to_global` | `graph_generation/method/helpers.py` | 333 / 357 |
| `compute_local_bases` | `graph_generation/method/helpers.py` | 382 |
| `compute_local_bases_for_leaves` | `graph_generation/method/helpers.py` | 477 |
| `compute_branch_angles_parent_centric` | `graph_generation/method/helpers.py` | 590 |
| `compute_geo_order` | `graph_generation/method/helpers.py` | 751 |
| `compute_geo_lr_for_new_leaves` (dead) | `graph_generation/method/helpers.py` | 865 |
| `compute_geo_angle_for_new_leaves` (test-only) | `graph_generation/method/helpers.py` | 998 |
| `precompute_full_geometry` | `graph_generation/method/helpers.py` | 1118 |
| `patch_geometry_for_noised_leaves` | `graph_generation/method/helpers.py` | 1200 |
| `MAX_CHILDREN = 23` | `graph_generation/method/expansion.py` | 17 |
| `Expansion.sample_graphs` | `graph_generation/method/expansion.py` | 85 |
| `Expansion.expand` | `graph_generation/method/expansion.py` | 174 |
| ↳ ordinal freeze block | `graph_generation/method/expansion.py` | 414 |
| `Expansion._assemble_diffusion_inputs` | `graph_generation/method/expansion.py` | 548 |
| `Expansion.get_loss` | `graph_generation/method/expansion.py` | 799 |
| `VFlowMatchingModel.forward` | `graph_generation/diffusion/flow_v.py` | 85 |
| `VFlowMatchingModel.sample` | `graph_generation/diffusion/flow_v.py` | 226 |
| `RandRedDataset._compute_apical_flag` | `graph_generation/data/reduction_dataset.py` | 35 |
| `num_root_children` computation | `graph_generation/data/reduction_dataset.py` | 119 |
| `ReducedGraphData` (field docs) | `graph_generation/data/data.py` | 9 |
| `SO2_EGNN_Network.forward` (conditioner injection) | `graph_generation/model/egnn_so2.py` | 752 |
| `_evaluate_rollout` (teacher-forced k) | `graph_generation/training.py` | 766 |

### Related documents

- `docs/APICAL_AXIAL_ERROR_MODE.md` — why the ordering criterion changed; the measured failure mode
- `docs/BUDGET_PARITY_SEMLAFLOW.md` — the parity protocol the `parity_*` configs implement
- `docs/TREE_DATASET_STATS.md` / `docs/NEURON_DATASET_STATS.md` — `pos_scale_factor` and `prior_std_pos` provenance
- `docs/VALIDATION_METRICS_SUMMARY.md` — what the eval suite measures
