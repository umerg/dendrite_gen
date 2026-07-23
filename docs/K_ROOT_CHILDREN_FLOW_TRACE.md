# K-Root Children Flow Trace: Training & Sampling

> **Update 2026-07-23 — `MAX_CHILDREN` raised 16 → 23.** The root-degree cap was removed:
> `expansion.py::MAX_CHILDREN = 23` = the max primary-dendrite count observed in the corpus, so no
> neuron is filtered by soma degree (dataset `neurons_conditional_full`). The mechanism below is
> unchanged — only the width. **Every "16" / `clamp(0, 15)` / "16-wide" / `[N, 16]` below is now 23 /
> `clamp(0, 22)` / 23-wide / `[N, 23]`** (the in-trace numbers are illustrative and were not rewritten).
>
> _Prior update 2026-07-06 — `MAX_CHILDREN` raised 10 → 16 (now superseded by 23 above)._

A comprehensive, line-by-line trace of how root nodes with k>2 children are handled across the entire pipeline — from data construction through geometry computation, diffusion noising/denoising, and loss/position recovery. Covers the design invariants (SO(2) equivariance, locked frames, ordinal features), the SO(2)-invariant ordering and relative-frame approach, and all edge cases for k=1, k=2, and k>2.

---

## Table of Contents

1. [Design Invariants and Key Decisions](#1-design-invariants)
2. [High-Level Overview: Training Path](#2-training-overview)
3. [High-Level Overview: Sampling Path](#3-sampling-overview)
4. [Phase 1: Data Pipeline — `num_root_children` and Reduction](#4-phase-1-data-pipeline)
5. [Phase 2: Shared Directions + Geometric Ordering — `_compute_tree_directions` + `compute_geo_order`](#5-phase-2-geometric-ordering)
   - [2a: SO(2)-Invariant Child Ordering via `_order_root_children_by_uhat`](#2a-so2-invariant-ordering)
   - [2b: Shared v_in and Reference Frame](#2b-shared-frame)
   - [2c: Ordinal Feature Assignment](#2c-ordinal-feature)
   - [2d: Binary Interior Children](#2d-binary-interior)
6. [Phase 3: Local Frames (Training) — `compute_local_bases`](#6-phase-3-training-frames)
7. [Phase 4: Full Geometry Precomputation — `precompute_full_geometry`](#7-phase-4-precompute)
8. [Phase 5: Training Feature Assembly — `get_loss()`](#8-phase-5-training-features)
   - [5a: `geo_ordinal` Feature (Replaces Binary L/R)](#5a-geo-ordinal-feature)
   - [5b: Local-Frame Target Conversion](#5b-local-frame-targets)
9. [Phase 6: Diffusion Forward (Training) — `DenoisingDiffusionModel.forward()`](#9-phase-6-diffusion-training)
   - [6a: Noise Sampling and Noising](#6a-noise-sampling)
   - [6b: Local-to-Global Conversion for P_t](#6b-local-to-global-training)
   - [6c: Geometry Patching — Root Children Frames Locked](#6c-geometry-patching)
   - [6d: Model Call and Loss](#6d-model-call-loss)
10. [Phase 7: Spawn Logic (Sampling) — `expand()`](#10-phase-7-spawn-logic)
    - [7a: Root Spawn Count from `num_root_children`](#7a-root-spawn-count)
    - [7b: Child Materialisation with Ordinal Tracking](#7b-child-materialisation)
    - [7c: Spawn-Order Ordinals for New Leaves](#7c-spawn-order-ordinals)
11. [Phase 8: Local Frames (Sampling) — `compute_local_bases_for_leaves`](#11-phase-8-sampling-frames)
    - [8a: Shared Random Frame for Root Children](#8a-shared-random-frame)
    - [8b: Legacy Fallback (No Ordinal Info)](#8b-legacy-fallback)
    - [8a: Precompute Geometry + Build Feature](#8a-precompute-sampling)
12. [Phase 9: Diffusion Denoising Loop (Sampling) — `DenoisingDiffusionModel.sample()`](#12-phase-9-diffusion-sampling)
    - [9a: Pure Noise Initialisation](#9a-noise-init)
    - [9b: Precomputed Geometry + Per-Step Patching](#9b-precompute-and-patch)
    - [9c: Local-to-Global Conversion at Each Step](#9c-local-to-global-sampling)
    - [9d: Geometry Patching vs Internal Computation](#9d-geometry-patching)
13. [Phase 10: Position Recovery](#13-phase-10-position-recovery)
15. [Training vs Sampling Consistency Analysis](#15-training-vs-sampling)
16. [Edge Cases: k=1, k=2, k>2](#16-edge-cases)
17. [Tensor Shape Reference](#17-tensor-shapes)
18. [Function Call Graph](#18-call-graph)

---

## 1. Design Invariants and Key Decisions {#1-design-invariants}

### SO(2) Equivariance — The Central Constraint

The entire pipeline preserves SO(2) equivariance around `uhat` (default: z-axis `[0,0,1]`). This means:

- **Neither ordering nor frames use global axes** (`global_inplane_basis`) for root children. Global axes are used **only** as a degenerate fallback (e.g., `v_in` parallel to `uhat` for non-root nodes) and for sampling frame generation (random angle needs a basis).
- **Child ordering is SO(2)-invariant**: child_0 is selected by lowest `uhat` component (tiebreak: largest perp-plane distance). Both criteria are unchanged by rotation around `uhat`.
- **The `forward` direction comes from geometry itself**: the direction from root to child_0, projected onto the plane perpendicular to `uhat`.
- Rotating the entire tree around `uhat` must produce identical ordinals, identical delta-theta values, and identical predictions (up to the same rotation).

### SO(2)-Invariant Ordering → Relative Frame

| Layer | Purpose | Uses Global Axes? | Enters Model? |
|-------|---------|-------------------|---------------|
| **Geometric ordering** | Label which child is child_0 (lowest uhat component) | **No** — uses `_order_root_children_by_uhat` (SO(2)-invariant) | No — only determines index assignment |
| **Shared frame** | Define `forward`/`sideways` basis for position prediction (shared by all root children) | No — uses root→child_0 direction | Yes — this is the local frame for diffusion |

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Spawn strategy** | All k children at once in one `expand()` step | Simplest; root has no prior children to reference |
| **k source** | From GT/metadata (`num_root_children` field) | No prediction head needed; k is a structural property |
| **Child_0 selection** | Lowest uhat component (tiebreak: largest perp distance) | SO(2)-invariant; backward-compatible with k=2 (lower Z = left = ordinal 0.0) |
| **Remaining children** | Ordered clockwise relative to child_0's perp direction | SO(2)-equivariant (relative to geometric reference, not global axis) |
| **Feature encoding** | 16D one-hot where child i lights up bit i (NOT `i/(k-1)` scalar, NOT angular `θ/2π`) | One-hot gives maximal separation between children regardless of k; does not leak GT angular information into features |
| **Reference direction** | Root → child_0 in perp plane | SO(2)-equivariant (co-rotates with positions) |
| **Frame stability** | Locked to P_0 positions during diffusion noising | Per-child frames must not drift with noise |
| **Sampling frames** | One random `forward` per root, shared by ALL children | SO(2)-equivariant — absolute orientation is free; model learns angular placement from one-hot child identity |
| **Global axes policy** | NEVER for root ordering/frames; ONLY for non-root degenerate fallbacks and sampling basis | Prevents SO(2) violation at the root level |
| **Backward compat** | Not needed | Full retrain acceptable |

### `global_inplane_basis` Usage Policy

After the SO(2) fix, `global_inplane_basis` is **never used for root child ordering or root child frames**. Its remaining usages are:

| Location | Status | Reason |
|----------|--------|--------|
| `_compute_tree_directions` (v_in_unit degenerate) | **KEPT** | True degenerate fallback when `nin ≤ eps` (v_in parallel to uhat or zero) |
| `compute_local_bases` standalone (forward ≈ 0) | **KEPT** | True degenerate fallback when called without `_directions` |
| `compute_local_bases_for_leaves` (sampling basis) | **KEPT** | Random rotation basis for sampling (equivariant in distribution since angle is random) |
| `compute_geo_angle_for_new_leaves` (non-root binary) | **KEPT** | Degenerate fallback for non-root binary parents with no grandparent |
| `compute_geo_lr_mask`, `compute_root_child_angles`, `compute_geo_lr_mask_f2` | **DELETED** | Absorbed into `_compute_tree_directions` + `compute_geo_order` |

### Ordinal Feature Table (16D One-Hot)

| k | Child indices | One-hot encoding (MAX_CHILDREN=16) |
|---|---------------|-------------------------------------|
| 1 | [0] | `[1,0,0,0,0,0,0,0,0,0]` |
| 2 | [0, 1] | `[1,0,...,0]`, `[0,1,0,...,0]` |
| 3 | [0, 1, 2] | bits 0, 1, 2 |
| 4 | [0, 1, 2, 3] | bits 0, 1, 2, 3 |
| k | [0, ..., k-1] | bit i for child i (k ≤ 16) |

Child index is **absolute, not normalized by k**: the 3rd child of k=5 and k=9 both encode as `[0,0,1,0,0,0,0,0,0,0]`. Binary interior children: left=bit 0, right=bit 1. Non-children (root, internal non-leaf) get all-zeros.

---

## 2. High-Level Overview: Training Path {#2-training-overview}

```
DataLoader (batch construction with num_root_children)
    |
    v
Expansion.get_loss(batch, model)
    |
    +-- decode_parent_indices(batch)              -> parent_idx [N], -1 for roots
    +-- build_directed_edge_index(parent_idx)     -> edge_index [2,E], edge_types [E]
    +-- select_training_leaf_indices(batch)        -> leaf_idx_train [L]
    +-- leaf_rel_targets(pos_gt, ...)             -> C_0 [L,3] (relative offsets in global frame)
    |
    +-- precompute_full_geometry(P_0, ...)        -> pre_geom_p0 dict (ONCE on clean P_0)
    |       |
    |       +-- _compute_tree_directions()        -> dirs dict (topology, v_in, v_out, projections, root ordering)
    |       |       |
    |       |       +-- _order_root_children_by_uhat() per root (SO(2)-invariant)
    |       |       +-- v_in[root_children] = fwd0 (shared frame, NOT per-child rotated)
    |       |       +-- v_in[interior] = pos[parent] - pos[grandparent]
    |       |       +-- perp projections + normalization (ONCE for all nodes)
    |       |
    |       +-- compute_geo_order(_directions=dirs) -> geo_ordinal [N], geo_delta_theta [N]
    |       |       +-- Root children: integer rank from root_ordering (0, 1, ..., k-1)
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
    |       (is_leaf, geo_onehot [16D] ★, new_leaf_flag, size_ratio, padding)
    |
    +-- Convert C_0 to local frame: global_to_local(C_0, leaf_fwd, leaf_side, uhat) -> leaf_rel_pos [L,3]
    |
    +-- DenoisingDiffusionModel.forward(...)
            |
            +-- sample sigma per graph
            +-- noise leaf_rel_pos -> C_t (in local frame, isotropic)
            +-- noise e_0 -> e_t
            +-- local_to_global(C_t) -> C_t_global
            +-- P_t[leaf_idx] = P_0[parent] + C_t_global
            +-- patch_geometry_for_noised_leaves(pre_geom_p0, P_t, ...)  -> pre_geom
            |       ★ Root children: v_in LOCKED to P_0-based frame (NOT updated with noised positions)
            +-- assemble x_in = [P_t | node_feats | e_feat | log_sigma]
            +-- model(x_in, ..., pre_geom=pre_geom) -> rel_pred, expansion_pred
            +-- loss = MSE(C_pred, C_0) + MSE(e_pred, e_0)
```

---

## 3. High-Level Overview: Sampling Path {#3-sampling-overview}

```
Expansion.sample_graphs(target_size, model, tmd, num_root_children)
    |
    +-- Initialize: pos=[0,0,0] per graph, leaf_expansion=1
    |
    +-- while not terminated:
    |     Expansion.expand(adj, batch, target_size, model, ..., num_root_children=nrc)
    |       |
    |       +-- Spawn count determination:
    |       |     ★ Root leaves: spawn_counts = num_root_children[graph_idx] (k children)
    |       |       Non-root: spawn_counts from leaf_expansion (0 or 2)
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
    |       |     geo_feat_all = geo_ordinal (internal nodes) + geo_angle_new (new leaves) → 16D one-hot
    |       |
    |       +-- Override pre_geom_p0['local_forward/sideways'][leaves] = leaf_fwd/side
    |       |     ★ Critical for root children at step 0: replaces deterministic fallback
    |       |       with random shared frame from compute_local_bases_for_leaves
    |       |
    |       +-- DenoisingDiffusionModel.sample(..., pre_geom_p0=pre_geom_p0)
    |       |     |
    |       |     +-- C = randn(L, 3) * sigma_max   (pure noise, local frame)
    |       |     +-- for each sigma step:
    |       |     |     C_global = local_to_global(C, leaf_fwd, leaf_side, uhat)
    |       |     |     P_cur[leaf] = parent_pos + C_global
    |       |     |     pre_geom_t = patch_geometry_for_noised_leaves(pre_geom_p0, P_cur, ...)
    |       |     |     model(P_cur, ..., pre_geom=pre_geom_t) -> C0_pred, e0_pred
    |       |     |     DDIM update: C = C0_pred + sigma_next * eps_C
    |       |     +-- return C0_pred, e0_pred (in local frame)
    |       |
    |       +-- Position update:
    |       |     rel_global = local_to_global(C0_pred, leaf_fwd, leaf_side, uhat)
    |       |     pos[leaf] = parent_pos + rel_global
    |       |
    |       +-- Expansion thresholding: leaf_expansion = (exp_pred > threshold) + 1
    |
    +-- Unbatch into list[nx.Graph]
```

---

## 4. Phase 1: Data Pipeline — `num_root_children` and Reduction {#4-phase-1-data-pipeline}

**File**: `graph_generation/data/reduction_dataset.py:63-65`
**File**: `graph_generation/data/data.py:26`

### 4.1 Computing `num_root_children` in the Dataset

When building a `ReducedGraphData` sample from a reduction sequence, the dataset computes how many children the root node has:

```python
# reduction_dataset.py:63-65 (_build_reduced_graph_data)
root_idx = graph._state.root
num_root_children = int(np.sum(parent_idx == root_idx)) if root_idx is not None else 0
```

This counts all nodes whose parent is the root. For a binary tree with root having 2 children: `num_root_children = 2`. For SWC neuron data where the soma (root) may have 3+ dendrite origins: `num_root_children = k`.

### 4.2 Storage in `ReducedGraphData`

```python
# data.py — ReducedGraphData fields (docstring line 26)
#   - num_root_children: scalar int, branching factor of the root node (k)
```

The field is passed to the `ReducedGraphData` constructor as a scalar int, which gets converted to a `th.long` tensor (line 47: `th.tensor(int(value), dtype=th.long)`).

### 4.3 Batching Behavior

`num_root_children` is a scalar tensor per graph. Under PyG batching, scalar tensors are concatenated into a 1D tensor of length `num_graphs`. No `__inc__` offset is needed (it's not an index).

**Result**: After batching, `batch.num_root_children` is a `[num_graphs]` LongTensor where `batch.num_root_children[g]` gives the root branching factor of graph `g`.

---

## 5. Phase 2: Shared Directions + Geometric Ordering {#5-phase-2-geometric-ordering}

**Files**: `graph_generation/method/helpers.py` — `_compute_tree_directions` (lines 111-233) and `compute_geo_order` (lines 724-832)

This phase is handled by two functions called in sequence from `precompute_full_geometry`:
1. **`_compute_tree_directions`**: Computes tree topology, root ordering, and ALL direction vectors (v_in, v_out, projections) **once**. Root children get a **shared** v_in = fwd0 (root→child_0 direction).
2. **`compute_geo_order`**: Assigns ordinal features for all children — root children from root ordering, binary interior from sinψ-based L/R.

### 2a: SO(2)-Invariant Child Ordering via `_order_root_children_by_uhat` {#2a-so2-invariant-ordering}

**Purpose**: Determine which child is child_0 using an **SO(2)-invariant** criterion. Called inside `_compute_tree_directions` for each root.

The helper `_order_root_children_by_uhat` (`helpers.py:235-305`) handles the ordering logic:

**Step-by-step**:
1. **Project onto uhat axis**: `uhat_components = (offsets · uhat)` → [k] scalar per child
2. **Project onto perp plane**: `offsets_perp = offsets - uhat_components * uhat` → [k, 3]
3. **Compute perp distances**: `perp_dist = ||offsets_perp||` → [k]
4. **Select child_0**: Lowest uhat component. Tiebreaker: largest perp distance. Both are SO(2)-invariant.
5. **Build reference direction**: `fwd0 = offsets_perp[child0]` → forward direction from root to child_0 in perp plane
6. **Sort remaining children**: clockwise by `atan2` relative to `fwd0`
7. **Compute delta_angles**: angle of each child relative to `fwd0` (child_0 gets 0.0)

**Why SO(2)-invariant**: uhat components, perp distances, and angles relative to fwd0 are all unchanged by rotation around uhat.

### 2b: Shared v_in and Reference Frame {#2b-shared-frame}

After root ordering, `_compute_tree_directions` sets the v_in for **all root children to the same fwd0**:

```python
# _compute_tree_directions (helpers.py:190-192)
# Shared frame: ALL children get the same fwd0 as v_in
for c in children.tolist():
    v_in[c] = fwd0_unit
```

This means ALL root children of a given root share the same `forward` direction (after perp projection and normalization). The 16D one-hot ordinal feature is the only cue telling the model which child it is.

**`geo_delta_theta`** is still computed (for ordinal ordering and post-diffusion refinement) but is **NOT used for frame rotation**. All children share fwd0 as their frame.

### 2c: Ordinal Feature Assignment {#2c-ordinal-feature}

`compute_geo_order` reads root ordering from `_directions` and assigns integer ordinals:

```python
# compute_geo_order (helpers.py:788-793)
for r, (sorted_idx, _fwd0, delta_angles, children) in root_ordering.items():
    k = children.numel()
    for rank, si in enumerate(sorted_idx.tolist()):
        c = children[si]
        geo_ordinal[c] = float(rank)  # integer child index (0, 1, ..., k-1)
        geo_delta_theta[c] = delta_angles[si]
```

| Rank | geo_ordinal | One-hot bit |
|------|-------------|-------------|
| 0 | 0.0 | bit 0 |
| 1 | 1.0 | bit 1 |
| 2 | 2.0 | bit 2 |
| ... | ... | ... |
| k-1 | k-1.0 | bit k-1 |

**Critical design choices**:
1. The ordinal encodes **rank in the geometric ordering**, NOT the actual angular position `θ/2π`. Using `θ/2π` would leak GT angular information into the feature.
2. The ordinal is an **absolute child index** — the 3rd child (rank=2) of k=5 and k=9 both get `geo_ordinal=2.0` → one-hot `[0,0,1,0,...,0]`. This is NOT normalized by k.
3. Converted to a **16D one-hot vector** in feature assembly (MAX_CHILDREN=16), giving maximal separation between any two children regardless of k.

### 2d: Binary Interior Children {#2d-binary-interior}

For non-root parents with exactly 2 children, `compute_geo_order` computes L/R from sinψ (using pre-computed `v_in_unit` and `v_out_unit` from shared directions):

```python
# compute_geo_order (helpers.py:799-830)
# Case 1: opposite-sign sines → left = sin > 0 → ordinal 0.0
# Case 2: same-sign or near-zero → left = larger atan2 angle → ordinal 0.0
geo_ordinal[case_nodes] = (~is_left).to(dtype)  # left→0.0, right→1.0
```

This is consistent with the one-hot scheme: left=index 0 (bit 0), right=index 1 (bit 1). The sinψ values come from the shared `_compute_tree_directions` output — no separate `compute_geo_lr_mask` call needed.

---

## 6. Phase 3: Local Frames (Training) — `compute_local_bases` {#6-phase-3-training-frames}

**File**: `graph_generation/method/helpers.py:358-450`

This function computes per-node local coordinate frames (`forward`, `sideways`) used for SO(2)-equivariant position prediction.

When called with `_directions` from `_compute_tree_directions` (the normal training path), this function is trivial:

```python
# helpers.py:383-391
if _directions is not None:
    forward = _directions['v_in_unit']
    sideways = uhat × forward
    return {'local_forward': forward, 'local_sideways': sideways}
```

All the heavy lifting — grandparent lookup, v_in computation, root children shared frame, perp projection, normalization, degenerate fallback — is done in `_compute_tree_directions`.

### How root children get their frame

All root children share `forward = fwd0` (root→child_0 direction, projected and normalized). This was set in `_compute_tree_directions`:

```python
# _compute_tree_directions (helpers.py:190-192)
for c in children.tolist():
    v_in[c] = fwd0_unit  # SAME for all children of this root
```

The 16D one-hot ordinal feature is the only thing that differentiates children for the model. During training, each child's target offset `C_0 = global_to_local(pos[child] - pos[root], fwd0, side0, uhat)` naturally has different forward/sideways components because children are at different angular positions. The model learns to predict these different offsets based on the one-hot child identity.

### Interior nodes

For non-root nodes with grandparent: `v_in = pos[parent] - pos[grandparent]` (unchanged, computed in `_compute_tree_directions`).

---

## 7. Phase 4: Full Geometry Precomputation — `precompute_full_geometry` {#7-phase-4-precompute}

**File**: `graph_generation/method/helpers.py:1079-1158`

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

**File**: `graph_generation/method/expansion.py:460-696`

### 5a: One-Hot Ordinal Feature (Replaces Scalar `i/(k-1)`) {#5a-geo-ordinal-feature}

```python
# expansion.py — get_loss() feature assembly
MAX_CHILDREN = 16
geo_ordinal = pre_geom_p0['geo_ordinal'].to(device=pos_gt.device, dtype=pos_gt.dtype)
geo_idx = geo_ordinal.long().clamp(0, MAX_CHILDREN - 1)
geo_onehot = pos_gt.new_zeros((N_nodes, MAX_CHILDREN))
child_mask = geo_ordinal >= 0  # sentinel -1 → all-zeros (non-children)
geo_onehot[child_mask] = geo_onehot[child_mask].scatter_(1, geo_idx[child_mask].unsqueeze(-1), 1.0)
features.append(geo_onehot)
feats_used += MAX_CHILDREN
```

**Feature vector layout** (per node, `avail_feats_dim` slots):

| Slots | Feature | Shape | Description |
|-------|---------|-------|-------------|
| 0 | `is_leaf` | [N, 1] | 1.0 if node is a leaf, 0.0 otherwise |
| 1-16 | `geo_onehot` | [N, 16] | One-hot child index: bit i = 1.0 for child i. All-zeros for non-children (sentinel -1). |
| 11 | `new_leaf_flag` | [N, 1] | 1.0 if node is a "new leaf from next level" |
| 12 | `size_ratio` | [N, 1] | `current_size / total_tree_size` |
| 13+ | padding | [N, ...] | zeros |

**Dimension budget**: `avail_feats_dim = feats_dim(64) - cond_dim(2) - tmd_hidden_dim(32) = 30`. Features use 13 slots, leaving 17 for padding.

**Change from previous**: Slots 1-16 were previously a single scalar `geo_ordinal.clamp(min=0.0)` — a continuous value `i/(k-1)` in [0, 1]. For k=8 this gave only 0.143 separation between adjacent children, too weak for the model to distinguish them. The 16D one-hot gives maximal (orthogonal) separation regardless of k.

### 5b: Local-Frame Target Conversion {#5b-local-frame-targets}

The model predicts parent-relative offsets in the **local frame**. Ground-truth targets must be converted:

```python
# expansion.py:544
leaf_rel_pos_global = leaf_rel_targets(pos_gt, leaf_idx_train, leaf_parent_idx)  # [L, 3]

# expansion.py:559-562
leaf_fwd = local_fwd[leaf_idx_train]      # [L, 3] from precomputed frames
leaf_side = local_side[leaf_idx_train]     # [L, 3]
leaf_rel_pos = global_to_local(leaf_rel_pos_global, leaf_fwd, leaf_side, uhat)  # [L, 3]
```

**For a root child**: `leaf_fwd` is `fwd0` (shared by all root children). The global offset `pos[child] - pos[root]` is decomposed into `(forward_component, sideways_component, axial_component)` in this shared frame. Each child gets different components because they're at different angular positions — the one-hot ordinal feature tells the model which angular offset to predict.

---

## 9. Phase 6: Diffusion Forward (Training) — `DenoisingDiffusionModel.forward()` {#9-phase-6-diffusion-training}

**File**: `graph_generation/diffusion/basic.py:30-134`

### 6a: Noise Sampling and Noising {#6a-noise-sampling}

```python
# basic.py:66-78
sigma_graph = (th.randn((num_graphs,)) * P_std + P_mean).exp()
sigma_graph = sigma_graph.clamp(sigma_min, sigma_max)   # per-graph noise level
sigma_leaf = sigma_graph[leaf_batch].view(-1, 1)         # per-leaf noise level

eps_pos = th.randn_like(C_0)       # [L, 3] position noise
eps_exp = th.randn_like(e_0)       # [L, 1] expansion noise

C_t = C_0 + sigma_leaf * eps_pos   # noised local-frame offsets
e_t = e_0 + sigma_leaf * eps_exp   # noised expansion values
```

**Key**: Noising is **isotropic in the local frame**. `C_0` is already in local coordinates (forward, sideways, axial), so adding isotropic Gaussian noise preserves the frame's meaning. Each root child's noise is independent in its own per-child frame.

### 6b: Local-to-Global Conversion for P_t {#6b-local-to-global-training}

```python
# basic.py:81-84
if local_forward is not None and local_sideways is not None and uhat is not None:
    C_t_global = local_to_global(C_t, local_forward, local_sideways, uhat)
    P_t[leaf_idx_train] = P_0[leaf_parent_idx] + C_t_global
```

Where `local_to_global` reconstructs:
```
offset_global = C_t[:, 0:1] * forward + C_t[:, 1:2] * sideways + C_t[:, 2:3] * uhat
```

**For root child i**: Its noised position is `pos[root] + local_to_global(C_t_i, fwd_i, side_i, uhat)` where `fwd_i` is the per-child forward rotated by `Δθ_i`.

### 6c: Geometry Patching — Root Children Frames Locked {#6c-geometry-patching}

```python
# basic.py:90-95
if pre_geom_p0 is not None:
    with th.no_grad():
        pre_geom = patch_geometry_for_noised_leaves(
            pre_geom_p0, P_t, leaf_idx_train, parent_idx,
            edge_index, model.uhat,
        )
```

Inside `patch_geometry_for_noised_leaves` (`helpers.py:1161-1275`):

**What gets patched** (for noised leaf positions):
- `v_out_new` = `P_t[leaf] - P_t[parent[leaf]]` — the outgoing direction is recomputed from noised positions
- `cospsi_leaf`, `sinpsi_leaf`, `cos_theta_leaf` — branch angles recomputed using `v_in_unit` from `local_forward` and `v_out_unit` from noised positions
- Edge-level `rel_coors`, `r_perp`, `rho`, `du` — recomputed for affected edges

**What stays locked** (the critical invariant):
```python
# helpers.py:1192-1194
# v_in direction reused from P_0 via local_forward (already projected,
# normalized, and degenerate-fallbacked — locked to P_0 frame).
v_in_unit = pre_geom_p0['local_forward'][leaf_idx_train]  # (L, 3)
```

The branch angle reference direction `v_in_unit` is read directly from `pre_geom_p0['local_forward']` — the per-node forward basis computed at P_0 time. This is already projected onto the perp plane, normalized, and has degenerate fallback applied. It is **reused unchanged** even though leaf positions have been noised.

**Why `local_forward` instead of raw `v_in`**: Previously, `patch_geometry` stored raw `v_in` in `pre_geom_p0` and re-projected/normalized it internally. Since `local_forward` is derived from `v_in` (it IS the projected, normalized version with degenerate fallback), using it directly eliminates redundant computation and ensures the degenerate fallback (for root children) is always applied consistently.

**Why this matters**: If `v_in` were updated with noised `v_out`, the local frame would drift with each noise realization, breaking the correspondence between the frame the model sees and the frame the targets were computed in.

### 6d: Model Call and Loss {#6d-model-call-loss}

```python
# basic.py:97-134
node_feats_t = th.cat([node_feats, e_feat, log_sigma_node], dim=-1)
x_in = th.cat([P_t, node_feats_t], dim=-1)

out = model(x=x_in, edge_index=edge_index, batch=batch,
            edge_attr=edge_attr, parent_idx=parent_idx,
            tmd=tmd, pre_geom=pre_geom)

C_pred = out["rel_pred"][leaf_idx_train]       # [L, 3] predicted local offsets
e_pred = out["expansion_pred"][leaf_idx_train]  # [L, 1] predicted expansion

pos_loss = F.mse_loss(C_pred, C_0)    # compare to clean local-frame targets
exp_loss = F.mse_loss(e_pred, e_0)    # compare to clean expansion values
```

The model receives noised positions `P_t` and the patched geometry (with locked root-child frames). It predicts the clean local-frame offset `C_0` and clean expansion `e_0`. The loss is MSE in the **local frame**.

---

## 10. Phase 7: Spawn Logic (Sampling) — `expand()` {#10-phase-7-spawn-logic}

**File**: `graph_generation/method/expansion.py:138-455`

### 7a: Root Spawn Count from `num_root_children` {#7a-root-spawn-count}

```python
# expansion.py:205-221
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
# expansion.py:246-265
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
# expansion.py:271-272
ordinal_t = th.tensor(ordinal_new, device=device, dtype=th.float)
geo_angle_new = ordinal_t  # raw integer child index: 0, 1, ..., k-1
```

This gives each new child its spawn-order ordinal as a raw integer index:
- k=1 child: 0
- k=2 children: 0, 1
- k=3 children: 0, 1, 2
- k=8 children: 0, 1, 2, 3, 4, 5, 6, 7

These integer indices are later converted to 16D one-hot vectors in feature assembly. New leaves are at placeholder positions, so no real geometry is available. For internal nodes (from previous steps), the ordinal feature comes from `precompute_full_geometry`'s `geo_ordinal` which is computed from real positions (see Phase 8a).

---

## 11. Phase 8: Per-Child Local Frames (Sampling) — `compute_local_bases_for_leaves` {#11-phase-8-sampling-frames}

**File**: `graph_generation/method/helpers.py:453-560`

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
# expansion.py:344-348
leaf_fwd, leaf_side = compute_local_bases_for_leaves(
    pos_new, parent_idx_new_0b, leaf_parent_idx_next, model.uhat,
    child_ordinal=child_ordinal_t,
    sibling_count=sib_count_long,
)
```

### Standard non-root children

For children whose parent has a grandparent (non-root):
```python
# helpers.py:316-324
gp = parent_idx[leaf_parent_idx]  # grandparent
v_in[sel] = pos[leaf_parent_idx[sel]] - pos[gp[sel]]  # grandparent → parent direction
```
This is unchanged — the incoming direction at the parent is inherited from the grandparent.

### Root children (degenerate case)

When k children are spawned at root, they are all at the parent (root) position initially. The parent (root) has no grandparent. This triggers the **degenerate** path where `v_in` is zero → `nin ≤ eps`.

```python
# helpers.py:326-340
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
# helpers.py:528-546
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

**For k=3 with random θ_0 = 0.7 rad**: ALL 3 children get `forward = cos(0.7)*e1 + sin(0.7)*e2`. The 16D one-hot ordinal (`[1,0,0,...,0]`, `[0,1,0,...,0]`, `[0,0,1,...,0]`) is the only cue telling the model which angular offset to predict for each child.

**SO(2) equivariance**: Since `θ_0` is random, the absolute orientation carries no information. The model learns angular placement from the one-hot child identity alone.

### 8b: Legacy Fallback (No Ordinal Info) {#8b-legacy-fallback}

```python
# helpers.py:547-555
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

### 8a: Precompute Geometry + Build Feature (NEW) {#8a-precompute-sampling}

After computing local bases, `expand()` precomputes full geometry and builds the ordinal node feature:

```python
# expansion.py:350-359
with th.no_grad():
    pre_geom_p0 = precompute_full_geometry(
        pos_new, parent_idx_new_0b, edge_index, model.uhat,
    )

# geo_ordinal for internal nodes (real positions), spawn ordinals for new leaves
geo_feat_all = pre_geom_p0['geo_ordinal'].clamp(min=0.0).clone()
geo_feat_all[leaf_idx_next] = geo_angle_new  # raw child index (0, 1, ..., k-1)

# Feature assembly: convert to 16D one-hot
MAX_CHILDREN = 16
geo_idx = geo_feat_all.long().clamp(0, MAX_CHILDREN - 1)
geo_onehot = pos_new.new_zeros((N, MAX_CHILDREN))
child_mask = pre_geom_p0['geo_ordinal'] >= 0
child_mask[leaf_idx_next] = True  # new leaves always get one-hot
geo_onehot[child_mask] = geo_onehot[child_mask].scatter_(1, geo_idx[child_mask].unsqueeze(-1), 1.0)
```

**Why override new leaves?** `precompute_full_geometry` computes `geo_ordinal` for ALL nodes, but new leaves are at placeholder positions (= parent pos), making their computed ordinals meaningless. The spawn-order indices (0, 1, 2 for k=3) are used instead.

**Why use `geo_ordinal` for internal nodes?** These nodes have finalized positions from prior expansion steps. `precompute_full_geometry` computes their ordinals from real geometry (SO(2)-invariant ordering for root children, sinψ-based L/R for binary interior). This replaces the old `geo_angle` state that was carried between expansion steps.

#### Critical: Override `local_forward`/`local_sideways` for Leaves

After precomputing geometry, `expand()` overrides the leaf entries in `pre_geom_p0` with the frames from `compute_local_bases_for_leaves`:

```python
# expansion.py:355-365
pre_geom_p0['local_forward'] = pre_geom_p0['local_forward'].clone()
pre_geom_p0['local_sideways'] = pre_geom_p0['local_sideways'].clone()
pre_geom_p0['local_forward'][leaf_idx_next] = leaf_fwd
pre_geom_p0['local_sideways'][leaf_idx_next] = leaf_side
```

**Why this is necessary**: `precompute_full_geometry` internally calls `compute_local_bases`, which handles degenerate root children (placeholder positions at step 0) with a **deterministic** `global_inplane_basis(e1)` fallback. But `compute_local_bases_for_leaves` handles the same case with a **random** shared frame (`base_theta = torch.rand(...)`). Without this override, `patch_geometry_for_noised_leaves` (which reads `pre_geom_p0['local_forward']` as its v_in reference) would use the deterministic `e1` direction, while the local↔global coordinate conversion in `diffusion.sample()` uses `leaf_fwd` (random direction). The model would see branch angles computed relative to one frame but produce predictions interpreted in a different frame — an inconsistency that corrupts the SO(2) geometric signal.

**For non-root steps (step > 0)**: Leaf positions are real (not placeholders), so both functions compute the same direction from grandparent→parent geometry. The override is a no-op.

**`pre_geom_p0`** is then passed to `diffusion.sample()` for efficient per-step geometry patching (see Phase 9).

---

## 12. Phase 9: Diffusion Denoising Loop (Sampling) — `DenoisingDiffusionModel.sample()` {#12-phase-9-diffusion-sampling}

**File**: `graph_generation/diffusion/basic.py:145-270`

### 9a: Pure Noise Initialisation {#9a-noise-init}

```python
# basic.py:185-189
sigmas = self.make_sigma_schedule(self.num_steps, self.sigma_max, self.sigma_min, device)
sigma_init = float(sigmas[0].item())

L = leaf_idx.numel()
C = th.randn((L, 3), device=device) * sigma_init   # [L, 3] pure noise in local frame
e = th.randn((L, 1), device=device) * sigma_init    # [L, 1] expansion noise
```

**Critical**: Diffusion starts from **pure noise** `C ~ N(0, σ_max²)`. The placeholder positions (all children at parent pos) are irrelevant — what matters is:
1. The local frame (`leaf_fwd`, `leaf_side`) defines how noise maps to 3D space
2. All root children share the SAME random forward (from Phase 8a), with independent noise

### 9b: Precomputed Geometry + Per-Step Patching {#9b-precompute-and-patch}

**Optimization**: Rather than recomputing ALL geometry from scratch at every diffusion step, `sample()` receives `pre_geom_p0` (precomputed on P_0 in `expand()`) and patches only leaf-affected quantities at each step:

```python
# basic.py:201-253
for step in range(self.num_steps):
    sigma_cur = float(sigmas[step].item())
    sigma_next = float(sigmas[step + 1].item())
    log_sigma = math.log(max(sigma_cur, 1e-12))

    P_cur = P_0.clone()

    # Convert local-frame C to global positions
    if local_forward is not None and local_sideways is not None and uhat is not None:
        C_global = local_to_global(C, local_forward, local_sideways, uhat)
        P_cur[leaf_idx] = parent_pos + C_global

    # Patch precomputed P_0 geometry for noised leaf positions
    pre_geom_t = None
    if pre_geom_p0 is not None:
        pre_geom_t = patch_geometry_for_noised_leaves(
            pre_geom_p0, P_cur, leaf_idx, parent_idx,
            edge_index, uhat,
        )

    # Assemble features and call model
    ...
    out = model(x=x_in, edge_index=edge_index, batch=batch,
                edge_attr=edge_attr, parent_idx=parent_idx,
                pre_geom=pre_geom_t, **model_kwargs)

    C0_pred = out["rel_pred"][leaf_idx]
    e0_pred = out["expansion_pred"][leaf_idx]

    # DDIM-style update
    inv_sigma = 1.0 / max(sigma_cur, 1e-12)
    eps_C = (C - C0_pred) * inv_sigma
    eps_e = (e - e0_pred) * inv_sigma
    C = C0_pred + sigma_next * eps_C
    e = e0_pred + sigma_next * eps_e

return C0_pred, e0_pred
```

**This mirrors the training path**: Training uses `precompute_full_geometry` on P_0 + `patch_geometry_for_noised_leaves` per noise sample. Sampling now does the same — geometry is computed once on P_0, then only leaf-affected quantities are patched at each denoising step.

### 9c: Local-to-Global Conversion at Each Step {#9c-local-to-global-sampling}

At each diffusion step, `C` (in local frame) is converted to global positions:

```python
C_global = local_to_global(C, local_forward, local_sideways, uhat)
P_cur[leaf_idx] = parent_pos + C_global
```

For root child i with shared random forward:
```
P_cur[child_i] = pos[root] + C[i,0]*forward + C[i,1]*sideways + C[i,2]*uhat
```

**The local frames are FIXED throughout all diffusion steps**. They were computed once in `compute_local_bases_for_leaves` and do not change. This is consistent with training, where frames are locked to P_0.

### 9d: Geometry Patching vs Internal Computation {#9d-geometry-patching}

When `pre_geom_p0` is provided (normal sampling path), `patch_geometry_for_noised_leaves` patches only:
- **Node-level**: Branch angles (cosψ, sinψ, cosθ) for leaves only, using `local_forward` as locked v_in reference
- **Edge-level**: `rel_coors`, `r_perp`, `rho`, `du` for edges touching leaves

Everything for internal nodes (positions fixed from prior expansion steps) is reused from `pre_geom_p0` without recomputation. The model skips its internal `_compute_static_so2_geometry()` when `pre_geom` is provided.

**Performance**: For a tree with N=100 nodes and L=10 new leaves, patching touches ~10 nodes and ~20 edges instead of recomputing all ~100 nodes and ~200 edges. Over ~60 diffusion steps, this is significant.

**Root children at step 0**: `pre_geom_p0['local_forward']` at leaf positions has been overridden with `leaf_fwd` from `compute_local_bases_for_leaves` (see Phase 8a). This ensures `patch_geometry_for_noised_leaves` uses the same random shared frame as the local↔global conversion — the branch angles and the coordinate transform are always consistent.

---

## 13. Phase 10: Position Recovery {#13-phase-10-position-recovery}

**File**: `graph_generation/method/expansion.py:419-423`

After diffusion returns `C0_pred` (clean offset prediction in local frame):

```python
# expansion.py:419-423
rel_pred_global = local_to_global(rel_pred, leaf_fwd, leaf_side, model.uhat)
parent_pos_for_children = pos_new[leaf_parent_idx_next]
pos_new[leaf_idx_next] = parent_pos_for_children + rel_pred_global
```

**For root child i** (all share same forward from Phase 8a):
```
final_pos[child_i] = pos[root] + C0_pred[i,0]*forward + C0_pred[i,1]*sideways + C0_pred[i,2]*uhat
```

The one-hot ordinal feature told the model which angular offset to predict for each child. All children share the same frame, but predict different offsets based on their one-hot identity.

**No post-diffusion ordinal refinement**: Previously, `compute_geo_angle_for_new_leaves()` was called here to re-order children based on final positions. This is no longer needed because `precompute_full_geometry` at the START of the next expansion step computes `geo_ordinal` from the now-finalized positions of these nodes. The ordinal feature for internal nodes is always fresh from `precompute_full_geometry`, not carried forward as state.

---

## 15. Training vs Sampling Consistency Analysis {#15-training-vs-sampling}

| Aspect | Training | Sampling |
|--------|----------|----------|
| **Frame source** | `_compute_tree_directions` → `compute_local_bases` | `compute_local_bases_for_leaves` with random θ_0 |
| **All root children forward** | fwd0 (root→child_0 direction from GT, shared) | Random direction in perp plane (shared) |
| **Feature** | 16D one-hot from integer `geo_ordinal` (from SO(2)-invariant GT ordering) | 16D one-hot: `geo_ordinal` for internal nodes (from `precompute_full_geometry`), raw spawn-order index for new leaves |
| **Noising** | C_0 + σ * ε in local frame | Start from pure noise σ_max * ε in local frame |
| **Frame stability** | Locked to P_0 (via `patch_geometry_for_noised_leaves`) | Locked to P_0 (via `patch_geometry_for_noised_leaves`) |
| **Geometry computation** | Precomputed once (`precompute_full_geometry`), patched per noise sample | Precomputed once (`precompute_full_geometry`), patched per diffusion step |
| **Branch angle reference** | `local_forward` from P_0 (real positions) | `local_forward` from `compute_local_bases_for_leaves` (random shared frame at step 0, real positions at step > 0) — overridden into `pre_geom_p0` before diffusion |

### Why the shared frame works

**Training**: All root children share fwd0. Each child's target offset `C_0 = global_to_local(pos[child]-pos[root], fwd0, side0, uhat)` has different forward/sideways components because children are at different angular positions. The one-hot identity tells the model which offset to predict.

**Sampling**: All root children share a random forward. The model has learned from training that one-hot bit 0 means "predict offset in the fwd0 direction" and bit 2 means "predict offset at some learned angle from fwd0". The absolute orientation is irrelevant (SO(2)-equivariant).

**The consistency invariant holds because**: Both training and sampling use the **same shared-frame structure** (all root children share one forward direction), the **same one-hot ordinal feature scheme** (child i → bit i, regardless of k), the **same local_to_global / global_to_local conversions**, and now the **same precompute + patch geometry pattern**.

---

## 16. Edge Cases: k=1, k=2, k>2 {#16-edge-cases}

### k=1 (Single Root Child)

| Phase | Behavior |
|-------|----------|
| **Ordering** | No sorting needed. `geo_ordinal = 0` (one-hot: bit 0), `geo_delta_theta = 0.0` |
| **Training frame** | `v_in = fwd0 = pos[child] - pos[root]` (perp-projected) |
| **Sampling frame** | Random forward (single child, degenerate path) |
| **Feature** | `[1,0,0,0,0,0,0,0,0,0]` (one-hot bit 0) |
| **Spawn** | `spawn_counts_final = 1 * has_capacity` |

### k=2 (Binary Root)

| Phase | Behavior |
|-------|----------|
| **Ordering** | Lowest uhat component = child_0. `geo_ordinal = [0, 1]` (one-hot: bits 0, 1) |
| **Training frame** | Both children share `fwd = fwd0` (root→child_0 direction) |
| **Sampling frame** | Both children share same random forward |
| **Feature** | `[1,0,...,0]` and `[0,1,0,...,0]` (one-hot bits 0 and 1) |
| **Spawn** | `spawn_counts_final = 2 * has_capacity` |

### k>2 (Multi-Child Root)

| Phase | Behavior |
|-------|----------|
| **Ordering** | Lowest uhat component = child_0, remaining clockwise relative to child_0. `geo_ordinal = [0, 1, 2, ..., k-1]` (one-hot: bits 0 through k-1) |
| **Training frame** | ALL children share `fwd = fwd0` (root→child_0 direction) |
| **Sampling frame** | ALL children share same random forward |
| **Feature** | 16D one-hot, bit i for child i — orthogonal separation regardless of k |
| **Spawn** | `spawn_counts_final = k * has_capacity` |

### Degenerate Cases

| Scenario | Handling |
|----------|----------|
| **Child on uhat axis** (offset ⊥ perp plane is zero) | `fwd0_norm ≤ eps` → sort by uhat ascending, `delta_angles = 0` |
| **All children at root** (sampling, pre-diffusion) | Degenerate path → shared random forward for all children |
| **`v_in` parallel to uhat** | `global_inplane_basis` fallback in `_compute_tree_directions` |
| **`num_root_children` = None** | Legacy spawn: exactly 1 child per root |

---

## 17. Tensor Shape Reference {#17-tensor-shapes}

### Training Tensors

| Tensor | Shape | Type | Source |
|--------|-------|------|--------|
| `pos_gt` (P_0) | [N, 3] | float32 | Batch |
| `parent_idx` | [N] | long | Decoded from `parent_idx_1b` |
| `geo_ordinal` | [N] | float32 | `compute_geo_order` |
| `geo_delta_theta` | [N] | float32 | `_compute_tree_directions` / `compute_geo_order` |
| `local_forward` | [N, 3] | float32 | `compute_local_bases` |
| `local_sideways` | [N, 3] | float32 | `compute_local_bases` |
| `leaf_rel_pos` (C_0) | [L, 3] | float32 | `global_to_local(leaf_rel_pos_global)` |
| `leaf_fwd` | [L, 3] | float32 | `local_forward[leaf_idx_train]` |
| `leaf_side` | [L, 3] | float32 | `local_sideways[leaf_idx_train]` |
| `geo_onehot` | [N, 16] | float32 | 16D one-hot from `geo_ordinal` integer index (MAX_CHILDREN=16) |
| `num_root_children` | [G] | long | Batch (per-graph scalar) |

### Sampling Tensors

| Tensor | Shape | Type | Source |
|--------|-------|------|--------|
| `pos_new` | [N', 3] | float32 | Accumulated, updated by diffusion |
| `geo_feat_all` | [N'] | float32 | `geo_ordinal` integer index for internal nodes (from `precompute_full_geometry`), raw child index for new leaves |
| `geo_angle_new` | [K] | float32 | Raw child index (0, 1, ..., k-1) for new leaves only |
| `pre_geom_p0` | dict | — | Full geometry precomputed on P_0, passed to `sample()` for per-step patching |
| `ordinal_new` | [K] | long | 0 to k-1 per child group |
| `sibling_count_new` | [K] | long | k for each child in group |
| `child_ordinal_t` | [K] | long | Passed to `compute_local_bases_for_leaves` |
| `sib_count_long` | [K] | long | Passed to `compute_local_bases_for_leaves` |
| `leaf_fwd` | [K, 3] | float32 | Shared frame per root (random forward) |
| `leaf_side` | [K, 3] | float32 | Shared frame per root |
| `C` | [K, 3] | float32 | Noised offset in local frame (evolves during diffusion) |
| `C0_pred` | [K, 3] | float32 | Clean offset prediction (local frame) |
| `nrc` | [G] | long | `num_root_children` per graph |

---

## 18. Function Call Graph {#18-call-graph}

### Training Path
```
Expansion.get_loss()
  ├── decode_parent_indices()
  ├── build_directed_edge_index()
  ├── select_training_leaf_indices()
  ├── leaf_rel_targets()
  ├── precompute_full_geometry()
  │     ├── _compute_tree_directions()                 (ONCE: topology, root ordering, v_in, v_out, projections)
  │     │     ├── _order_root_children_by_uhat() per root
  │     │     └── v_in[root_children] = fwd0 (shared frame)
  │     ├── compute_geo_order(_directions=dirs)        (ordinals: root from ordering, interior from sinψ)
  │     ├── compute_branch_angles_parent_centric(_directions=dirs)
  │     ├── assign_branch_angles_to_edges()
  │     ├── assign_parent_scalar_to_edges()
  │     └── compute_local_bases(_directions=dirs)      (trivial: forward = v_in_unit)
  ├── global_to_local()                                (targets: global → local)
  ├── [feature assembly with 16D one-hot from geo_ordinal]
  └── DenoisingDiffusionModel.forward()
        ├── local_to_global()                          (C_t: local → global for P_t)
        ├── patch_geometry_for_noised_leaves()         (v_in_unit = local_forward, locked to P_0)
        ├── model.forward()
        └── MSE loss in local frame
```

### Sampling Path
```
Expansion.sample_graphs(num_root_children=nrc)
  └── while not terminated:
        Expansion.expand(num_root_children=nrc)
          ├── spawn count: k children for root
          ├── ordinal tracking: ordinal_new, sib_count
          ├── geo_angle_new = i (raw child index) per new child
          ├── build_directed_edge_index()
          ├── compute_local_bases_for_leaves(
          │     child_ordinal=..., sibling_count=...)
          │     └── shared random frame: all root children get same θ_0
          ├── precompute_full_geometry()                  (ONCE on P_0)
          │     ├── _compute_tree_directions()
          │     ├── compute_geo_order()                   → geo_ordinal for internal nodes
          │     ├── compute_branch_angles_parent_centric()
          │     └── compute_local_bases()                 → local_forward for patch_geometry
          ├── override pre_geom_p0['local_forward/sideways'][leaves] = leaf_fwd/side
          │     └── critical for root children at step 0: replaces deterministic
          │       fallback with random shared frame from compute_local_bases_for_leaves
          ├── geo_feat_all = geo_ordinal (internal) + geo_angle_new (new leaves) → 16D one-hot
          ├── [feature assembly with 16D one-hot geo_onehot]
          ├── DenoisingDiffusionModel.sample(pre_geom_p0=...)
          │     ├── C = randn * sigma_max               (pure noise, local frame)
          │     └── for each step:
          │           ├── local_to_global(C)             (local → global for P_cur)
          │           ├── patch_geometry_for_noised_leaves()  (CHEAP: leaves only)
          │           ├── model.forward(pre_geom=...)    (skips internal geometry)
          │           └── DDIM update
          ├── local_to_global(C0_pred)                   (final: local → global)
          └── pos update
```

---

## Summary of Current Function Architecture

| Function | File | Role |
|----------|------|------|
| `_compute_tree_directions` | helpers.py:111 | **Shared base**: tree topology, root ordering via `_order_root_children_by_uhat`, v_in/v_out, projections, normalization. Root children get shared fwd0 as v_in. |
| `_order_root_children_by_uhat` | helpers.py:235 | SO(2)-invariant root child ordering: child_0 = lowest uhat component, clockwise sort |
| `compute_geo_order` | helpers.py:724 | Unified integer ordinal: root children get rank (0, 1, ..., k-1) from root_ordering, binary interior get 0/1 from sinψ L/R. Converted to 16D one-hot in feature assembly. |
| `compute_branch_angles_parent_centric` | helpers.py:567 | cosψ, sinψ, cosθ from shared directions (or standalone) |
| `compute_local_bases` | helpers.py:358 | forward/sideways from shared v_in_unit (or standalone) |
| `compute_local_bases_for_leaves` | helpers.py:453 | Sampling: shared random frame for root children |
| `precompute_full_geometry` | helpers.py:1079 | Orchestrates: `_compute_tree_directions` → `compute_geo_order` → branch angles → local bases |
| `patch_geometry_for_noised_leaves` | helpers.py:1161 | Patches leaf geometry for diffusion; uses `local_forward` as locked v_in reference (both training and sampling) |
| `compute_geo_angle_for_new_leaves` | helpers.py:964 | Post-diffusion ordinal refinement using `_order_root_children_by_uhat` (used in training via `get_loss`, no longer called in sampling) |
| `Expansion.sample_graphs` | expansion.py:52 | Added `num_root_children` param; no `geo_angle` state (ordinals from `precompute_full_geometry`) |
| `Expansion.expand` | expansion.py:138 | k-child spawn; precomputes geometry; 16D one-hot from `geo_ordinal` + spawn-order index; shared frame via `compute_local_bases_for_leaves` |
| `Expansion.get_loss` | expansion.py:460 | 16D one-hot from integer `geo_ordinal` (child i → bit i) |
| `ReducedGraphData` | data.py:9 | Added `num_root_children` field |

### Deleted Functions
| Function | Reason |
|----------|--------|
| `compute_geo_lr_mask` | Absorbed into `compute_geo_order` |
| `compute_root_child_angles` | Absorbed into `compute_geo_order` |
| `compute_geo_lr_mask_f2` | Dead code (was never called) |
