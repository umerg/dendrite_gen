# K-Root Children Flow Trace: Training & Sampling

A comprehensive, line-by-line trace of how root nodes with k>2 children are handled across the entire pipeline — from data construction through geometry computation, diffusion noising/denoising, and loss/position recovery. Covers the design invariants (SO(2) equivariance, locked frames, ordinal features), the SO(2)-invariant ordering and relative-frame approach, and all edge cases for k=1, k=2, and k>2.

---

## Table of Contents

1. [Design Invariants and Key Decisions](#1-design-invariants)
2. [High-Level Overview: Training Path](#2-training-overview)
3. [High-Level Overview: Sampling Path](#3-sampling-overview)
4. [Phase 1: Data Pipeline — `num_root_children` and Reduction](#4-phase-1-data-pipeline)
5. [Phase 2: Geometric Ordering — `compute_root_child_angles`](#5-phase-2-geometric-ordering)
   - [2a: SO(2)-Invariant Child Ordering via `_order_root_children_by_uhat`](#2a-so2-invariant-ordering)
   - [2b: Relative-Frame Angles (SO(2)-Equivariant)](#2b-relative-frame-angles)
   - [2c: Ordinal Feature Assignment](#2c-ordinal-feature)
   - [2d: Binary Interior Children (Unchanged)](#2d-binary-interior)
6. [Phase 3: Per-Child Local Frames (Training) — `compute_local_bases`](#6-phase-3-training-frames)
   - [3a: Child 0 Frame](#3a-child0-frame)
   - [3b: Children 1..k-1 Rotated Frames](#3b-rotated-frames)
   - [3c: Legacy Fallback (No `geo_delta_theta`)](#3c-legacy-fallback)
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
    - [7c: Initial `geo_angle` Computation](#7c-initial-geo-angle)
11. [Phase 8: Per-Child Local Frames (Sampling) — `compute_local_bases_for_leaves`](#11-phase-8-sampling-frames)
    - [8a: Random Forward for Child 0](#8a-random-forward)
    - [8b: Structured Rotation for Children 1..k-1](#8b-structured-rotation)
    - [8c: Legacy Fallback (No Ordinal Info)](#8c-legacy-fallback)
12. [Phase 9: Diffusion Denoising Loop (Sampling) — `DenoisingDiffusionModel.sample()`](#12-phase-9-diffusion-sampling)
    - [9a: Pure Noise Initialisation](#9a-noise-init)
    - [9b: Per-Step Denoising in Local Frame](#9b-per-step-denoise)
    - [9c: Local-to-Global Conversion at Each Step](#9c-local-to-global-sampling)
    - [9d: Model Geometry (Internal, No Precompute)](#9d-model-geometry)
13. [Phase 10: Position Recovery and `geo_angle` Refinement](#13-phase-10-position-recovery)
14. [Phase 11: `compute_geo_angle_for_new_leaves` (Post-Diffusion)](#14-phase-11-geo-angle-refine)
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
| **Relative frame** | Define per-child `forward`/`sideways` basis for position prediction | No — uses root→child_0 direction | Yes — this is the local frame for diffusion |

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Spawn strategy** | All k children at once in one `expand()` step | Simplest; root has no prior children to reference |
| **k source** | From GT/metadata (`num_root_children` field) | No prediction head needed; k is a structural property |
| **Child_0 selection** | Lowest uhat component (tiebreak: largest perp distance) | SO(2)-invariant; backward-compatible with k=2 (lower Z = left = ordinal 0.0) |
| **Remaining children** | Ordered clockwise relative to child_0's perp direction | SO(2)-equivariant (relative to geometric reference, not global axis) |
| **Feature encoding** | Ordinal `i/(k-1)`, NOT angular `θ/2π` | Ordinal does not leak GT angular information into features |
| **Reference direction** | Root → child_0 in perp plane | SO(2)-equivariant (co-rotates with positions) |
| **Frame stability** | Locked to P_0 positions during diffusion noising | Per-child frames must not drift with noise |
| **Sampling frames** | Random `forward` for child 0, then `2πi/k` rotations | SO(2)-equivariant — absolute orientation is free |
| **Global axes policy** | NEVER for root ordering/frames; ONLY for non-root degenerate fallbacks and sampling basis | Prevents SO(2) violation at the root level |
| **Backward compat** | Not needed | Full retrain acceptable |

### `global_inplane_basis` Usage Policy

After the SO(2) fix, `global_inplane_basis` is **never used for root child ordering or root child frames**. Its remaining usages are:

| Location | Status | Reason |
|----------|--------|--------|
| `compute_root_child_angles` | **REMOVED** | Replaced by `_order_root_children_by_uhat` |
| `compute_geo_angle_for_new_leaves` (root branch) | **REMOVED** | Replaced by `_order_root_children_by_uhat` |
| `compute_geo_lr_mask` / `_f2` (root fallback) | **REMOVED** | Replaced with `v_in = v_out` (geometric) |
| `compute_geo_lr_mask` / `_f2` (v_in_perp ≈ 0 fallback) | **KEPT** | True degenerate fallback for non-root nodes (geometric rarity) |
| `compute_local_bases` (forward ≈ 0 fallback) | **KEPT** | True degenerate fallback (v_in parallel to uhat) |
| `compute_local_bases_for_leaves` (sampling basis) | **KEPT** | Random rotation basis for sampling (equivariant in distribution since angle is random) |
| `compute_geo_angle_for_new_leaves` (non-root binary) | **KEPT** | Degenerate fallback for non-root binary parents with no grandparent |

### Ordinal Feature Table

| k | Child indices | Ordinal values `i/(k-1)` |
|---|---------------|--------------------------|
| 1 | [0] | [0.0] |
| 2 | [0, 1] | [0.0, 1.0] |
| 3 | [0, 1, 2] | [0.0, 0.5, 1.0] |
| 4 | [0, 1, 2, 3] | [0.0, 0.333, 0.667, 1.0] |
| k | [0, ..., k-1] | [0.0, ..., 1.0] |

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
    |       +-- compute_geo_lr_mask()             -> geo_lr_mask [N] bool (binary L/R, still needed for interior)
    |       +-- compute_root_child_angles()       -> geo_ordinal [N], geo_delta_theta [N]  ★ NEW
    |       |       |
    |       |       +-- _order_root_children_by_uhat() (SO(2)-invariant, no global axes)
    |       |       +-- child_0 = lowest uhat component (tiebreak: largest perp dist)
    |       |       +-- Relative angles Δθ_i from child 0's direction
    |       |       +-- Ordinal feature i/(k-1)
    |       |
    |       +-- compute_branch_angles_parent_centric() -> (cospsi, sinpsi, cos_theta) [N,1]
    |       +-- edge SO(2) decomposition          -> rel_coors, r_perp, rho, du
    |       +-- assign_branch_angles_to_edges()
    |       +-- compute_local_bases(geo_delta_theta=...) -> local_forward [N,3], local_sideways [N,3]  ★ MODIFIED
    |               |
    |               +-- Per-child frames for root children using geo_delta_theta
    |               +-- Child 0: forward = root→child_0 direction
    |               +-- Child i: forward = child_0's forward rotated by Δθ_i
    |
    +-- assemble node_feats [N, avail_feats_dim]
    |       (is_leaf, geo_ordinal ★, new_leaf_flag, size_ratio, padding)
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
    +-- Initialize: pos=[0,0,0] per graph, geo_angle=-1.0 (sentinel), leaf_expansion=1
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
    |       |       geo_angle_new[i] = i / max(k-1, 1)
    |       |
    |       +-- compute_local_bases_for_leaves(pos, parent_idx, ...,
    |       |       child_ordinal=ordinal_t, sibling_count=sib_count_long)  ★ MODIFIED
    |       |     |
    |       |     +-- Root children (degenerate: all at parent pos):
    |       |     |     Child 0: random forward in perp plane
    |       |     |     Child i: forward = child_0's forward rotated by 2πi/k
    |       |     +-- Non-root: grandparent→parent direction (standard)
    |       |
    |       +-- DenoisingDiffusionModel.sample(...)
    |       |     |
    |       |     +-- C = randn(L, 3) * sigma_max   (pure noise, local frame)
    |       |     +-- for each sigma step:
    |       |     |     C_global = local_to_global(C, leaf_fwd, leaf_side, uhat)
    |       |     |     P_cur[leaf] = parent_pos + C_global
    |       |     |     model(P_cur, ...) -> C0_pred, e0_pred
    |       |     |     DDIM update: C = C0_pred + sigma_next * eps_C
    |       |     +-- return C0_pred, e0_pred (in local frame)
    |       |
    |       +-- Position update:
    |       |     rel_global = local_to_global(C0_pred, leaf_fwd, leaf_side, uhat)
    |       |     pos[leaf] = parent_pos + rel_global
    |       |
    |       +-- compute_geo_angle_for_new_leaves(pos, parent_idx, leaf_idx)  ★ NEW
    |       |     Refine geo_angle using actual (denoised) positions
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

## 5. Phase 2: Geometric Ordering — `compute_root_child_angles` {#5-phase-2-geometric-ordering}

**File**: `graph_generation/method/helpers.py:671-796`

This function implements SO(2)-invariant ordering for root children, then relative-frame angles for frame construction. **No global axes are used** — ordering is based entirely on uhat-component projections and geometric relationships.

**Signature**:
```python
def compute_root_child_angles(
    pos: th.Tensor,           # [N, 3] node positions (GT, clean P_0)
    parent_idx: th.Tensor,    # [N] 0-based, -1 for roots
    uhat: th.Tensor,          # [3] SO(2) axis
    geo_lr_mask: th.Tensor,   # [N] bool, True = left (from compute_geo_lr_mask)
    *,
    eps: float = 1e-8,
) -> Tuple[th.Tensor, th.Tensor]:  # (geo_ordinal [N], geo_delta_theta [N])
```

**Returns**:
- `geo_ordinal [N]`: float tensor. `i/(k-1)` for root children, `0.0`/`1.0` for binary L/R, `-1.0` sentinel for root/parentless nodes.
- `geo_delta_theta [N]`: float tensor. Relative angle from child 0 for root children (radians). `0.0` for all other nodes.

### 2a: SO(2)-Invariant Child Ordering via `_order_root_children_by_uhat` {#2a-so2-invariant-ordering}

**Purpose**: Determine which child is child_0 using an **SO(2)-invariant** criterion. No global axes are involved — child selection and ordering are based on uhat-component projections and perpendicular-plane distances, both of which are unchanged by rotation around `uhat`.

The shared helper `_order_root_children_by_uhat` (`helpers.py:111-181`) handles all the logic:

```python
# helpers.py:111-181
def _order_root_children_by_uhat(
    offsets: th.Tensor,  # [k, 3] offsets from root to each child
    uhat: th.Tensor,     # [3]
    eps: float = 1e-8,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """SO(2)-invariant ordering of root children.
    child_0 = lowest uhat component (tiebreak: largest perp-plane distance).
    Returns: sorted_idx [k], fwd0_unit [3], delta_angles [k]
    """
```

For each root node `r` with children, `compute_root_child_angles` calls this helper:

```python
# helpers.py:817-836
for r in root_indices.tolist():
    children = (parent == r).nonzero(as_tuple=False).flatten()
    k = children.numel()

    offsets = pos[children] - pos[r].unsqueeze(0)  # [k, 3]
    sorted_idx, _fwd0, delta_angles = _order_root_children_by_uhat(
        offsets, uhat, eps=eps,
    )

    for rank, si in enumerate(sorted_idx.tolist()):
        c = children[si]
        geo_ordinal[c] = rank / max(k - 1, 1)
        geo_delta_theta[c] = delta_angles[si]
```

**Inside `_order_root_children_by_uhat` — step-by-step**:

1. **Project onto uhat axis**: `uhat_components = (offsets · uhat)` → [k] scalar per child
2. **Project onto perp plane**: `offsets_perp = offsets - uhat_components * uhat` → [k, 3]
3. **Compute perp distances**: `perp_dist = ||offsets_perp||` → [k]
4. **Select child_0**: The child with the **lowest uhat component**. Tiebreaker: largest perpendicular-plane distance.
   ```python
   min_uhat = uhat_components.min()
   is_min = uhat_components <= min_uhat + eps
   tied_perp = th.where(is_min, perp_dist, -1.0)
   child0_local = tied_perp.argmax()
   ```
5. **Build reference direction**: `fwd0 = offsets_perp[child0]` → forward direction from root to child_0 in perp plane
6. **Build local 2D basis**: `side0 = uhat × fwd0_unit`
7. **Compute angles**: For each child, `angle = atan2(proj_side, proj_fwd)` relative to `fwd0`
8. **Sort clockwise**: `sorted_idx = argsort(-angles)`, ensuring child_0 is first

**Why this is SO(2)-invariant**:
- `uhat_component = (offset · uhat)` is unchanged by rotation around `uhat` ✓
- `perp_distance = ||offset_perp||` is unchanged by rotation around `uhat` ✓
- `fwd0 = offsets_perp[child0]` co-rotates with positions (equivariant) ✓
- Angles measured relative to `fwd0` are invariant ✓

**Degenerate case**: If `fwd0_norm ≤ eps` (all children on the uhat axis), children are sorted by ascending uhat component, `fwd0_unit` is zero, and all `delta_angles` are 0.0.

**Backward compatibility with k=2**: Lowest uhat component = child_0 = ordinal 0.0. This matches the pre-existing binary convention where lower Z = left = ordinal 0.0.

### 2b: Relative-Frame Angles (SO(2)-Equivariant) {#2b-relative-frame-angles}

The `_order_root_children_by_uhat` helper returns both `sorted_idx` and `delta_angles` together. The relative angles are computed as part of the ordering:

```python
# Inside _order_root_children_by_uhat (helpers.py:161-178)
fwd0_unit = fwd0 / fwd0_norm
side0 = th.cross(uhat, fwd0_unit)

# Angle of each child relative to fwd0
proj_fwd = (offsets_perp * fwd0_unit).sum(dim=-1)    # [k]
proj_side = (offsets_perp * side0).sum(dim=-1)        # [k]
angles_from_fwd = th.atan2(proj_side, proj_fwd)       # [k]

# delta_angles[child0] = 0.0 by construction
delta_angles = angles_from_fwd.clone()
delta_angles[child0_local] = 0.0
```

**Step-by-step**:
1. `fwd0` = root → child_0 direction projected onto perp plane. This is the **reference direction** — it comes from geometry, not global axes.
2. `side0` = `uhat × fwd0_unit` — completes the local 2D basis in the perp plane.
3. For each child i, `Δθ_i = atan2(proj_side, proj_fwd)` gives the angle relative to `fwd0`.
4. Child 0 always has `Δθ_0 = 0.0` by construction.

**Key invariant**: `geo_delta_theta` is computed from **GT positions (P_0)** and remains fixed throughout the training forward pass. It is NOT recomputed when leaves are noised.

### 2c: Ordinal Feature Assignment {#2c-ordinal-feature}

```python
# helpers.py:785
geo_ordinal[c] = rank / max(k - 1, 1)
```

| Rank | k=1 | k=2 | k=3 | k=4 |
|------|-----|-----|-----|-----|
| 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1 | — | 1.0 | 0.5 | 0.333 |
| 2 | — | — | 1.0 | 0.667 |
| 3 | — | — | — | 1.0 |

**Critical design choice**: The ordinal feature encodes **rank in the geometric ordering**, NOT the actual angular position `θ/2π`. Using `θ/2π` would leak GT angular information into the feature, making the model memorize angles rather than learn to predict them.

### 2d: Binary Interior Children (Unchanged) {#2d-binary-interior}

For non-root parents with exactly 2 children, the existing `geo_lr_mask` is mapped to ordinals:

```python
# helpers.py:724-727
binary_interior = has_parent & ~parent_is_root & (counts[parent_clamped] == 2)
if binary_interior.any():
    geo_ordinal[binary_interior] = (~geo_lr_mask[binary_interior]).to(dtype)
    # left (True) → 0.0, right (False) → 1.0
```

This is consistent with the k=2 ordinal scheme: left=0.0, right=1.0.

---

## 6. Phase 3: Per-Child Local Frames (Training) — `compute_local_bases` {#6-phase-3-training-frames}

**File**: `graph_generation/method/helpers.py:160-277`

This function computes per-node local coordinate frames (`forward`, `sideways`) used for SO(2)-equivariant position prediction.

**Signature** (modified):
```python
def compute_local_bases(
    pos, parent_idx, uhat, geo_lr_mask, eps=1e-8,
    geo_delta_theta: th.Tensor | None = None,  # ★ NEW parameter
) -> dict:  # {'local_forward': [N,3], 'local_sideways': [N,3]}
```

### Standard nodes (has grandparent)

For any node `i` with parent `p` and grandparent `gp`:
```
v_in[i] = pos[p] - pos[gp]   (incoming direction at parent)
```
This is the same as before — no change for interior nodes.

### 3a: Child 0 Frame {#3a-child0-frame}

For root children, child 0 (the one with `geo_delta_theta ≈ 0`) gets:

```python
# helpers.py:227-233
if geo_delta_theta is not None:
    child0_local_idx = geo_delta_theta[children].abs().argmin()
child0 = children[child0_local_idx]
ref_dir = pos[child0] - pos[r]   # root → child_0 direction (3D)
```

This `ref_dir` is then projected onto the perp plane and used as the forward direction:

```python
# helpers.py:237-243
du = (ref_dir * uhat.view(-1)).sum()
ref_perp = ref_dir - du * uhat
ref_norm = ref_perp.norm()
if ref_norm > eps:
    fwd0 = ref_perp / ref_norm                # forward for child 0
    side0 = torch.cross(uhat, fwd0)           # sideways for child 0
    side0 = side0 / (side0.norm() + eps)
```

Then `v_in[child0] = ref_dir` — which after the projection step (lines 258-262) yields `forward[child0] = fwd0`.

### 3b: Children 1..k-1 Rotated Frames {#3b-rotated-frames}

Each subsequent child gets a **rotated** forward direction:

```python
# helpers.py:244-246
for c in children.tolist():
    dt = geo_delta_theta[c].item()
    v_in[c] = torch.cos(torch.tensor(dt)) * fwd0 + torch.sin(torch.tensor(dt)) * side0
```

**Geometric interpretation**: Child i's `v_in` is `fwd0` rotated by `Δθ_i` around `uhat` in the perp plane. After the global projection/normalization step, this becomes:
- `forward[child_i]` = fwd0 rotated by Δθ_i
- `sideways[child_i]` = uhat × forward[child_i]

**Result**: Each root child has its **own** local frame. The model sees positions expressed in these distinct frames, allowing it to learn child-specific offsets.

### 3c: Legacy Fallback (No `geo_delta_theta`) {#3c-legacy-fallback}

When `geo_delta_theta` is None (backward compatibility), all root children share the left-child direction:

```python
# helpers.py:252-254
# Legacy fallback: all children share left-child direction
for c in children.tolist():
    v_in[c] = ref_dir
```

### Final Projection (All Nodes)

After v_in is set, all nodes undergo the same projection:

```python
# helpers.py:257-277
uhat_vec = uhat.view(1, -1)
du_in = (v_in * uhat_vec).sum(dim=-1, keepdim=True)
v_in_perp = v_in - du_in * uhat_vec
nin = v_in_perp.norm(dim=-1, keepdim=True)
forward = v_in_perp / (nin + eps)

# Degenerate fallback
degenerate = nin.squeeze(-1) <= eps
if degenerate.any():
    e1, _ = global_inplane_basis(uhat, eps=eps)
    forward[degenerate] = e1.unsqueeze(0)      # ← ONLY for degenerate cases

sideways = torch.cross(uhat_vec.expand_as(forward), forward, dim=-1)
sideways = sideways / (sideways.norm(dim=-1, keepdim=True) + eps)
```

**Note**: `global_inplane_basis` is used ONLY as a degenerate fallback (when v_in is parallel to uhat or zero). In normal operation, the forward direction comes from geometry.

---

## 7. Phase 4: Full Geometry Precomputation — `precompute_full_geometry` {#7-phase-4-precompute}

**File**: `graph_generation/method/helpers.py:1169-1245`

Called once on clean P_0 positions during training. Orchestrates all geometry computation:

```python
def precompute_full_geometry(pos, parent_idx, edge_index, uhat, *, eps=1e-8, tol=1e-6, debug=False):
    # 1. Binary L/R assignment (still needed for interior children)
    geo_lr_mask = compute_geo_lr_mask(pos, parent_idx, uhat=uhat, ...)

    # 1b. Root child angular ordering and per-child relative angles  ★ NEW
    geo_ordinal, geo_delta_theta = compute_root_child_angles(
        pos, parent_idx, uhat, geo_lr_mask, eps=eps,
    )

    # 2. Branch angles with intermediates
    cospsi_node, sinpsi_node, cos_theta_node, intermediates = \
        compute_branch_angles_parent_centric(pos, parent_idx, uhat, return_intermediates=True)

    # 3. Edge SO(2) decomposition
    src, dst = edge_index
    rel_coors = pos[dst] - pos[src]
    du = (rel_coors @ uhat)
    r_par = du[:, None] * uhat
    r_perp = rel_coors - r_par
    rho = r_perp.norm(dim=-1, keepdim=True).clamp_min(eps)

    # 4. Edge angle assignment
    cospsi_edge, sinpsi_edge = assign_branch_angles_to_edges(edge_index, parent_idx, ...)
    cos_theta_edge = assign_parent_scalar_to_edges(edge_index, parent_idx, ...)

    # 5. Local bases with per-child root frames  ★ MODIFIED
    local_bases = compute_local_bases(
        pos, parent_idx, uhat, geo_lr_mask, eps=eps,
        geo_delta_theta=geo_delta_theta,             # ★ passed through
    )

    return {
        # Edge-level (for SO2_EGNN layers)
        'rel_coors', 'r_perp', 'rho', 'du',
        'cospsi_edge', 'sinpsi_edge', 'cos_theta_edge',
        # Node-level
        'cospsi_node', 'sinpsi_node', 'cos_theta_node',
        'geo_lr_mask',
        'geo_ordinal',       # ★ NEW
        'geo_delta_theta',   # ★ NEW
        # Intermediates for patching
        'v_in', 'v_out', 'has_gp',
        # Local bases
        'local_forward', 'local_sideways',
    }
```

**Critical**: This is computed **once** on P_0 and cached. The `geo_ordinal` and `geo_delta_theta` values are derived from GT positions and do NOT change during diffusion noising.

---

## 8. Phase 5: Training Feature Assembly — `get_loss()` {#8-phase-5-training-features}

**File**: `graph_generation/method/expansion.py:460-696`

### 5a: `geo_ordinal` Feature (Replaces Binary L/R) {#5a-geo-ordinal-feature}

```python
# expansion.py:600-605
# Geometry-derived ordinal feature for siblings (replaces binary L/R)
if feats_used < avail_feats_dim:
    geo_ordinal = pre_geom_p0['geo_ordinal'].to(device=pos_gt.device, dtype=pos_gt.dtype)
    geo_feat = geo_ordinal.clamp(min=0.0).unsqueeze(-1)  # sentinel -1.0 → 0.0
    features.append(geo_feat)
    feats_used += 1
```

**Feature vector layout** (per node, `avail_feats_dim` slots):

| Slot | Feature | Shape | Description |
|------|---------|-------|-------------|
| 0 | `is_leaf` | [N, 1] | 1.0 if node is a leaf, 0.0 otherwise |
| 1 | `geo_ordinal` | [N, 1] | `i/(k-1)` clamped to [0, 1] (sentinel -1 → 0.0) |
| 2 | `new_leaf_flag` | [N, 1] | 1.0 if node is a "new leaf from next level" |
| 3 | `size_ratio` | [N, 1] | `current_size / total_tree_size` |
| 4+ | padding | [N, ...] | zeros |

**Change from previous**: Slot 1 was previously `geo_lr_mask.float()` (binary 0.0/1.0). Now it's `geo_ordinal.clamp(min=0.0)` — a continuous value in [0, 1] that gracefully handles any k.

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

**For a root child**: `leaf_fwd` is the per-child forward direction (rotated by `Δθ_i` from child 0). The global offset `pos[child] - pos[root]` is decomposed into `(forward_component, sideways_component, axial_component)` in this child's own frame.

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

Inside `patch_geometry_for_noised_leaves` (`helpers.py:1248-1362`):

**What gets patched** (for noised leaf positions):
- `v_out_new` = `P_t[leaf] - P_t[parent[leaf]]` — the outgoing direction is recomputed from noised positions
- `cospsi_leaf`, `sinpsi_leaf`, `cos_theta_leaf` — branch angles recomputed
- Edge-level `rel_coors`, `r_perp`, `rho`, `du` — recomputed for affected edges

**What stays locked** (the critical invariant):
```python
# helpers.py:1288-1290
# Root-child leaves: v_in is locked to the P_0-based per-child frame
# (computed in compute_local_bases with geo_delta_theta). Do NOT update
# with noised v_out — the frame must remain stable through diffusion.
```

The `v_in` for root children comes from `pre_geom_p0['v_in']` which was computed from P_0 positions in `compute_branch_angles_parent_centric`. This is **reused unchanged** even though leaf positions have been noised.

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

### 7c: Initial `geo_angle` Computation {#7c-initial-geo-angle}

```python
# expansion.py:282-289
ordinal_t = th.tensor(ordinal_new, device=device, dtype=th.float)
sib_count_t = th.tensor(sibling_count_new, device=device, dtype=th.float)
geo_angle_new = ordinal_t / sib_count_t.clamp(min=2.0).sub(1.0)  # i / (k-1)

# For k=1: sib_count_t - 1 = 0 → clamp avoids div by zero; result is 0.0
single_child_mask = sib_count_t == 1
if single_child_mask.any():
    geo_angle_new[single_child_mask] = 0.0
```

This gives each new child its initial ordinal feature:
- k=1 child: 0.0
- k=2 children: 0.0 and 1.0
- k=3 children: 0.0, 0.5, 1.0

The `geo_angle_next` tensor is then passed to the feature assembly.

---

## 11. Phase 8: Per-Child Local Frames (Sampling) — `compute_local_bases_for_leaves` {#11-phase-8-sampling-frames}

**File**: `graph_generation/method/helpers.py:280-391`

Called during `expand()` to compute local frames for newly spawned leaf nodes.

**Signature** (modified):
```python
def compute_local_bases_for_leaves(
    pos, parent_idx, leaf_parent_idx, uhat, eps=1e-8,
    child_ordinal: th.Tensor | None = None,   # ★ NEW
    sibling_count: th.Tensor | None = None,    # ★ NEW
) -> tuple[th.Tensor, th.Tensor]:  # (leaf_fwd [L, 3], leaf_side [L, 3])
```

**Call site** in `expand()`:
```python
# expansion.py:395-399
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

### 8a: Random Forward for Child 0 {#8a-random-forward}

The degenerate fallback with ordinal info:

```python
# helpers.py:355-377
if child_ordinal is not None and sibling_count is not None:
    degen_sel = degenerate.nonzero(as_tuple=False).flatten()
    degen_parents = leaf_parent_idx[degen_sel]
    unique_parents, inv = torch.unique(degen_parents, return_inverse=True)

    e1_v, e2_v = global_inplane_basis(uhat, eps=eps)
    # One random angle per unique degenerate parent
    base_theta = torch.rand(unique_parents.numel(), device=device) * (2 * torch.pi)
```

**Step-by-step**:
1. Group degenerate leaves by parent (each root is a unique parent)
2. For each root, sample **one random angle** `θ_0` ∈ [0, 2π) — this is the absolute orientation of child 0's frame
3. `e1_v, e2_v` from `global_inplane_basis` are used as the canonical basis for angle parameterization

### 8b: Structured Rotation for Children 1..k-1 {#8b-structured-rotation}

```python
# helpers.py:367-377
for j, up in enumerate(unique_parents.tolist()):
    group_mask = inv == j
    group_indices = degen_sel[group_mask]
    k = int(sibling_count[group_indices[0]].item())
    theta0 = base_theta[j]
    ordinals = child_ordinal[group_indices]
    angles = theta0 + 2 * torch.pi * ordinals.to(dtype=pos.dtype) / max(k, 1)
    forward[group_indices] = (
        angles.cos().unsqueeze(-1) * e1_v.unsqueeze(0)
        + angles.sin().unsqueeze(-1) * e2_v.unsqueeze(0)
    )
```

**For k=3 with random θ_0 = 0.7 rad**:
- Child 0 (ordinal 0): angle = 0.7 rad
- Child 1 (ordinal 1): angle = 0.7 + 2π/3 ≈ 2.79 rad
- Child 2 (ordinal 2): angle = 0.7 + 4π/3 ≈ 4.89 rad

**Result**: Children are evenly spaced at 2π/k intervals starting from a random angle. Child 0 is in the `forward` direction by construction.

**SO(2) equivariance**: Since `θ_0` is random, the absolute orientation carries no information. The relative spacing (2π/k) is the structural prior.

### 8c: Legacy Fallback (No Ordinal Info) {#8c-legacy-fallback}

```python
# helpers.py:378-386
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

Without ordinal info, each degenerate leaf gets an independent random frame. This is the pre-k>2 behavior, retained for backward compatibility.

---

## 12. Phase 9: Diffusion Denoising Loop (Sampling) — `DenoisingDiffusionModel.sample()` {#12-phase-9-diffusion-sampling}

**File**: `graph_generation/diffusion/basic.py:145-258`

### 9a: Pure Noise Initialisation {#9a-noise-init}

```python
# basic.py:182-189
sigmas = self.make_sigma_schedule(self.num_steps, self.sigma_max, self.sigma_min, device)
sigma_init = float(sigmas[0].item())

L = leaf_idx.numel()
C = th.randn((L, 3), device=device) * sigma_init   # [L, 3] pure noise in local frame
e = th.randn((L, 1), device=device) * sigma_init    # [L, 1] expansion noise
```

**Critical**: Diffusion starts from **pure noise** `C ~ N(0, σ_max²)`. The placeholder positions (all children at parent pos) are irrelevant — what matters is:
1. The local frame (`leaf_fwd`, `leaf_side`) defines how noise maps to 3D space
2. Each root child starts with independent noise in its **own** per-child frame

### 9b: Per-Step Denoising in Local Frame {#9b-per-step-denoise}

```python
# basic.py:201-251
for step in range(self.num_steps):
    sigma_cur = float(sigmas[step].item())
    sigma_next = float(sigmas[step + 1].item())
    log_sigma = math.log(max(sigma_cur, 1e-12))

    P_cur = P_0.clone()

    # Convert local-frame C to global positions
    if local_forward is not None and local_sideways is not None and uhat is not None:
        C_global = local_to_global(C, local_forward, local_sideways, uhat)
        P_cur[leaf_idx] = parent_pos + C_global

    # Assemble features and call model
    e_feat = P_0.new_zeros((N, 1))
    e_feat[leaf_idx] = e
    log_sigma_feat = P_0.new_full((N, 1), log_sigma)
    node_feats_t = th.cat([node_feats, e_feat, log_sigma_feat], dim=-1)
    x_in = th.cat([P_cur, node_feats_t], dim=-1)

    out = model(x=x_in, edge_index=edge_index, batch=batch,
                edge_attr=edge_attr, parent_idx=parent_idx, **model_kwargs)

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

### 9c: Local-to-Global Conversion at Each Step {#9c-local-to-global-sampling}

At each diffusion step, `C` (in local frame) is converted to global positions:

```python
C_global = local_to_global(C, local_forward, local_sideways, uhat)
P_cur[leaf_idx] = parent_pos + C_global
```

For root child i with forward_i (rotated by 2πi/k from random θ_0):
```
P_cur[child_i] = pos[root] + C[i,0]*forward_i + C[i,1]*sideways_i + C[i,2]*uhat
```

**The local frames are FIXED throughout all diffusion steps**. They were computed once in `compute_local_bases_for_leaves` and do not change. This is consistent with training, where frames are locked to P_0.

### 9d: Model Geometry (Internal, No Precompute) {#9d-model-geometry}

During sampling, the model computes its own SO(2) geometry **internally** via `_compute_static_so2_geometry()` — there is no `precompute_full_geometry` or `patch_geometry_for_noised_leaves` call.

The model sees the current `P_cur` positions (with noised leaf positions in global frame) and computes edge decompositions, branch angles, etc. from scratch at each step.

**Key difference from training**: The model's internal geometry computation uses the **current noised** positions, not patched P_0 geometry. However, the **local frames** passed to `local_to_global` are still the fixed per-child frames.

---

## 13. Phase 10: Position Recovery and `geo_angle` Refinement {#13-phase-10-position-recovery}

**File**: `graph_generation/method/expansion.py:419-434`

After diffusion returns `C0_pred` (clean offset prediction in local frame):

```python
# expansion.py:421-423
rel_pred_global = local_to_global(rel_pred, leaf_fwd, leaf_side, model.uhat)
parent_pos_for_children = pos_new[leaf_parent_idx_next]
pos_new[leaf_idx_next] = parent_pos_for_children + rel_pred_global
```

**For root child i**:
```
final_pos[child_i] = pos[root] + C0_pred[i,0]*fwd_i + C0_pred[i,1]*side_i + C0_pred[i,2]*uhat
```

Where `fwd_i` was the per-child forward direction from Phase 8 (rotated by 2πi/k from random θ_0).

---

## 14. Phase 11: `compute_geo_angle_for_new_leaves` (Post-Diffusion) {#14-phase-11-geo-angle-refine}

**File**: `graph_generation/method/helpers.py:928-1038`

After positions are finalized by diffusion, the `geo_angle` is refined using actual positions:

```python
# expansion.py:426-433
if leaf_idx_next.numel() > 0:
    geo_angle_refined, valid = compute_geo_angle_for_new_leaves(
        pos_new, parent_idx_new_0b, leaf_idx_next, uhat=model.uhat,
    )
    if valid.any():
        geo_angle_next[leaf_idx_next[valid]] = geo_angle_refined[valid]
```

Inside `compute_geo_angle_for_new_leaves`:

### Root children (k≥1): SO(2)-invariant ordering

```python
# helpers.py:1034-1040
if is_root_parent:
    # SO(2)-invariant ordering by uhat component
    sorted_idx, _fwd0, _delta = _order_root_children_by_uhat(
        offsets, uhat, eps=eps,
    )
    for rank, si in enumerate(sorted_idx.tolist()):
        geo_angle[group_indices[si]] = rank / max(k - 1, 1)
```

Now using **actual denoised positions** (not placeholders), children are re-ordered using the same SO(2)-invariant `_order_root_children_by_uhat` helper as training — lowest uhat component = child_0. No global axes involved.

### Binary children (k=2, non-root): Parity logic

```python
# helpers.py:1041-1059
elif k == 2:
    # Use existing sin-based parity
    gp = parent_idx[parent_node]
    if gp >= 0:
        v_in = pos[parent_node] - pos[gp]
    else:
        # Non-root degenerate fallback: global axis acceptable here
        e1, _ = global_inplane_basis(uhat, eps=eps)
        v_in = e1
    # ... cross product, sin decision ...
    geo_angle[gi] = 0.0 if sin_val > 0 else 1.0
```

Note: The non-root binary path uses `global_inplane_basis` **only** as a degenerate fallback (parent has no grandparent AND is not root). This is geometrically rare and acceptable — the rule is that **root children never use global axes**, while non-root degenerate cases may.

### Single child (k=1):
```python
# helpers.py:981-984
if k == 1:
    geo_angle[group_indices[0]] = 0.0
    valid[group_indices[0]] = True
    continue
```

**Why refine?** The initial `geo_angle` was assigned based on ordinal index (pre-diffusion). After diffusion places children at their final positions, the geometric ordering may differ from the ordinal assignment. Refinement ensures the feature reflects the actual geometric layout for the **next** expansion step.

---

## 15. Training vs Sampling Consistency Analysis {#15-training-vs-sampling}

| Aspect | Training | Sampling |
|--------|----------|----------|
| **Frame source** | `compute_local_bases` with `geo_delta_theta` from GT P_0 | `compute_local_bases_for_leaves` with random θ_0 + structured rotations |
| **Child 0 forward** | Root → child_0 direction (child_0 = lowest uhat component from GT) | Random direction in perp plane |
| **Child i forward** | fwd_0 rotated by Δθ_i (GT angle) | fwd_0 rotated by 2πi/k (uniform) |
| **Feature** | `geo_ordinal` = `i/(k-1)` (from SO(2)-invariant GT ordering) | `geo_angle` = `i/(k-1)` (from ordinal index, then refined via `_order_root_children_by_uhat`) |
| **Noising** | C_0 + σ * ε in local frame | Start from pure noise σ_max * ε in local frame |
| **Frame stability** | Locked to P_0 (via `patch_geometry_for_noised_leaves`) | Fixed throughout diffusion steps |
| **Geometry computation** | Precomputed once, patched for noised leaves | Computed internally by model at each step |

### Why the difference in child spacing is acceptable

**Training**: Children are at their GT angular positions. Δθ_i is the actual angle between child i and child 0. The frame reflects where the child really is.

**Sampling**: Children start with uniform 2π/k spacing from a random orientation. The diffusion process can refine positions — the model learns to move children from their initial evenly-spaced positions to a more realistic distribution.

**The consistency invariant holds because**: Both training and sampling use the **same ordinal feature scheme** (`i/(k-1)`), the **same local frame structure** (per-child frames), and the **same local_to_global / global_to_local conversions**. The SO(2) equivariance ensures that the absolute orientation of the random frame in sampling is irrelevant — only relative positions matter.

---

## 16. Edge Cases: k=1, k=2, k>2 {#16-edge-cases}

### k=1 (Single Root Child)

| Phase | Behavior |
|-------|----------|
| **Ordering** | No sorting needed. `geo_ordinal = 0.0`, `geo_delta_theta = 0.0` |
| **Training frame** | `v_in = pos[child] - pos[root]` (same as legacy) |
| **Sampling frame** | Random forward (same as legacy single-child fallback) |
| **Feature** | 0.0 (same as legacy left-child feature) |
| **Spawn** | `spawn_counts_final = 1 * has_capacity` |

### k=2 (Binary Root)

| Phase | Behavior |
|-------|----------|
| **Ordering** | Lowest uhat component = child_0. `geo_ordinal = [0.0, 1.0]`. Backward-compatible with legacy L/R convention (lower Z = left = 0.0) |
| **Training frame** | Child 0: `fwd = root→child_0`. Child 1: `fwd = fwd_0 rotated by Δθ_1` |
| **Sampling frame** | Child 0: random fwd. Child 1: fwd rotated by π (= 2π × 1/2) |
| **Feature** | [0.0, 1.0] — matches legacy L=0.0, R=1.0 |
| **Spawn** | `spawn_counts_final = 2 * has_capacity` |

### k>2 (Multi-Child Root)

| Phase | Behavior |
|-------|----------|
| **Ordering** | Lowest uhat component = child_0, remaining clockwise relative to child_0's perp direction. `geo_ordinal = [0, 1/(k-1), 2/(k-1), ..., 1.0]` |
| **Training frame** | Each child i: `fwd = fwd_0 rotated by Δθ_i` (actual GT angle) |
| **Sampling frame** | Each child i: `fwd = fwd_0 rotated by 2πi/k` (uniform spacing) |
| **Feature** | `i/(k-1)` spread evenly in [0, 1] |
| **Spawn** | `spawn_counts_final = k * has_capacity` |

### Degenerate Cases

| Scenario | Handling |
|----------|----------|
| **Child on uhat axis** (offset ⊥ perp plane is zero) | `fwd0_norm ≤ eps` → sort by uhat ascending, `delta_angles = 0` |
| **All children at root** (sampling, pre-diffusion) | Degenerate path → structured rotation with random θ_0 |
| **`v_in` parallel to uhat** | `global_inplane_basis` fallback for `forward` (line 267-269) |
| **`num_root_children` = None** | Legacy spawn: exactly 1 child per root |

---

## 17. Tensor Shape Reference {#17-tensor-shapes}

### Training Tensors

| Tensor | Shape | Type | Source |
|--------|-------|------|--------|
| `pos_gt` (P_0) | [N, 3] | float32 | Batch |
| `parent_idx` | [N] | long | Decoded from `parent_idx_1b` |
| `geo_ordinal` | [N] | float32 | `compute_root_child_angles` |
| `geo_delta_theta` | [N] | float32 | `compute_root_child_angles` |
| `local_forward` | [N, 3] | float32 | `compute_local_bases` |
| `local_sideways` | [N, 3] | float32 | `compute_local_bases` |
| `leaf_rel_pos` (C_0) | [L, 3] | float32 | `global_to_local(leaf_rel_pos_global)` |
| `leaf_fwd` | [L, 3] | float32 | `local_forward[leaf_idx_train]` |
| `leaf_side` | [L, 3] | float32 | `local_sideways[leaf_idx_train]` |
| `geo_feat` | [N, 1] | float32 | `geo_ordinal.clamp(min=0.0)` |
| `num_root_children` | [G] | long | Batch (per-graph scalar) |

### Sampling Tensors

| Tensor | Shape | Type | Source |
|--------|-------|------|--------|
| `pos_new` | [N', 3] | float32 | Accumulated, updated by diffusion |
| `geo_angle` | [N'] | float32 | `i/(k-1)` per child, refined post-diffusion |
| `ordinal_new` | [K] | long | 0 to k-1 per child group |
| `sibling_count_new` | [K] | long | k for each child in group |
| `child_ordinal_t` | [K] | long | Passed to `compute_local_bases_for_leaves` |
| `sib_count_long` | [K] | long | Passed to `compute_local_bases_for_leaves` |
| `leaf_fwd` | [K, 3] | float32 | Per-child frames (structured rotation) |
| `leaf_side` | [K, 3] | float32 | Per-child frames |
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
  ├── precompute_full_geometry()                       ★
  │     ├── compute_geo_lr_mask()                      (binary L/R, interior children)
  │     ├── compute_root_child_angles()                ★ NEW
  │     │     ├── _order_root_children_by_uhat()       (SO(2)-invariant, no global axes)
  │     │     │     ├── child_0 = lowest uhat component (tiebreak: largest perp dist)
  │     │     │     ├── clockwise sort relative to child_0's perp direction
  │     │     │     └── delta_angles relative to child_0
  │     │     └── ordinal = rank / max(k-1, 1)
  │     ├── compute_branch_angles_parent_centric()
  │     ├── assign_branch_angles_to_edges()
  │     ├── assign_parent_scalar_to_edges()
  │     └── compute_local_bases(geo_delta_theta=...)   ★ MODIFIED
  │           └── per-child rotation by Δθ_i
  ├── global_to_local()                                (targets: global → local)
  ├── [feature assembly with geo_ordinal]              ★ MODIFIED
  └── DenoisingDiffusionModel.forward()
        ├── local_to_global()                          (C_t: local → global for P_t)
        ├── patch_geometry_for_noised_leaves()         ★ MODIFIED (v_in locked for root children)
        ├── model.forward()
        └── MSE loss in local frame
```

### Sampling Path
```
Expansion.sample_graphs(num_root_children=nrc)         ★ MODIFIED
  └── while not terminated:
        Expansion.expand(num_root_children=nrc)         ★ MODIFIED
          ├── spawn count: k children for root          ★ NEW
          ├── ordinal tracking: ordinal_new, sib_count  ★ NEW
          ├── geo_angle = i/(k-1) per child             ★ NEW
          ├── [feature assembly with geo_angle]          ★ MODIFIED
          ├── compute_local_bases_for_leaves(            ★ MODIFIED
          │     child_ordinal=..., sibling_count=...)
          │     └── structured rotation: θ_0 + 2πi/k
          ├── DenoisingDiffusionModel.sample()
          │     ├── C = randn * sigma_max               (pure noise, local frame)
          │     └── for each step:
          │           ├── local_to_global(C)             (local → global for P_cur)
          │           ├── model.forward()                (internal geometry)
          │           └── DDIM update
          ├── local_to_global(C0_pred)                   (final: local → global)
          ├── pos update
          └── compute_geo_angle_for_new_leaves()         ★ NEW (refinement)
```

---

## Summary of All Modified Functions

| Function | File | What Changed |
|----------|------|-------------|
| `compute_root_child_angles` | helpers.py:747 | **NEW** — SO(2)-invariant ordering via `_order_root_children_by_uhat` + relative angles + ordinal feature |
| `compute_local_bases` | helpers.py:160 | Added `geo_delta_theta` param; per-child root frames |
| `compute_local_bases_for_leaves` | helpers.py:280 | Added `child_ordinal`/`sibling_count`; structured rotation |
| `precompute_full_geometry` | helpers.py:1169 | Calls `compute_root_child_angles`; passes `geo_delta_theta` to `compute_local_bases`; returns `geo_ordinal`/`geo_delta_theta` |
| `compute_geo_lr_mask` / `_f2` | helpers.py:597/1086 | Replaced `v_in[fallback_mask] = global_e1` with `v_in[fallback_mask] = v_out` (geometric, no global axis for root children) |
| `patch_geometry_for_noised_leaves` | helpers.py:1248 | Removed `v_in = v_out` override for root children; frames locked to P_0 |
| `_order_root_children_by_uhat` | helpers.py:111 | **NEW** — shared SO(2)-invariant root child ordering helper (no global axes) |
| `compute_geo_angle_for_new_leaves` | helpers.py:970 | **NEW** — post-diffusion ordinal refinement using `_order_root_children_by_uhat` for root children |
| `Expansion.sample_graphs` | expansion.py:52 | Added `num_root_children` param; uses `geo_angle` (float) instead of `geo_lr_assign` (int) |
| `Expansion.expand` | expansion.py:138 | k-child spawn; ordinal tracking; `geo_angle` feature; passes ordinal to frame computation |
| `Expansion.get_loss` | expansion.py:460 | Uses `geo_ordinal.clamp(min=0.0)` instead of `geo_lr_mask.float()` |
| `ReducedGraphData` | data.py:9 | Added `num_root_children` field |
| `_build_reduced_graph_data` | reduction_dataset.py:25 | Computes and passes `num_root_children` |
