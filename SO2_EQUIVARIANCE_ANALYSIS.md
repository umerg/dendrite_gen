# SO(2) Equivariance Analysis

**Scope**: Systematic audit of whether the model, features, and geometry pipeline maintain SO(2) equivariance (invariance/equivariance under rotations around `uhat`).

**Definition**: A system is SO(2)-equivariant w.r.t. axis `uhat` if rotating all input positions by angle α around `uhat` produces correspondingly rotated outputs. Scalar features fed to the model must be SO(2)-**invariant** (unchanged by rotation). Vector predictions must transform correctly.

---

## 1. Edge-Level Geometric Features (Model Inputs)

These are computed in `SO2_EGNN.forward()` (lines 294-328) or precomputed via `_compute_static_so2_geometry()` (lines 750-786) / `precompute_full_geometry()` (helpers.py:1169-1245).

| Feature | Definition | SO(2) Status |
|---------|-----------|--------------|
| `rel_coors` | `pos[dst] - pos[src]` | **Equivariant** (rotates with input) — used only for message weighting via `r_perp`, not directly as scalar |
| `rho` | `\|\|r_perp\|\|` = norm of projection onto plane ⊥ `uhat` | **Invariant** ✓ — rotation around `uhat` preserves in-plane distances |
| `du` | `rel_coors · uhat` | **Invariant** ✓ — axial component unchanged by rotation around axis |
| `r_perp` | `rel_coors - du * uhat` | **Equivariant** — rotates in-plane, but only its norm (`rho`) enters the edge MLP |
| `cosψ_edge` | Cosine of in-plane branch angle | **Invariant** ✓ — relative angle between two projected vectors |
| `sinψ_edge` | Sine of in-plane branch angle | **Invariant** ✓ — same reasoning |
| `cosθ_edge` | `du / \|\|v_out\|\|` (elevation angle) | **Invariant** ✓ — ratio of axial to total length |

**Verdict**: All scalar quantities entering the edge MLP are SO(2)-invariant. ✓

---

## 2. Branch Angle Computation (`compute_branch_angles_parent_centric`)

**File**: `helpers.py:394-464`

For each non-root node `i` with parent `p`:
- `v_in[i]` = incoming direction at parent = `pos[p] - pos[grandparent(p)]`
- `v_out[i]` = outgoing direction = `pos[i] - pos[p]`

Both are geometric vectors derived purely from relative positions. The angles (cosψ, sinψ, cosθ) are computed from their projections onto the plane ⊥ `uhat`.

### Root-children fallback (line 424-427):
```python
fallback_mask = has_parent & ~has_gp  # children of root (no grandparent)
v_in[sel] = coors[sel] - coors[parent[sel]]  # = v_out direction
```

This sets `v_in = v_out` for root children → cosψ = 1.0, sinψ = 0.0. Since both vectors are geometric (no global axes), this is **invariant**. ✓

**Verdict**: Branch angles are fully SO(2)-invariant. ✓

---

## 3. Node Feature: `geo_ordinal`

**File**: `helpers.py:671-796` (`compute_root_child_angles`)

### 3a. Binary interior children (line 725-727)
```python
geo_ordinal[binary_interior] = (~geo_lr_mask[binary_interior]).to(dtype)
# left (True) → 0.0, right (False) → 1.0
```

The L/R assignment comes from `compute_geo_lr_mask` which uses:
- **Interior nodes**: sinψ sign (relative angle) — **invariant** ✓
- **Root children (k=2)**: Z-coordinate comparison (`pos[:, -1]`) — this is the `uhat` component. Rotation around `uhat` does not change the `uhat` component → **invariant** ✓

### 3b. Root children (k ≥ 2): Angular clockwise ordering (lines 733-796)

```python
e1, e2 = global_inplane_basis(uhat, eps=eps)   # FIXED global basis
...
proj_e1 = (offsets_perp * e1.unsqueeze(0)).sum(dim=-1)
proj_e2 = (offsets_perp * e2.unsqueeze(0)).sum(dim=-1)
angles_global = th.atan2(proj_e2, proj_e1)
_, sorted_idx = (-angles_global).sort()         # clockwise from e1
```

**VIOLATION**: The ordering of root children is determined by their absolute angle relative to a **fixed global basis** (`e1`, `e2`). Rotating all positions by α around `uhat` shifts these angles by α, potentially changing which child is "child 0" (leftmost) and reassigning all ordinals.

**Impact**: The `geo_ordinal` feature for root children is **NOT SO(2)-invariant**. The same tree rotated produces different ordinal assignments → different node features → potentially different model predictions.

**Severity**: **HIGH** — this feature is slot 1 in the node feature vector, fed to every MPNN layer, and also drives the LR_offset_head routing (Section 5).

---

## 4. Node Feature: `geo_lr_mask` (via `compute_geo_lr_mask`)

**File**: `helpers.py:521-668`

### Interior binary children (lines 621-652)
Uses sinψ sign or atan2 angle — both derived from projected relative vectors. **Invariant** ✓

### Root children with k=2 (lines 606-612)
```python
z_rc = pos[multi_root_ch, -1]                    # Z = uhat component
max_z = scatter(z_rc, p_rc, ...)
lr_mask[multi_root_ch] = z_rc < max_z[p_rc] - 1e-7
```
Compares the `uhat`-component of positions. Rotation around `uhat` doesn't change this. **Invariant** ✓

### Root children with k=1 (line 605)
Single child → always left. **Invariant** (trivially) ✓

### Fallback `v_in` for root children (lines 559-561)
```python
v_in[fallback_mask] = global_e1.view(1, -1)  # FIXED global direction
```
**Latent violation**: Uses global `e1` as incoming direction for root children. However, root children are handled by the separate Z-comparison path (lines 592-612), so this fallback value is **never used** in the final L/R decision for root children. The sinψ computed from this `v_in` only matters in the `binary_parents` path (line 622: `~parent_is_root_node`), which explicitly excludes root children.

**Verdict**: `geo_lr_mask` is **SO(2)-invariant** in practice, despite the latent global-basis fallback. ✓

---

## 5. LR_offset_head Routing

**File**: `egnn_so2.py:662-736`

```python
class_feature_idx = self.pos_dim + 1  # = slot 1 = geo_ordinal
class_feature = x[:, class_feature_idx].clone()
...
class_mask = (class_feature > 0.5)  # True → head 1, False → head 0
```

The LR_offset_head routes each node's features through one of two decoder MLPs based on `geo_ordinal > 0.5`.

- **Binary interior children**: geo_ordinal ∈ {0.0, 1.0} → deterministic routing. Since geo_ordinal is invariant for these nodes → **invariant** ✓
- **Root children (k ≥ 3)**: geo_ordinal = i/(k-1) ∈ [0, 1]. Since root-child ordinals depend on global ordering (Section 3b), the routing is **NOT SO(2)-invariant**. A rotation could swap which child goes through head 0 vs head 1.
- **Root children (k = 2)**: geo_ordinal ∈ {0.0, 1.0} from the Z-comparison L/R. This IS invariant ✓

**Verdict**: LR_offset_head routing **inherits the SO(2) violation** from `geo_ordinal` for root children with k ≥ 3.

---

## 6. Local Coordinate Frames (Training Path)

**File**: `helpers.py:160-277` (`compute_local_bases`)

### Interior nodes with grandparent (lines 211-215)
```python
v_in[sel] = pos[parent[sel]] - pos[gp[sel]]
```
Purely geometric. Frame = normalize(project(v_in, ⊥uhat)). **Equivariant** ✓ — the frame rotates with the input, so local-frame coordinates are invariant.

### Root children with `geo_delta_theta` (lines 219-254)
```python
child0 = children[sorted_idx[0]]     # child 0 from geo_delta_theta ordering
ref_dir = pos[child0] - pos[r]       # geometric direction to child 0
...
for c in children:
    v_in[c] = cos(dt) * fwd0 + sin(dt) * side0  # rotate ref_dir by Δθ
```

The reference direction (`ref_dir`) is geometric ✓. The rotation by `geo_delta_theta` is also geometric ✓. **However**, which child is "child 0" depends on `geo_delta_theta`, which in turn depends on `compute_root_child_angles` (the global-basis ordering from Section 3b).

**Subtle issue**: The frame assignment for root children depends on which child is labeled "child 0". Under SO(2) rotation, the same physical child might get a different label (and therefore a different frame). But since:
1. The frame is defined relative to child 0's geometric direction
2. All children's frames are rotated versions of child 0's frame
3. The relative geometry is preserved

The **physical predictions** (in global frame) will be the same IF the model has learned to produce outputs that are consistent across ordinal assignments. In practice, this is coupled to the ordinal feature violation (Section 3b).

### Degenerate fallback (lines 266-269)
```python
if degenerate.any():
    e1, _ = global_inplane_basis(uhat, eps=eps)
    forward[degenerate] = e1.unsqueeze(0)
```

**VIOLATION**: When `v_in` is parallel to `uhat` or zero, the forward direction falls back to a **fixed global vector** `e1`. Rotating positions changes which direction `e1` points relative to the geometry.

**Severity**: **LOW** — this triggers only when a node's incoming direction has zero in-plane component (exactly aligned with `uhat`), which is geometrically rare. For typical dendrite morphologies where branches spread in the plane, this case almost never occurs.

**Verdict**: Local frames are SO(2)-equivariant except for rare degenerate cases and the root-child labeling coupling.

---

## 7. Local Coordinate Frames (Sampling Path)

**File**: `helpers.py:280-391` (`compute_local_bases_for_leaves`)

### Non-root leaves with grandparent
Same as training: `v_in = pos[parent] - pos[grandparent]`. **Equivariant** ✓

### Root children — structured rotation (lines 355-377)
```python
e1_v, e2_v = global_inplane_basis(uhat)
base_theta = torch.rand(...) * (2 * torch.pi)  # RANDOM base angle
angles = theta0 + 2 * torch.pi * ordinals / k
forward[group_indices] = cos(angle) * e1_v + sin(angle) * e2_v
```

The base angle `theta0` is **random**. While the expression uses the global basis (`e1_v`, `e2_v`), the randomness makes the distribution rotation-invariant: rotating all positions by α is equivalent to shifting `theta0` by α, but since `theta0` is uniformly random, the distribution is unchanged.

**Verdict**: **SO(2)-equivariant in distribution** ✓ (stochastic equivariance). Any particular sample is not equivariant, but the ensemble is.

### Legacy fallback (no ordinal info, lines 379-386)
Same pattern — random angle per leaf. **SO(2)-equivariant in distribution** ✓

---

## 8. Diffusion Geometry: Training (`patch_geometry_for_noised_leaves`)

**File**: `helpers.py:1248-1362`

Patches edge-level and node-level geometry for noised leaf positions. All quantities are recomputed from the actual noised positions using the same geometric formulas (relative vectors, projections, cross products). No global axes are introduced.

Key: `v_in` for leaves is reused from P_0 (parent/grandparent are internal nodes, unchanged by noising). This is correct since the incoming direction at the parent doesn't change.

**Verdict**: Patch geometry preserves SO(2) invariance. ✓

---

## 9. Diffusion Geometry: Sampling (No `pre_geom`)

**File**: `basic.py:201-233` (sampling loop)

During the DDIM loop, no `pre_geom` is passed to the model. The model recomputes geometry internally via `_compute_static_so2_geometry()` from the current `P_cur` positions. This uses the same `compute_branch_angles_parent_centric` function.

**Verdict**: Sampling geometry is SO(2)-invariant (same geometric computations, no global axes). ✓

---

## 10. Other Node Features

| Feature | Source | SO(2) Status |
|---------|--------|--------------|
| `is_leaf` | Structural (leaf_mask) | **Invariant** ✓ — topology, not geometry |
| `new_leaf_mask` | Structural | **Invariant** ✓ |
| `size_ratio` | Node count / target size | **Invariant** ✓ — purely combinatorial |
| `e_t` (noised expansion) | Random noise on scalar | **Invariant** ✓ |
| `log_sigma` | Noise schedule scalar | **Invariant** ✓ |
| `edge_attr` (edge type) | Parent→child / child→parent | **Invariant** ✓ — topology |
| TMD embedding | Per-graph topological descriptor | **Invariant** ✓ — topology |

---

## 11. Global Attention (`GlobalLinearAttention_Sparse`)

**File**: `egnn_so2.py:115-154`

Operates on feature vectors only (not positions). Features are SO(2)-invariant scalars → attention is invariant. The learned global tokens are position-independent.

**Verdict**: **Invariant** ✓

---

## 12. Output Decode and Local-to-Global Conversion

### Training (loss computation)
```python
# helpers.py:111-132 (global_to_local)
leaf_rel_pos = global_to_local(leaf_rel_pos_global, leaf_fwd, leaf_side, uhat)
```
Target offsets are projected into the local frame. Since the local frame co-rotates with positions (for non-degenerate cases), the local-frame targets are invariant. Model predicts in local frame → loss is invariant.

### Sampling (position reconstruction)
```python
# expansion.py:421
rel_pred_global = local_to_global(rel_pred, leaf_fwd, leaf_side, model.uhat)
pos_new[leaf_idx_next] = parent_pos + rel_pred_global
```
Local predictions are rotated back to global frame. Since the frame co-rotates → global output is equivariant. ✓

---

## 13. Post-Diffusion Ordinal Refinement (Sampling)

**File**: `expansion.py:426-433` → `helpers.py:928-1038` (`compute_geo_angle_for_new_leaves`)

After diffusion places leaves at their final positions, `compute_geo_angle_for_new_leaves` recomputes ordinals from actual geometry. For root children (line 996-1003):

```python
e1, e2 = global_inplane_basis(uhat)
proj_e1 = (offsets_perp * e1.unsqueeze(0)).sum(dim=-1)
proj_e2 = (offsets_perp * e2.unsqueeze(0)).sum(dim=-1)
angles = th.atan2(proj_e2, proj_e1)
_, sorted_idx = (-angles).sort()
```

Same global-basis ordering as `compute_root_child_angles`. **Same SO(2) violation** as Section 3b. The refined ordinals for root children depend on absolute orientation.

**Verdict**: Post-diffusion ordinal refinement for root children is **NOT SO(2)-invariant**.

---

## Summary of Findings

### Confirmed SO(2) Violations

| # | Location | Description | Severity | Nodes Affected |
|---|----------|-------------|----------|----------------|
| **V1** | `compute_root_child_angles` (helpers.py:734, 760-765) | Global `e1`/`e2` basis used for clockwise ordering of root children → `geo_ordinal` depends on absolute orientation | **HIGH** | Root children (k ≥ 3) |
| **V2** | `LR_offset_head` routing (egnn_so2.py:664-736) | Routes through different decoder heads based on `geo_ordinal > 0.5`, inheriting V1 for root children k ≥ 3 | **HIGH** | Root children (k ≥ 3) |
| **V3** | `compute_geo_angle_for_new_leaves` (helpers.py:996-1003) | Post-diffusion ordinal refinement uses same global basis ordering | **MEDIUM** | Root children at sampling time |
| **V4** | `compute_local_bases` degenerate fallback (helpers.py:267-269) | Falls back to global `e1` when `v_in ⊥ uhat`-projection is zero | **LOW** | Rare degenerate geometry |

### Confirmed SO(2)-Invariant/Equivariant Components

| Component | Status |
|-----------|--------|
| Edge features (ρ, du, cosψ, sinψ, cosθ) | Invariant ✓ |
| Branch angle computation | Invariant ✓ |
| geo_lr_mask (binary interior L/R) | Invariant ✓ |
| geo_lr_mask (root children k=2, Z-comparison) | Invariant ✓ |
| Local frames (non-degenerate, interior nodes) | Equivariant ✓ |
| Local frames (sampling, random base angle) | Equivariant in distribution ✓ |
| Diffusion geometry patching | Invariant ✓ |
| All other node features (is_leaf, size_ratio, etc.) | Invariant ✓ |
| Global attention | Invariant ✓ |
| TMD embedding | Invariant ✓ |
| Local↔Global coordinate conversion | Equivariant ✓ |

---

## 14. Analysis: Practical Impact

### Why V1/V2 matter
The `geo_ordinal` feature for root children with k ≥ 3 means the model receives different inputs for the same physical tree at different orientations. During training, this creates a many-to-one mapping (multiple ordinal assignments → same geometry), which the model must average over. This can:
1. Reduce prediction sharpness for root children
2. Make the model sensitive to training data orientation distribution
3. Create inconsistency between training (GT-derived ordinal) and sampling (arbitrary then refined)

### Why V1/V2 may be acceptable in practice
- **Binary trees (k=2)**: The Z-comparison L/R is fully invariant. V1 only triggers for k ≥ 3.
- **Training data augmentation**: If random SO(2) rotations are applied during training, the model sees all ordinal assignments and learns to average → effective invariance.
- **Root is a single node**: The violation only affects the first expansion step (root → children). All subsequent expansions (interior binary splits) are fully invariant.

### V4 is negligible
The degenerate fallback requires a parent's incoming direction to be exactly aligned with `uhat` (zero in-plane projection). For real dendrite data with non-trivial branching geometry, this is vanishingly rare.

---

## 15. Recommendations

### If exact SO(2) equivariance for k-ary root children is desired:

**Option A — Relative angular ordering**: Instead of ordering root children by absolute angle from `e1`, order them relative to a geometrically-defined reference (e.g., the child with largest `uhat`-component, or a canonical child chosen by a rotation-invariant criterion). This makes the ordinal assignment invariant.

**Option B — Remove ordinal for root children**: Use a uniform ordinal (e.g., all root children get `geo_ordinal = 0.5`) and rely on the diffusion process + branch angles to resolve their positions. This sacrifices the ordering signal but guarantees invariance.

**Option C — Data augmentation** (current implicit approach): Apply random SO(2) rotations during training so the model sees all possible ordinal assignments. The model learns rotation-invariant behavior empirically rather than architecturally.

### For V4 (degenerate fallback):
Replace `global_inplane_basis` fallback with a geometry-derived direction (e.g., the direction to the nearest sibling, or a hash-based but deterministic direction derived from local topology). Low priority given rarity.
