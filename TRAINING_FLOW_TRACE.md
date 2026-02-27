# Training Flow Trace: Geometry-Aware Denoising Diffusion

A detailed, line-by-line trace of the training forward pass from batch construction through loss computation. Covers geometry precomputation, the compute-once/patch-leaves optimisation, and all root/single-child edge cases.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Phase 1: Batch Construction (DataLoader)](#2-phase-1-batch-construction)
3. [Phase 2: `Expansion.get_loss()` Entry](#3-phase-2-expansionget_loss-entry)
4. [Phase 3: Parent Index Decoding](#4-phase-3-parent-index-decoding)
5. [Phase 4: Edge Index Construction](#5-phase-4-edge-index-construction)
6. [Phase 5: Leaf Selection and Expansion Targets](#6-phase-5-leaf-selection-and-expansion-targets)
7. [Phase 6: Relative Position Targets](#7-phase-6-relative-position-targets)
8. [Phase 7: Full Geometry Precomputation on P_0](#8-phase-7-full-geometry-precomputation-on-p_0)
   - [7a: geo_lr_mask (Left/Right Sibling Assignment)](#7a-geo_lr_mask)
   - [7b: Branch Angles (cospsi, sinpsi, cos_theta)](#7b-branch-angles)
   - [7c: Edge-Level SO(2) Decomposition](#7c-edge-level-so2-decomposition)
   - [7d: Angle Assignment to Edges](#7d-angle-assignment-to-edges)
9. [Phase 8: Node Feature Assembly](#9-phase-8-node-feature-assembly)
10. [Phase 9: Diffusion Forward Pass](#10-phase-9-diffusion-forward-pass)
    - [9a: Noise Sampling](#9a-noise-sampling)
    - [9b: Noising Leaf Positions and Expansions](#9b-noising)
    - [9c: Geometry Patching for Noised Leaves](#9c-geometry-patching)
    - [9d: Diffusion Conditioning Features](#9d-diffusion-conditioning)
    - [9e: Model Input Assembly](#9e-model-input-assembly)
11. [Phase 10: Model Forward Pass (SO2_EGNN_Network)](#11-phase-10-model-forward)
    - [10a: TMD Embedding](#10a-tmd-embedding)
    - [10b: Geometry Routing (pre_geom)](#10b-geometry-routing)
    - [10c: MPNN Layer Loop](#10c-mpnn-layers)
    - [10d: SO2_EGNN Single Layer Detail](#10d-so2-egnn-layer)
    - [10e: Offset Head Decoding](#10e-offset-head)
12. [Phase 11: Loss Computation](#12-phase-11-loss-computation)
13. [Root and Single-Child Edge Cases](#13-root-and-single-child-edge-cases)
14. [Geometry Compute-Once / Patch-Leaves Summary](#14-geometry-compute-once-summary)

---

## 1. High-Level Overview

```
DataLoader (batch construction)
    |
    v
Expansion.get_loss(batch, model)
    |
    +-- decode_parent_indices(batch)         -> parent_idx [N], -1 for roots
    +-- build_directed_edge_index(parent_idx) -> edge_index [2,E], edge_types [E]
    +-- select_training_leaf_indices(batch)   -> leaf_idx_train [L]
    +-- leaf_rel_targets(pos_gt, ...)         -> C_0 [L,3] (relative offsets)
    +-- precompute_full_geometry(P_0, ...)    -> pre_geom_p0 dict (ONCE on clean P_0)
    |       |
    |       +-- compute_geo_lr_mask()         -> geo_lr_mask [N] bool
    |       +-- compute_branch_angles_parent_centric() -> (cospsi, sinpsi, cos_theta) [N,1] each
    |       +-- edge SO(2) decomposition      -> rel_coors, r_perp, rho, du
    |       +-- assign_branch_angles_to_edges()
    |       +-- assign_parent_scalar_to_edges()
    |
    +-- assemble node_feats [N, avail_feats_dim]
    |       (is_leaf, geo_lr, new_leaf_flag, size_ratio, padding)
    |
    +-- DenoisingDiffusionModel.forward(...)
            |
            +-- sample sigma per graph
            +-- noise C_0 -> C_t, e_0 -> e_t
            +-- construct P_t (clone P_0, scatter noised leaf positions)
            +-- patch_geometry_for_noised_leaves(pre_geom_p0, P_t, ...)  -> pre_geom
            +-- assemble diffusion conditioning (e_feat, log_sigma)
            +-- concat [P_t | node_feats | e_feat | log_sigma] -> x_in
            |
            +-- model(x_in, edge_index, ..., pre_geom=pre_geom)
            |       |
            |       +-- SO2_EGNN_Network.forward()
            |       |   +-- (optional) TMD embedding -> cat to features
            |       |   +-- skip internal geometry (pre_geom provided)
            |       |   +-- for each MPNN layer:
            |       |   |     SO2_EGNN.forward(x, edge_index, ..., pre_geom)
            |       |   |       -> uses pre_geom for rho, du, angles
            |       |   |       -> edge_mlp([feats_i, feats_j, edge_scalar_feats])
            |       |   |       -> aggregate -> node_mlp -> residual update
            |       |   +-- offset_head(feats) -> [rel_pred[N,3], expansion_pred[N,1]]
            |       |
            |       +-- returns dict {"rel_pred", "expansion_pred", "node_state"}
            |
            +-- extract predictions at leaf indices
            +-- MSE loss: pos_loss = MSE(C_pred, C_0), exp_loss = MSE(e_pred, e_0)
            +-- return (exp_loss, pos_loss)
    |
    +-- loss = pos_loss + expansion_loss_weight * exp_loss
    +-- return loss, metrics
```

---

## 2. Phase 1: Batch Construction

**File**: `graph_generation/data/reduction_dataset.py`

Each training sample is a `ReducedGraphData` (PyG `Data` subclass) built from a random reduction sequence of a full tree. Key fields per sample:

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `pos` | `[n, 3]` | float32 | Absolute 3D node positions (ground truth) |
| `parent_idx_1b` | `[n]` | int64 | 1-based parent indices (0 = root, safe for PyG offset batching) |
| `leaf_idx` | `[L_all]` | int64 | Indices of all leaf nodes |
| `leaf_mask` | `[n]` | bool | Boolean mask: `leaf_mask[i] = True` iff node `i` is a leaf |
| `leaf_expansion` | `[L_all]` | int64 | Expansion labels in {1, 2}: 1=terminal, 2=will branch |
| `new_leaf_idx_from_next` | `[L_new]` | int64 | Subset of leaves: the "new" leaves from the next reduction level |
| `new_leaf_mask_from_next` | `[n]` | bool | Boolean mask for `new_leaf_idx_from_next` |
| `total_tree_size` | scalar | int64 | Total nodes in the original unreduced tree |
| `tmd` | `[1, D]` | float32 | Tree Morphology Descriptor (optional, global per graph) |
| `adj` | `[n, n]` | SparseTensor | Adjacency matrix (undirected) |

**PyG Batching** (`Batch.from_data_list`):
- Concatenates per-node tensors along dim 0
- Auto-creates `batch` vector `[N]` (graph assignment) and `ptr` vector `[B+1]` (cumulative node counts)
- Applies `__inc__()` offsets to index tensors (`parent_idx_1b`, `leaf_idx`, `new_leaf_idx_from_next`) so they index into the concatenated node dimension

After batching, `batch.pos` is `[N, 3]` where `N = sum of all n_i`, and index tensors point into this concatenated space.

---

## 3. Phase 2: `Expansion.get_loss()` Entry

**File**: `graph_generation/method/expansion.py:469`

```python
def get_loss(self, batch, model: th.nn.Module):
```

Entry point called by the training loop. Receives the PyG batch and the `SO2_EGNN_Network` model.

---

## 4. Phase 3: Parent Index Decoding

**File**: `expansion.py:486` -> `helpers.py:694-710`

```python
parent_idx = decode_parent_indices(batch).to(device=batch.pos.device)  # [N], -1 for roots
```

`decode_parent_indices` converts the 1-based `parent_idx_1b` to 0-based indexing with -1 for roots:

```python
def decode_parent_indices(batch) -> th.Tensor:
    parent_idx_1b = batch.parent_idx_1b          # [N], 1-based, offsets already applied by PyG
    parent_idx = parent_idx_1b - 1                # shift to 0-based
    batch_vec = getattr(batch, "batch", None)
    ptr = getattr(batch, "ptr", None)
    if batch_vec is not None and ptr is not None:
        offsets = ptr[batch_vec]                  # per-node graph offset
        root_mask = parent_idx_1b == offsets       # roots: their 1-based idx equals their graph's offset
    else:
        root_mask = parent_idx_1b == 0
    if root_mask.any():
        parent_idx = parent_idx.clone()
        parent_idx[root_mask] = -1                # mark roots with -1
    return parent_idx
```

**Why 1-based**: PyG's `Batch.from_data_list` auto-increments index tensors by cumulative node counts (`ptr`). A root with `parent_idx=0` in graph 0 would be shifted to `parent_idx=ptr[1]` in graph 1, which is wrong. Using 1-based (root=0 becomes `parent_idx_1b=0`, children have `parent_idx_1b=parent+1`), the offsets work correctly because 0 stays 0 after adding any offset... wait, actually: **root nodes have `parent_idx_1b = 0`** before batching. PyG's `__inc__` returns `self.num_nodes` for `parent_idx_1b`, so after batching root of graph `g` has `parent_idx_1b = ptr[g]` (the cumulative offset). In `decode_parent_indices`, `root_mask = (parent_idx_1b == ptr[batch_vec])` correctly identifies roots across all graphs.

**Result**: `parent_idx` is `[N]`, where `parent_idx[i] = j` means node `j` is the parent of node `i`, and `parent_idx[i] = -1` means `i` is a root.

---

## 5. Phase 4: Edge Index Construction

**File**: `expansion.py:490-494` -> `helpers.py:14-42`

```python
edge_index, edge_types = build_directed_edge_index(
    parent_idx,
    edge_parent_to_child=self.EDGE_PARENT_TO_CHILD,  # 0
    edge_child_to_parent=self.EDGE_CHILD_TO_PARENT,   # 1
)
```

For every child-parent pair, creates two directed edges:
- `parent -> child` with type `0`
- `child -> parent` with type `1`

```python
for child, parent in enumerate(parent_idx.tolist()):
    if parent < 0:
        continue                           # skip roots (no parent edge)
    src_list.append(parent)                # parent -> child
    dst_list.append(child)
    type_list.append(edge_parent_to_child) # 0
    src_list.append(child)                 # child -> parent
    dst_list.append(parent)
    type_list.append(edge_child_to_parent) # 1
```

**Result**: `edge_index [2, E]` where `E = 2 * (N - num_roots)`. Each tree edge appears twice (bidirectional). `edge_types [E]` in {0, 1}.

---

## 6. Phase 5: Leaf Selection and Expansion Targets

**File**: `expansion.py:497-532`

### 5a: Get all leaves and their expansion labels

```python
leaf_idx_all = batch.leaf_idx                         # [L_total] all leaves across all graphs
leaf_expansion_all = batch.leaf_expansion - 1         # [L_total] map {1,2} -> {0,1}
```

### 5b: Select training leaves

```python
leaf_idx_train = select_training_leaf_indices(batch)  # [L] subset for training
```

<!-- <CHNAGE> THE FOLLOWING MAY CUASE SILENT FAILURE - CHANGE THIS TO BREAK IF NEW LEAVES NOT PRESENT -->
`select_training_leaf_indices` (`helpers.py:713-736`) checks for `batch.new_leaf_idx_from_next`:
- If present and non-empty, uses those indices (the "new" leaves from the next reduction level)
- Falls back to `batch.leaf_idx` if not available
- Filters out any indices where `leaf_mask` is False (safety check). 

### 5c: Get parent indices for training leaves

```python
leaf_parent_idx = parent_idx[leaf_idx_train]          # [L] parent of each training leaf
assert (leaf_parent_idx >= 0).all()                   # leaves must have valid parents (not roots)
```

### 5d: Map expansion labels to training leaves

```python
leaf_targets_per_node = leaf_expansion_all.new_full((pos_gt.size(0),), -1)  # [N] all -1
leaf_targets_per_node[leaf_idx_all] = leaf_expansion_all.view(-1)           # scatter labels
leaf_expansion = leaf_targets_per_node[leaf_idx_train]                      # [L] gather for train leaves
```

This two-step scatter-gather handles the mapping from "all leaves" to "training leaves" even though their index sets may differ.

---

## 7. Phase 6: Relative Position Targets

**File**: `expansion.py:553` -> `helpers.py:739-748`

```python
leaf_rel_pos = leaf_rel_targets(pos_gt, leaf_idx_train, leaf_parent_idx)  # [L, 3]
```

```python
def leaf_rel_targets(pos_gt, leaf_idx, leaf_parent_idx):
    parent_pos = pos_gt[leaf_parent_idx]              # [L, 3]
    return pos_gt[leaf_idx] - parent_pos              # C_0 = leaf_pos - parent_pos
```

**Result**: `C_0 [L, 3]` -- the ground-truth parent-relative offset for each training leaf. This is what the model must learn to predict (denoised).

---

## 8. Phase 7: Full Geometry Precomputation on P_0

**File**: `expansion.py:556-562` -> `helpers.py:513-572`

```python
uhat = model.uhat                                     # [3] SO(2) rotation axis (default [0,0,1])
pre_geom_p0 = precompute_full_geometry(
    pos_gt, parent_idx, edge_index, uhat,
    debug=getattr(self, "debug", False),
)
geo_lr_mask = pre_geom_p0['geo_lr_mask']               # [N] bool
```

This is the **key optimisation**: compute ALL geometry once on the clean ground-truth positions P_0. Later, only leaf-affected quantities are patched for the noised positions P_t.

`precompute_full_geometry` does four things:

### 7a: geo_lr_mask (Left/Right Sibling Assignment) {#7a-geo_lr_mask}

**File**: `helpers.py:529` -> `helpers.py:238-382`

```python
geo_lr_mask = compute_geo_lr_mask(pos, parent_idx, uhat=uhat, debug=debug, eps=eps, tol=tol)
```

This assigns a boolean left/right label to every node based on its geometric relationship to its sibling relative to the parent. The label is used as a **node feature** (not an edge feature) to break the symmetry between siblings.

**Step-by-step**:

1. **Grandparent computation** (lines 256-263):
   ```python
   gp = parent.new_full((N,), -1)
   gp[has_parent] = parent[parents[positive_mask]].clamp(min=-1)
   ```
   For each node with a parent, find its grandparent. Root children have `gp = -1`.

2. **Incoming direction `v_in`** (lines 271-278):
   ```python
   # If node has grandparent: v_in = parent_pos - grandparent_pos
   v_in[sel] = pos[parent[sel]] - pos[gp[sel]]
   # If node has parent but NO grandparent (root children): v_in = global_e1
   v_in[fallback_mask] = global_e1.view(1, -1)
   ```
   `global_e1` is a fixed reference vector in the plane orthogonal to `uhat` (computed via `global_inplane_basis`). This is the `compute_geo_lr_mask`-specific fallback for root children -- it uses a **global reference direction** rather than `v_out`.

3. **Outgoing direction `v_out`** (lines 280-283):
   ```python
   v_out[sel] = pos[sel] - pos[parent[sel]]   # child - parent
   ```

4. **Project onto plane orthogonal to uhat** (lines 285-297):
   ```python
   du_in = (v_in * uhat_vec).sum(dim=-1, keepdim=True)
   v_in_perp = v_in - du_in * uhat_vec
   v_out_perp = v_out - du_out * uhat_vec
   v_in_unit = v_in_perp / (nin + eps)
   v_out_unit = v_out_perp / (nout + eps)
   ```

5. **Compute in-plane angle (cospsi, sinpsi)** (lines 299-304):
   ```python
   cospsi = (v_in_unit * v_out_unit).sum(dim=-1, keepdim=True)
   cross = th.cross(v_in_unit, v_out_unit, dim=-1)
   sinpsi = (cross * uhat_vec).sum(dim=-1, keepdim=True)
   ```
   <!-- <CLARIFY> WHEN VIEWED ALONG uhat? SHOULD THIS NOT BE IN THE PLANE PERPENDICULAR TO uhat? -->
   `sinpsi` is the signed angle component: positive means "left" of the incoming direction when viewed along `uhat`.

6. **Root children handling** (lines 309-327):

   **CRITICAL DECISION -- Single root child**:
   ```python
   single_root_ch = parent_is_root_node & (sibling_count == 1)
   lr_mask = lr_mask.masked_fill(single_root_ch, True)   # always assigned LEFT (=True)
   ```
   A single child of the root is **arbitrarily assigned the left label (True/1)**. There is no sibling to distinguish from, so this is a fixed convention.

   **Two root children**:
   ```python
   multi_root_ch = parent_is_root_node & (sibling_count > 1)
   lr_mask[multi_root_ch] = (
       pos[multi_root_ch, -1] >= pos[parent_clamped[multi_root_ch], -1]
   )
   ```
   <!-- <CLARIFY> THE GOAL OF THIS COMPARISON SHOULD HAVE BEEN TO COMPARE THE TWO SIBLINGS BUT IT FEELS LIKE WE ARE COMPARING EACH SIBLING WITH ROOT? -->
   For two children of the root, the child with **higher z-coordinate** (last component, aligned with `uhat=[0,0,1]`) is labeled left. This is a **position-based** assignment using the uhat axis directly, NOT the sinpsi angle (because root children have no grandparent to define a meaningful `v_in`).

7. **Binary (non-root) parent handling** (lines 329-366):

   For all non-root parents with exactly 2 children:
   - **Case 1**: If `sin(psi_child0) * sin(psi_child1) < -tol` (opposite signs), the child with `sinpsi > 0` is left. This is the clean case where siblings are on opposite sides of the incoming direction.
   - **Case 2**: Same-sign or near-zero sines -- use `atan2(sinpsi, cospsi)` and the child with the larger angle is left.

**Result**: `geo_lr_mask [N]` boolean tensor. `True` = left child, `False` = right child. Root nodes and non-binary-parent nodes are `False` (default).

### 7b: Branch Angles (cospsi, sinpsi, cos_theta) {#7b-branch-angles}

**File**: `helpers.py:532-533` -> `helpers.py:111-181`

```python
cospsi_node, sinpsi_node, cos_theta_node, intermediates = compute_branch_angles_parent_centric(
    pos, parent_idx, uhat, eps=eps, return_intermediates=True,
)
```

Per-node angles describing the branch `parent(i) -> i` relative to the incoming direction at `parent(i)`:

1. **Grandparent lookup** (same as geo_lr_mask):
   ```python
   gp[has_parent] = parent_idx[parent[has_parent]].clamp(min=-1)
   ```

2. **`v_in` computation** (lines 136-144):
   ```python
   # Has grandparent: v_in = parent_pos - grandparent_pos
   v_in[sel] = coors[parent[sel]] - coors[gp[sel]]
   # No grandparent (root children): v_in = v_out (child - parent direction)
   v_in[sel] = coors[sel] - coors[parent[sel]]
   ```

   **CRITICAL DIFFERENCE from geo_lr_mask**: Here root children use `v_in = v_out` (the child-to-parent direction itself), NOT a global reference direction. This means for root children:
   - `v_in_perp` and `v_out_perp` are parallel
   - `cospsi = 1`, `sinpsi = 0` (angle between identical directions is 0)
   - Only `cos_theta` (tilt angle relative to uhat) is meaningful and position-dependent

3. **Perpendicular projections and angles** (lines 152-168):
   ```python
   du_in = (v_in @ uhat).unsqueeze(-1)
   du_out = (v_out @ uhat).unsqueeze(-1)
   v_in_perp = v_in - du_in * uhat
   v_out_perp = v_out - du_out * uhat

   # In-plane angle
   cospsi = (v_in_unit * v_out_unit).sum(dim=-1, keepdim=True)
   sinpsi = (cross * uhat).sum(dim=-1, keepdim=True)

   # Tilt angle (how much v_out tilts relative to uhat axis)
   v_out_norm = (nout.pow(2) + du_out.pow(2)).sqrt()
   cos_theta = du_out / (v_out_norm + eps)
   ```

4. **Root/parentless fallback** (lines 170-172):
   ```python
   cospsi = torch.where(has_parent, cospsi, ones)    # root: cospsi = 1
   sinpsi = torch.where(has_parent, sinpsi, zeros)   # root: sinpsi = 0
   cos_theta = torch.where(has_parent, cos_theta, ones)  # root: cos_theta = 1
   ```

5. **Return intermediates** for leaf patching (lines 174-180):
   ```python
   intermediates = {
       'v_in': v_in,       # [N, 3] incoming direction per node
       'v_out': v_out,     # [N, 3] outgoing direction per node
       'has_gp': has_gp,   # [N] bool: does node have a grandparent?
   }
   ```
   These are needed by `patch_geometry_for_noised_leaves` to efficiently recompute angles only for noised leaves.

**Result**: `cospsi_node [N,1]`, `sinpsi_node [N,1]`, `cos_theta_node [N,1]` -- per-node SO(2) angle features.

### 7c: Edge-Level SO(2) Decomposition {#7c-edge-level-so2-decomposition}

**File**: `helpers.py:537-543`

```python
src, dst = edge_index
rel_coors = pos[dst] - pos[src]                       # [E, 3] relative position vectors
du = (rel_coors @ uhat)                                # [E] component along uhat axis
r_par = du[:, None] * uhat                             # [E, 3] parallel component
r_perp = rel_coors - r_par                             # [E, 3] perpendicular component
rho = r_perp.norm(dim=-1, keepdim=True).clamp_min(eps) # [E, 1] perpendicular distance
du = du[:, None]                                       # [E, 1]
```

Each edge vector `r_ij = pos[j] - pos[i]` is decomposed into:
- **`du`**: signed distance along the SO(2) axis (uhat). Positive = j is "above" i along uhat.
- **`rho`**: perpendicular distance (in the plane orthogonal to uhat). Always positive.
- **`r_perp`**: the 3D perpendicular component (direction in the plane).

These are the SO(2)-equivariant edge features: `rho` and `du` are invariant to rotations around uhat, while `r_perp` transforms equivariantly.

### 7d: Angle Assignment to Edges {#7d-angle-assignment-to-edges}

**File**: `helpers.py:546-551`

```python
cospsi_edge, sinpsi_edge = assign_branch_angles_to_edges(
    edge_index, parent_idx, cospsi_node, sinpsi_node,
)
cos_theta_edge = assign_parent_scalar_to_edges(
    edge_index, parent_idx, cos_theta_node,
)
```

`assign_branch_angles_to_edges` (`helpers.py:184-210`):
For each directed edge, assigns the branch angle of the **child node** in the parent-child relationship:
```python
# Parent -> Child edges: use child's angle
mask_parent_to_child = (parent_idx[dst] == src)
cos_edge[mask_parent_to_child] = cospsi_node[dst[mask_parent_to_child]]

# Child -> Parent edges: use child's (=src's) angle
mask_child_to_parent = (parent_idx[src] == dst)
cos_edge[mask_child_to_parent] = cospsi_node[src[mask_child_to_parent]]
```

Both directions of the same tree edge get the **same** angle features (the child's angles). This is because the angles describe the branch geometry at the parent junction, and both edge directions should see the same geometric context.

### 7e: Full pre_geom_p0 Dictionary

The returned dictionary:

```python
{
    # Edge-level (used by SO2_EGNN layers for message passing)
    'rel_coors': rel_coors,           # [E, 3] dst_pos - src_pos
    'r_perp': r_perp,                 # [E, 3] perpendicular component
    'rho': rho,                       # [E, 1] perpendicular distance (>= eps)
    'du': du,                         # [E, 1] signed axial component
    'cospsi_edge': cospsi_edge,       # [E, 1] in-plane angle cosine
    'sinpsi_edge': sinpsi_edge,       # [E, 1] in-plane angle sine
    'cos_theta_edge': cos_theta_edge, # [E, 1] tilt angle cosine

    # Node-level (needed for patching)
    'cospsi_node': cospsi_node,       # [N, 1]
    'sinpsi_node': sinpsi_node,       # [N, 1]
    'cos_theta_node': cos_theta_node, # [N, 1]

    # Feature
    'geo_lr_mask': geo_lr_mask,       # [N] bool

    # Intermediates for leaf patching
    'v_in': v_in,                     # [N, 3] incoming direction per node
    'v_out': v_out,                   # [N, 3] outgoing direction per node
    'has_gp': has_gp,                 # [N] bool: has grandparent?
}
```

---

## 9. Phase 8: Node Feature Assembly

**File**: `expansion.py:566-637`

The model expects input `x [N, pos_dim + feats_dim]`. The feature channels are allocated:

```
feats_dim = avail_feats_dim + cond_dim + tmd_hidden_dim
```

Where:
- `avail_feats_dim` = channels for structural features (built here)
- `cond_dim = 2` = reserved for diffusion conditioning (e_t, log_sigma -- added later in diffusion)
- `tmd_hidden_dim` = reserved for TMD embedding (added later in model forward)

**Feature channels (in order)**:

| Channel | Dim | Description | Values |
|---------|-----|-------------|--------|
| `is_leaf` | 1 | Whether node is a leaf | 0.0 or 1.0 |
| `geo_lr` | 1 | Left/right sibling label | 0.0 (right/root) or 1.0 (left) |
| `new_leaf_flag` | 1 | Whether node is a "new" training leaf | 0.0 or 1.0 |
| `size_ratio` | 1 | `current_graph_size / total_tree_size` per node | float in (0, 1] |
| padding | remaining | Zero padding to fill `avail_feats_dim` | 0.0 |

```python
features = []
# 1. Is-leaf indicator
is_leaf = pos_gt.new_zeros((pos_gt.size(0), 1))
is_leaf[batch.leaf_idx] = 1.0
features.append(is_leaf)                              # [N, 1]

# 2. Geometry L/R label (from geo_lr_mask)
geo_left = geo_lr_mask.to(dtype=pos_gt.dtype).unsqueeze(-1)
features.append(geo_left)                              # [N, 1]

# 3. New-leaf flag
new_flag = pos_gt.new_zeros((pos_gt.size(0), 1))
new_flag[leaf_idx_next] = 1.0
features.append(new_flag)                              # [N, 1]

# <CLARIFY> IS SIZE STILL OPTIONAL?

# 4. Size ratio
ratio_graph = node_counts / target_sizes.clamp_min(1.0)
ratio_nodes = ratio_graph[batch_vec].unsqueeze(-1)
features.append(ratio_nodes)                           # [N, 1]

# 5. Zero padding
pad = pos_gt.new_zeros((N, avail_feats_dim - feats_used))
features.append(pad)

node_feats = th.cat(features, dim=-1)                  # [N, avail_feats_dim]
```
<!-- <CLARIFY> THIS IS INTERESTING - CATEGORICAL EDGE FEATS ARE EMBEDDED INSIDE THE SO2 MODEL - IS IT CORRECT TO CONVERT TO FLOAT? IS THIS WHAT WE DID PREVIOUSLY? -->
**Edge attributes**:
```python
edge_attr = edge_types.unsqueeze(-1).to(pos_gt.dtype)  # [E, 1] values 0.0 or 1.0
```

---

## 10. Phase 9: Diffusion Forward Pass

**File**: `expansion.py:658-672` -> `basic.py:27-122`

```python
expansion_loss, position_loss = self.diffusion(
    node_feats=node_feats,         # [N, avail_feats_dim]
    edge_index=edge_index,         # [2, E]
    batch=batch.batch,             # [N]
    edge_attr=edge_attr,           # [E, 1]
    P_0=pos_gt,                    # [N, 3] clean ground truth
    C_0=leaf_rel_pos,              # [L, 3] clean relative offsets
    parent_idx=parent_idx,         # [N] with -1 for roots
    leaf_idx_train=leaf_idx_train, # [L]
    leaf_expansion=leaf_expansion, # [L] in {0, 1}
    leaf_parent_idx=leaf_parent_idx, # [L]
    model=model,                   # SO2_EGNN_Network
    tmd=tmd,                       # [B, D] or None
    pre_geom_p0=pre_geom_p0,       # dict from precompute_full_geometry
)
```

### 9a: Noise Sampling {#9a-noise-sampling}

**File**: `basic.py:44-67`

```python
# Map expansion labels from {0,1} to {-1,+1} for symmetric denoising
leaf_expansion = leaf_expansion.to(dtype=P_0.dtype).view(-1, 1)
e_0 = 2.0 * leaf_expansion - 1.0                      # [L, 1] in {-1.0, +1.0}

# Sample one sigma per graph from log-normal distribution
num_graphs = int(batch.max().item()) + 1
sigma_graph = (
    th.randn((num_graphs,), device=device) * self.P_std + self.P_mean  # P_mean=-1.2, P_std=1.2
).exp()
sigma_graph = sigma_graph.clamp(self.sigma_min, self.sigma_max)        # clamp to [0.002, 4.0]
log_sigma_graph = sigma_graph.log()

# Broadcast sigma to leaves
leaf_batch = batch[leaf_idx_train]                     # [L] which graph each leaf belongs to
sigma_leaf = sigma_graph[leaf_batch].view(-1, 1)       # [L, 1]
```

The noise level `sigma` is sampled per-graph (not per-node). All leaves in the same graph share the same sigma. The log-normal sampling concentrates most sigma values around `exp(-1.2) ~ 0.3`.

### 9b: Noising Leaf Positions and Expansions {#9b-noising}

**File**: `basic.py:69-75`

```python
eps_pos = th.randn_like(C_0)                           # [L, 3] position noise
eps_exp = th.randn_like(e_0)                           # [L, 1] expansion noise
C_t = C_0 + sigma_leaf * eps_pos                       # [L, 3] noised relative offset
e_t = e_0 + sigma_leaf * eps_exp                       # [L, 1] noised expansion signal

P_t = P_0.clone()                                      # [N, 3] start from clean positions
P_t[leaf_idx_train] = P_0[leaf_parent_idx] + C_t       # scatter noised leaf positions
```

**Key insight**: Only leaf positions change. `P_t` is identical to `P_0` for all internal nodes. The noised leaf position is `parent_pos + C_t` where `C_t = C_0 + sigma * noise`.

### 9c: Geometry Patching for Noised Leaves {#9c-geometry-patching}

**File**: `basic.py:77-83` -> `helpers.py:575-691`

```python
pre_geom = None
if pre_geom_p0 is not None:
    pre_geom = patch_geometry_for_noised_leaves(
        pre_geom_p0, P_t, leaf_idx_train, parent_idx,
        edge_index, model.uhat,
    )
```

This is the **patch step** of the compute-once optimisation. Instead of recomputing ALL geometry on P_t from scratch, we selectively update only the quantities affected by leaf position changes.

**Step 1: Identify affected edges** (`helpers.py:598-600`):
```python
leaf_set = th.zeros(N, dtype=th.bool, device=device)
leaf_set[leaf_idx_train] = True
affected = leaf_set[src] | leaf_set[dst]               # [E] bool: edges touching any noised leaf
```

An edge is "affected" if either endpoint is a noised leaf. For a tree with L noised leaves, this is at most 2L edges (the parent-child edge in both directions for each leaf).

**Step 2: Patch node-level angles for leaf nodes** (`helpers.py:603-652`):

```python
# New outgoing direction for leaves (parent pos unchanged since parents are internal)
v_out_new = P_t[leaf_idx_train] - P_t[parent_clamped[leaf_idx_train]]  # [L, 3]

# v_in reused from P_0 (parent and grandparent are internal, positions unchanged)
v_in_leaf = v_in_p0[leaf_idx_train]                    # [L, 3]

# SPECIAL CASE: root-child leaves (no grandparent)
is_root_child_leaf = ~has_gp_p0[leaf_idx_train] & (parent[leaf_idx_train] >= 0)
if is_root_child_leaf.any():
    v_in_leaf = v_in_leaf.clone()
    v_in_leaf[is_root_child_leaf] = v_out_new[is_root_child_leaf]  # v_in = v_out fallback
```

**Root-child leaf patching**: When a leaf is a direct child of the root (no grandparent), `v_in = v_out` is the fallback used by `compute_branch_angles_parent_centric`. Since `v_out` changes when the leaf is noised, `v_in` must also be updated to `v_out_new`. This preserves the invariant that root-child angles have `cospsi=1, sinpsi=0` (because `v_in = v_out` means they're parallel), and only `cos_theta` changes based on the new tilt.

```python
# Recompute perpendicular projections and angles for leaves
du_in = (v_in_leaf @ uhat).unsqueeze(-1)
du_out = (v_out_new @ uhat).unsqueeze(-1)
v_in_perp = v_in_leaf - du_in * uhat
v_out_perp = v_out_new - du_out * uhat

cospsi_leaf = (v_in_unit * v_out_unit).sum(dim=-1, keepdim=True)
cross = th.cross(v_in_unit, v_out_unit, dim=-1)
sinpsi_leaf = (cross * uhat).sum(dim=-1, keepdim=True)
v_out_norm = (nout.pow(2) + du_out.pow(2)).sqrt()
cos_theta_leaf = du_out / (v_out_norm + eps)

# Clone and scatter
cospsi_node = pre_geom_p0['cospsi_node'].clone()
cospsi_node[leaf_idx_train] = cospsi_leaf              # overwrite leaf angles only
# (same for sinpsi_node, cos_theta_node)
```

**Step 3: Patch edge-level quantities** (`helpers.py:654-678`):

```python
# Recompute rel_coors for affected edges
rel_coors = pre_geom_p0['rel_coors'].clone()
rel_coors_new = P_t[dst[affected]] - P_t[src[affected]]
rel_coors[affected] = rel_coors_new

# Recompute rho, du for affected edges
du_new = (rel_coors_new @ uhat).unsqueeze(-1)
r_perp_new = rel_coors_new - du_new * uhat
rho_new = r_perp_new.norm(dim=-1, keepdim=True).clamp_min(eps)

du_edge[affected] = du_new
rho_edge[affected] = rho_new
r_perp_edge[affected] = r_perp_new

# Reassign ALL edge angles from patched node angles
# (cheaper than selective reassignment and handles edge cases correctly)
cospsi_edge, sinpsi_edge = assign_branch_angles_to_edges(
    edge_index, parent_idx, cospsi_node, sinpsi_node,
)
cos_theta_edge = assign_parent_scalar_to_edges(
    edge_index, parent_idx, cos_theta_node,
)
```

**Note**: Edge angle reassignment is done for ALL edges (not just affected), because a leaf's parent might have edges to other children where the angle assignment is based on node-level values that haven't changed. This is cheap (just indexing) and correct.

**Result**: `pre_geom` dict with the same format as `pre_geom_p0`, but with leaf-related quantities updated for P_t. The `geo_lr_mask` is NOT included in the patched dict (it's only needed as a node feature, already assembled).

### 9d: Diffusion Conditioning Features {#9d-diffusion-conditioning}

**File**: `basic.py:85-89`

```python
N = P_0.size(0)
e_feat = P_0.new_zeros((N, 1))
e_feat[leaf_idx_train] = e_t                           # [N, 1] noised expansion (0 for non-leaves)

log_sigma_node = log_sigma_graph[batch].view(N, 1)     # [N, 1] log(sigma) broadcast to all nodes

node_feats_t = th.cat([node_feats, e_feat, log_sigma_node], dim=-1)
# node_feats_t: [N, avail_feats_dim + 2]
# The +2 is cond_dim = 2 (e_t channel + log_sigma channel)
```

These two extra channels are the diffusion conditioning:
- `e_feat`: The current noised expansion signal. Zero for non-leaves, tells the model the current expansion state at each leaf.
- `log_sigma_node`: The noise level. Tells the model how much noise was added, so it can calibrate its denoising.

### 9e: Model Input Assembly {#9e-model-input-assembly}

**File**: `basic.py:91`

```python
x_in = th.cat([P_t, node_feats_t], dim=-1)
# x_in: [N, 3 + avail_feats_dim + 2]
# = [N, pos_dim + feats_dim - tmd_hidden_dim]
```

The full input tensor layout:

```
x_in[i] = [ P_t_x, P_t_y, P_t_z,     # 3 dims: (noised) position
             is_leaf,                    # 1 dim: leaf indicator
             geo_lr,                     # 1 dim: left/right label
             new_leaf_flag,              # 1 dim: training leaf indicator
             size_ratio,                 # 1 dim: graph completion ratio
             ...padding...,              # remaining avail_feats_dim
             e_t,                        # 1 dim: noised expansion signal
             log_sigma ]                 # 1 dim: noise level
```

TMD embedding channels are NOT here yet -- they're added inside the model's forward pass.

---

## 11. Phase 10: Model Forward Pass (SO2_EGNN_Network)

**File**: `basic.py:95-103` -> `egnn_so2.py:626-745`

```python
out = model(
    x=x_in,                   # [N, pos_dim + feats_dim - tmd_hidden_dim]
    edge_index=edge_index,     # [2, E]
    batch=batch,               # [N]
    edge_attr=edge_attr,       # [E, 1]
    parent_idx=parent_idx,     # [N]
    tmd=tmd,                   # [B, D] or None
    pre_geom=pre_geom,         # dict (patched for P_t)
)
```

### 10a: TMD Embedding {#10a-tmd-embedding}

**File**: `egnn_so2.py:637-660`

If `tmd_hidden_dim > 0` and TMD data is provided:

```python
tmd_emb = self.tmd_mlp(tmd)                           # [B, tmd_hidden_dim]
# tmd_mlp: Linear(tmd_in_dim -> tmd_hidden_dim) -> SiLU -> Linear(tmd_hidden_dim -> tmd_hidden_dim)

tmd_nodes = tmd_emb[batch]                             # [N, tmd_hidden_dim] broadcast per node
feats = torch.cat([feats, tmd_nodes], dim=-1)          # now feats has full feats_dim
x = torch.cat([coors, feats], dim=-1)                  # [N, pos_dim + feats_dim]
```

After this, `x[:, :pos_dim]` = positions, `x[:, pos_dim:]` = features of dimension `feats_dim`.

### 10b: Geometry Routing (pre_geom) {#10b-geometry-routing}

**File**: `egnn_so2.py:672-678`

```python
if pre_geom is None:
    static_coords = all(
        (not getattr(L, 'update_coors', True)) for L in self._iter_egnn_layers()
    )
    if parent_idx is not None and static_coords:
        pre_geom = self._compute_static_so2_geometry(
            x[:, :self.pos_dim], edge_index, parent_idx
        )
```

Since `pre_geom` was passed from the diffusion (already patched for P_t), this block is **skipped entirely**. The model reuses the externally provided geometry. This is the compute-once payoff: the model does NOT call `_compute_static_so2_geometry` internally.

If `pre_geom` were `None` (e.g., during sampling where we don't use the optimisation), the model would compute it internally using the same angle functions but called on the current positions.

### 10c: MPNN Layer Loop {#10c-mpnn-layers}

**File**: `egnn_so2.py:681-718`

```python
for i, layer in enumerate(self.mpnn_layers):
    # Edge embedding (only first time)
    if edges_need_embedding:
        edge_attr = embedd_token(edge_attr, ...)
        edges_need_embedding = False

    # Optional global attention + EGNN
    if isinstance(layer, nn.ModuleList):  # [GlobalLinearAttention, SO2_EGNN]
        # (a) Make per-graph global tokens
        tokens, tokens_batch = self._make_global_tokens(self.global_tokens, batch)
        # (b) ISAB on features (cross-attention with tokens)
        feats, _ = layer[0](feats, tokens, x_batch=batch, q_batch=tokens_batch)
        # (c) Merge and pass through EGNN
        x = torch.cat([coors, feats], dim=-1)
        x = layer[1](x, edge_index, edge_attr, batch=batch, parent_idx=parent_idx, pre_geom=pre_geom)
    else:
        # Regular EGNN layer
        x = layer(x, edge_index, edge_attr, batch=batch, parent_idx=parent_idx, pre_geom=pre_geom)
```

Each layer receives `pre_geom` and reuses it. Positions don't update (`update_coors=False`), so the geometry stays valid.

### 10d: SO2_EGNN Single Layer Detail {#10d-so2-egnn-layer}

**File**: `egnn_so2.py:281-357`

```python
def forward(self, x, edge_index, edge_attr=None, batch=None, parent_idx=None, pre_geom=None):
    coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]

    # Use precomputed geometry
    if pre_geom is not None:
        rel_coors = pre_geom['rel_coors']
        r_perp = pre_geom['r_perp']
        rho = pre_geom['rho']
        du = pre_geom['du']
        cospsi_edge = pre_geom.get('cospsi_edge')
        sinpsi_edge = pre_geom.get('sinpsi_edge')
        cos_theta_edge = pre_geom.get('cos_theta_edge')
    else:
        # Compute from scratch (fallback)
        ...
```

**Edge feature assembly**:
```python
# Base edge scalars (SO(2) invariants)
base_feats = [rho_feat, du_feat]                      # [E, 1] each (or [E, rbf_k] with RBF)

# Local branch angles (if enabled)
if self.add_local_angles:
    base_feats.extend([cospsi_edge, sinpsi_edge, cos_theta_edge])  # [E, 1] each

# Combine with edge_attr (direction type: 0=parent->child, 1=child->parent)
edge_attr_feats = torch.cat([edge_attr] + base_feats, dim=-1)
# edge_attr_feats: [E, 1 + 2 + 3] = [E, 6]  (without RBF, with angles)
```

**Message passing**:
```python
# Message computation (per edge)
def message(self, x_i, x_j, edge_attr):
    m_ij = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
    # input: [feats_i || feats_j || edge_scalars]
    # edge_mlp: Linear -> Dropout -> SiLU -> Linear -> SiLU
    return m_ij                                        # [E, m_dim]

# Aggregation and node update
m_i = self.aggregate(m_ij, ...)                        # [N, m_dim] sum/mean over neighbors
hidden_out = self.node_mlp(torch.cat([feats, m_i], dim=-1))
hidden_out = feats + hidden_out                        # residual connection
```

**Output**: `torch.cat([coors_out, hidden_out], dim=-1)` where `coors_out = coors` (unchanged).

### 10e: Offset Head Decoding {#10e-offset-head}

**File**: `egnn_so2.py:720-745`

After all MPNN layers:

```python
coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]

if not self.LR_offset_head:
    # Single head
    offset_state = self.offset_head(feats)             # [N, 4]
    # offset_head: Linear(feats_dim -> hidden) -> SiLU -> Linear(hidden -> 4)
else:
    # Dual head (class 0 = right children, class 1 = left children)
    class_feature = x[:, self.pos_dim + 1]             # geo_lr feature (2nd feature channel)
    class_mask = (class_feature > 0.5).unsqueeze(-1)   # True = left child
    head0 = self.offset_head_class0(feats)             # [N, 4]
    head1 = self.offset_head_class1(feats)             # [N, 4]
    offset_state = torch.where(class_mask, head1, head0)

rel_pred = offset_state[:, :3]                         # [N, 3] predicted relative offset
expansion_pred = offset_state[:, 3:4]                  # [N, 1] predicted expansion signal
```

**Return**:
```python
return {
    "node_state": x,              # [N, pos_dim + feats_dim] final embeddings
    "rel_pred": rel_pred,          # [N, 3] predicted C_0 for all nodes
    "expansion_pred": expansion_pred,  # [N, 1] predicted e_0 for all nodes
}
```

---

## 12. Phase 11: Loss Computation

**File**: `basic.py:112-122`

```python
# Extract predictions only at training leaf positions
C_pred = rel_pred_all[leaf_idx_train]                  # [L, 3]
e_pred = exp_pred_all[leaf_idx_train]                  # [L, 1]

# MSE losses
pos_loss = F.mse_loss(C_pred, C_0)                     # predicted offset vs clean offset
exp_loss = F.mse_loss(e_pred, e_0)                     # predicted expansion vs clean expansion
return exp_loss, pos_loss
```

Back in `expansion.py:680`:
```python
loss = position_loss + self.expansion_loss_weight * expansion_loss
```

The model is trained to denoise: given noised leaf positions and expansion signals, predict the clean (denoised) values. The loss is the MSE between predictions and ground truth.

**What the model learns**: Given `(P_t, e_t, sigma)`, predict `(C_0, e_0)`. At each sigma level, the model sees the current noisy state and must output what the clean state should be. This is the "denoiser" formulation of diffusion models.

---

## 13. Root and Single-Child Edge Cases

### Case 1: Root has a single child

**Tree**: `root(0) -> node(1) -> {node(2), node(3)}`

**geo_lr_mask** (`compute_geo_lr_mask`):
- Node 1 is a root child with `sibling_count == 1`
- `single_root_ch` mask is True for node 1
- `lr_mask[1] = True` -- **arbitrarily assigned LEFT**
- This is a fixed convention: single root children are always "left"

**Branch angles** (`compute_branch_angles_parent_centric`):
- Node 1 has no grandparent (`has_gp[1] = False`)
- Fallback: `v_in[1] = coors[1] - coors[parent[1]] = coors[1] - coors[0]`
- This equals `v_out[1]`, so:
  - `cospsi[1] = 1.0` (parallel directions)
  - `sinpsi[1] = 0.0` (no angular deviation)
  - `cos_theta[1]` = axial tilt, depends on actual positions

**Patching** (`patch_geometry_for_noised_leaves`):
- If node 1's children (2, 3) are noised leaves:
  - `v_in` for nodes 2 and 3 is `coors[1] - coors[0]` (parent - grandparent, unchanged since both are internal)
  - Only `v_out` changes (noised child positions)
  - `is_root_child_leaf` check does NOT apply (nodes 2, 3 have grandparent = node 0)
- If node 1 itself is somehow a noised leaf (edge case):
  - `is_root_child_leaf[1] = True` (no grandparent AND parent >= 0)
  - `v_in_leaf[1] = v_out_new[1]` (preserving the `v_in = v_out` fallback)
  - Result: `cospsi = 1`, `sinpsi = 0`, only `cos_theta` changes

### Case 2: Root has two children

**Tree**: `root(0) -> {node(1), node(2)}`

**geo_lr_mask**:
- `multi_root_ch` is True for nodes 1 and 2
- Assignment: `lr_mask[i] = pos[i, -1] >= pos[parent[i], -1]`
- The child with higher z-coordinate (along uhat) is "left"
- This uses **absolute position** rather than angular geometry (no meaningful `v_in` for root children)

**Branch angles**:
- Both nodes 1 and 2 have `has_gp = False`
- `v_in = v_out` for both (same fallback as single-child case)
- `cospsi = 1, sinpsi = 0` for both
- `cos_theta` differs based on their different positions

**Key insight**: For root children, the geo_lr_mask uses a **different method** than the branch angles. The L/R label comes from z-coordinate comparison (absolute, position-based), while the angles use the `v_in = v_out` fallback (giving trivial cospsi=1, sinpsi=0). These are independent features serving different purposes: geo_lr breaks symmetry as a node feature, while angles provide edge-level geometric context.

### Case 3: Two-node tree (root + one leaf)

**Tree**: `root(0) -> leaf(1)`

- Node 1 is both a root child and a leaf
- `geo_lr_mask[1] = True` (single root child convention)
- `cospsi[1] = 1, sinpsi[1] = 0` (v_in = v_out fallback)
- If node 1 is noised during diffusion:
  - `is_root_child_leaf = True`
  - `v_in_leaf = v_out_new` (updated fallback)
  - Only `cos_theta` and edge-level `rho, du, rel_coors` are affected

---

## 14. Geometry Compute-Once / Patch-Leaves Summary

### Why it works

During training diffusion, only leaf positions are noised:
```python
P_t = P_0.clone()
P_t[leaf_idx_train] = P_0[leaf_parent_idx] + C_t       # only leaves change
```

For any leaf node `l`:
- Its **parent** `p = parent[l]` is always an internal node (leaves can't be parents of training leaves)
- Its **grandparent** `gp = parent[p]` (if it exists) is also internal
- Therefore `v_in[l] = pos[p] - pos[gp]` is **unchanged** between P_0 and P_t (both p and gp are internal)
- Only `v_out[l] = P_t[l] - P_t[p]` changes (because P_t[l] is noised)

This means:
1. **geo_lr_mask**: Computed once on P_0 and NEVER patched. It's used as a node feature that's fixed for the batch. (The diffusion doesn't change which child is "left" -- the feature is based on clean geometry.)
2. **Node angles for internal nodes**: Unchanged (all their relevant positions are unchanged)
3. **Node angles for leaf nodes**: Only `v_out` changes, `v_in` is reused from P_0 (except root-child leaves where `v_in = v_out` must be updated)
4. **Edge-level quantities**: Only edges touching a noised leaf are affected (at most 2L edges out of 2(N-1) total)
5. **Edge angle assignment**: Rerun on all edges from the patched node angles (cheap indexing operation)

### What's saved

Without optimisation (old path):
- `compute_geo_lr_mask(P_0)` in `expansion.py` -- full O(N) computation
- `_compute_static_so2_geometry(P_t)` inside `SO2_EGNN_Network.forward()` -- full O(N+E) computation

With optimisation (new path):
- `precompute_full_geometry(P_0)` -- one O(N+E) computation (replaces both old calls)
- `patch_geometry_for_noised_leaves(pre_geom_p0, P_t)` -- O(L) node patches + O(L) edge patches

For typical dendrite trees where L << N (training leaves are a small fraction of all nodes), this is a significant saving.

### What's NOT optimised

- **Sampling** (`DenoisingDiffusionModel.sample`): Positions change at every denoising step (not just leaves), so the compute-once optimisation doesn't apply. The model computes geometry internally via `_compute_static_so2_geometry` at each step.
- **geo_lr_mask during expansion** (`Expansion.expand`): Recomputed from scratch after each expansion step because positions are genuinely new (denoised leaf positions are placed).

---

## Appendix: Tensor Shape Summary

For a batch with N total nodes, E edges, L training leaves, B graphs:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `P_0` / `pos_gt` | `[N, 3]` | Ground truth positions |
| `P_t` | `[N, 3]` | Noised positions (leaves modified) |
| `parent_idx` | `[N]` | 0-based parent indices, -1 for roots |
| `edge_index` | `[2, E]` | Directed edges (E = 2*(N - num_roots)) |
| `edge_attr` | `[E, 1]` | Edge type (0=parent->child, 1=child->parent) |
| `batch` | `[N]` | Graph assignment vector |
| `leaf_idx_train` | `[L]` | Indices of training leaves |
| `leaf_parent_idx` | `[L]` | Parent indices for training leaves |
| `C_0` | `[L, 3]` | Clean relative offsets (leaf - parent) |
| `C_t` | `[L, 3]` | Noised relative offsets |
| `e_0` | `[L, 1]` | Clean expansion signal in {-1, +1} |
| `e_t` | `[L, 1]` | Noised expansion signal |
| `sigma_graph` | `[B]` | Noise level per graph |
| `node_feats` | `[N, avail_feats_dim]` | Structural features (pre-diffusion) |
| `node_feats_t` | `[N, avail_feats_dim + 2]` | Structural + diffusion conditioning |
| `x_in` | `[N, 3 + avail_feats_dim + 2]` | Full model input |
| `rel_pred` | `[N, 3]` | Predicted relative offsets (all nodes) |
| `expansion_pred` | `[N, 1]` | Predicted expansion signal (all nodes) |
| `geo_lr_mask` | `[N]` | Boolean left/right sibling label |
| `cospsi_node` | `[N, 1]` | In-plane branch angle cosine |
| `sinpsi_node` | `[N, 1]` | In-plane branch angle sine |
| `cos_theta_node` | `[N, 1]` | Axial tilt angle cosine |
| `rho` | `[E, 1]` | Perpendicular edge distance |
| `du` | `[E, 1]` | Signed axial edge component |
| `rel_coors` | `[E, 3]` | Full edge displacement vectors |
