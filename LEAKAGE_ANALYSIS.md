# Information Leakage Analysis: Training vs Sampling Feature Consistency

This document audits every piece of information that reaches the model (EGNN) during
**training** (`get_loss` -> `diffusion.forward` -> `model(...)`) and **sampling**
(`expand` -> `diffusion.sample` -> `model(...)`), checking for leakage of ground-truth
information that the model should not have access to.

---

## 1. Executive Summary

| Channel | Training | Sampling | Verdict |
|---------|----------|----------|---------|
| **Positions `x[:, :3]`** | P_t (noised leaves) | P_cur (noised leaves) | OK |
| **Feature slot 0: is_leaf** | From GT `leaf_idx` | From `leaf_mask_next` | OK |
| **Feature slot 1: geo_ordinal** | From GT positions (P_0) | From initial ordinal assignment | **REVIEW** (see 3) |
| **Feature slot 2: new_leaf_mask** | From `new_leaf_mask_from_next` (GT) | From newly spawned nodes | OK |
| **Feature slot 3: size_ratio** | From GT `total_tree_size` | From `target_size` | OK |
| **Feature slot 4+: padding** | Zeros | Zeros | OK |
| **Cond slot 0: e_t** | Noised expansion label | Noised expansion estimate | OK |
| **Cond slot 1: log_sigma** | Per-graph log sigma | Per-step log sigma | OK |
| **Edge attr** | Edge type (P->C / C->P) | Edge type (P->C / C->P) | OK |
| **Edge geometry (rho, du)** | From P_t (patched) | From P_cur (recomputed) | OK |
| **Edge angles (cospsi, sinpsi, cos_theta)** | From P_t (patched) | From P_cur (recomputed) | **REVIEW** (see 4) |
| **Local frames (forward, sideways)** | From GT P_0 | From parent geometry | OK |
| **parent_idx** | From GT tree | From constructed tree | OK |
| **pre_geom dict** | Patched from P_0 | Recomputed from P_cur | **REVIEW** (see 4) |
| **TMD** | From GT graph | From GT graph (given) | OK (conditioning) |
| **LR_offset_head class feature** | geo_ordinal (slot 1) | geo_ordinal (slot 1) | OK |

---

## 2. Detailed Channel-by-Channel Analysis

### 2.1 Positions: `x[:, :3]` = P_t / P_cur

**Training** (`basic.py:77-86`):
```python
C_t = C_0 + sigma_leaf * eps_pos        # noised local-frame offset
C_t_global = local_to_global(C_t, ...)  # convert to global
P_t[leaf_idx_train] = P_0[leaf_parent_idx] + C_t_global
```
- Internal (non-leaf) nodes retain their **GT positions** P_0.
- Leaf nodes get noised positions: parent_pos + noised_offset.
- The model sees P_t, where leaf positions are corrupted by Gaussian noise scaled by sigma.

**Sampling** (`basic.py:208-215`):
```python
P_cur = P_0.clone()                      # parents at their current positions
C_global = local_to_global(C, ...)       # current denoised estimate
P_cur[leaf_idx] = parent_pos + C_global
```
- Internal nodes retain their previously-placed positions.
- Leaf nodes get current denoised estimate positions.

**Verdict: OK.** In both cases, the model sees noised/estimated leaf positions, never
GT leaf positions. Internal node positions are "GT" in training but equivalently "placed"
in sampling -- the model needs to know where the tree built so far is located. This is
consistent: in sampling the internal nodes are the result of prior expansion steps.

### 2.2 Feature Slot 0: `is_leaf` (binary indicator)

**Training** (`expansion.py:596-598`):
```python
is_leaf = zeros(N, 1)
is_leaf[batch.leaf_idx] = 1.0           # marks ALL leaves in the reduction level
```

**Sampling** (`expansion.py:352-354`):
```python
is_leaf = leaf_mask_next.float().unsqueeze(-1)  # marks all current leaf nodes
```

**Verdict: OK.** Both tell the model which nodes are leaves. The model needs this to know
which nodes are "active" (need predictions). In training, leaf status comes from the
reduction; in sampling, from the expansion state. Both are structural, not positional GT.

### 2.3 Feature Slot 1: `geo_ordinal` (sibling ordering)

**Training** (`expansion.py:602-606`):
```python
geo_ordinal = pre_geom_p0['geo_ordinal']        # computed from GT P_0
geo_feat = geo_ordinal.clamp(min=0.0).unsqueeze(-1)
```
- `geo_ordinal` is computed by `compute_root_child_angles` (helpers.py:671-796) from **GT positions P_0**.
- For root children: clockwise angular ordering from x-axis, ordinal `i/(k-1)`.
- For binary interior children: left=0.0, right=1.0 (from `compute_geo_lr_mask` using GT P_0).
- Sentinel -1.0 for root nodes, clamped to 0.0.

**Sampling** (`expansion.py:290-296`):
```python
ordinal_t = tensor(ordinal_new, ...)
sib_count_t = tensor(sibling_count_new, ...)
geo_angle_new = ordinal_t / sib_count_t.clamp(min=2.0).sub(1.0)
```
- For root children: ordinal assigned as `i / max(k-1, 1)` where `i` is the spawn order.
  The spawn order is arbitrary (sequential 0..k-1 as they're created in the loop).
- For binary children: similarly `0 / 1` and `1 / 1` = 0.0 and 1.0.
- Post-diffusion refinement via `compute_geo_angle_for_new_leaves` re-orders based on
  actual denoised positions (helpers.py:928-1038).

**Is there leakage?**

The geo_ordinal in training encodes the **relative geometric ordering of siblings** from
GT positions. This is a labeling convention -- it tells the model "you are child 0" or
"you are child 1 of 3" etc. The model uses this to differentiate between siblings sharing
the same parent.

During sampling, the initial ordinal assignment (before diffusion) is arbitrary/sequential.
After diffusion places the children, `compute_geo_angle_for_new_leaves` refines the
ordinals based on actual positions -- establishing the same clockwise-from-x convention.

**Key question:** Does the GT ordinal in training leak positional information?

The ordinal `i/(k-1)` only encodes **rank** (which sibling is "leftmost" clockwise), not
the actual angle or distance. For k=2, it's always 0.0 and 1.0. The model can't recover
GT positions from an ordinal rank.

However, there is a **subtle inconsistency**: in training, the ordinal is computed from
GT P_0 and stays fixed even as leaves are noised. In sampling, the ordinal is initially
arbitrary, then refined after diffusion. But **within** the diffusion loop, the ordinal
used as a feature is the **initial arbitrary assignment**, not the refined one.

**Impact:** Low. The ordinal primarily serves as a sibling disambiguation signal. The
model sees the same value at every diffusion step in both training and sampling.

**Verdict: ACCEPTABLE.** The ordinal is a labeling convention, not positional GT. The
user confirmed this is the intended design: "the geometric ordering which we pass from
the GT in training, but it is arbitrary in sampling so it should be fine."

### 2.4 Feature Slot 2: `new_leaf_mask` (newly expanded indicator)

**Training** (`expansion.py:609-623`):
```python
new_mask = batch.new_leaf_mask_from_next     # from dataset: which nodes were just expanded
```
- Comes from `reduction_dataset.py`, marking nodes that appeared when going from level L+1
  to level L (i.e., nodes that are "new" at this reduction level).

**Sampling** (`expansion.py:358-360`):
```python
new_flag = zeros(N, 1)
new_flag[leaf_idx_next] = 1.0               # newly spawned children
```

**Verdict: OK.** Both mark the nodes that were just created. In training, from reduction
metadata; in sampling, from the expansion step. Structurally equivalent.

### 2.5 Feature Slot 3: `size_ratio` (graph progress indicator)

**Training** (`expansion.py:626-634`, calls `size_ratio_feature_from_batch`):
```python
size_ratio = current_node_count / total_tree_size
```
- `total_tree_size` comes from `batch.total_tree_size` (GT value from dataset).
- `current_node_count` = number of nodes at this reduction level.

**Sampling** (`expansion.py:362-367`):
```python
ratio_graph = node_counts_per_graph / target_size.clamp_min(1.0)
ratio_nodes = ratio_graph[batch_new]
```
- Uses `target_size` (the generation target) and current node count.

**Verdict: OK.** In training, `total_tree_size` is the number of nodes in the original
tree. In sampling, `target_size` is the generation target. Both serve the same purpose:
telling the model how far along the generation process is. The model doesn't learn to
"cheat" from this -- it's a progress indicator.

### 2.6 Conditioning Slot 0: `e_t` (noised expansion state)

**Training** (`basic.py:60-61, 76-78`):
```python
e_0 = 2.0 * leaf_expansion - 1.0   # GT: {0,1} -> {-1, +1}
e_t = e_0 + sigma_leaf * eps_exp   # noised
e_feat[leaf_idx_train] = e_t       # only at leaf positions
```

**Sampling** (`basic.py:189, 218-219`):
```python
e = randn(L, 1) * sigma_init      # pure noise initially
e_feat[leaf_idx] = e               # current estimate
```

**Verdict: OK.** In training, the GT expansion label is noised. In sampling, it starts
from pure noise. The model denoises both identically. No leakage.

### 2.7 Conditioning Slot 1: `log_sigma` (noise level indicator)

**Training** (`basic.py:100`):
```python
log_sigma_node = log_sigma_graph[batch].view(N, 1)   # per-graph sigma, broadcast to nodes
```

**Sampling** (`basic.py:220`):
```python
log_sigma_feat = P_0.new_full((N, 1), log_sigma)     # per-step sigma, same for all nodes
```

**Verdict: OK.** In training, sigma is randomly sampled per graph. In sampling, sigma
follows a deterministic schedule. Both tell the model the current noise level. Standard
diffusion conditioning.

### 2.8 Edge Attributes: `edge_attr`

**Training** (`expansion.py:645-648`):
```python
edge_attr = edge_types.unsqueeze(-1).to(pos_gt.dtype)
# edge_types: 0 = parent->child, 1 = child->parent
```

**Sampling** (`expansion.py:385-388`):
```python
edge_attr = edge_types.unsqueeze(-1).to(pos_new.dtype)
# Same encoding
```

**Verdict: OK.** Both encode edge direction type. Derived from `parent_idx` structure.

---

## 3. Geometric Features (Edge-Level) -- CRITICAL SECTION

### 3.1 Training: Precompute on P_0, then Patch for P_t

**Step 1** -- `precompute_full_geometry(P_0, ...)` (`helpers.py:1169-1245`):
- Computes ALL geometry from GT positions P_0.
- Returns: `rel_coors, rho, du, r_perp` (edge), `cospsi, sinpsi, cos_theta` (node/edge),
  `v_in, v_out, has_gp` (intermediates), `local_forward, local_sideways` (local frames),
  `geo_lr_mask, geo_ordinal, geo_delta_theta`.

**Step 2** -- `patch_geometry_for_noised_leaves(pre_geom_p0, P_t, ...)` (`helpers.py:1248-1362`):
- **Only patches quantities affected by leaf noising.**
- **v_out** for leaves: recomputed from P_t (noised) positions.
- **v_in** for leaves: **REUSED FROM P_0** (parent/grandparent positions unchanged).
- **Root-child leaves: v_in locked to P_0-based frame** (lines 1288-1290).
- **cospsi, sinpsi, cos_theta** for leaves: recomputed using P_t v_out + P_0 v_in.
- **Edge quantities** (rel_coors, du, rho, r_perp): recomputed only for edges touching leaves.
- **Edge angles**: re-assigned from patched node angles.

**What the model receives:**
```
pre_geom = patched dict from P_t
```
- For edges between internal nodes: P_0 geometry (correct -- these nodes aren't noised).
- For edges touching leaves: geometry from P_t (noised positions).
- Node angles at leaves: computed from P_t v_out + P_0 v_in.

### 3.2 Sampling: No Pre-Geometry, Recomputed from P_cur

**In `diffusion.sample`** (basic.py:226-233):
```python
out = model(
    x=x_in,          # contains P_cur in first 3 dims
    edge_index=edge_index,
    batch=batch,
    edge_attr=edge_attr,
    parent_idx=parent_idx,
    **model_kwargs,
)
```
**No `pre_geom` is passed.** The model falls through to the `else` branch
(`egnn_so2.py:305-316`):
```python
rel_coors = coors[dst] - coors[src]   # from P_cur
du = (rel_coors @ self.uhat)
r_perp = rel_coors - du[:, None] * self.uhat
rho = r_perp.norm(...)
```
And if `add_local_angles=True` (`egnn_so2.py:317-328`):
```python
cospsi_node, sinpsi_node, cos_theta_node = compute_branch_angles_parent_centric(
    coors, parent_idx, self.uhat, ...
)
```
**This recomputes ALL geometry from P_cur**, including for internal nodes.

### 3.3 Consistency Analysis: Training vs Sampling Edge Geometry

| Quantity | Training (internal edges) | Training (leaf edges) | Sampling (all edges) |
|----------|--------------------------|----------------------|---------------------|
| rel_coors | From P_0 (GT) | From P_t (noised) | From P_cur (estimated) |
| rho, du | From P_0 | From P_t | From P_cur |
| cospsi, sinpsi | From P_0 | From P_t v_out + P_0 v_in | From P_cur |
| cos_theta | From P_0 | From P_t | From P_cur |

**Key observation about internal-node edges in training:**

In training, edges between internal (non-leaf) nodes have their geometry computed from
**GT P_0 positions**. In sampling, these same edges have geometry from the positions
placed by **prior expansion steps** (which are imperfect).

**Is this leakage?** No, because:
1. Internal node positions ARE the GT positions in training (they were never noised).
2. In sampling, internal node positions are the result of prior denoising steps.
3. Both are the "best available" positions for those nodes. The model doesn't get
   extra information about leaf positions from internal-node edge geometry.

### 3.4 v_in for Leaves: P_0 vs Recomputed

**Training**: `v_in` for leaves is reused from P_0 (parent→grandparent direction).
Since parents and grandparents are internal nodes (not noised), this is consistent --
the parent/grandparent positions in P_t are identical to P_0.

**Sampling**: `v_in` is computed from P_cur. Since parent/grandparent positions are
their placed positions (from prior steps), this is also the parent→grandparent direction
at placed positions.

**Verdict: OK.** The v_in direction comes from the same source (parent/grandparent
positions) in both cases. In training these happen to be GT; in sampling they're placed.

### 3.5 Root-Child v_in Locking

**Training** (`helpers.py:1288-1290`): Root-child leaves have v_in locked to the P_0-based
per-child frame. This frame was computed from `geo_delta_theta` in `compute_local_bases`.

**Sampling**: Root children are spawned with structured rotation frames from
`compute_local_bases_for_leaves` (random base angle + 2pi*i/k rotation). These frames
are passed as `local_forward`/`local_sideways` and remain fixed through all diffusion steps.

**Are these consistent?** Yes -- in both cases, root-child frames are fixed before
diffusion begins and never updated during denoising. The frames define the local coordinate
system, not positional information.

---

## 4. Local Frames: Training vs Sampling

### 4.1 Training Local Frames

Computed in `precompute_full_geometry` -> `compute_local_bases` (helpers.py:160-277):

For node i with parent p and grandparent g:
- `v_in = pos[p] - pos[g]` (from GT P_0)
- `forward = normalize(project_perp(v_in, uhat))`
- `sideways = uhat x forward`

For root children with `geo_delta_theta`:
- Child 0: `forward = normalize(project_perp(pos[child0] - pos[root], uhat))`
- Child i: `forward = rotate(forward_child0, delta_theta_i, uhat)`

These frames are then passed to `diffusion.forward` as `local_forward`, `local_sideways`
and used to:
1. Convert GT C_0 (global) -> local frame
2. Noise in local frame (isotropic)
3. Convert noised C_t (local) -> global for P_t construction
4. Compute loss: model predicts C_pred in local frame, compared to C_0 in local frame

**The frames are derived from P_0 (GT) but this is NOT leakage because:**
- The frames define the **coordinate system** for the denoising target
- They are derived from **parent/grandparent** geometry, not from the leaf itself
- The model predicts offsets in this frame -- it doesn't receive the frame as a feature
  (the frame is used externally for coordinate conversion, not fed to the network)

Wait -- does `pre_geom` contain the local frames? Let's check:

The `pre_geom` dict contains `local_forward` and `local_sideways` in the return of
`precompute_full_geometry` (helpers.py:1243-1244). However, `patch_geometry_for_noised_leaves`
does **NOT** include local frames in its return dict (helpers.py:1351-1362). So patched
`pre_geom` passed to the model does NOT contain local frames.

The model (`egnn_so2.py:297-304`) only uses:
- `rel_coors, r_perp, rho, du` (edge-level)
- `cospsi_edge, sinpsi_edge, cos_theta_edge` (angle features)

**Local frames are NOT passed to the model.** They are only used externally for
local<->global coordinate conversion of the denoising target.

### 4.2 Sampling Local Frames

Computed in `compute_local_bases_for_leaves` (helpers.py:280-391):

For leaves with grandparent available:
- `v_in = pos[parent] - pos[grandparent]` (from current positions)

For root children (degenerate, all at parent position):
- Random base angle + structured rotation by `2pi * ordinal / k`

These frames are passed to `diffusion.sample` as `local_forward`, `local_sideways`
and used to convert between local and global frames at each diffusion step.

**Verdict: OK.** Local frames are never fed to the model. They're external scaffolding.

---

## 5. Branch Angle Features -- DETAILED ANALYSIS

The model receives `cospsi_edge`, `sinpsi_edge`, `cos_theta_edge` as edge features.
These encode the **branch angle** between v_in and v_out at each node, projected onto
the plane perpendicular to uhat.

### 5.1 Definition

For a node n with parent p and grandparent g:
- `v_in = pos[p] - pos[g]` (direction coming into parent)
- `v_out = pos[n] - pos[p]` (direction going out to child)
- `psi` = angle between v_in_perp and v_out_perp in the plane perpendicular to uhat
- `theta` = elevation angle of v_out relative to the perp plane
- `cospsi = dot(v_in_perp_unit, v_out_perp_unit)`
- `sinpsi = (v_in_perp_unit x v_out_perp_unit) . uhat`
- `cos_theta = du / ||v_out||`

### 5.2 Training: Angle Features Under Noising

In `patch_geometry_for_noised_leaves` (helpers.py:1275-1323):

For **leaf nodes**:
- `v_out_new = P_t[leaf] - P_t[parent[leaf]]` — uses **noised** leaf position
- `v_in_leaf = v_in_p0[leaf_idx_train]` — uses **P_0** v_in (parent→grandparent unchanged)
- Angles recomputed from noised v_out + GT v_in

For **internal nodes**: angles unchanged from P_0.

**What information do these angles carry for leaf nodes?**
- `v_out_new` is derived from P_t (noised), so it carries noised positional info
- `v_in` is from P_0, but this is the parent→grandparent direction, which is unchanged
  in P_t (parents are internal, not noised)

So the angles at leaves encode the **noised** branch angle, not the GT angle.

### 5.3 Sampling: Angle Features

In `diffusion.sample`, no `pre_geom` is passed. The model recomputes from P_cur
(`egnn_so2.py:320-328`):
```python
cospsi_node, sinpsi_node, cos_theta_node = compute_branch_angles_parent_centric(
    coors, parent_idx, self.uhat, ...
)
```

This uses `coors` = P_cur positions, including noised leaf positions.

**Consistency check:** Both training and sampling compute branch angles from the
current (noised/estimated) positions. The v_in in training comes from P_0 for internal
nodes, which equals P_t for internal nodes. The v_in in sampling comes from P_cur for
internal nodes, which are placed positions from prior steps.

**Are they equivalent?** Yes -- in both cases, v_in is computed from parent/grandparent
positions, which are fixed (not being denoised in this step).

### 5.4 A Subtle Point: `compute_branch_angles_parent_centric` in Sampling

During sampling, `compute_branch_angles_parent_centric` is called on **all** nodes,
including internal ones. For internal nodes, it computes v_in from their actual
grandparent positions and v_out from their actual child positions. For leaf nodes,
v_out uses the noised position.

**But wait**: in the sampling diffusion loop, the leaf positions are being denoised.
At step t, the leaf positions are at P_cur = parent_pos + C_global, where C is the
current noisy estimate. The angle features at each step reflect the current noisy
state -- exactly as in training where P_t has noised leaf positions.

**Verdict: OK.** Angles are consistently computed from current (noised) positions.

---

## 6. What About `pre_geom` in Training vs Its Absence in Sampling?

### 6.1 The Asymmetry

In training, `diffusion.forward` receives `pre_geom` (patched) and passes it to the model.
In sampling, `diffusion.sample` does NOT pass `pre_geom` -- the model recomputes geometry.

### 6.2 Does This Matter?

The `pre_geom` dict provides precomputed edge-level geometry to avoid recomputation.
When it's absent, the model computes the exact same quantities from `coors` (P_cur).

**Potential discrepancy:** In training, patching is selective -- only edges touching
leaves get recomputed. Internal-internal edges retain P_0 geometry. In sampling,
ALL edges get recomputed from P_cur.

**Is this a problem?** No, because:
- Internal node positions are identical in P_0 and P_t (training) -- only leaves are noised
- So internal-internal edge geometry from P_0 == from P_t
- Leaf-touching edges are patched in training and recomputed in sampling -- from the same
  positions (P_t / P_cur)

**The optimization is equivalent to full recomputation.** The patching is a performance
optimization, not a semantic difference.

**Verdict: OK.** No leakage from this asymmetry.

---

## 7. The `parent_idx` Question

### 7.1 Training

`parent_idx` comes from the GT tree structure via `batch.parent_idx_1b` (reduced to
0-based by `decode_parent_indices`). This is the **full tree structure** of the
reduction level being trained on.

### 7.2 Sampling

`parent_idx` is built incrementally as children are spawned:
```python
parent_idx_new_0b = cat([parent_idx, tensor(new_parents, ...)])
```

### 7.3 Is This Leakage?

The model needs `parent_idx` to:
1. Build directed edges (P->C, C->P)
2. Compute branch angles (v_in, v_out)
3. Decode parent-relative offsets

In training, the tree structure is GT. In sampling, it's the tree built so far.
The model uses `parent_idx` to define the **message-passing graph**, not to predict
anything. This is the graph structure -- equivalent to the adjacency matrix.

**Verdict: OK.** The graph structure is necessary input, not leaked GT.

---

## 8. Expansion Targets and the LR_offset_head

### 8.1 How the Model Routes Predictions

If `LR_offset_head=True` (`egnn_so2.py:662-667`):
```python
class_feature_idx = self.pos_dim + 1   # = index 4 (after 3 pos dims + is_leaf feat)
class_feature = x[:, class_feature_idx].clone()   # THIS IS geo_ordinal!
```
Later (`egnn_so2.py:731-736`):
```python
class_mask = (class_feature > 0.5)     # routes to head 0 or head 1
head0 = self.offset_head_class0(feats)
head1 = self.offset_head_class1(feats)
offset_state = where(class_mask, head1, head0)
```

**This means `geo_ordinal > 0.5` routes to head 1, else head 0.**

For binary children: left=0.0 -> head 0, right=1.0 -> head 1.
For k=3 root children: child 0 (0.0) -> head 0, child 1 (0.5) -> head 0 (or edge case),
child 2 (1.0) -> head 1.

**Is this problematic?** The class routing is a design choice -- the model has two
separate decoders for left-like and right-like children. The routing signal comes from
the same `geo_ordinal` feature analyzed in Section 2.3.

**Verdict: OK.** This is intentional architectural design, not leakage.

---

## 9. The Critical Question: Does P_0 Geometry for Internal Nodes Leak Leaf Information?

### 9.1 The Concern

In training, internal nodes have their GT positions. Edge features between internal
nodes encode GT geometry. Could the model "read" information about where leaves
**should** go from internal-node features?

### 9.2 Analysis

Internal nodes are the **non-leaf** nodes at this reduction level. Their positions
define the skeleton of the tree so far. The model's job is to predict where to place
new leaves (children) given this skeleton.

In sampling, the internal nodes are the result of prior expansion steps. Their
positions are imperfect but represent the "best estimate" of the skeleton.

The model learns: "given a tree skeleton (internal nodes), predict child offsets."
The internal-node geometry is the **input context**, not leaked output. Just as a
language model sees prompt tokens to predict the next token, the EGNN sees internal
nodes to predict leaf offsets.

**Verdict: OK.** Internal-node geometry is input context, not leakage.

---

## 10. Edge Case: Root Node Special Handling

### 10.1 Root Has No Parent

The root node has `parent_idx = -1`. Its features:
- `is_leaf`: 1.0 (initially) then 0.0 (after spawning children)
- `geo_ordinal`: -1.0 clamped to 0.0
- `v_in`: zero vector (no grandparent)
- Branch angles: default (cospsi=1, sinpsi=0, cos_theta=1)

This is consistent between training and sampling.

### 10.2 Root Children Have No Grandparent

Root children have parent = root, grandparent = -1.
- Training: `v_in` set from `geo_delta_theta` (per-child rotated frame) via
  `compute_local_bases`.
- Sampling: `v_in` set from structured rotation in `compute_local_bases_for_leaves`.
- Branch angles: computed using the fallback v_in direction.

The branch angles at root children may differ between training and sampling because
the v_in direction differs (GT-derived vs random-rotated). However:
- In training, `patch_geometry_for_noised_leaves` preserves the P_0-based v_in (locked frame).
- In sampling, `compute_branch_angles_parent_centric` computes v_in from P_cur, which
  for root children falls into the "no grandparent" case.

**POTENTIAL INCONSISTENCY**: In training, root-child v_in comes from the GT-derived
per-child frame (locked). In sampling's diffusion loop, v_in for root children
falls into the `no grandparent` fallback in `compute_branch_angles_parent_centric`.

Let me check what that fallback does:

(`helpers.py:394+`, `compute_branch_angles_parent_centric`):
```python
has_gp_mask = gp >= 0
if has_gp_mask.any():
    sel = has_gp_mask.nonzero()
    v_in[sel] = pos[parent[sel]] - pos[gp[sel]]
# fallback: v_in stays zero for nodes without grandparent
```

For root children (no grandparent), `v_in = [0,0,0]`. This makes:
- `v_in_perp = [0,0,0]` -> degenerate
- `cospsi = 1.0, sinpsi = 0.0` (default for degenerate case)

**In training** (patched geometry): root-child leaves keep their P_0 v_in from the
`compute_local_bases` call, which used `geo_delta_theta` to set proper per-child
directions. But wait -- `patch_geometry_for_noised_leaves` reuses `v_in_p0` from
`pre_geom_p0['v_in']`, which comes from `intermediates['v_in']` in
`compute_branch_angles_parent_centric`. This is the **angle computation's** v_in,
NOT the local frame's v_in!

Let me re-check. In `precompute_full_geometry` (helpers.py:1192-1195):
```python
cospsi_node, sinpsi_node, cos_theta_node, intermediates = compute_branch_angles_parent_centric(
    pos, parent_idx, uhat, eps=eps, return_intermediates=True,
)
```
The `intermediates['v_in']` is from `compute_branch_angles_parent_centric`, where
root children have v_in = zero (no grandparent fallback).

Then in `patch_geometry_for_noised_leaves` (helpers.py:1284-1286):
```python
v_in_p0 = pre_geom_p0['v_in']           # from compute_branch_angles, root children = zero
v_in_leaf = v_in_p0[leaf_idx_train]      # root-child leaves get zero v_in
```

So in training's patched geometry, root children also have v_in = zero, giving
cospsi=1.0, sinpsi=0.0 -- same as sampling!

**Wait, but the local frame computation is separate from the branch angle computation.**
The local frames (used for coordinate conversion) use `compute_local_bases` with
`geo_delta_theta`, which gives per-child frames. But the branch angle features
passed to the model (`cospsi, sinpsi, cos_theta`) come from
`compute_branch_angles_parent_centric`, which uses its own v_in (zero for root children).

**These are two different v_in vectors:**
1. **Branch angle v_in** (from `compute_branch_angles_parent_centric`): zero for root children.
   Used for: edge angle features (`cospsi, sinpsi, cos_theta`) fed to the model.
2. **Local frame v_in** (from `compute_local_bases`): per-child rotated direction.
   Used for: coordinate conversion (local <-> global), NOT fed to the model.

**Verdict: OK.** The branch angle features are consistent (both use zero v_in for root
children). The local frames are different between training and sampling but are not
model inputs.

---

## 11. Summary of Findings

### No Leakage Found

After exhaustive analysis, **no information leakage was identified**. Every channel
that reaches the model carries consistent information in training and sampling:

1. **Positions**: Noised in training, estimated in sampling. Internal nodes are fixed context.
2. **Node features**: Structural indicators (is_leaf, new_leaf_mask, size_ratio) + ordinal.
3. **Edge geometry**: Computed from current (noised/estimated) positions consistently.
4. **Branch angles**: Same computation from current positions. Root children get default angles.
5. **Local frames**: Not fed to model. Used externally for coordinate conversion only.
6. **Expansion state**: Noised in training, estimated in sampling. Standard diffusion.

### Acceptable Design Choices (Not Leakage)

1. **geo_ordinal from GT**: Encodes sibling rank, not positions. Arbitrary in sampling.
   This was a deliberate design choice.
2. **Internal-node GT positions**: These are the tree context, not leaked targets.
3. **TMD conditioning**: Explicitly provided in both training and sampling as a conditioning
   signal for generating morphologically-correct trees.

### Minor Inconsistencies (Low Impact)

1. **geo_ordinal during diffusion loop**: In sampling, the ordinal used during diffusion
   steps is the initial arbitrary assignment. Post-diffusion refinement updates it. In
   training, the ordinal is GT-derived and fixed. The model sees a consistent ordinal
   within each forward pass in both cases; the difference is that training's ordinal
   reflects true geometric ordering while sampling's reflects arbitrary ordering within
   the diffusion loop. This could cause the model to receive slightly different
   disambiguation signals at training vs inference time.

   **Mitigation**: The ordinal's primary purpose is sibling disambiguation. For k=2
   (the most common case), the model always sees 0.0 and 1.0 regardless. For k>2,
   the arbitrary ordering during sampling means the model can't rely on ordinal to
   encode positional information -- which is actually the desired behavior (preventing
   the model from memorizing positions from ordinals).

---

## 12. Information Flow Diagram

```
                    TRAINING                                   SAMPLING
                    ========                                   ========

  GT Tree (P_0, parent_idx, leaf_expansion)          Prior expand steps + target_size
         |                                                    |
         v                                                    v
  precompute_full_geometry(P_0)                      expand() spawns children
         |                                                    |
         |-- geo_ordinal (from GT P_0) -----+        +-- geo_angle (arbitrary ordinal)
         |-- local_forward/sideways -+       |        |
         |-- branch angles (P_0)     |       |        +-- local_forward/sideways (random rot)
         v                           |       |        |
  DIFFUSION FORWARD                  |       |        v
  - Sample sigma                     |       |   DIFFUSION SAMPLE
  - Noise: C_t = C_0 + sigma*eps    |       |   - Schedule: sigma_max -> sigma_min
  - Convert C_t to P_t via frames --+       |   - Init: C = randn * sigma_init
  - Patch geometry for P_t                   |   - Each step:
         |                                   |     - Convert C to P_cur via frames
         v                                   |     - Recompute geometry from P_cur
  MODEL SEES:                                |            |
  +----------------------------------+       |            v
  | x = [P_t | node_feats | e_t |   |       |     MODEL SEES:
  |      log_sigma]                  |       |     +----------------------------------+
  | edge_attr = [type]               |       |     | x = [P_cur | node_feats | e |   |
  | pre_geom = {rho, du, cospsi,    |       |     |      log_sigma]                  |
  |             sinpsi, cos_theta}   |       |     | edge_attr = [type]               |
  +----------------------------------+       |     | NO pre_geom (recomputed inside)  |
         |                                   |     +----------------------------------+
         v                                   |            |
  Model predicts C_pred, e_pred              |            v
  Loss = MSE(C_pred, C_0)                   |     Model predicts C0_pred, e0_pred
  + MSE(e_pred, e_0)                        |     DDIM update: C = C0_pred + sigma_next * eps
                                             |
  NODE FEATURES (both paths):               |
  +----------------------------------+      |
  | [0] is_leaf          (binary)    | <----+  Same semantic meaning in both
  | [1] geo_ordinal      (0-1 float) | <----+  GT in training, arbitrary in sampling
  | [2] new_leaf_mask    (binary)    | <----+  Same semantic meaning in both
  | [3] size_ratio       (0-1 float) | <----+  Same semantic meaning in both
  | [4+] zeros           (padding)   |
  +----------------------------------+
  | [avail] e_t/e        (float)     |  Noised GT / noised estimate
  | [avail+1] log_sigma  (float)     |  Noise level indicator
  +----------------------------------+
```

---

## 13. Recommendations

1. **No immediate changes needed.** The current design is leak-free.

2. **For future development**: If adding new features, always verify:
   - Is this computed from GT P_0 or from noised P_t/P_cur?
   - Is the same information available to the model during sampling?
   - Could the model use this to "shortcut" the denoising task?

3. **Testing suggestion**: To empirically verify no leakage, train with shuffled
   `geo_ordinal` for root children. If the model still converges similarly for k=2
   cases, it confirms the ordinal doesn't carry positional information.
