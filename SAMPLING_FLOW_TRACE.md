# Sampling / Inference / Validation Flow Trace

A detailed, line-by-line trace of the sampling and validation forward pass — from the training loop's validation trigger through iterative graph expansion, diffusion-based denoising, and metric evaluation. Covers geometry computation differences from training, the multi-step expansion loop, and all root/single-child edge cases during generation.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Phase 1: Validation Trigger (`Trainer.run_validation`)](#2-phase-1-validation-trigger)
3. [Phase 2: Graph Generation Entry (`Trainer.evaluate`)](#3-phase-2-graph-generation-entry)
4. [Phase 3: `Expansion.sample_graphs()` — Iterative Generation](#4-phase-3-sample_graphs)
5. [Phase 4: `Expansion.expand()` — Single Generation Step](#5-phase-4-expand)
   - [4a: Spawn Count Determination](#4a-spawn-counts)
   - [4b: Child Node Materialisation](#4b-child-materialisation)
   - [4c: Adjacency Rebuild (SparseTensor)](#4c-adjacency-rebuild)
   - [4d: Node Feature Assembly](#4d-node-feature-assembly)
   - [4e: Edge Index Construction](#4e-edge-construction)
   - [4f: Diffusion Sampling Call](#4f-diffusion-sample-call)
   - [4g: Position Update from Predictions](#4g-position-update)
   - [4h: geo_lr_mask Recomputation (Post-Positioning)](#4h-geo-lr-recompute)
   - [4i: Expansion Thresholding](#4i-expansion-threshold)
   - [4j: Termination Check](#4j-termination)
6. [Phase 5: Diffusion Denoising Loop (`DenoisingDiffusionModel.sample`)](#6-phase-5-diffusion-sample)
   - [5a: Sigma Schedule](#5a-sigma-schedule)
   - [5b: Noisy Initialisation](#5b-noisy-init)
   - [5c: Per-Step Model Call](#5c-per-step-model)
   - [5d: DDIM-style Update](#5d-ddim-update)
7. [Phase 6: Model Forward Pass During Sampling](#7-phase-6-model-forward-sampling)
   - [6a: Geometry Computation (Internal, Not Precomputed)](#6a-geometry-internal)
   - [6b: TMD Embedding](#6b-tmd-embedding)
   - [6c: MPNN Layers](#6c-mpnn-layers)
   - [6d: Offset Head Decode](#6d-offset-head)
8. [Phase 7: Unbatching to NetworkX Graphs](#8-phase-7-unbatch)
9. [Phase 8: Metric Evaluation](#9-phase-8-metrics)
10. [Phase 9: Plotting](#10-phase-9-plotting)
11. [Key Differences: Training vs Sampling](#11-key-differences)
12. [Root and Edge Case Handling During Sampling](#12-root-edge-cases)
13. [Geometry Computation: Training vs Sampling Summary](#13-geometry-summary)
14. [Tensor Shape Reference](#14-tensor-shapes)

---

## 1. High-Level Overview

```
Trainer.train()
  └─ every cfg.validation.interval steps:
       Trainer.run_validation()
         └─ for each EMA beta:
              Trainer.evaluate(eval_graphs, beta)
                └─ for each batch of target sizes:
                     Expansion.sample_graphs(target_size, model, tmd)
                       └─ while not terminated:
                            Expansion.expand(...)
                              ├─ determine spawn counts from expansion predictions
                              ├─ materialise child nodes at parent positions
                              ├─ build node features (is_leaf, geo_lr, new_leaf, size_ratio, padding)
                              ├─ build directed edge_index
                              ├─ call DenoisingDiffusionModel.sample(...)
                              │    └─ for each sigma step:
                              │         ├─ construct P_cur from parent_pos + C
                              │         ├─ build conditioning: [e_feat, log_sigma]
                              │         ├─ call SO2_EGNN_Network.forward(x_in, ...)
                              │         │    └─ _compute_static_so2_geometry() internally
                              │         └─ DDIM update: C, e
                              │    return C0_pred, e0_pred
                              ├─ update positions: P[leaf] = parent_pos + C0_pred
                              ├─ compute_geo_lr_mask on UPDATED positions
                              └─ threshold expansion → next leaf_expansion labels
                       └─ unbatch into list[nx.Graph]
                └─ compute metrics (if enabled)
                └─ save plots (if enabled)
```

**Critical difference from training**: During sampling there is **no** `precompute_full_geometry()` or `patch_geometry_for_noised_leaves()`. The model computes its own SO(2) geometry **internally** at each diffusion step via `_compute_static_so2_geometry()`. The `geo_lr_mask` is computed **after** leaf positions are finalised (not before, as in training).

---

## 2. Phase 1: Validation Trigger

**File**: `graph_generation/training.py`, `Trainer.run_validation()` (lines 304–358)

The training loop triggers validation at configurable intervals:

```python
# training.py:223-230
if self.cfg.validation.interval > 0 and (
    self.step >= self.cfg.validation.first_step
    and self.step % self.cfg.validation.interval == 0
    or last_step
):
    if self.device == "cuda":
        th.cuda.empty_cache()
    self.run_validation()
```

Inside `run_validation()`:

```python
# training.py:304-358
def run_validation(self):
    val_results = {}
    test_results = {}
    enable_metrics = getattr(self.cfg.validation, 'enable_metrics', True)

    for beta in self.cfg.ema.betas:
        # 1. Generate graphs via evaluate()
        val_results[f"ema_{beta}"] = self.evaluate(self.validation_graphs, beta)

        if enable_metrics:
            # 2. Extract validation score (UniqueNovelValid or 1/Ratio)
            validation_score = val_results[f"ema_{beta}"][unique_novel_valid_keys[0]]

            # 3. If improved, ALSO run on test set
            if validation_score >= self.best_validation_scores[beta]:
                self.best_validation_scores[beta] = validation_score
                test_results[f"ema_{beta}"] = self.evaluate(self.test_graphs, beta)

    # 4. Log + pickle results
    self.log({"validation": val_results, "test": test_results})
    if self.cfg.training.save_checkpoint:
        # pickle to output_dir/validation/step_X.pkl
        # pickle to output_dir/test/step_X.pkl (if test triggered)
```

**Key detail**: The model used for evaluation is the **EMA model** (not the raw training model). `self.ema_models[beta]` wraps the model with exponential moving average of weights.

---

## 3. Phase 2: Graph Generation Entry

**File**: `graph_generation/training.py`, `Trainer.evaluate()` (lines 360–537)

```python
# training.py:360-407
@th.no_grad()
def evaluate(self, eval_graphs: list[nx.Graph], beta):
    model = self.ema_models[beta]   # EMA-smoothed model weights

    # 1. Shuffle eval graphs (random permutation for uniform size batching)
    pred_perm = self.rng.permutation(np.arange(len(eval_graphs)))
    target_size = np.array([len(g) for g in eval_graphs])[pred_perm]

    # 2. Optional TMD computation for conditioning
    tmd_hidden_dim = getattr(model, "tmd_hidden_dim", 0)
    tmds = None
    if tmd_hidden_dim > 0:
        tmds = np.stack(
            [compute_tmd_mixed(g) for g in eval_graphs], axis=0
        )[pred_perm]

    # 3. Split into batches of target sizes
    bs = self.cfg.validation.batch_size or self.cfg.training.batch_size
    batches = [target_size[i : i + bs] for i in range(0, len(target_size), bs)]

    # 4. Generate one batch at a time
    pred_graphs = []
    cursor = 0
    for batch in batches:
        tmd_batch = None
        if tmds is not None:
            tmd_batch = th.from_numpy(tmds[cursor : cursor + len(batch)]).to(self.device)
        pred_graphs_batch = self.method.sample_graphs(
            target_size=th.tensor(batch, device=self.device),
            model=model,
            tmd=tmd_batch,
        )
        pred_graphs += pred_graphs_batch
        cursor += len(batch)

    # 5. Reorder back to original eval_graphs order
    inv_perm = np.empty_like(pred_perm)
    inv_perm[pred_perm] = np.arange(len(pred_perm))
    results["pred_graphs"] = [pred_graphs[i] for i in inv_perm]
```

**What `target_size` means**: For each eval graph with N nodes, we tell the generator "produce a graph with N nodes". This enables fair comparison between reference and generated graphs at the same size.

**TMD (Tree Morphology Descriptor)**: Topological feature vector computed from the reference graph. Passed as a conditioning signal so the model generates a graph with similar topological properties to the reference.

---

## 4. Phase 3: `Expansion.sample_graphs()` — Iterative Generation

**File**: `graph_generation/method/expansion.py`, lines 50–124

This is the outer loop that grows graphs from a single root node to the target size via repeated binary branching.

### Initialisation (lines 59–76)

```python
# expansion.py:59-76
device = target_size.device
num_graphs = int(target_size.numel())

# One root node per graph, at the origin
pos = th.zeros((num_graphs, 3), device=device)              # [G, 3]

# Empty adjacency (no edges yet)
adj = SparseTensor(
    row=th.tensor([], dtype=th.long, device=device),
    col=th.tensor([], dtype=th.long, device=device),
    value=th.tensor([], dtype=th.float, device=device),
    sparse_sizes=(num_graphs, num_graphs),
)

batch = th.arange(num_graphs, device=device, dtype=th.long)  # [G]
parent_idx_1b = th.zeros_like(batch)       # all zeros → roots (1-based: 0 = root sentinel)
leaf_idx = batch.clone()                   # roots are the initial leaves
leaf_expansion = th.ones_like(leaf_idx)    # overridden in expand()
geo_lr_assign = th.full((num_graphs,), -1, device=device, dtype=th.long)  # unassigned
leaf_mask = th.ones((num_graphs,), device=device, dtype=th.bool)
```

**Initial state**: G independent single-root-node graphs. Each root is at position `[0,0,0]`, has `parent_idx = -1` (encoded as 0 in 1-based), is a leaf, and has no L/R assignment (`-1`).

### Iterative Expansion Loop (lines 78–108)

```python
# expansion.py:78-108
max_steps = int(target_size.max().item() * 2)
step = 0
terminated = False

while not terminated and step < max_steps:
    (
        adj, pos, leaf_idx, leaf_expansion, parent_idx_1b,
        batch, geo_lr_assign, leaf_mask, terminated,
    ) = self.expand(
        adj, batch, target_size, model,
        pos=pos, leaf_idx=leaf_idx, leaf_expansion=leaf_expansion,
        parent_idx_1b=parent_idx_1b, geo_lr_assign=geo_lr_assign,
        leaf_mask=leaf_mask, tmd=tmd, step=step,
        ensure_progress=False,
    )
    step += 1
```

Each call to `expand()` grows the tree by one generation: current leaves that have `expansion=2` spawn two children, and `expansion=1` leaves are terminal (no children).

**Safety cap**: `max_steps = 2 * max(target_size)` prevents infinite loops if expansion predictions never terminate.

---

## 5. Phase 4: `Expansion.expand()` — Single Generation Step

**File**: `graph_generation/method/expansion.py`, lines 126–464

### 4a: Spawn Count Determination

```python
# expansion.py:156-198
parent_idx = parent_idx_1b - 1   # shift to 0-based, roots → -1

# Map expansion labels: {1→terminal, 2→binary branch}
spawn_counts = (leaf_expansion == 2).long() * 2  # [L], values in {0, 2}
leaf_batch = batch_reduced[leaf_idx]
spawn_counts_final = spawn_counts.clone()

# Root nodes always spawn exactly 1 child (non-binary root)
is_root_leaf = parent_idx[leaf_idx] < 0
if is_root_leaf.any():
    root_should_spawn = (target_size[leaf_batch] > 1).long()
    spawn_counts_final = th.where(is_root_leaf, root_should_spawn, spawn_counts_final)
```

**Root special case**: Roots ALWAYS spawn exactly 1 child (not 2). This creates a single trunk node below the root. The first generation therefore produces one child per root, regardless of expansion predictions.

**Subsequent generations**: Non-root leaves with `expansion=2` spawn 2 children (binary branching); leaves with `expansion=1` are terminal.

### Optional: Ensure Progress (lines 225–240)

```python
# expansion.py:225-240
if ensure_progress and (remaining_capacity >= 2).any():
    # Force at least one leaf per graph to expand if capacity allows
    # Prevents stalling when all predictions are terminal
```

This is called with `ensure_progress=False` from `sample_graphs`, so it's inactive during normal sampling.

### Early Termination (lines 242–254)

```python
# expansion.py:242-254
total_new_children = int(spawn_counts_final.sum().item())
if total_new_children == 0:
    return (..., True)  # terminated=True
```

If no leaf wants to expand, the generation halts.

### 4b: Child Node Materialisation

```python
# expansion.py:256-286
base_N = adj_reduced.size(0)
leaf_mask_updated = leaf_mask.clone()
expanded_mask = spawn_counts_final > 0
if expanded_mask.any():
    leaf_mask_updated[leaf_idx[expanded_mask]] = False  # parents are no longer leaves

new_positions = []
new_parents = []
new_batches = []
parent_child_edges = []
lr_assign_new = []
running_child_index = 0

for leaf_global, sc in zip(leaf_idx.tolist(), spawn_counts_final.tolist()):
    if sc == 0:
        continue
    parent_pos = pos[leaf_global].unsqueeze(0)
    placeholder = parent_pos.expand(sc, -1).clone()  # children start AT parent position
    new_positions.append(placeholder)
    for local_child in range(sc):
        global_child = base_N + running_child_index
        parent_child_edges.append((leaf_global, global_child))
        new_parents.append(leaf_global)
        new_batches.append(int(batch_reduced[leaf_global].item()))
        lr_assign_new.append(0 if local_child == 0 else 1)  # first child → 0, second → 1
        running_child_index += 1
```

**Key details**:
- New children are placed **exactly at their parent's position** initially (placeholder). The diffusion sampler will later predict relative offsets.
- L/R assignment during child creation: first child → 0 (left), second child → 1 (right). This is a **preliminary** assignment; the definitive geo_lr is recomputed AFTER positioning (Phase 4h).
- For roots spawning 1 child: that single child gets `lr_assign = 0`.

### Tensor Concatenation (lines 287–299)

```python
# expansion.py:287-299
pos_new = th.cat([pos, new_pos_tensor], dim=0)              # [N+C, 3]
parent_idx_new_0b = th.cat([parent_idx, new_parents_tensor]) # [N+C]
parent_idx_1b_new = parent_idx_new_0b + 1
batch_new = th.cat([batch_reduced, new_batches_tensor])      # [N+C]
geo_lr_assign_next = th.cat([geo_lr_assign, lr_assign_new_tensor])  # [N+C]
```

### 4c: Adjacency Rebuild

```python
# expansion.py:302-323
row_old, col_old, val_old = adj_reduced.coo()
new_rows, new_cols, new_vals = [], [], []
for p, c in parent_child_edges:
    new_rows.extend([p, c])      # undirected: p→c and c→p
    new_cols.extend([c, p])
    new_vals.extend([1.0, 1.0])

adj_new = SparseTensor(
    row=th.cat([row_old, th.tensor(new_rows, ...)]),
    col=th.cat([col_old, th.tensor(new_cols, ...)]),
    value=th.cat([val_old, th.tensor(new_vals, ...)]),
    sparse_sizes=(pos_new.size(0), pos_new.size(0)),
)
```

### New Leaf Set (line 325)

```python
# expansion.py:325
leaf_idx_next = th.arange(base_N, base_N + total_new_children, device=device)
```

All newly created children become the next generation's leaves.

### 4d: Node Feature Assembly

```python
# expansion.py:348-387
feats_total = getattr(model, "feats_dim", 0)
tmd_hidden_dim = getattr(model, "tmd_hidden_dim", 0)
cond_dim = getattr(self.diffusion, "cond_dim", 0)      # = 2 (e_t + log_sigma)
avail_feats_dim = feats_total - cond_dim - tmd_hidden_dim

features = []
feats_used = 0

# Feature 1: is_leaf [N+C, 1] — 1.0 for leaves, 0.0 for internal
is_leaf = leaf_mask_next.to(dtype=pos_new.dtype).unsqueeze(-1)
features.append(is_leaf)
feats_used += 1

# Feature 2: geo_lr [N+C, 1] — geometric L/R class (0.0 or 1.0; -1 → 0.0)
geo_feat = pos_new.new_zeros((pos_new.size(0), 1))
mask = geo_lr_assign_next >= 0
if mask.any():
    geo_feat[mask] = (geo_lr_assign_next[mask] == 0).to(pos_new.dtype).unsqueeze(-1)
features.append(geo_feat)
feats_used += 1

# Feature 3: new_leaf_flag [N+C, 1] — 1.0 for nodes in this generation's leaf set
new_flag = pos_new.new_zeros((pos_new.size(0), 1))
new_flag[leaf_idx_next] = 1.0
features.append(new_flag)
feats_used += 1

# Feature 4: size_ratio [N+C, 1] — current_node_count / target_size per graph
ratio_graph = node_counts_per_graph / target_size.clamp_min(1.0)
ratio_nodes = ratio_graph[batch_new].unsqueeze(-1)
features.append(ratio_nodes)
feats_used += 1

# Padding to fill remaining dims
if feats_used < avail_feats_dim:
    pad = pos_new.new_zeros((pos_new.size(0), avail_feats_dim - feats_used))
    features.append(pad)

node_feats = th.cat(features, dim=-1)  # [N+C, avail_feats_dim]
```

**Feature layout** (same structure as training):
```
[is_leaf | geo_lr | new_leaf_flag | size_ratio | padding... | (reserved for e_t) | (reserved for log_sigma)]
                                                               ← cond_dim=2, filled by diffusion →
```

**Difference from training**: In training, `geo_lr` is computed on clean P_0 positions via `compute_geo_lr_mask`. During sampling, `geo_lr_assign` is a **running tracker** that gets updated after each expansion step based on the model's predicted positions. At this point (before diffusion), the new children have preliminary L/R assignments (first child=0, second=1) that will be overwritten in Phase 4h.

### 4e: Edge Index Construction

```python
# expansion.py:389-397
edge_index, edge_types = build_directed_edge_index(
    parent_idx_new_0b,
    edge_parent_to_child=self.EDGE_PARENT_TO_CHILD,  # 0
    edge_child_to_parent=self.EDGE_CHILD_TO_PARENT,   # 1
)
edge_attr = edge_types.unsqueeze(-1).to(pos_new.dtype)  # [2E, 1]
```

Same `build_directed_edge_index` as training — generates parent→child (type 0) and child→parent (type 1) directed edges for all non-root nodes.

### 4f: Diffusion Sampling Call

```python
# expansion.py:399-413
leaf_parent_idx_next = parent_idx_new_0b[leaf_idx_next]
model_kwargs = {"tmd": tmd} if tmd is not None else None

rel_pred, exp_pred = self.diffusion.sample(
    node_feats=node_feats,
    edge_index=edge_index,
    batch=batch_new,
    edge_attr=edge_attr,
    P_0=pos_new,                         # positions with children AT parent
    parent_idx=parent_idx_new_0b,
    leaf_idx=leaf_idx_next,
    leaf_parent_idx=leaf_parent_idx_next,
    model=model,
    model_kwargs=model_kwargs,
)
```

**What P_0 means here**: Unlike training where P_0 is ground-truth, during sampling `P_0` contains the current tree state with new children initialised at their parent positions. The diffusion sampler will predict offsets from these parent positions.

Returns:
- `rel_pred`: `[L, 3]` — predicted parent-relative offsets for leaves
- `exp_pred`: `[L, 1]` — predicted expansion scores for leaves

### 4g: Position Update from Predictions

```python
# expansion.py:416-417
parent_pos_for_children = pos_new[leaf_parent_idx_next]
pos_new[leaf_idx_next] = parent_pos_for_children + rel_pred
```

Leaf positions are updated to `parent_position + predicted_offset`. This is the final geometric position for this generation's leaves.

### 4h: geo_lr_mask Recomputation (Post-Positioning)

```python
# expansion.py:419-436
if leaf_idx_next.numel() > 0:
    geo_lr_mask = compute_geo_lr_mask(
        pos_new, parent_idx_new_0b,
        debug=getattr(self, "debug", False),
    )
    parent_new = parent_idx_new_0b[leaf_idx_next]
    counts = scatter(
        th.ones_like(parent_new),
        parent_new, dim=0, dim_size=pos_new.size(0),
    )
    valid = counts[parent_new] == 2   # only assign L/R where parent has exactly 2 children
    if valid.any():
        geo_left = geo_lr_mask[leaf_idx_next][valid]
        geo_lr_assign_next = geo_lr_assign_next.clone()
        geo_lr_assign_next[leaf_idx_next[valid]] = (~geo_left).to(dtype=geo_lr_assign_next.dtype)
```

**Critical difference from training**: The `compute_geo_lr_mask` call here:
1. Uses **predicted positions** (after diffusion update), not ground-truth P_0
2. Is called with **no `uhat` argument** — defaults to `[0, 0, 1]` z-axis internally
3. Only updates `geo_lr_assign` for binary siblings (`counts == 2`), leaving single root children with their preliminary assignment

**L/R assignment logic during sampling** (from `compute_geo_lr_mask`):
- **Root children with 1 sibling** (single root child): `lr_mask[single_root_ch] = True` (arbitrarily left)
- **Root children with 2 siblings**: `lr_mask = pos[child, -1] >= pos[parent, -1]` (z-coordinate comparison)
- **Non-root binary siblings**: sinψ-based left/right from the full `compute_geo_lr_mask` algorithm

**Note on the inversion**: `geo_lr_assign_next[...] = (~geo_left).to(...)` — the `geo_lr_mask` returns `True` for the "left" child, but the feature encoding uses `0` for left (class 0) and `1` for right (class 1). The `~` (NOT) flips: `left=True` → `assign=0`, `left=False` → `assign=1`. Actually looking more carefully: `(geo_lr_assign == 0).to(float)` is the feature value, so `assign=0` → feature `1.0` and `assign=1` → feature `0.0`. Combined with the `~geo_left`: if `geo_left=True` then `assign = 0` → feature = `1.0`. This makes the "geometrically left" child have feature value `1.0`.

### 4i: Expansion Thresholding

```python
# expansion.py:446-449
if exp_pred.dim() == 1:
    exp_pred = exp_pred.unsqueeze(-1)
expansion_score = exp_pred.squeeze(-1)
leaf_expansion_next = (expansion_score > map_threshold).long() + 1
```

**map_threshold = 0.0** (default from `expand()` signature, line 143).

The model predicts raw expansion scores (not sigmoid-normalised in the diffusion path). Values > 0.0 → `expansion = 2` (binary branch), values ≤ 0.0 → `expansion = 1` (terminal).

Recall from training: the target `e_0 = 2 * expansion - 1`, mapping `{0, 1} → {-1, +1}`. So the model learns to predict values in roughly `[-1, +1]`. The threshold at 0.0 is the natural decision boundary.

### 4j: Termination Check

```python
# expansion.py:451-452
remaining_capacity_new = target_size.to(device) - node_counts_per_graph
terminated = leaf_idx_next.numel() == 0
```

Termination occurs when no new children were produced (all expansion predictions were terminal, or no capacity). Note: the capacity-based termination (`(remaining_capacity_new < 2).all()`) is commented out — the loop relies on the model predicting terminal labels to stop.

---

## 6. Phase 5: Diffusion Denoising Loop

**File**: `graph_generation/diffusion/basic.py`, `DenoisingDiffusionModel.sample()` (lines 133–239)

### 5a: Sigma Schedule

```python
# basic.py:167-168
sigmas = self.make_sigma_schedule(self.num_steps, self.sigma_max, self.sigma_min, device)
# sigmas: [num_steps + 1] tensor, monotonically decreasing, ending with 0
# Example for num_steps=1: [sigma_max, 0.0]
# Example for num_steps=5: [sigma_max, ..., sigma_min, 0.0]
sigma_init = float(sigmas[0].item())  # = sigma_max = 4.0
```

The schedule is geometric:
```python
# basic.py:125-131
@staticmethod
def make_sigma_schedule(num_steps, sigma_max, sigma_min, device):
    steps = max(int(num_steps), 1)
    ramp = th.linspace(0.0, 1.0, steps=steps, device=device)
    sigmas = sigma_max * (sigma_min / sigma_max) ** ramp
    sigmas = th.cat([sigmas, sigmas.new_zeros(1)], dim=0)
    return sigmas
```

For `num_steps=1`: `sigmas = [4.0, 0.0]` (single denoising step).

### 5b: Noisy Initialisation

```python
# basic.py:170-177
L = leaf_idx.numel()
N = P_0.size(0)
parent_pos = P_0[leaf_parent_idx]              # [L, 3] fixed parent positions

C = th.randn((L, 3), device=device) * sigma_init    # [L, 3] initial noisy offsets
e = th.randn((L, 1), device=device) * sigma_init    # [L, 1] initial noisy expansion

C0_pred = th.zeros_like(C)    # will be overwritten
e0_pred = th.zeros_like(e)    # will be overwritten
```

**C** is the parent-relative offset vector; **e** is the expansion score. Both start as Gaussian noise scaled by `sigma_max`.

### 5c: Per-Step Model Call

```python
# basic.py:186-215
for step in range(self.num_steps):
    sigma_cur = float(sigmas[step].item())
    sigma_next = float(sigmas[step + 1].item())
    sigma_cur_clamped = max(sigma_cur, 1e-12)
    log_sigma = math.log(sigma_cur_clamped)

    # 1. Construct current positions
    P_cur = P_0.clone()                              # [N, 3]
    P_cur[leaf_idx] = parent_pos + C                 # place leaves at parent + current offset

    # 2. Build conditioning features
    e_feat = P_0.new_zeros((N, 1))
    e_feat[leaf_idx] = e                             # current noisy expansion at leaves
    log_sigma_feat = P_0.new_full((N, 1), log_sigma) # UNIFORM log_sigma for all nodes
    node_feats_t = th.cat([node_feats, e_feat, log_sigma_feat], dim=-1)

    # 3. Assemble model input: [positions | features]
    x_in = th.cat([P_cur, node_feats_t], dim=-1)    # [N, 3 + feats_dim + 2]

    # 4. Model forward — NO pre_geom passed!
    out = model(
        x=x_in,
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        parent_idx=parent_idx,
        **model_kwargs,     # may contain tmd
    )
```

**Key observation**: The model call does **NOT** receive `pre_geom`. This is intentional — during sampling, positions change at each diffusion step (leaves move as C is refined), so geometry must be recomputed fresh. The model will compute it internally via `_compute_static_so2_geometry()`.

**log_sigma is UNIFORM**: Unlike training where `sigma_graph` is per-graph, during sampling the sigma schedule is deterministic and the same for all graphs. `log_sigma_feat` is broadcast to all nodes (not just leaves).

### 5d: DDIM-style Update

```python
# basic.py:217-232
    rel_pred_all = out["rel_pred"]          # [N, 3]
    exp_pred_all = out["expansion_pred"]    # [N, 1]

    C0_pred = rel_pred_all[leaf_idx]        # [L, 3] model's denoised offset prediction
    e0_pred = exp_pred_all[leaf_idx]        # [L, 1] model's denoised expansion prediction
    if e0_pred.dim() == 1:
        e0_pred = e0_pred.unsqueeze(-1)

    # Estimate noise direction
    inv_sigma = 1.0 / sigma_cur_clamped
    eps_C = (C - C0_pred) * inv_sigma       # [L, 3] estimated noise for offsets
    eps_e = (e - e0_pred) * inv_sigma       # [L, 1] estimated noise for expansion

    # DDIM deterministic step: x_{t-1} = x0_pred + sigma_next * noise_estimate
    C = C0_pred + sigma_next * eps_C
    e = e0_pred + sigma_next * eps_e
```

**Interpretation**: The model predicts the clean signal (`C0_pred`, `e0_pred`). The noise component is estimated as `(noisy - clean) / sigma`. The next iterate interpolates between the clean prediction and the noise direction, weighted by the next (smaller) sigma.

**Final step** (`sigma_next = 0`): `C = C0_pred`, `e = e0_pred` — the output is exactly the model's denoised prediction.

**For `num_steps=1`**: There is only one step, `sigma_cur = sigma_max`, `sigma_next = 0`, so the output is the single-step denoised prediction directly.

After the loop:
```python
# basic.py:234-239
return C0_pred, e0_pred   # [L, 3], [L, 1]
```

---

## 7. Phase 6: Model Forward Pass During Sampling

**File**: `graph_generation/model/egnn_so2.py`, `SO2_EGNN_Network.forward()` (lines 626–745)

### 6a: Geometry Computation (Internal)

```python
# egnn_so2.py:672-678
# pre_geom is None during sampling (not passed from diffusion.sample)
if pre_geom is None:
    static_coords = all(
        (not getattr(L, 'update_coors', True)) for L in self._iter_egnn_layers()
    )
    if parent_idx is not None and static_coords:
        pre_geom = self._compute_static_so2_geometry(
            x[:, :self.pos_dim], edge_index, parent_idx
        )
```

The model detects that no external `pre_geom` was provided and computes geometry internally:

```python
# egnn_so2.py:750-786
def _compute_static_so2_geometry(self, coors, edge_index, parent_idx):
    src, dst = edge_index
    rel_coors = coors[dst] - coors[src]                # (E, 3)
    du = (rel_coors @ self.uhat)                        # (E,)
    r_par = du[:, None] * self.uhat
    r_perp = rel_coors - r_par
    rho = r_perp.norm(dim=-1, keepdim=True).clamp_min(self.eps)
    du = du[:, None]

    cospsi_node, sinpsi_node, cos_theta_node = compute_branch_angles_parent_centric(
        coors, parent_idx, self.uhat, eps=self.eps
    )
    cospsi_edge, sinpsi_edge = assign_branch_angles_to_edges(
        edge_index, parent_idx, cospsi_node, sinpsi_node
    )
    cos_theta_edge = assign_parent_scalar_to_edges(
        edge_index, parent_idx, cos_theta_node
    )
    return { 'rel_coors', 'r_perp', 'rho', 'du',
             'cospsi_edge', 'sinpsi_edge', 'cos_theta_edge',
             'cospsi_node', 'sinpsi_node', 'cos_theta_node' }
```

**Key difference from training**:
- Uses `compute_branch_angles_parent_centric` with `return_intermediates=False` (no need for patching intermediates)
- Computed on `P_cur` (noised positions where leaves are at `parent + C`), not on clean P_0
- This is called at **every diffusion step** (in multi-step mode), so geometry is fresh for the current leaf positions
- Uses `self.uhat` from the model's registered buffer

**Branch angle behaviour for root children during sampling**:
- Root children have `parent >= 0` (the root) but `grandparent = -1` (no grandparent)
- `compute_branch_angles_parent_centric` applies fallback: `v_in = v_out` for nodes without grandparent
- Result: `cospsi = 1.0`, `sinpsi = 0.0` (in-plane angle is trivially aligned since v_in ∥ v_out)
- `cos_theta` is computed from actual `v_out` projection onto uhat (reflects real position)

### 6b: TMD Embedding

```python
# egnn_so2.py:637-660
if self.tmd_hidden_dim > 0:
    tmd_emb = self.tmd_mlp(tmd)                 # [B, tmd_hidden_dim]
    tmd_nodes = tmd_emb[batch]                    # [N, tmd_hidden_dim] broadcast per node
    feats = torch.cat([feats, tmd_nodes], dim=-1) # append to node features
    x = torch.cat([coors, feats], dim=-1)
```

During sampling, `tmd` comes from the reference eval graph's topology descriptor (computed in `evaluate()`). It conditions generation to match the reference graph's topological properties.

### 6c: MPNN Layers

```python
# egnn_so2.py:681-706
for i, layer in enumerate(self.mpnn_layers):
    if isinstance(layer, nn.ModuleList):   # [GlobalAttn, EGNN]
        # (a) ISAB global attention on features
        tokens, tokens_batch = self._make_global_tokens(self.global_tokens, batch)
        coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]
        feats, _ = layer[0](feats, tokens, x_batch=batch, q_batch=tokens_batch)
        x = torch.cat([coors, feats], dim=-1)
        # (b) EGNN message passing
        x = layer[1](x, edge_index, edge_attr, batch=batch,
                      parent_idx=parent_idx, pre_geom=pre_geom)
    else:
        # Regular EGNN layer
        x = layer(x, edge_index, edge_attr, batch=batch,
                  parent_idx=parent_idx, pre_geom=pre_geom)
```

Each SO2_EGNN layer receives the same `pre_geom` computed once for this diffusion step. Inside each layer:

```python
# egnn_so2.py:297-304 (SO2_EGNN.forward)
if pre_geom is not None:
    rel_coors = pre_geom['rel_coors']
    r_perp = pre_geom['r_perp']
    rho = pre_geom['rho']
    du = pre_geom['du']
    cospsi_edge = pre_geom.get('cospsi_edge')
    sinpsi_edge = pre_geom.get('sinpsi_edge')
    cos_theta_edge = pre_geom.get('cos_theta_edge')
```

Edge features assembled as:
```
[edge_attr(1) | rho(1) | du(1) | cospsi(1) | sinpsi(1) | cos_theta(1)] → edge_mlp → m_ij
```

Message passing: `m_i = aggregate(edge_mlp(cat[x_i, x_j, edge_feats]))` → node update via `node_mlp(cat[x, m_i])`.

### 6d: Offset Head Decode

```python
# egnn_so2.py:720-745
coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]

if not self.LR_offset_head:
    offset_state = self.offset_head(feats)             # single head
else:
    class_mask = (class_feature > 0.5).unsqueeze(-1)   # class_feature captured at input
    head0 = self.offset_head_class0(feats)             # "right" head
    head1 = self.offset_head_class1(feats)             # "left" head
    offset_state = torch.where(class_mask, head1, head0)

rel_pred = offset_state[:, :3]           # [N, 3] parent-relative offsets
expansion_pred = offset_state[:, 3:4]    # [N, 1] expansion scores
return {"node_state": x, "rel_pred": rel_pred, "expansion_pred": expansion_pred}
```

**Dual-head routing**: When `LR_offset_head=True`, the `class_feature` (= the geo_lr feature at index `pos_dim + 1` in the input) routes each node to either `offset_head_class0` (geo_lr=0 → right) or `offset_head_class1` (geo_lr=1 → left).

During sampling, `class_feature` is captured from the input at the start of `forward()`:
```python
# egnn_so2.py:663-667
class_feature = None
if self.LR_offset_head:
    class_feature_idx = self.pos_dim + 1   # second feature channel (geo_lr)
    class_feature = x[:, class_feature_idx].clone()
```

---

## 8. Phase 7: Unbatching to NetworkX Graphs

**File**: `graph_generation/method/expansion.py`, lines 110–124

After the expansion loop terminates:

```python
# expansion.py:110-124
row, col, _ = adj.coo()
graphs = []
for g in range(num_graphs):
    mask = batch == g
    node_ids = mask.nonzero(as_tuple=False).flatten()
    local_map = {int(n.item()): i for i, n in enumerate(node_ids)}
    G = nx.Graph()
    for i_local, n_global in enumerate(node_ids.tolist()):
        G.add_node(i_local, pos=pos[n_global].detach().cpu().numpy())
    for r, c in zip(row.tolist(), col.tolist()):
        if r in local_map and c in local_map:
            if local_map[r] <= local_map[c]:   # avoid duplicate undirected edges
                G.add_edge(local_map[r], local_map[c])
    graphs.append(G)
return graphs
```

Each node gets a `pos` attribute containing its 3D position. Node indices are remapped to local (per-graph) 0-based indices.

---

## 9. Phase 8: Metric Evaluation

**File**: `graph_generation/training.py`, `evaluate()` (lines 424–449)

```python
# training.py:424-449
enable_metrics = getattr(self.cfg.validation, 'enable_metrics', True)
if enable_metrics:
    for metric in self.metrics:
        results[str(metric)] = metric(
            reference_graphs=eval_graphs,       # ground-truth eval set
            predicted_graphs=pred_graphs,       # generated graphs
            train_graphs=self.train_graphs,     # for novelty check
        )

    # Optional per-size breakdown
    if self.cfg.validation.per_graph_size:
        for n in set(target_size):
            eval_n = [g for g in eval_graphs if len(g) == n]
            pred_n = [g for g in pred_graphs if len(g) == n]
            for metric in self.metrics:
                results[f"size_{n}"][str(metric)] = metric(...)
```

Metrics include: NodeNumDiff, NodeDegree, ClusteringCoefficient, OrbitCount, Spectral, Wavelet, Ratio, Uniqueness, Novelty, ValidTree, UniqueNovelValidTree.

---

## 10. Phase 9: Plotting

**File**: `graph_generation/training.py`, `evaluate()` (lines 456–535)

When `enable_plots=True`:
1. **Generated graph plots**: Up to 8 examples, XY projection of node positions with edges
2. **Side-by-side comparison**: Reference vs predicted graph for each example
3. Saved to `output_dir/eval_plots/step_X_beta_Y.png` and `step_X_beta_Y_compare.png`

---

## 11. Key Differences: Training vs Sampling

| Aspect | Training | Sampling |
|--------|----------|----------|
| **Positions** | Ground-truth P_0 (fixed) | Evolving P_cur (predicted each step) |
| **Geometry precomputation** | `precompute_full_geometry(P_0, ...)` once | None — model computes internally |
| **Geometry patching** | `patch_geometry_for_noised_leaves(P_0→P_t)` | N/A — full recompute each step |
| **pre_geom to model** | Passed from diffusion.forward() | Not passed; model calls `_compute_static_so2_geometry()` |
| **geo_lr_mask timing** | Before diffusion (on P_0) | After diffusion (on predicted positions) |
| **geo_lr_mask uhat** | `model.uhat` passed explicitly | Defaults to `[0,0,1]` (no uhat arg) |
| **Sigma** | Random per-graph (log-normal) | Deterministic schedule (same for all graphs) |
| **Expansion targets** | Ground truth from dataset (`leaf_expansion`) | Thresholded from model predictions |
| **e_0 mapping** | `e_0 = 2*expansion - 1` (target for MSE) | Raw score thresholded at 0.0 |
| **Gradient** | Enabled (training) | `@th.no_grad()` (inference) |
| **Model** | Raw model (`self.model`) | EMA model (`self.ema_models[beta]`) |
| **Iterative** | Single forward pass per batch | Multi-step expand loop |

---

## 12. Root and Edge Case Handling During Sampling

### Root Node (Step 0)

**State**: Single root per graph at `[0,0,0]`, `parent_idx = -1`, `leaf_expansion = 1` (overridden).

**Spawn behaviour**: Root always spawns exactly 1 child regardless of expansion prediction:
```python
is_root_leaf = parent_idx[leaf_idx] < 0
root_should_spawn = (target_size[leaf_batch] > 1).long()  # 1 if target > 1, else 0
spawn_counts_final = th.where(is_root_leaf, root_should_spawn, spawn_counts_final)
```

So the root produces one child placed at `[0,0,0]` initially. The diffusion sampler predicts the offset from root to this first child.

### Single Root Child After Step 0

After step 0, there is one root child. Its geo_lr handling:

1. **In the expansion child creation loop**: The single child gets `lr_assign_new = 0` (first child → 0)
2. **In `compute_geo_lr_mask` (after positioning)**: Single root children get `lr_mask = True` (arbitrarily "left"). But since `counts[parent] == 1` (only 1 child), the `valid = counts[parent_new] == 2` check **fails**, so `geo_lr_assign` is NOT updated from the `compute_geo_lr_mask` result. The preliminary `lr_assign = 0` persists.
3. **Feature encoding**: `geo_feat = (geo_lr_assign == 0).float()` → `1.0` for this child

### Binary Root Children (Steps after root's child expands)

When the root's child expands (step 1+), it produces two grandchildren of the root. These grandchildren:

1. Get preliminary `lr_assign = [0, 1]` during child creation
2. After positioning, `compute_geo_lr_mask` assigns L/R based on the full sinψ/atan2 algorithm (their parent is not the root, so the non-root binary-parent path is used)
3. Since `counts[parent] == 2`, the `valid` mask passes and `geo_lr_assign` IS updated

### Root-Child Leaf in Later Steps (if root child is still a leaf)

If the root child is still considered a leaf in later steps (hasn't been expanded yet):
- It gets `v_in = v_out` fallback in `compute_branch_angles_parent_centric` (no grandparent)
- cospsi = 1.0, sinpsi = 0.0, cos_theta computed from actual position
- These values feed into the SO2_EGNN edge features for the root↔child edge

---

## 13. Geometry Computation: Training vs Sampling Summary

### Training Path
```
Expansion.get_loss()
  │
  ├─ precompute_full_geometry(P_0, parent_idx, edge_index, model.uhat)
  │    ├─ compute_geo_lr_mask(P_0, parent_idx, uhat=model.uhat)    → geo_lr_mask
  │    ├─ compute_branch_angles_parent_centric(P_0, ..., return_intermediates=True)
  │    │    → cospsi_node, sinpsi_node, cos_theta_node, {v_in, v_out, has_gp}
  │    ├─ edge-level SO(2): rel_coors, r_perp, rho, du
  │    └─ assign angles to edges → cospsi_edge, sinpsi_edge, cos_theta_edge
  │
  ├─ DenoisingDiffusionModel.forward()
  │    ├─ noise P_0 → P_t (only leaf positions change)
  │    ├─ patch_geometry_for_noised_leaves(pre_geom_p0, P_t, leaf_idx, ...)
  │    │    ├─ recompute v_out for leaves only (v_in reused from P_0)
  │    │    ├─ recompute cospsi, sinpsi, cos_theta for leaves only
  │    │    ├─ recompute rel_coors, rho, du for affected edges only
  │    │    └─ reassign edge angles from patched node angles
  │    └─ model(x_in, pre_geom=patched_pre_geom)   ← PATCHED geometry passed
  │
  └─ MSE loss on predictions vs clean targets
```

### Sampling Path
```
Expansion.expand()
  │
  ├─ create children at parent positions (no geometry yet)
  ├─ build node features (geo_lr from running tracker, not computed fresh)
  │
  ├─ DenoisingDiffusionModel.sample()
  │    └─ for each sigma step:
  │         ├─ construct P_cur = P_0 clone; P_cur[leaves] = parent + C
  │         ├─ model(x_in, ...)   ← NO pre_geom passed
  │         │    └─ SO2_EGNN_Network.forward():
  │         │         └─ _compute_static_so2_geometry(P_cur, edge_index, parent_idx)
  │         │              ├─ compute_branch_angles_parent_centric(P_cur, ...)
  │         │              │    → v_in=v_out fallback for root children
  │         │              ├─ edge SO(2): rel_coors, r_perp, rho, du from P_cur
  │         │              └─ assign angles to edges
  │         └─ DDIM update: C, e
  │
  ├─ update positions: P[leaves] = parent + C0_pred
  │
  ├─ compute_geo_lr_mask(pos_updated, parent_idx)   ← AFTER positioning
  │    └─ only updates geo_lr_assign for binary siblings (counts == 2)
  │
  └─ threshold expansion → leaf_expansion_next
```

### Key Geometry Differences

1. **Compute frequency**:
   - Training: Once on P_0, patched for P_t (efficient)
   - Sampling: Full recompute at every diffusion step (positions change)

2. **Who computes**:
   - Training: `helpers.py` functions called from `expansion.py` and `basic.py`
   - Sampling: `egnn_so2.py::_compute_static_so2_geometry()` called internally by model

3. **v_in fallback for root children**:
   - Training `compute_geo_lr_mask`: uses `global_e1` (fixed reference vector orthogonal to uhat)
   - Training `compute_branch_angles_parent_centric`: uses `v_in = v_out` (self-referential)
   - Sampling `compute_branch_angles_parent_centric`: uses `v_in = v_out` (same as training angles)
   - Sampling `compute_geo_lr_mask`: uses `global_e1` (same as training geo_lr)

4. **uhat threading**:
   - Training: `model.uhat` explicitly passed to `compute_geo_lr_mask` and `compute_branch_angles_parent_centric`
   - Sampling `compute_branch_angles_parent_centric`: uses `self.uhat` from model buffer
   - Sampling `compute_geo_lr_mask` (in `expand()`): **no uhat passed** → defaults to `[0,0,1]`

   **Potential discrepancy**: If `model.uhat ≠ [0,0,1]`, the geo_lr_mask during sampling uses a different axis than during training. Currently `so2_axis=(0,0,1)` is the default, so they match, but this is a latent inconsistency if the axis is changed.

---

## 14. Tensor Shape Reference

### At expansion step with N existing nodes and C new children

| Tensor | Shape | Description |
|--------|-------|-------------|
| `pos` | `[N, 3]` → `[N+C, 3]` | Node positions (children start at parent pos) |
| `batch` | `[N]` → `[N+C]` | Graph membership per node |
| `parent_idx` | `[N+C]` | 0-based parent (-1 for roots) |
| `leaf_idx` | `[L_prev]` → `[C]` | Leaf node indices (previous → new children) |
| `leaf_expansion` | `[C]` | Expansion labels {1, 2} from model predictions |
| `geo_lr_assign` | `[N+C]` | Running L/R tracker (-1=unassigned, 0=left, 1=right) |
| `leaf_mask` | `[N+C]` | Boolean: True for current leaves |
| `edge_index` | `[2, 2E]` | Directed edges (rebuilt each step) |
| `edge_attr` | `[2E, 1]` | Edge type (0=parent→child, 1=child→parent) |
| `node_feats` | `[N+C, avail_feats_dim]` | `[is_leaf\|geo_lr\|new_leaf\|size_ratio\|pad]` |

### Inside diffusion.sample()

| Tensor | Shape | Description |
|--------|-------|-------------|
| `C` | `[L, 3]` | Current parent-relative offset (evolving) |
| `e` | `[L, 1]` | Current expansion score (evolving) |
| `P_cur` | `[N+C, 3]` | Positions with `P_cur[leaves] = parent + C` |
| `e_feat` | `[N+C, 1]` | Zeros except leaves have current `e` |
| `log_sigma_feat` | `[N+C, 1]` | Uniform log(σ) for all nodes |
| `x_in` | `[N+C, 3 + feats_dim]` | `[P_cur \| node_feats_t]` |
| `sigmas` | `[num_steps+1]` | Sigma schedule (decreasing, ends with 0) |
| `C0_pred` | `[L, 3]` | Model's denoised offset prediction |
| `e0_pred` | `[L, 1]` | Model's denoised expansion prediction |

### Inside model (per diffusion step)

| Tensor | Shape | Description |
|--------|-------|-------------|
| `pre_geom['rel_coors']` | `[2E, 3]` | `P_cur[dst] - P_cur[src]` |
| `pre_geom['rho']` | `[2E, 1]` | Perpendicular distance |
| `pre_geom['du']` | `[2E, 1]` | Parallel (axis) distance |
| `pre_geom['r_perp']` | `[2E, 3]` | Perpendicular displacement vector |
| `pre_geom['cospsi_edge']` | `[2E, 1]` | In-plane branch angle cosine |
| `pre_geom['sinpsi_edge']` | `[2E, 1]` | In-plane branch angle sine |
| `pre_geom['cos_theta_edge']` | `[2E, 1]` | Axis tilt angle cosine |

---

## Appendix: Complete Expand Return Tuple

```python
return (
    adj_new,               # SparseTensor [N+C, N+C] — updated undirected adjacency
    pos_new,               # [N+C, 3] — positions with leaves at predicted locations
    leaf_idx_next,         # [C] — indices of new children (next generation's leaves)
    leaf_expansion_next,   # [C] — {1,2} expansion labels from thresholded predictions
    parent_idx_1b_new,     # [N+C] — 1-based parent indices
    batch_new,             # [N+C] — graph membership
    geo_lr_assign_next,    # [N+C] — updated running L/R assignments
    leaf_mask_next,        # [N+C] — True for current-generation leaves
    terminated,            # bool — True if no new children produced
)
```
