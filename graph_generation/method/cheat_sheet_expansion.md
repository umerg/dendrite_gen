# High-level contract

* **Inputs (key ones):**

  * `adj_reduced`: sparse adjacency for current graphs (N×N, undirected).
  * `batch_reduced`: `(N,)` graph id for each node (0…G-1).
  * `target_size`: `(G,)` desired final node count per graph.
  * `pos`: `(N,D)` current node coordinates (usually D=3).
  * `leaf_idx`: `(L,)` indices of current leaf nodes in `[0, N)`.
  * `leaf_expansion`: `(L,)` labels in `{1,2}` where `2` = “branch (spawn 2 kids)”.
  * `parent_idx_1b`: `(N,)` **1-based** parent indices for each node (0/1 means “no parent”), converted to 0-based internally.
  * `model`: must return `{"rel_pred": (N',D), "expansion_pred": (N',) or (N',1)}`.

* **Outputs:**

  * `adj_new`: sparse adjacency after adding children.
  * `pos_new`: `(N+C, D)` updated positions (new children refined by model).
  * `leaf_idx_next`: `(C,)` indices of the brand-new children (the next frontier).
  * `leaf_expansion_next`: `(C,)` labels in `{1,2}` for the next step (thresholded).
  * `parent_idx_1b_new`: `(N+C,)` updated 1-based parents including new kids.
  * `batch_new`: `(N+C,)` updated batch vector.
  * `terminated`: `bool` flag—no more room for binary branching or no leaves.

> Notation:
> N = current #nodes, G = #graphs, L = #leaves, D = position dim (typically 3),
> C = total new children this step (sum over leaves), E = #edges.

---

# Step-by-step (with shapes)

### 0) `@th.no_grad()`

* Entire function runs in inference mode—no gradient tracking. Good for iterative generation loops.

### 1) Validate required tensors

```python
if not all(x is not None for x in (pos, leaf_idx, leaf_expansion, parent_idx_1b)):
    raise ValueError(...)
```

* Ensures **leaf mode** is active (you gave all the inputs needed to grow).

### 2) Basic prep

```python
device = pos.device
parent_idx = parent_idx_1b - 1  # (N,) convert to 0-based
```

* `parent_idx` becomes `(N,)` with `-1` meaning “no parent”.

### 3) Per-graph size & capacity

```python
size_per_graph = scatter(th.ones_like(batch_reduced), batch_reduced)  # (G,)
remaining_capacity = target_size.to(device) - size_per_graph          # (G,)
```

* `size_per_graph[g]` = current node count in graph g.
* Early exit if no capacity anywhere or there are no leaves:

```python
if (remaining_capacity <= 0).all() or leaf_idx.numel() == 0:
    return ... , True
```

### 4) Map labels → spawn counts

```python
spawn_counts = (leaf_expansion == 2).long() * 2  # (L,), values in {0,2}
leaf_batch = batch_reduced[leaf_idx]            # (L,), per-leaf graph id
```

* You’re doing **binary branching**: only “no spawn” or “spawn 2”.

### 5) Capacity enforcement (all-or-nothing per leaf)

For each graph `g`:

* If `cap < 2`, disable **all** expansions in that graph.
* Otherwise:

  * `expanders` = indices of leaves in `g` with `spawn_counts==2`.
  * `needed = 2 * (#expanders)`.
  * If `needed > cap`, keep only the **first** `cap // 2` leaves (deterministic by index order), zero out the rest.

```python
spawn_counts_final = spawn_counts.clone()       # (L,)
...
spawn_counts_final[disable] = 0
```

**Intuition:** Each expanding leaf costs **2 slots**. Don’t partially expand a single leaf—either add both kids or none.

### 6) Ensure progress (optional)

If there’s capacity (≥2 somewhere) but all expansions were suppressed, force the **first eligible leaf** in the **first graph with room** to expand (set its count to 2). This prevents stagnation:

```python
if ensure_progress and spawn_counts_final.sum()==0 and (remaining_capacity >= 2).any():
    ... set one leaf to 2 ...
```

### 7) Count children & early exit

```python
total_new_children = int(spawn_counts_final.sum().item())  # C
if total_new_children == 0:
    return ... , True
```

### 8) Materialize children (positions, parents, batches)

* `base_N = adj_reduced.size(0)` is the starting number of nodes (N).
* For each leaf i with `sc ∈ {0,2}`:

  * `parent_pos = pos[li]` → `(D,)`
  * Sample `noise ~ N(0, sigma^2 I)`, shape `(sc, D)`.
  * **Clip** each noise vector to length `leaf_noise_clip` if provided.
  * `child_pos = parent_pos[None,:] + noise` → `(sc, D)`
  * Assign global child indices sequentially `base_N ... base_N+C-1`.
  * Record undirected parent↔child edges.

After the loop:

```python
new_child_positions_tensor: (C, D)
pos_new = cat([pos, new_child_positions_tensor])             # (N+C, D)

new_child_parents: (C,)  # each stores the leaf index 'li' (0-based)
parent_idx_new_0b = cat([parent_idx, new_child_parents])     # (N+C,)
parent_idx_1b_new  = parent_idx_new_0b + 1                   # (N+C,)  # back to 1-based

new_child_batches: (C,)
batch_new = cat([batch_reduced, new_child_batches])          # (N+C,)
```

**Noise clipping detail:** For each row `v` in `noise`, you compute

```
scale = min(1, clip / ||v||)
noise := noise * scale
```

so the L2 norm never exceeds `leaf_noise_clip`.

### 9) Rebuild adjacency with new parent–child edges

* Get old COO:

```python
row_old, col_old, val_old = adj_reduced.coo()
```

* Append `(p,c)` and `(c,p)` with value `1.0` for every new edge.
* Create

```python
adj_new = SparseTensor(
  row=row_all, col=col_all, value=val_all,
  sparse_sizes=(N+C, N+C)
)
```

**Notes**

* Old edges (weights `val_old`) are preserved.
* Self-loops are not added here.
* You’re treating this as unweighted for new edges (weight=1).

### 10) Define next leaf set

```python
leaf_idx_next = arange(base_N, base_N + C, device=device)  # (C,)
```

* By construction, **only the brand-new children** are the next frontier.
* If `C==0` (shouldn’t happen after step 7), early exit.

### 11) Model forward: refine child positions + predict next expansion

**Node features:**

* If `model.feats_dim > 0`:

  * `is_leaf_flag`: `(N+C,1)` ones at `leaf_idx_next`, zeros elsewhere.
  * `extra`: `(N+C, feats_dim-1)` all zeros (placeholder).
  * `x_in = cat([pos_new[:, :pos_dim], node_feats], -1)` → `(N+C, pos_dim + feats_dim)`
* Else:

  * `x_in = pos_new[:, :pos_dim]` → `(N+C, pos_dim)`

**Edges for message passing:**

```python
edge_index, _ = to_edge_index(adj_new)  # (2, E')
```

**Forward pass & slicing to leaves:**

```python
out = model(x=x_in, edge_index=edge_index, batch=batch_new, edge_attr=None, parent_idx=parent_idx_new_0b)
rel_pred_all        # (N+C, D)    (assumed)
expansion_pred_all  # (N+C,) or (N+C,1)

rel_pred_leaves       = rel_pred_all[leaf_idx_next]          # (C, D)
expansion_pred_leaves = expansion_pred_all[leaf_idx_next]    # (C,) or (C,1)
if 1D: unsqueeze to (C,1)
```

**Refine children coordinates (relative to parent):**

```python
parent_pos_for_children = pos_new[parent_idx_new_0b[leaf_idx_next]]  # (C, D)
pos_new[leaf_idx_next] = parent_pos_for_children + rel_pred_leaves    # (C, D)
```

* This enforces the **relative** geometry: child = parent + predicted offset.

**Next expansion labels:**

```python
# expansion_prob = sigmoid(expansion_pred_leaves.squeeze(-1))  # (C,)
leaf_expansion_next = (expansion_prob > map_threshold).long() + 1  # (C,), in {1,2}
```

* Threshold at `map_threshold` (default 0.5) to map probs → labels (then shift to {1,2}).

### 12) Termination check

```python
size_per_graph_new = scatter(ones_like(batch_new), batch_new)  # (G,)
remaining_capacity_new = target_size - size_per_graph_new      # (G,)
terminated = (remaining_capacity_new < 2).all() or leaf_idx_next.numel() == 0
```

* You need at least 2 free slots somewhere to do a binary branch in the next step.
* Return everything including `terminated`.

---

## Shape cheat-sheet (typical D=3)

* `pos`: `(N,3)` → `pos_new`: `(N+C,3)`
* `leaf_idx`: `(L,)`  → `leaf_idx_next`: `(C,)` (these are the **new** nodes)
* `leaf_expansion`: `(L,)` in `{1,2}` → `leaf_expansion_next`: `(C,)` in `{1,2}`
* `spawn_counts_final`: `(L,)` in `{0,2}`, `sum = C`
* `parent_idx_1b`: `(N,)` → `parent_idx_1b_new`: `(N+C,)`
* `batch_reduced`: `(N,)` → `batch_new`: `(N+C,)`
* `edge_index`: `(2, E')` after rebuild from `adj_new`

---

## Tiny worked example (just to visualize)

Say **G=2** graphs, target sizes `target_size=[12,10]`. Current:

* `N=8` nodes total, `size_per_graph=[5,3]` → `remaining_capacity=[7,7]`.
* `leaf_idx` has `L=3` leaves: `leaf_idx=[1,4,6]` with `leaf_batch=[0,0,1]`.
* `leaf_expansion=[1,2,2]` → `spawn_counts=[0,2,2]`.
* Capacity enforcement: both graphs have cap≥2, total needed=4 ≤ cap in both, so keep both expansions.
* `C = 4` new children.
* Create 4 children with noise `(4,3)`, indices `[8,9,10,11]`.
* New parent edges: add `(p,c)` and `(c,p)` for each.
* `pos_new` is `(12,3)`, `batch_new` is `(12,)`.
* `leaf_idx_next = [8,9,10,11]`.
* Forward pass:

  * `rel_pred_leaves`: `(4,3)`; `expansion_pred_leaves`: `(4,)`.
  * Update `pos_new[8:12] = pos_new[parents_of_8:12] + rel_pred_leaves`.
  * Threshold to get `leaf_expansion_next` in `{1,2}`.
* `size_per_graph_new` becomes `[?]` depending on which graphs those 4 kids belong to (2 per expanding leaf). Recompute capacity and check `terminated`.

---

## Subtle behaviors & gotchas

1. **All-or-nothing per leaf.** You never create a single child; if a leaf expands, it adds **two**. Good for symmetric binary growth.

2. **Deterministic trimming.** When capacity is tight, `expanders[max_leaves:] = 0` keeps the earliest leaves (by index). If you want stochastic fairness, shuffle before trimming.

3. **Ensure progress.** If all expansions are blocked but `cap≥2` exists, you force exactly one leaf to expand. This avoids deadlocks due to per-graph capacity mis-matches.

4. **Noise clipping.** You clip each child’s noise vector by L2 norm to `leaf_noise_clip`. This bounds the initial placement before refinement.

5. **Adjacency rebuild.** You copy old COO and append new edges as weight `1.0`. If the old graph had weighted edges, you preserve them.

6. **Feature gating.** `feats_dim` just sets a leaf-indicator and zero padding. If your model expects richer node features, this is the hook.

7. **Parent indices (1-based externally).** Internally 0-based for indexing; you return 1-based to stay consistent with the rest of your code.

8. **Termination logic ties to binary branching.** Requires `remaining_capacity < 2` everywhere; if you ever allow 1-child spawns, you’d adjust this.

9. **Devices & dtypes.** You consistently allocate new tensors on `device` and match dtypes—prevents cross-device errors.

10. **No gradients.** Because of `@no_grad`, your model’s refinement doesn’t backprop through the expand step (as intended for sampling/generation).
