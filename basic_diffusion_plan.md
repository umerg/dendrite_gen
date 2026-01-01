# Basic diffusion plan (training only)

This document describes the **noise-conditioned (σ-conditioned) diffusion training** used to replace one-shot leaf prediction in `expansion_oneshot.py`.

We focus **only on training-time logic** (loss computation). Sampling/rollout will be handled later.

---

## 0. Goal

We generate geometric binary trees by iterative leaf expansion. At each expansion step we need to predict, for the *new* leaves created in the previous step:

1. **Relative position offset** from parent: \(C_0 \in \mathbb{R}^{L\times 3}\)
2. **Expansion state** (whether the leaf will expand next step): \(e_0 \in \mathbb{R}^{L\times 1}\)

where \(L\) is the number of **training leaves** for the current step (subset of `batch.leaf_idx`).

The EGNN model consumes **absolute positions** \(P\in\mathbb{R}^{N\times 3}\) and node features, so during diffusion we:
- corrupt **only** the new/train leaves in relative space (\(C\), \(e\)),
- rebuild a noisy absolute position tensor \(P_t\) to pass through the model.

---

## 1. Notation and batch fields

### Core tensors
- `N`: number of nodes in the (batched) tree graph
- `L`: number of new/train leaves we train on for this forward pass
- `P_0`: absolute GT positions, shape `[N, 3]`  (from `batch.pos`)
- `parent_idx`: shape `[N]`, **0-based**, with `-1` for roots (decoded from `batch.parent_idx_1b`)
- `leaf_idx_all`: indices of leaves in the current reduced graph, shape `[L_total]` (from `batch.leaf_idx`)
- `leaf_idx_train`: subset of leaves we are training on at this step, shape `[L]`
- `leaf_parent_idx`: parent indices aligned with `leaf_idx_train`, shape `[L]`
- `C_0`: GT parent-relative offsets for train leaves, shape `[L, 3]`
  - `C_0 = P_0[leaf_idx_train] - P_0[leaf_parent_idx]`
- `leaf_expansion`: leaf expansion labels aligned to `leaf_idx_train`, shape `[L]` or `[L,1]`
  - In `expansion.py`, these are mapped to `{0,1}` via `batch.leaf_expansion - 1`.
  - For diffusion we treat expansion as a **continuous variable**; we use:
    - `e_0 = 2 * leaf_expansion - 1` so `e_0 ∈ {-1, +1}` with shape `[L,1]`.

### Graph / model inputs
- `edge_index`: directed edges built from `parent_idx`, shape `[2, E]`
- `edge_attr`: directed edge type id, shape `[E, 1]` (parent→child vs child→parent)
- `batch.batch`: PyG batch vector, shape `[N]`
- `node_feats_base`: base node features built in `expansion.py`, shape `[N, F_base]`
  - includes at least: `is_leaf`, `geo_lr_mask` (if available), optional `new_leaf_mask`, `size_ratio`, padding zeros.

---

## 2. High-level training strategy (Choice B)

We use **σ-conditioned denoising** (continuous noise level). This is consistent with your existing scaffolding:

```python
sigma = (rnd_normal * self.P_std + self.P_mean).exp()
```

Equivalently: `log(sigma) ~ Normal(P_mean, P_std^2)`.

### What we do (now)
- Sample a noise level **σ** (often named `t` in code).
- Create noisy leaf variables:
  - `C_t = C_0 + σ * ε_C`
  - `e_t = e_0 + σ * ε_e`
- Condition the network on σ via a feature (recommended: `log_sigma`).
- Train with **plain MSE** (unweighted) on the denoised targets (`x0`-prediction):
  - predict `C_0` and `e_0` from the noisy inputs.

### What we explicitly skip (for now)
We **do not** apply EDM loss-weighting or EDM preconditioning yet.
- i.e. set `weight = 1.0` everywhere.
- keep σ-conditioning so EDM can be added later with minimal changes.

---

## 3. What is diffused?

Only the **new/train leaves**, in *relative* space:

- Position diffusion variable: `C` (leaf parent-relative offset)
- Expansion diffusion variable: `e` (continuous scalar, derived from discrete label)

All other nodes remain “clean” (their absolute coordinates are not perturbed) in training.

This mirrors the one-shot masking idea, but replaces “masked with small Gaussian around parent” with “corrupted with controlled σ-noise”.

---

## 4. Model conditioning and inputs

### 4.1 Build noisy absolute coordinates `P_t` for the model

The EGNN takes absolute positions, so we construct `P_t`:

- Start with `P_t = P_0.clone()`
- Overwrite only train leaves:
  - `P_t[leaf_idx_train] = P_0[leaf_parent_idx] + C_t`

All other nodes keep GT abs coords:
- `P_t[i] = P_0[i]` for `i` not in `leaf_idx_train`.

### 4.2 Add diffusion conditioning features

We append two diffusion-specific features:

1) **Noisy expansion value per node**
- `e_feat = zeros([N,1]); e_feat[leaf_idx_train] = e_t`

2) **Noise level feature**
- `log_sigma` broadcast per node:
  - `log_sigma_node = log(sigma_graph)[batch]`  → shape `[N,1]`
  - (or per-leaf; per-graph is simplest & stable)

Then:
- `node_feats_t = cat([node_feats_base, e_feat, log_sigma_node], dim=-1)`

Finally model input:
- `x_in = cat([P_t, node_feats_t], dim=-1)`  (same pattern as oneshot)

---

## 5. Model outputs and targets

The model is expected to output (either as dict or tuple):

- `rel_pred_all`: shape `[N,3]`, predicted denoised **relative offsets** for all nodes
- `expansion_pred_all`: shape `[N,1]` (or `[N]`), predicted denoised expansion scalar for all nodes

We index to train leaves:
- `C_pred = rel_pred_all[leaf_idx_train]`  → `[L,3]`
- `e_pred = expansion_pred_all[leaf_idx_train]` → `[L,1]`

Targets are:
- `C_0` → `[L,3]`
- `e_0` → `[L,1]`

---

## 6. Losses

### 6.1 Leaf-only diffusion loss (plain MSE)

We compute losses **only** for `leaf_idx_train`:

- `pos_loss = mean( (C_pred - C_0)^2 )`  (over `L×3`)
- `exp_loss = mean( (e_pred - e_0)^2 )`  (over `L×1`)

Total loss:
- `loss = pos_loss + λ_exp * exp_loss`

Notes:
- If you want the combined Frobenius over `L×4`, you can match it by choosing:
  - `λ_exp = 1.0` if both terms are computed as **sums**
  - `λ_exp = 3.0` if you compute **mean over dims** separately but want equal per-dimension weighting
- In practice, keep `λ_exp` as a hyperparameter (start with `1.0`).

### 6.2 Why splitting is safe

Let \(A = [C, e]\) with shapes `C: [L,3]`, `e: [L,1]`. Then:
\[
\|A_\theta - A_0\|_F^2 = \|C_\theta - C_0\|_F^2 + \|e_\theta - e_0\|_2^2
\]
So “combined Frobenius MSE” equals “position MSE + expansion MSE” up to the exact reduction/normalization convention.

---

## 7. Implementation plan

### 7.1 `expansion.py` (already mostly correct)

`Expansion.get_loss(batch, model)` should:

1) Decode parents:
- `parent_idx = self._decode_parent_indices(batch)`  (0-based; -1 for roots)

2) Build directed graph:
- `edge_index, edge_types = self._build_directed_edge_index(parent_idx)`
- `edge_attr = edge_types.unsqueeze(-1).float()`

3) Select train leaves:
- `leaf_idx_train = self._select_training_leaf_indices(batch)`
- `leaf_parent_idx = parent_idx[leaf_idx_train]`
- Filter any invalids (root) defensively (already asserts in code)

4) Build targets:
- `C_0 = self._leaf_rel_targets(P_0, leaf_idx_train, leaf_parent_idx)`
- `leaf_expansion` aligned to node indices, then indexed by `leaf_idx_train` (already done)

5) Build base node features:
- `node_feats_base` from `is_leaf`, `geo_lr_mask`, optional `new_leaf_mask_from_next`, `size_ratio`, padding zeros.

6) Call diffusion:
```python
exp_loss, pos_loss = self.diffusion(
    node_feats=node_feats_base,
    edge_index=edge_index,
    batch=batch.batch,
    edge_attr=edge_attr,
    P_0=P_0,
    C_0=C_0,
    parent_idx=parent_idx,
    leaf_idx_train=leaf_idx_train,
    leaf_expansion=leaf_expansion,
    leaf_parent_idx=leaf_parent_idx,
    model=model,
)
loss = pos_loss + lambda_exp * exp_loss
```

#### Important: feature dimensionality contract
Because diffusion appends `[e_feat, log_sigma]` **inside `basic.py`**, the model’s `feats_dim` should equal:
- `F_total = F_base + 2`

To keep `expansion.py` consistent:
- either set `F_base = model.feats_dim - 2` when diffusion is enabled,
- or have the model expose `model.feats_dim_base` and use that in `expansion.py`.

**Recommended quick fix** (minimal invasive):
- In `expansion.py`, when building `node_feats`, do:
  - `feats_dim_total = getattr(model, "feats_dim", 0)`
  - `cond_dim = getattr(self.diffusion, "cond_dim", 2)`  (default 2)
  - `feats_dim = max(feats_dim_total - cond_dim, 0)` for base feature construction

Then diffusion always appends exactly `cond_dim` features to reach `feats_dim_total`.

### 7.2 `basic.py` (main work)

Implement `DenoisingDiffusionModel.__call__` / `forward` (currently `get_loss`) properly.

#### Step-by-step inside `basic.py`

**Inputs**: the same signature already present in scaffolding:
- `node_feats, edge_index, batch, edge_attr, P_0, C_0, parent_idx, leaf_idx_train, leaf_expansion, leaf_parent_idx, model`

**1) Prepare expansion targets**
- Ensure `leaf_expansion` is float `[L,1]`
- Convert `{0,1} -> {-1,+1}`:
  - `e_0 = 2 * leaf_expansion.float().view(-1,1) - 1`

**2) Sample σ (t)**
- Sample per-graph:
  - `G = batch.max() + 1`
  - `sigma_graph = exp(N(P_mean, P_std))`  shape `[G]`
- Clamp (recommended):
  - `sigma_graph = sigma_graph.clamp(self.sigma_min, self.sigma_max)`
- Map to leaves:
  - `leaf_batch = batch[leaf_idx_train]`
  - `sigma_leaf = sigma_graph[leaf_batch].view(-1,1)`  shape `[L,1]`

**3) Corrupt leaf variables**
- `eps_C ~ N(0,1)` shape `[L,3]`
- `eps_e ~ N(0,1)` shape `[L,1]`
- `C_t = C_0 + sigma_leaf * eps_C`
- `e_t = e_0 + sigma_leaf * eps_e`

**4) Build `P_t`**
- `P_t = P_0.clone()`
- `P_t[leaf_idx_train] = P_0[leaf_parent_idx] + C_t`

**5) Build diffusion features**
- `e_feat = zeros([N,1]); e_feat[leaf_idx_train] = e_t`
- `log_sigma_node = log(sigma_graph)[batch].view(N,1)`
- `node_feats_t = cat([node_feats, e_feat, log_sigma_node], dim=-1)`

**6) Run model**
- `x_in = cat([P_t, node_feats_t], dim=-1)`
- `out = model(x=x_in, edge_index=edge_index, batch=batch, edge_attr=edge_attr, parent_idx=parent_idx)`
- Get:
  - `rel_pred_all = out["rel_pred"]`
  - `exp_pred_all = out["expansion_pred"]`

**7) Leaf-only losses (unweighted)**
- `C_pred = rel_pred_all[leaf_idx_train]`
- `e_pred = exp_pred_all[leaf_idx_train].view(-1,1)`
- `pos_loss = mean((C_pred - C_0)**2)`
- `exp_loss = mean((e_pred - e_0)**2)`

Return `exp_loss, pos_loss` (scalars).

#### Notes / pitfalls to avoid
- **Do not** use `weight[batch]` for leaf losses. If you later re-enable weights, you must use `leaf_batch`.
- Ensure you reduce to scalars inside diffusion (so `expansion.py` can log `.item()` safely).
- Keep everything leaf-only; otherwise the model can “cheat” by using clean GT leaf info.

---

## 8. Minimal pseudocode (training)

```python
# expansion.py
parent_idx = decode_parent(batch.parent_idx_1b)      # [N]
edge_index, edge_types = build_edges(parent_idx)     # [2,E], [E]
leaf_idx_train = select_train_leaves(batch)          # [L]
leaf_parent_idx = parent_idx[leaf_idx_train]         # [L]
C_0 = P_0[leaf_idx_train] - P_0[leaf_parent_idx]     # [L,3]
leaf_expansion = map_leaf_expansion_to_nodes(...)[leaf_idx_train]  # [L] in {0,1}

node_feats_base = build_base_feats(...)
exp_loss, pos_loss = diffusion(...)

loss = pos_loss + lambda_exp * exp_loss
```

```python
# basic.py
e_0 = 2*leaf_expansion.float().view(L,1) - 1

sigma_graph = exp(N(P_mean, P_std)).clamp(sigma_min, sigma_max)   # [G]
sigma_leaf  = sigma_graph[batch[leaf_idx_train]].view(L,1)

C_t = C_0 + sigma_leaf * randn(L,3)
e_t = e_0 + sigma_leaf * randn(L,1)

P_t = P_0.clone()
P_t[leaf_idx_train] = P_0[leaf_parent_idx] + C_t

e_feat = zeros(N,1); e_feat[leaf_idx_train] = e_t
log_sigma_node = log(sigma_graph)[batch].view(N,1)

node_feats_t = cat([node_feats_base, e_feat, log_sigma_node], -1)
x_in = cat([P_t, node_feats_t], -1)

out = model(x=x_in, edge_index=edge_index, batch=batch, edge_attr=edge_attr, parent_idx=parent_idx)
C_pred = out["rel_pred"][leaf_idx_train]
e_pred = out["expansion_pred"][leaf_idx_train].view(L,1)

pos_loss = mean((C_pred - C_0)**2)
exp_loss = mean((e_pred - e_0)**2)
return exp_loss, pos_loss
```

---

## 9. Debug / sanity checks

Add these checks early to avoid silent failure:

1) **Leaf corruption actually applied**
- Verify `P_t[leaf_idx_train] != P_0[leaf_idx_train]` for typical σ.

2) **No root in train leaves**
- Ensure `(leaf_parent_idx >= 0).all()`.

3) **Shapes**
- `C_0: [L,3]`, `e_0: [L,1]`, predictions same.

4) **Conditioning**
- `log_sigma_node` is present; training should degrade if removed.

5) **Loss computed only over `leaf_idx_train`**
- No indexing mistakes / broadcasting with `batch`.

---

## 10. Future: adding EDM later (no code churn)

Once the above is stable, adding EDM is localized to `basic.py`:

- enable EDM loss weighting as a function of σ,
- optionally add EDM preconditioning (skip+scales),
- keep the same corruption form `x = x0 + σ ε`,
- sampling will be implemented in `sample()`.

Because we already use σ-conditioning and relative-space corruption, EDM can be integrated without changing `expansion.py` or the target definitions.
