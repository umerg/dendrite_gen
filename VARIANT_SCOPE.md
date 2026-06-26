# Augmented edges on the current architecture (standalone) — scope + recon

> Scope & file-map only. The implementation plan is written separately on this worktree.
> Branch: `variant/augmented-edges`, off `diagnostics/l1-l2-stratified-loss` @ 7ff1c90.
> **Deliberately decoupled from the SO(2) angle refactor** — this rides the *current* model
> as-is, so it's a light, config-toggled change. No `egnn_so2.py` angular rewrite.

## ⚠️ Live flow (build on these)

| Layer | Class | File | Selected by |
|---|---|---|---|
| Model | `SO2_EGNN_Network` | `graph_generation/model/egnn_so2.py` | `cfg.model.name=egnn` |
| Method | `Expansion` (diffusion-wrapped) | `graph_generation/method/expansion.py` | `cfg.method.name=expansion` + a diffusion |
| Diffusion | `FlowMatchingModel` | `graph_generation/diffusion/flow.py` | `name in {flow, flow_v}` |

**Dormant — reuse the *idea*, not the file:** `expansion_oneshot_augmented_edges.py` has the sibling
edge logic (`_build_augmented_edge_index`, 64-112, `EDGE_SIBLING=2`) but it's bolted onto the dead
one-shot method. Port the logic into the live path; don't depend on that class.

## Why this is light (the key recon finding)

1. **Edge construction is one function, two call sites.** `build_directed_edge_index(parent_idx)`
   (`helpers.py:14-42`) is called only at `expansion.py:335` (sampling `expand`) and `:502`
   (training `get_loss`). Both wrap the result into `edge_attr = edge_types.unsqueeze(-1)`
   (`:340-343`, `:671-674`) and pass `edge_index`+`edge_attr` to `precompute_full_geometry` and the model.
2. **The model already embeds categorical edge types.** `egnn_so2.py:579-593` builds
   `nn.Embedding` layers over `edge_embedding_nums`; `main.py:90-93` already sets `[3]` (vs `[2]`)
   when augmentation is requested. A 3rd edge type "just works" through the embedding.
3. **Sibling/neighbour edges get SO(2) geometry for free.** The per-edge decomposition into
   `du` (along `uhat`), `rho` (perp-plane radius), `r_perp` is computed for **every** edge from
   `rel_coors = pos[dst]-pos[src]` (`egnn_so2.py:376-398`), independent of edge type.
   `assign_branch_angles_to_edges` (`helpers.py:668-719`) returns **0** for non-parent/child edges
   (its masks only match `parent_idx[dst]==src` etc.) — safe, no crash, no angle work required.
   → Siblings carry: categorical type embedding + `du`/`rho`/`r_perp` distance geometry. That's the
   whole point — useful geometric locality without touching the angular machinery.

## Scope of work

### Edge builder
- Add an augmented builder (sibling edges = type 2; optionally geometric-locality **neighbour** edges
  = type 3) — port/adapt `_build_augmented_edge_index`. Keep it returning `(edge_index, edge_types)`
  so it's a drop-in for `build_directed_edge_index`.
- Swap it in behind a flag at the two call sites: `expansion.py:335` and `:502`.

### Config flick (matches "just flick on in config")
- Cleanest: add a boolean `augment_edges` (and maybe `neighbour_k`) to the **plain `expansion`**
  method config, rather than the `expansion_augmented` name — because `expansion_augmented` + diffusion
  currently **raises** (`main.py:159-163` requires `method_name=="expansion"` for diffusion).
- In `main.py`: when `augment_edges`, set `edge_embedding_nums=[3]` (or `[4]` with neighbours) and pass
  the flag into the `Expansion` constructor (`main.py:166-172`); `Expansion` picks the augmented builder.

### Things to verify in the plan (not blockers, design choices)
- **Coordinate updates**: if any layer has `update_coors=True`, sibling edges also feed the weighted
  `rel_coors` position update (`egnn_so2.py` coors aggregation). Decide whether siblings should affect
  positions or only messages. (Live runs may be static-geometry / offset-head only — confirm.)
- **`patch_geometry_for_noised_leaves`** (`helpers.py:1161`) patches edges touching leaves per diffusion
  step; sibling edges among newly-spawned leaves touch leaves on both ends → confirm they're patched
  consistently (they take `edge_index` as input, so they should flow through).
- **Symmetry/count**: sibling edges are O(k²) per parent (fine for dendrites' small branching factor).

## Relationship to the other worktrees
- Independent of `variant/so2-edge-angles` (the angle refactor). This tree proves out augmented edges
  on the current arch **without** that refactor, so the angle change becomes an optional, separate
  experiment rather than a prerequisite for augmented edges.
- Only overlap with other trees is `main.py` wiring (trivial).
