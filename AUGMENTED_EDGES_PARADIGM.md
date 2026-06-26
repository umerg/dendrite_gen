# Augmented Edges — Design Paradigm (agreed)

> **Status: implemented** behind `method.augment_edges` (default `False`). Builder +
> bearing assigner in `helpers.py`; gated swap in `expansion.py`; wiring in `main.py`
> + `config/method/expansion.yaml`. Tests: `tests/test_augmented_edges.py`.
>
> Companion to `VARIANT_SCOPE.md` (scope & recon). This is the **paradigm** we agreed on:
> how neighbourhoods are created, why it's train/sample-equivalent, how angular
> information is assigned, and where it plugs into the live stack.
> Branch: `variant/augmented-edges`. Live stack: `egnn_so2.py` + diffusion `Expansion` + `flow.py`.

---

## 0. The three architectural facts this rides on

1. **`update_coors = False`** (`egnn_so2.py:341`, "false for us"). EGNN layers update *features only*;
   the prediction comes from `offset_head` on the final node features. → Augmented edges can only
   influence **messages**, never directly push positions. (VARIANT_SCOPE's coordinate-update worry is moot.)
2. **The model reads all edge geometry from `pre_geom`** (`egnn_so2.py:367-398`), and the live path
   *always* provides it (`precompute_full_geometry` → `patch_geometry_for_noised_leaves`, used identically
   in `flow.py` and `basic.py`). → New edges must have their geometry produced in those two functions.
   `du`/`rho`/`r_perp` are already computed for **every** edge; `assign_branch_angles_to_edges` returns
   **0** for non-parent/child edges (no crash; the angle slots are simply empty for new edge types).
3. **Edges are a function of `parent_idx`/topology, fixed across the whole diffusion trajectory.**
   `patch_geometry` re-derives *geometry* per step on a *fixed* edge set; it never changes the edge set.

---

## 1. The unifying invariant

> **An edge may *exist* only if its existence is fixed by topology + clean (finalized) positions.
> Its *geometry* may then involve a diffusing endpoint — that's what `patch_geometry` handles.**

This is what guarantees train/sample equivalence. The moment an edge's *existence* depends on a
diffusing position (a leaf's own coordinate, which is clean P₀ in training but **unknown** at sampling),
the two paths diverge. Every edge below satisfies the invariant.

Edge taxonomy that falls out of the invariant:

| Pair | Class | Existence determined by | In v1? |
|---|---|---|---|
| leaf ↔ leaf (same parent) | topological (**sibling**) | `parent_idx` only | ✅ |
| leaf → internal | geometric, **parent-anchored** | the leaf's fixed parent's k-NN over fixed internal nodes | ✅ |
| internal ↔ internal | geometric, fully static | both fixed positions | ⏸ deferred |
| leaf ↔ leaf (cousins) | topological | `parent_idx` only | ⏸ deferred |

We never create a geometric edge whose existence rides on a diffusing position.

---

## 2. Edge types (v1)

| id | meaning | direction | added when |
|---|---|---|---|
| 0 | parent → child | directed | existing |
| 1 | child → parent | directed | existing |
| 2 | sibling ↔ sibling | both directions, same type | **new** (topological) |
| 3 | neighbour: internal → leaf | **directed internal→leaf only** | **new** (geometric, parent-anchored) |

→ `edge_embedding_nums = [4]` (extends the existing `main.py:86-93` pattern that already flips `[2]→[3]`).

**Direction rationale (decision: internal→leaf only).** Messages aggregate at `dst`. `internal→leaf`
lets the leaf *read* the map of nearby built structure without a noised leaf broadcasting its noise into
finalized nodes (which would then leak into other leaves). Siblings stay bidirectional (both leaves diffuse
symmetrically; this is the same regime as the existing parent↔leaf edges).

---

## 3. Neighbourhood construction (parent-anchored, radius-then-cap)

For each current-step **leaf** ℓ with parent `p`:

1. Candidate set = **internal nodes** (any node not a current-step leaf), excluding `p` itself
   (already connected via the parent edge) and ℓ's siblings (already connected via sibling edges).
2. **Radius filter**: keep candidates `q` with `‖pos[p] − pos[q]‖ ≤ R_leaf`. Anchor is the **parent's**
   position — never ℓ's own position (R_leaf is the *larger* boundary: the leaf will land ~one branch-length
   off `p`, so we inflate the parent's neighbourhood to cover where ℓ might go).
3. **Cap**: of those, keep the nearest `k_leaf` (nearest to `p`). Bounded degree → no blow-up on dense neurons.
4. Add directed edge `q → ℓ`, type 3.

**Critical discipline (leak-free even in training):** at training the leaf's clean P₀ position is technically
available, but the builder must use **only `pos[p]` (parent) and `pos[q]` (internal)** — never `pos[ℓ]`.
That is exactly what makes the leaf's neighbour *set* the same deterministic function of clean state in both paths.

**Where it's built:** the augmented builder runs at the *same point* as today's `build_directed_edge_index`
(`expansion.py:335` sampling, `:502` training), i.e. **before diffusion**, on positions where every internal
node and every parent is clean. So neighbour edges are static across all diffusion sub-steps; only their
geometry is patched as ℓ moves. The builder's signature extends to take `pos` (for the k-NN) in addition to
`parent_idx`; it still returns `(edge_index, edge_types)` as a drop-in.

**Step 0 (root spawns k children):** no internal nodes exist yet besides the root, so the neighbour set is
empty — only sibling edges. The augmentation degrades gracefully and gets richer as the tree grows.

Hyperparameters: `R_leaf`, `k_leaf`. (Deferred internal↔internal would add a tighter `R_internal`, `k_internal`.)

---

## 4. Angular information — bearing in the receiver's parent-relative frame

The existing angles (`cosψ, sinψ, cosθ`) are a **child-node property in the parent's incoming frame**,
stamped onto the parent↔child edge. A sibling/neighbour pair has no intrinsic incoming direction — which is
why the existing assigner correctly returns 0 for them. We fill those empty slots with a **per-edge bearing**.

**Reference frame = `local_forward[dst]`** (the receiver / leaf endpoint's parent-relative frame — the same
frame the leaf's diffusion target `C_0` already lives in). For sibling edges `local_forward[i] == local_forward[j]`
(siblings share the parent's incoming frame), so it is unambiguous; for `internal→leaf` it is the leaf's own
locked predict-frame.

For an edge `src→dst` with `rel = pos[dst] − pos[src]` and `f = local_forward[dst]`:

```
rel_perp      = rel - (rel·uhat) uhat
rel_perp_unit = rel_perp / ‖rel_perp‖
cosφ = rel_perp_unit · f
sinφ = (f × rel_perp_unit) · uhat          # signed, in the SO(2) plane
cosθ = (rel·uhat) / ‖rel‖                   # axial tilt (optional; fills cos_theta slot)
```

**Properties**

- **SO(2)-invariant.** Rotate the tree by R → `rel` and `f` both rotate by R → `(cosφ, sinφ, cosθ)` and
  `du`, `rho` are unchanged.
- **Generalizes the existing machinery, doesn't replace it.** For a `parent→child` edge,
  `rel = v_out(child)` and `f = local_forward[child] = v_in(child)`, so this bearing **equals the existing ψ**.
  We are computing the same quantity for every edge; it just happened to be defined only on tree edges before.
- **Not GemNet / not the deferred refactor.** We use a single, pre-existing per-node reference frame; we never
  compute angles between adjacent edges (no triplets). The full directional-MP construction stays deferred.
- **Auto-patched.** `patch_geometry` already locks `local_forward` (reads from `pre_geom_p0`) and patches
  `rel_coors` for leaf-touching edges → `(cosφ, sinφ)` update as the leaf moves while the frame stays fixed.
  This is the same "lock v_in, patch v_out" pattern that governs branch angles today.

**Implementation discipline (zero regression):** *do not* touch `assign_branch_angles_to_edges` for
parent/child edges. Add a new assigner that writes `(cosφ, sinφ, cosθ)` **only into the currently-zero rows**
for type-2/3 edges. Reuse the existing `cospsi_edge / sinpsi_edge / cos_theta_edge` channels → **no model
feature-dim change**; the edge-type embedding disambiguates "ψ of a branch" vs "φ of an augmented edge."

**Shared-frame subtlety (root children, step 0):** in sampling `local_forward` is the *shared random* frame
vs `fwd0` (GT) in training. φ is still consistent — the siblings co-rotate with whichever shared frame is
chosen, so φ is invariant to that choice. Same SO(2) argument that already justifies the shared-frame target scheme.

---

## 5. Effect on flow dynamics

Adding these edges is a **pure denoiser-architecture change**: the noising process, the targets `C_0`, the
local frames, and the loss are all untouched. Only the vector field's receptive field changes —

- **Siblings** exchange messages in 1 hop instead of 2-hops-through-the-parent (helps coordinate branch fans,
  avoid overlap).
- **Leaves** gain awareness of the already-built structure around them (collision/density cues).

Because `update_coors=False`, all of this stays in feature space and cannot destabilize the offset head.
At high σ the augmented-edge geometry is noise-dominated, but that's the same regime as the existing noised
parent→leaf edges; the `log_sigma` conditioning already lets the model down-weight unreliable geometry early
and lean on it near the data manifold.

---

## 6. Train ↔ sample equivalence (summary)

| Element | Why equivalent |
|---|---|
| sibling edge set | function of `parent_idx` only → bit-identical in both paths |
| neighbour edge set | function of fixed parent + fixed internal nodes only (never the leaf's own position) |
| edge geometry (`du,rho,r_perp`) | same `precompute_full_geometry` + `patch_geometry` in both paths |
| bearing reference frame | `local_forward[dst]`, locked to P₀ in both paths (random-but-equivariant for root step 0) |
| noising / targets / loss | unchanged from the current diffusion |

---

## 7. Plug-in points (light, config-toggled)

| Hook | File:line | Change |
|---|---|---|
| Augmented builder | new helper (port `_build_augmented_edge_index`, `expansion_oneshot_augmented_edges.py:64-112`) | add sibling (type 2, topological) + parent-anchored neighbour (type 3, radius-then-cap, internal→leaf); take `pos`+`leaf_idx`; return `(edge_index, edge_types)` |
| Swap behind flag | `expansion.py:335` (sample), `:502` (train) | call augmented builder when `augment_edges` set; still wrap `edge_attr = edge_types.unsqueeze(-1)` |
| Bearing assigner | new fn in `helpers.py`, called inside `precompute_full_geometry` (`:~1079`) and `patch_geometry_for_noised_leaves` (`:1161`) | fill `cospsi/sinpsi/cos_theta` zero-rows for type 2/3 with φ/θ in `local_forward[dst]` |
| Edge embedding count | `main.py:86-93` | `edge_embedding_nums = [4]` when `augment_edges` |
| Config flag | plain `expansion` method config | `augment_edges: bool`, `neighbour_k` (`k_leaf`), `neighbour_radius` (`R_leaf`) — avoids the `expansion_augmented`+diffusion guard at `main.py:159-163` |

---

## 8. Deferred (post-v1 toggles)

- internal ↔ internal geometric edges (tighter `R_internal`/`k_internal`) — second-order (don't touch
  predicted leaves directly); add as type 4 when wanted.
- topological cousin edges (leaf↔leaf across sibling-parents).
- bidirectional / direction-typed neighbour edges (if internal nodes should adapt to leaves).
- full directional-MP (GemNet) triplet angles — the separate SO(2) angle refactor on the other worktree.

---

## Open hyperparameters to set before first run
- `R_leaf` (neighbour radius around parent) and `k_leaf` (cap). Start generous on R, modest on k (e.g. k≈8–16).
