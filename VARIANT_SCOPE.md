# Variant 2 → 3 — SO(2) edge-angle simplification, then augmented edges (scope + recon)

> Scope & file-map only. Per-variant implementation plans are written separately on this worktree.
> Branch: `variant/so2-edge-angles`, off `diagnostics/l1-l2-stratified-loss` @ 7ff1c90.
> V2 first; V3 as follow-on commits on this same branch, augmentation toggled by config.

## ⚠️ Live flow vs dormant code (read first)

The **active** stack is:

| Layer | Class | File | Selected by |
|---|---|---|---|
| Model | `SO2_EGNN_Network` | `graph_generation/model/egnn_so2.py` | `cfg.model.name=egnn` |
| Method | `Expansion` (diffusion-wrapped) | `graph_generation/method/expansion.py` | `cfg.method.name=expansion` **with** a diffusion |
| Diffusion | `FlowMatchingModel` | `graph_generation/diffusion/flow.py` | `name in {flow, flow_v}` (reference: `basic.py`) |

**DORMANT — do NOT build on:** `egnn_simple`/`egnn_so2_simple.py` (it's just inlined helpers +
a non-batched attention loop — *not* the frame refactor), `Expansion_OneShot`
(`expansion_oneshot.py`), and `Expansion_OneShot_Augmented` (`expansion_oneshot_augmented_edges.py`).

---

## Variant 2 — encode angular info at the edge level, drop the relative-frame complexity

### Where the SO(2) machinery lives today
- **Encoder edge features** (`egnn_so2.py`): per-edge SO(2) decomposition into `du` (along `uhat`),
  `rho` (perp-plane radius), `r_perp` (376-398); edge MLP input assembly + `edge_input_dim`
  (303-422); optional local angles `(cosψ, sinψ, cosθ)` gated by `add_local_angles`.
- **Angle computation** (`helpers.py`): `_compute_tree_directions` (111-233),
  `compute_branch_angles_parent_centric` (567-665), `assign_branch_angles_to_edges` /
  `assign_parent_scalar_to_edges` (668-719). So angles are **already assigned at the edge level** —
  V2 is largely a *re-parameterization/consolidation*, not a from-scratch build.
- **Static geometry cache** `_compute_static_so2_geometry` (`egnn_so2.py:820-856`).

### 🔑 The critical fork — relative frames serve DUAL duty
The relative-frame system is used for **two distinct things**:
1. **Encoder geometry** — the edge angular/scalar features fed to the GNN (what your V2 idea targets:
   distances along `uhat` + in perp plane, angles to parent/sibling/neighbour, angle to the axis).
2. **Decoder target frame** — positions are predicted as offsets in each node's local
   `(forward, sideways, uhat)` frame; GT targets are converted via `global_to_local`, diffusion noises
   isotropically in that frame, and `local_to_global` reconstructs (`compute_local_bases`,
   `flow.py:134`, trace Phases 5b/6b/13). The diffusion contract depends on this frame.

**Decision V2 must make:** does it simplify *only the encoder edge features* (1), or also rework the
*decoder target frame* (2)? Recommended path: **simplify the encoder angular encoding first, keep the
local-frame target system intact** (it's the diffusion contract and is orthogonal to how edges are
encoded). Revisit the target frame only after the encoder change is validated.

### Recommended structure
- New model class `egnn_so2_edge.py` registered as `cfg.model.name="egnn_edge"` (wire in
  `main.py:100-139`) so the base `egnn` stays runnable for A/B comparison — **don't mutate `egnn_so2.py`
  in place.**
- Keep everything SO(2) by feeding the edge MLP only SO(2)-**invariant** scalars (distances + angles).
  Payoff: SO(2)↔SO(3) becomes "which axis-angles you include," not a frame rebuild.
- **`patch_geometry_for_noised_leaves` (`helpers.py:1161`) must stay consistent** — it patches the same
  edge features per diffusion step. Any change to the edge-feature set must be mirrored in the patch path,
  or training/sampling geometry will diverge.

---

## Variant 3 — augmented edges (sibling + geometric-locality neighbours)

### Status of existing scaffolding
- `expansion_oneshot_augmented_edges.py` has the reusable **idea** — `_build_augmented_edge_index`
  (64-112) adds an `EDGE_SIBLING=2` edge type — but it's built on the **dormant one-shot** method.
- `main.py:90-93` bumps `edge_embedding_nums=[3]` for `expansion_augmented` (generic to the model — reusable).
- **Blocker:** `expansion_augmented` + diffusion currently **raises** (`main.py:159-163` requires
  `method_name=="expansion"` for diffusion runs). V3 must allow the augmented method under diffusion.

### V3 work
- Port augmented-edge construction (sibling, plus new **geometric-neighbour** edges) into the **live**
  `expansion.py` build path (`build_directed_edge_index` usage), not the one-shot file.
- Feed V2's edge-level angular features over the augmented edges (this is *why* V3 sits on top of V2 —
  the angular encoding makes augmented-edge angles cheap to add).
- Toggle augmentation via config flag; default off so the base remains the comparison point.

### Sequencing
V3's angular payoff depends on V2's edge-feature parameterization → keep V2 then V3 on this one branch.
The edge-*construction* plumbing alone could be prototyped in parallel, but integration needs V2.

## Compartmentalization
- V2/V3 touch `egnn_so2.py` (or a new `egnn_so2_edge.py`), `helpers.py`, `expansion.py`, configs, `main.py`.
- Only `main.py` wiring overlaps with the V1 worktree.
