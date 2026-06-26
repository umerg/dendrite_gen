# Variant 1 — Topology-given / positions-only (scope + recon)

> Scope & file-map only. The implementation plan is written separately on this worktree.
> Branch: `variant/positions-only`, off `diagnostics/l1-l2-stratified-loss` @ 7ff1c90.

## ⚠️ Live flow vs dormant code (read first)

The **active** training/sampling stack (per `K_ROOT_CHILDREN_FLOW_TRACE.md` and the
`flow*`/`small_trees_run`/`neuron_dataset_run`/`det_synth_run` configs) is:

| Layer | Class | File | Selected by |
|---|---|---|---|
| Model | `SO2_EGNN_Network` | `graph_generation/model/egnn_so2.py` | `cfg.model.name=egnn` |
| Method | `Expansion` (diffusion-wrapped) | `graph_generation/method/expansion.py` | `cfg.method.name=expansion` **with** a diffusion |
| Diffusion | `FlowMatchingModel` | `graph_generation/diffusion/flow.py` | `cfg.diffusion.name in {flow, flow_v}` |
| (reference) | `DenoisingDiffusionModel` | `graph_generation/diffusion/basic.py` | `name=basic` — the trace doc traces this; flow.py is a drop-in with the same `expansion.py` + local-frame contract |

**DORMANT — do NOT build on these:** `Expansion_OneShot` (`expansion_oneshot.py`),
`egnn_simple`/`SO2_EGNN_Sparse_Network_Simple` (`egnn_so2_simple.py`),
`Expansion_OneShot_Augmented` (`expansion_oneshot_augmented_edges.py`). Any earlier notes
pointing at `expansion_oneshot.py` line numbers are for the wrong (one-shot) path.

## How topology + geometry are currently coupled (live flow)

- Positions are predicted as **local-frame offsets** `C` in each node's `(forward, sideways, uhat)` frame.
- The expansion decision is **also flowed jointly** with positions: `e_0 = 2*leaf_expansion - 1`
  (`flow.py:112`), `e_t` is conditioned in as a node feature (`flow.py:150`, `cond_dim=2`), and the
  model emits a 4th channel `expansion_pred` alongside `rel_pred`.
- Two data-prediction MSE losses: `pos_loss = MSE(C_pred, C_0)` (`flow.py:180`),
  `exp_loss = MSE(e_pred, e_0)` (`flow.py:181`); returned as `(exp_loss, pos_loss, diag)` and
  combined in `expansion.get_loss` (`expansion.py:~699-725`, `leaf_expansion_loss`).

## V1 changes (positions-only training; given-topology sampling)

### Training (`predict_positions_only`)
- `flow.py:forward` (77-198): drop the `exp_loss` term (181); instead of conditioning on the
  **noised** `e_t` (150), feed the **clean GT** expansion label as the conditioning feature
  (topology is given, so there's no reason to corrupt it). Keep `pos_loss` only.
- The model still emits `expansion_pred` (4-ch head) — leave it unsupervised/ignored.
- `expansion.get_loss` (`expansion.py:481-725`): gate out the expansion-loss combination.
- Mirror the loss change in `basic.py:forward` if/when `name=basic` runs are used.

### Sampling (`given_topology`)
- `expansion.expand` (137-471): the topology decision today is
  `leaf_expansion_next = (expansion_score > map_threshold) + 1` (`456`). Replace with the **GT**
  expansion label for that reduction level.
- `expand()` **already accepts a `leaf_expansion` arg** (`expansion.py:146`) — the plumbing hook
  exists. The real work is in `sample_graphs` (52): thread the GT per-level expansion schedule
  (from the reduction sequence / `batch.leaf_expansion`, see `get_loss:522-538`) and **align GT
  leaves ↔ spawned leaves across levels** so the tree forms exactly as GT while only positions roll out.
- Optional TF-sampling mode: also pin GT positions of already-placed nodes (diffusion rebuilds
  from `P_0` each step anyway, so this is a feature-assembly tweak).

### Config + wiring
- New method config flags `predict_positions_only` / `given_topology` (a `method/*.yaml`).
- Thread through the `Expansion` constructor in `main.py:166-172`.

## Independence
- Touches `expansion.py`, `flow.py` (and `basic.py` mirror), a method config, `main.py`.
- **Model (`egnn_so2.py`) untouched.** Only `main.py` wiring overlaps with the V2 worktree — trivial.
