# Viewing-Each-Step Evaluation Plan

## Objective
Leverage the existing `graph_generation` training stack to probe the model **step-by-step on ground-truth reduction sequences**. Instead of sampling entire graphs from scratch, we will:
1. Take a real graph (from disk or a synthetic generator),
2. Run the same reduction pipeline (`ReductionFactory` → `ReducedGraphData`) used for training,
3. Feed each reduced state through a **new eval-only algorithm** that mirrors `Expansion_OneShot.get_loss`,
4. Capture predicted offsets + expansion logits along with the GT metadata for that step,
5. Visualize GT vs. prediction per step inside a notebook (slider for steps, toggle GT vs. predicted leaves, probability table).

## Building Blocks to Reuse
- `graph_generation/reduction.py` → `ReductionFactory` and `CherryReducer` already supply `reduction_level`, leaves, parents, sibling order, etc. exactly as needed.
- `graph_generation/data/reduction_dataset.py` (`RandRedDataset` and friends) already bundles a reducer state into `ReducedGraphData`. We can call `get_random_reduction_sequence` directly to obtain a full trajectory for one graph.
- `graph_generation/data/data.py` → `ReducedGraphData` defines the PyG attributes required by `Expansion_OneShot.get_loss`.
- `graph_generation/method/expansion_oneshot.py` provides the masking, feature construction, and forwards logic we must mirror. We mainly need the internals of `_make_masked_positions`, `_size_ratio_feature_from_batch`, expansion head outputs, etc.
- `graph_generation/training.py` (and `main.py`) defines how we instantiate models/methods/datasets from Hydra configs; `sampling_experiments/loaders/checkpoint_loader.py` already mirrors those pieces for interactive sampling and can be extended for this workflow.

## Proposed Workflow
1. **Notebook input** (config path + graph path): the notebook will compose a lightweight request (cfg path, overrides, checkpoint path, graph identifier) and call a helper in `sampling_experiments/` instead of reimplementing Hydra logic.
2. **Helper setup script** (e.g., `sampling_experiments/loaders/sequence_setup.py`):
   - Load Hydra config (same as `run_interactive_sampling.py`),
   - Instantiate `ReductionFactory`, EGNN model, and expansion method (reuse `_instantiate_model/_instantiate_method` from `checkpoint_loader.py`),
   - Load checkpoint weights,
   - Convert the provided graph into adjacency/position (reuse `utils.data_loading.nx_graph_to_adj_pos` or accept a `.pkl/.gpickle` path).
   - Produce **one reduction sequence** via `ReductionFactory(adj).get_random_reduction_sequence(...)`, returning ordered `ReducedGraphData` objects.
3. **Eval-time `get_loss` clone** (new module, e.g., `sampling_experiments/interactive_methods/expansion_eval_step.py` or `sampling_experiments/eval/step_eval.py`):
   - Accept a `Batch` of `ReducedGraphData` (multiple reduction levels batched for efficiency),
   - Reuse the logic from `Expansion_OneShot.get_loss` up through the forward pass, but instead of computing losses, return:
     - Absolute GT leaf positions,
     - Masked inputs,
     - Parent indices, sibling order, `reduction_level`,
     - Predicted relative offsets + absolute positions (`parent_pos + pred_rel`),
     - Expansion logits & probabilities for each leaf,
     - Any auxiliary masks (new leaf mask, size ratio, etc.).
   - Provide the per-step metadata in an easy-to-serialize structure (dataclass or dict) keyed by `step_idx` / `reduction_level`.
   - Still compute the original losses (for reference) but keep them separate from visualization payload.
4. **Sequence runner** (new orchestrator under `sampling_experiments/runners/`):
   - Accepts a `SamplingContext`, the single-graph reduction sequence, and an optional batch size (to chunk steps).
   - Forms `torch_geometric.data.Batch` objects from slices of the sequence (e.g., 8 steps per forward pass),
   - Calls the eval-only method, collects results per step, and caches them structure like:
     ```python
     StepEvalRecord(
         step_idx: int,                       # matches reduction_level
         reduced_graph_metadata: {...},       # adjacency, num_nodes, leaves
         gt_leaf_positions: np.ndarray,       # [L,3]
         pred_leaf_positions: np.ndarray,     # [L,3]
         expansion_logits: np.ndarray,        # [L]
         expansion_probs: np.ndarray,         # [L]
         leaf_indices: list[int],             # original node IDs
         parent_indices: list[int],
         masked_positions: np.ndarray,        # what the network saw
         losses: dict,                        # optional (pos, expansion, sibling)
     )
     ```
   - Persist the list of `StepEvalRecord` objects (e.g., pickle/JSON) alongside any convenience tensors for the notebook.
5. **Notebook visualization** (new `sampling_experiments/notebooks/view_each_step.ipynb`):
   - Load the serialized `StepEvalRecord`s,
   - Provide widgets:
     - Graph selector (if we later support >1),
     - Step slider,
     - Toggle buttons for GT vs. predicted leaves (colors: e.g., blue nodes for GT leaves, orange markers for predictions),
     - Table showing `[node_id, gt_prob(label), pred_prob]` plus the logits/probabilities.
   - Use Plotly or Matplotlib 3D scatter to render nodes/edges per step (edges already available as adjacency from `ReducedGraphData.adj`).
   - Display textual stats: losses, reduction level, number of leaves expanded, etc.

## Implementation Steps
1. **Setup helper module**
   - Create `sampling_experiments/loaders/sequence_setup.py`.
   - Move/shared logic from `checkpoint_loader.py` for instantiating model/method to avoid duplication (possibly expose a `build_method_and_model(cfg, method_cls=None)` helper).
   - Provide a function `prepare_sequence_context(config_path, graph_path, checkpoint_path, overrides=None, ema_beta=None, device='cpu')` that returns `(SamplingContext, List[ReducedGraphData])`.
   - Inside, handle both NetworkX `.gpickle` files and raw adjacency/position arrays; rely on `nx_graph_to_adj_pos` if the user supplies an SWC-derived graph path.

2. **Reduction sequence extraction**
   - Expose a helper `build_reduction_sequence(graph, red_factory)` that mirrors `RandRedDataset.get_random_reduction_sequence`.
   - Ensure we store all `ReducedGraphData` attributes (parent_idx_1b, sibling_order, new_leaf masks, etc.) because `get_loss` expects them.
   - Attach a monotonically increasing `step_idx` (maybe via `data.step_idx = th.tensor(step_number)`) so we can reference it later in the notebook; `ReducedGraphData.reduction_level` already serves as a base.

3. **Eval-time method**
   - Create `sampling_experiments/eval/expansion_step_evaluator.py` (name TBD) that defines `ExpansionStepEvaluator`.
   - This class either subclasses `Expansion_OneShot` or wraps it; it should expose `collect_step_predictions(batch, model)` which internally calls the shared logic from `Expansion_OneShot.get_loss`.
   - Refactor the common code paths (masking, feature building, forward pass) into reusable functions to avoid copy/paste (e.g., extend `Expansion_OneShot` with protected helpers). If refactoring inside `graph_generation/method/expansion_oneshot.py` is too invasive, encapsulate the replication carefully and keep the functions next to the evaluator.
   - Outputs per-leaf predictions + metadata. Optionally compute standard losses for diagnostics but return them along with per-leaf info instead of aggregating.

4. **Sequence runner + serialization**
   - Add `sampling_experiments/runners/run_stepwise_eval.py` (or similar) with functions:
     - `evaluate_sequence(context, reduction_sequence, batch_size=8)` → returns `List[StepEvalRecord]`.
     - `save_stepwise_results(run_dir, step_records, metadata)` storing pickles/JSON for notebook consumption.
   - Metadata should include config path, checkpoint info, graph source, RNG seeds, etc.
   - Align output folder structure with existing `sampling_experiments/artifacts/`, e.g., `artifacts/view_each_step/<graph_name>/<timestamp>/`.

5. **Notebook + visualization helpers**
   - Under `sampling_experiments/notebooks/`, add `view_each_step.ipynb`.
   - Provide a small helper module (`sampling_experiments/visualization/step_plots.py`) that:
     - Converts `StepEvalRecord` into scatter traces (GT nodes vs. predicted leaves),
     - Builds HTML tables for probabilities,
     - Handles toggles (GT vs. prediction) to simplify notebook code.
   - Notebook workflow:
     1. Select artifact folder,
     2. Load records,
     3. Use widgets (ipywidgets slider + toggle buttons) to update the 3D plot + table.

6. **Testing / sanity checks**
   - Add a smoke test (e.g., `tests/test_step_eval_pipeline.py`) that:
     - Generates a tiny synthetic tree,
     - Runs the stepwise evaluator on CPU with a randomly initialized model,
     - Asserts that we get one `StepEvalRecord` per reduction level and that shapes match (same number of leaves in GT vs. predictions, etc.).
   - Optionally compare the aggregated losses from the evaluator against the original `get_loss` outputs to ensure parity.

7. **Documentation**
   - Update `sampling_experiments/README.md` with a new section describing the “View Each Step” workflow, CLI invocation, and notebook usage (where to put `config`, `graph_path`, `checkpoint`).
   - Document required inputs (graph format, expected config overrides) and tips for large graphs (batch size for steps, GPU vs. CPU).

## Notes / Considerations
- `ReducedGraphData` already carries `reduction_level` and `total_tree_size`; reuse these fields instead of inventing new names when possible.
- Masking logic must remain identical to training so the evaluation is faithful; consider factoring `_make_masked_positions` and feature construction into reusable utilities rather than duplicating inside the evaluator.
- We are working with a **single graph** at a time, but the evaluator should be general enough to batch multiple graphs (useful for future comparisons). Design the API accordingly.
- Keep all new files inside `sampling_experiments/` to avoid touching the core training loop; if small refactors are needed in `graph_generation/method/expansion_oneshot.py`, gate them carefully with backward-compatible helpers.
