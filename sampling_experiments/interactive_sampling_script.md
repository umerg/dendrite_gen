# Interactive Sampling Script Plan

## Objectives
- Run evaluation-only sampling from trained checkpoints while tweaking sampling parameters (expansion threshold, deterministic flag, enforced progress, etc.).
- Mirror the logic of `Expansion_OneShot` and `Expansion_OneShot_Augmented` but emit every intermediate graph state, node metadata, and model predictions so we can inspect how each expansion unfolds.
- Provide both a scripted entry point (for reproducible sweeps/headless runs) and a notebook front-end for interactive visualization of per-step graphs, predicted leaf positions, and expansion probabilities.
- Keep the new work isolated under `sampling_experiments/` so it does not interfere with the training pipeline.

## Current Pipeline Touchpoints
- `main.py`: owns Hydra config loading, dataset construction, reduction factory/dataloader setup, model+method instantiation, and dispatch to `Trainer`. We need the model/method creation logic (especially the EGNN variants and edge-embedding settings) as well as the reduction factory to build consistent train/eval sequences.
- `graph_generation/training.py`: exposes `Trainer.evaluate`, which loads EMA checkpoints, batches validation/test graphs, and calls `method.sample_graphs`. It also handles Hydra output dirs/checkpoints. We mainly need: (1) how checkpoint state_dicts are structured (`model`, `model_ema_*`, optimizer, scheduler) and (2) the batching logic that feeds `method.sample_graphs` target sizes.
- `graph_generation/method/expansion_oneshot.py` & `expansion_oneshot_augmented_edges.py`: define `sample_graphs`/`expand` plus the training `get_loss`. Our interactive runner will largely reuse their logic but wrap `expand` to capture metadata (noise samples, rel_pred, expansion logits, sibling order, adjacency snapshots) per graph/per step.

## Directory Layout (new)
```
sampling_experiments/
├── interactive_sampling_script.md        # this plan
├── configs/
│   └── sampling_eval.yaml                # lightweight overrides referencing Hydra configs
├── loaders/
│   └── checkpoint_loader.py              # utilities to hydrate model/method from checkpoints
├── interactive_methods/
│   ├── expansion_interactive.py          # normal expansion instrumentation
│   └── expansion_augmented_interactive.py# augmented version instrumentation
├── runners/
│   ├── run_interactive_sampling.py       # CLI entry: load config, run sampling, persist traces
│   └── eval_helpers.py                   # wrappers that mimic Trainer.evaluate batching logic
├── artifacts/
│   └── <timestamped_run>/...             # serialized graph sequences, metadata, quick plots
└── notebooks/
    └── interactive_sampling.ipynb        # optional front-end for plotting/step-through
```
(Exact module names can shift, but we want clean separation between config loading, instrumented methods, runners, and visualization.)

## Implementation Plan

### 1. Bootstrap environment & helper utilities
1. Reuse Hydra configs: import `hydra.initialize`/`compose` or load from YAML to keep parity with `main.py`. Provide CLI flags for config path, overrides, checkpoint path, EMA beta to use, and sampling batch size.
2. Add a helper in `loaders/checkpoint_loader.py` that:
   - Calls `gg.reduction.ReductionFactory`/`gg.data.InfiniteRandRedDataset` only if we need additional reduction-based metadata; otherwise, we can skip dataset builds for eval-only runs.
   - Instantiates the correct EGNN model variant using `cfg.model.name` plus any edge embedding overrides for augmented methods (
`edge_embedding_nums = [2]` or `[3]`, dims `[4]`, etc.).
   - Builds both `Expansion_OneShot` and `Expansion_OneShot_Augmented` objects (mirroring `get_expansion_items`) so the runner can switch between them.
   - Loads checkpoint weights (`model` or `model_ema_beta`) from a `.pt` file and pushes them to the requested device.
3. Implement a `SamplingContext` data class encapsulating `cfg`, `model`, `method`, `device`, `ema_beta`, `checkpoint_step`, etc., so downstream tools can annotate outputs with provenance.

### 2. Instrumented expansion methods
1. Create subclasses/wrappers (`InteractiveExpansionOneShot`, `InteractiveExpansionOneShotAugmented`) that:
   - Inherit from the respective base classes or wrap them compositionally.
   - Override `sample_graphs` / `expand` to call the base logic but intercept intermediate tensors.
   - Emit, for every `step`:
     - Current per-graph node list & adjacency (`SparseTensor` or edge list), positions, batch vector.
     - IDs of leaves expanded this step, their expansion logits/probabilities, the deterministic/random selection decisions, noise vectors, predicted relative offsets (`rel_pred`), sibling order assignments, and any enforced adjustments (capacity trimming, ensure_progress triggers).
     - Derived statistics (remaining capacity, map_threshold used, termination flag, number of spawned nodes).
2. Decide on serialization schema for traces, e.g.:
   ```python
   GraphStepTrace(
       step_idx: int,
       graph_id: int,
       node_positions: np.ndarray,
       edges: list[tuple[int,int]],
       leaf_ids: list[int],
       expansion_logits: list[float],
       expansion_probs: list[float],
       leaf_parent_ids: list[int],
       rel_pred: list[list[float]],
       noise: list[list[float]],
       capacity: int,
       sibling_order: list[int],
       enforced_progress: bool,
   )
   ```
   Use dataclasses or Pydantic-style dicts for JSON/pkl dumping. Provide toggles for how much to store (full tensors vs. sampled stats) to control disk usage.
3. Capture both reduced (`adj_reduced`, `leaf_idx`, etc.) and expanded states so we can replay the entire progression. Optionally store a `networkx.Graph` snapshot at each step for quick plotting.
4. For the augmented variant, also log the richer edge types (parent/child/sibling) and any sibling distance regularizer terms if/when we compute them.

### 3. Sampling runner & parameter sweeps
1. `runners/run_interactive_sampling.py` responsibilities:
   - Parse arguments (config path, checkpoint, ema beta, device, method type, batch size, sampling target sizes or dataset split to mimic).
   - Build `SamplingContext` via helpers.
   - Create batches of target sizes: either by loading graphs from disk (similar to `Trainer.evaluate`) or by accepting `--target-sizes 64 96 128` CLI input. Provide a convenience flag to mirror validation/test splits (loads SWC graphs using `utils.data_loading.load_swc_graphs_from_dir`).
   - Call the interactive method's `sample_graphs_with_trace(...)` to obtain both final graphs and the per-step metadata.
   - Persist outputs under `sampling_experiments/artifacts/<run_name>/` as:
     - `config.yaml` (resolved Hydra config + overrides),
     - `sampling_summary.json` (high-level metrics: nodes per graph, steps taken, thresholds),
     - `graph_<idx>_sequence.pkl` (list of `GraphStepTrace`),
     - quicklook PNGs or GIFs (optional) for each graph.
2. Add hooks for live inspection: print per-step summaries to stdout (e.g., `Graph 0 | Step 3 | leaves 5 -> 2 expansions | map_threshold=0.35 | remaining_cap=12`).
3. Implement `runners/eval_helpers.py` with functions to build `target_size` tensors from real graphs, respecting batching logic from `Trainer.evaluate` (permutations, chunking, etc.), so sampling experiments can replicate evaluation orderings.

### 4. Notebook for visualization
1. Create `notebooks/interactive_sampling.ipynb` that:
   - Loads a chosen artifact run, deserializes graph sequences, and provides widgets/sliders (e.g., ipywidgets) to step through `GraphStepTrace` objects.
   - Plots each step using Matplotlib/Plotly: show parent-child edges, highlight leaves being expanded, annotate expansion probabilities, and display predicted offsets vs. actual offsets.
   - Optionally compute aggregated diagnostics (histogram of expansion probs, spatial drift, capacity utilization) to compare normal vs augmented methods.
2. Provide convenience helpers (e.g., `plot_step(graph_trace, step_idx)`) in `sampling_experiments/runners/eval_helpers.py` or a dedicated `visualization.py` so both the script and the notebook reuse the same plotting code.

### 5. Parameter exploration workflow
1. Support runtime overrides for:
   - `map_threshold`
   - `leaf_noise_sigma` / `leaf_noise_clip`
   - `ensure_progress` flag
   - deterministic vs stochastic shuffling
   - optional sibling matching / loss weights (used only for logging but can be toggled to see effects on inference)
2. Implement a sweep utility (could be a simple shell script or Python loop) to iterate across thresholds/noise levels, saving each run under a unique artifact folder, enabling later comparison inside the notebook.
3. Provide summary scripts (e.g., `summarize_runs.py`) to collate stats per run (avg steps, termination reason, node count distribution, invalid graphs) for quick selection before deep dives.

### 6. Integration checkpoints & testing
1. Unit-like tests: craft a tiny synthetic dataset (3–5 node trees) and run the interactive sampler to ensure traces align with manual expectations (node counts, parent-child correctness, stored metadata shapes). Store tests under `tests/test_interactive_sampling.py` if we want automated checks.
2. Dry-run the runner on CPU with a small checkpoint (or randomly initialized weights) to validate serialization paths, Hydra override handling, and notebook compatibility.
3. Document usage in `sampling_experiments/README.md` (future step) outlining CLI + notebook instructions, expected inputs, and troubleshooting tips (e.g., large trace files, GPU memory).

## Notebook vs Script Decision
- **Script** (`run_interactive_sampling.py`): best for reproducibility, sweeps, and integration with existing configs/checkpoints. It can dump rich traces for later consumption.
- **Notebook** (`notebooks/interactive_sampling.ipynb`): ideal for interactive visualization once traces are generated. Rather than re-implementing sampling logic in the notebook (which would reintroduce Hydra/device complexity), we keep heavy lifting in the script and let the notebook load artifacts.
- This split also enables headless experimentation (script) plus interactive analysis (notebook) without duplicating logic.

## Immediate Next Steps
1. Scaffold the directory structure under `sampling_experiments/` (create placeholder modules/files matching the plan) so future diffs are focused.
2. Implement the checkpoint/model loader helper.
3. Fork the expansion methods into interactive subclasses and ensure they produce identical final graphs before adding tracing features.
4. Build the CLI runner + artifact writer, then layer notebook tooling on top.
