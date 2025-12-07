# Sampling Experiments Playground

This folder contains everything needed to run interactive sampling sessions on top of the
existing expansion methods, load traces from checkpoints, and visualize each expansion step
inside a notebook. It stays separate from the training pipeline so you can iterate on sampling
parameters without touching `main.py`.

## Directory layout

- `configs/`: Light-weight Hydra configs/overrides for sampling-only runs.
- `loaders/`: Utilities for instantiating the correct EGNN model/method combination and loading
  checkpoints (`SamplingContext`) plus the single-graph sequence setup helper.
- `eval/`: Eval-time helpers such as `ExpansionStepEvaluator` that mirror the training loss but
  return metadata for every reduced graph.
- `interactive_methods/`: Instrumented subclasses of the expansion methods that capture model
  outputs, noise samples, sibling state, etc., at every expansion step.
- `runners/`: The CLI entrypoint (`run_interactive_sampling.py`) plus helpers for deriving target
  graph sizes and the stepwise evaluation utilities under `view_each_step.py`.
- `artifacts/`: Per-run folders containing final graphs, per-step trace pickles, summaries, and
  resolved configs. This is what the notebook consumes.
- `notebooks/`: `interactive_sampling.ipynb` lets you browse saved runs with widgets and plots.

## Running an interactive sampling job

1. Pick a Hydra config (reusing training configs works). Example:
   ```bash
   CONFIG=config/small_trees_run.yaml
   CHECKPOINT=outputs/2024-06-01_egnn/checkpoints/step_2000.pt
   ```
2. Decide how to specify target graph sizes:
   - `--target-sizes 64 96 128` for explicit capacities (quick sanity runs).
   - `--dataset-split val --max-graphs 32` to mimic validation/test splits. The runner
     loads graphs using the dataset settings in the config and uses their node counts as
     sampling targets.
3. Run the CLI:
   ```bash
   python sampling_experiments/runners/run_interactive_sampling.py \
       --config "$CONFIG" \
       --checkpoint "$CHECKPOINT" \
       --dataset-split val \
       --max-graphs 16 \
       --batch-size 4 \
       --device cuda:0 \
       --ema-beta 1 \
       --map-threshold 0.35
   ```
   Key flags:
   - `--method` to force `expansion` vs `expansion_augmented` (defaults to config).
   - `--batch-size` (optional) controls how many graphs are sampled at once; falls back to
     validation batch size if omitted.
   - `--target-sizes` and `--dataset-split` are mutually exclusive; one must be provided.
   - Results land under `sampling_experiments/artifacts/<method>_<timestamp>/`.

## Artifact contents
Each run directory contains:
- `run_summary.json`: metadata (checkpoint path, EMA beta, method, target source, basic graph stats).
- `config_resolved.yaml`: resolved Hydra config for reproducibility.
- `graph_<idx>_final.gpickle`: the final NetworkX graph for each sampled item.
- `graph_<idx>_trace.pkl`: list of `GraphStepTrace` / `AugmentedGraphStepTrace` objects capturing
  every expansion step with node positions, active leaves, logits/probabilities, noise samples, etc.

## Visualizing in the notebook
1. Launch Jupyter (e.g. `jupyter lab` at repo root).
2. Open `sampling_experiments/notebooks/interactive_sampling.ipynb`.
3. Run all cells; the run dropdown auto-populates from `sampling_experiments/artifacts`.
4. Use the widgets to select a run, graph, and step. The notebook plots the graph, highlights
   current leaves, and displays the per-leaf probabilities/logits for that step.
5. If you create new runs, rerun the widget cell (or restart kernel) to refresh the dropdown.

## Extending / debugging
- Instrumentation lives in `interactive_methods/`. To capture additional stats (e.g., sibling loss,
  per-leaf distance), add them to the debug payload returned by the custom `expand` overrides.
- If you need custom evaluation configs, drop them under `sampling_experiments/configs/` and point
  `--config` to that file.
- Because everything reuses Hydra configs + checkpoints, you can experiment with different map
  thresholds, noise sigmas, or deterministic flags without retraining.

## Step-by-step evaluation for a single graph
When you want to examine the model on **ground-truth reduction sequences** instead of sampling
entire graphs from scratch:

1. Call `prepare_sequence_setup` (in `loaders/sequence_setup.py`) with a Hydra config, checkpoint,
   and graph file (`.gpickle`, `.pkl`, or `.swc`). It returns the instantiated model +
   `ExpansionStepEvaluator` plus a `ReductionSequenceBundle` containing every reduced graph.
2. Feed that bundle into `evaluate_sequence_records` (in `runners/view_each_step.py`). It batches
   the reduced graphs, reuses the training-time masking, and yields one `StepEvalRecord` per step
   with GT vs predicted leaf positions, masked inputs, expansion logits/probabilities, sibling info,
   and per-step losses.
3. Optionally call `save_stepwise_results` to persist everything under
   `sampling_experiments/artifacts/view_each_step/<graph>_<timestamp>/`, ready for a notebook that
   renders GT vs predictions with a step slider and probability tables.

Example CLI-free run:
```bash
python - <<'PY'
from pathlib import Path
from sampling_experiments.runners import run_view_each_step

run_view_each_step(
    config_path=Path("config/small_trees_run.yaml"),
    checkpoint_path=Path("outputs/example/checkpoints/step_2000.pt"),
    graph_path=Path("debug_graphs/sample_tree.gpickle"),
    device="cpu",
    batch_size=8,
)
PY
```

Each artifact folder stores:
- `reduction_sequence.pkl` (pickled `ReductionSequenceBundle` with the NetworkX graph + PyG steps),
- `step_records.pkl` (list of `StepEvalRecord` entries),
- `step_summary.json` (quick glance at node/leaf counts and per-step losses),
- `metadata.json` and `config_resolved.yaml` for provenance.

You can import the same helpers from a notebook to run everything in-memory and build 3D plots,
GT/pred toggles, and GT vs predicted expansion probability tables without relying on the CLI.
