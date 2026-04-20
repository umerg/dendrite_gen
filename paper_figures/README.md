# Paper figures

Small, paper-facing plotting utilities for the `dendrite_gen` project.

This folder is meant to stay separate from the older code in
`dendrite_gen/validation/` and provide a cleaner place for publication-quality
figures and supporting tables.

## Current Structure

- `common.py`
  - shared GT/pred loading, pairing, selection, and output-root helpers
- `run_all_plots.py`
  - simple orchestrator for the current paper plot set
- `run_qualitative.py`
  - qualitative 2D figure runner
- `run_tree_stats.py`
  - tree-level statistics runner
- `run_distribution_stats.py`
  - within-tree distribution runner
- `qualitative/`
  - currently contains 2D qualitative plots
- `stats/`
  - graph-based tree-level and within-tree statistics for loaded NetworkX trees
- `utils/`
  - shared IO helpers and plotting style constants
- `stats.py`
  - thin compatibility re-export for the main stats helpers

## Currently Implemented

### Runners

- `run_qualitative.py`
  - `simple2d`, `overlay2d`, `gallery2d`
- `run_tree_stats.py`
  - tree-level histogram grid
- `run_distribution_stats.py`
  - pooled, tree-averaged, or single-tree distribution histograms
- `run_all_plots.py`
  - calls the current family runners with simple defaults

All runners write into the same output root, with one subfolder per runner.

### Inputs

The current plotting path assumes:

- GT trees come from a directory of SWC files
- predicted trees come from a validation pickle containing `pred_graphs`
- optional EMA selection can be provided with `--ema-key`
- GT and predicted graphs are currently paired by index order

### Statistics

The current stats layer is graph-based:

- GT SWCs are converted to NetworkX once during loading
- predicted trees are read from the validation pickle as NetworkX graphs
- both tree-level scalar stats and within-tree distributions are computed from
  the same graph representation

## Example Commands

Generate the current full paper plot set:

```bash
python -m dendrite_gen.paper_figures.run_all_plots \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/validation/step_30000.pkl \
  --ema-key ema_1 \
  --projection xy \
  --max-pairs 6 \
  --out-dir dendrite_gen/outputs/paper_figures
```

Generate one side-by-side 2D plot:

```bash
python -m dendrite_gen.paper_figures.run_qualitative \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/validation/step_30000.pkl \
  --figure simple2d \
  --projection xy \
  --max-pairs 1 \
  --out-dir dendrite_gen/outputs/paper_figures
```

Generate all qualitative plots:

```bash
python -m dendrite_gen.paper_figures.run_qualitative \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/validation/step_30000.pkl \
  --ema-key ema_1 \
  --all \
  --projection xy \
  --max-pairs 6 \
  --out-dir dendrite_gen/outputs/paper_figures
```

Generate a tree-level histogram grid:

```bash
python -m dendrite_gen.paper_figures.run_tree_stats \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/validation/step_30000.pkl \
  --ema-key ema_1 \
  --max-pairs 64 \
  --ncols 3 \
  --out-dir dendrite_gen/outputs/paper_figures
```

Generate pooled within-tree distribution histograms:

```bash
python -m dendrite_gen.paper_figures.run_distribution_stats \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/validation/step_30000.pkl \
  --ema-key ema_1 \
  --distribution-mode pooled \
  --max-pairs 64 \
  --ncols 3 \
  --out-dir dendrite_gen/outputs/paper_figures
```

Show the CLI:

```bash
python -m dendrite_gen.paper_figures.run_all_plots --help
```
