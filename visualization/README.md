# Visualization

Lightweight plotting and visualization utilities for the `dendrite_gen`
project. The goal is to make it easy to generate several useful views of a
GT/predicted tree run from one command, while still keeping the reusable
plotting functions small and script-friendly.

## Main command

Generate the current visualization set:

```bash
python3 -m dendrite_gen.visualization.run_all_plots \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/validation/step_30000.pkl \
  --ema-key ema_1 \
  --max-pairs 6 \
  --stats-max-pairs 128 \
  --out-dir dendrite_gen/outputs/visualization
```

`run_all_plots.py` currently writes:

- qualitative 2D views for `xy`, `xz`, and `yz`
- tree-level metric histograms
- within-tree distribution histograms
- TMD persistence-diagram grids

Use `--skip-tmd` when you want a faster pass without persistence diagrams.
Use `--projections xy` or `--projections xy xz` to limit the qualitative views.

## Structure

- `run_all_plots.py`
  - main entrypoint for the current visualization set
- `run_qualitative.py`
  - individual 2D tree views and galleries
- `run_tree_stats.py`
  - tree-level statistics runner
- `run_distribution_stats.py`
  - within-tree distribution runner
- `run_tmd_figures.py`
  - TMD persistence-diagram runner
- `common.py`
  - shared GT/pred loading, pairing, selection, and output-root helpers
- `qualitative/`
  - 2D tree plotting helpers
- `stats/`
  - graph-based tree-level and within-tree statistics
- `tmd/`
  - TMD plotting helpers
- `utils/`
  - shared IO helpers and plotting style constants

## Inputs

The current plotting path assumes:

- GT trees come from a directory of SWC files
- predicted trees come from a validation pickle containing `pred_graphs`
- optional EMA selection can be provided with `--ema-key`
- GT and predicted graphs are paired by index order

## Individual commands

Generate all qualitative plots:

```bash
python3 -m dendrite_gen.visualization.run_qualitative \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/validation/step_30000.pkl \
  --ema-key ema_1 \
  --all \
  --max-pairs 6 \
  --out-dir dendrite_gen/outputs/visualization
```

Generate one side-by-side 2D plot:

```bash
python3 -m dendrite_gen.visualization.run_qualitative \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/validation/step_30000.pkl \
  --figure simple2d \
  --projection xy \
  --max-pairs 1 \
  --out-dir dendrite_gen/outputs/visualization
```

Generate a tree-level histogram grid:

```bash
python3 -m dendrite_gen.visualization.run_tree_stats \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/validation/step_30000.pkl \
  --ema-key ema_1 \
  --max-pairs 64 \
  --ncols 3 \
  --out-dir dendrite_gen/outputs/visualization
```

Generate pooled within-tree distribution histograms:

```bash
python3 -m dendrite_gen.visualization.run_distribution_stats \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/validation/step_30000.pkl \
  --ema-key ema_1 \
  --distribution-mode pooled \
  --max-pairs 64 \
  --ncols 3 \
  --out-dir dendrite_gen/outputs/visualization
```

Generate TMD persistence-diagram figures:

```bash
python3 -m dendrite_gen.visualization.run_tmd_figures \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/validation/step_30000.pkl \
  --ema-key ema_1 \
  --filtrations path height rho \
  --point-alpha 0.5 \
  --max-pairs 6 \
  --out-dir dendrite_gen/outputs/visualization
```

Show the main CLI:

```bash
python3 -m dendrite_gen.visualization.run_all_plots --help
```
