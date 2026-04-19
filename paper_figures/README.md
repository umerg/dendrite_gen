# Paper figures

Small, paper-facing plotting utilities for the `dendrite_gen` project.

This folder is meant to stay separate from the older code in
`dendrite_gen/validation/` and provide a cleaner place for publication-quality
figures and supporting tables.

## Current Structure

- `run_plots.py`
  - top-level plotting entrypoint
- `qualitative/`
  - currently contains 2D qualitative plots
- `stats/`
  - graph-based tree-level and within-tree statistics for loaded NetworkX trees
- `utils/`
  - shared IO helpers and plotting style constants
- `stats.py`
  - thin compatibility re-export for the main stats helpers

## Currently Implemented

### Plots

`run_plots.py` currently supports:

- `simple2d`
  - side-by-side GT / predicted 2D plot
- `overlay2d`
  - single-axis GT / predicted 2D overlay
- `gallery2d`
  - small qualitative gallery of 2D overlays
- `treelevel_hist`
  - grid of overlaid GT / predicted histograms for tree-level scalar metrics
- `distribution_hist`
  - grid of GT / predicted histograms for within-tree distribution metrics
- `--all`
  - runs all currently implemented plot targets

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

Generate all currently implemented plots:

```bash
python -m dendrite_gen.paper_figures.run_plots \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/validation/step_30000.pkl \
  --ema-key ema_1 \
  --all \
  --projection xy \
  --max-pairs 6 \
  --out-dir dendrite_gen/outputs/paper_figures/all_xy
```

Generate one side-by-side 2D plot:

```bash
python -m dendrite_gen.paper_figures.run_plots \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/validation/step_30000.pkl \
  --figure simple2d \
  --projection xy \
  --max-pairs 1 \
  --out-dir dendrite_gen/outputs/paper_figures/debug
```

Generate one 2D overlay:

```bash
python -m dendrite_gen.paper_figures.run_plots \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/validation/step_30000.pkl \
  --ema-key ema_1 \
  --figure overlay2d \
  --projection xy \
  --max-pairs 1 \
  --out-dir dendrite_gen/outputs/paper_figures/overlay2d_xy
```

Generate a small 2D gallery:

```bash
python -m dendrite_gen.paper_figures.run_plots \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/validation/step_30000.pkl \
  --ema-key ema_1 \
  --figure gallery2d \
  --projection xy \
  --max-pairs 6 \
  --out-dir dendrite_gen/outputs/paper_figures/gallery2d_xy
```

Generate a tree-level histogram grid:

```bash
python -m dendrite_gen.paper_figures.run_plots \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/validation/step_30000.pkl \
  --ema-key ema_1 \
  --figure treelevel_hist \
  --max-pairs 64 \
  --ncols 3 \
  --out-dir dendrite_gen/outputs/paper_figures/treelevel_hist
```

Generate pooled within-tree distribution histograms:

```bash
python -m dendrite_gen.paper_figures.run_plots \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/validation/step_30000.pkl \
  --ema-key ema_1 \
  --figure distribution_hist \
  --distribution-mode pooled \
  --max-pairs 64 \
  --ncols 3 \
  --out-dir dendrite_gen/outputs/paper_figures/distribution_hist
```

Show the CLI:

```bash
python -m dendrite_gen.paper_figures.run_plots --help
```
