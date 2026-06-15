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
- `run_tmd_figures.py`
  - TMD persistence-diagram runner
- `run_curated_offset_gallery.py`
  - TOML-driven runner for hand-tuned offset galleries
- `qualitative/`
  - currently contains 2D qualitative plots
- `specs/shared/`
  - committed example and template TOMLs
- `specs/local/`
  - local machine-specific TOMLs for custom figure specs
- `stats/`
  - graph-based tree-level and within-tree statistics for loaded NetworkX trees
- `utils/`
  - shared IO helpers and plotting style constants
- `stats.py`
  - thin compatibility re-export for the main stats helpers

## Currently Implemented

### Runners

- `run_qualitative.py`
  - `simple2d`, `offset2d`, `overlay2d`, `gallery2d`, `offset_gallery2d`
- `run_tree_stats.py`
  - tree-level histogram grid
- `run_distribution_stats.py`
  - pooled, tree-averaged, or single-tree distribution histograms
- `run_tmd_figures.py`
  - GT/pred persistence-diagram overlays for selected filtrations
- `run_all_plots.py`
  - calls the current family runners with simple defaults
- `run_curated_offset_gallery.py`
  - builds a custom offset gallery from a TOML spec

All runners write into the same output root, with one subfolder per runner.
When qualitative runners are used with `--all`, they emit separate outputs for
`xy`, `xz`, and `yz`.
In `run_all_plots.py`, `--max-pairs` controls the qualitative example subset,
while tree/distribution statistics use all paired trees unless
`--stats-max-pairs` is set.
The exception is per-tree distribution figures: those follow `--max-pairs`,
while pooled distribution summaries follow `--stats-max-pairs` (or all pairs
by default).

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
  --stats-max-pairs 128 \
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

Generate the offset side-by-side qualitative figure:

```bash
python -m dendrite_gen.paper_figures.run_qualitative \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/validation/step_30000.pkl \
  --ema-key ema_1 \
  --figure offset2d \
  --projection xy \
  --max-pairs 3 \
  --out-dir dendrite_gen/outputs/paper_figures
```

Generate an offset gallery with custom offsets:

```bash
python -m dendrite_gen.paper_figures.run_qualitative \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/validation/step_30000.pkl \
  --ema-key ema_1 \
  --figure offset_gallery2d \
  --projection xy \
  --x-gap-scale 0.08 \
  --y-offset-scale 0.15 \
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

Generate TMD persistence-diagram figures:

```bash
python -m dendrite_gen.paper_figures.run_tmd_figures \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/validation/step_30000.pkl \
  --ema-key ema_1 \
  --filtrations path height rho \
  --point-alpha 0.5 \
  --max-pairs 6 \
  --out-dir dendrite_gen/outputs/paper_figures
```

Generate a curated offset gallery from a TOML spec:

```bash
python -m dendrite_gen.paper_figures.run_curated_offset_gallery \
  --spec /Users/speltonen/Documents/projects/generating-trees/dendrite_gen/paper_figures/specs/local/my_offset_gallery.toml
```

The template spec lives at:

```text
/Users/speltonen/Documents/projects/generating-trees/dendrite_gen/paper_figures/specs/shared/offset_gallery_template.toml
```

Show the CLI:

```bash
python -m dendrite_gen.paper_figures.run_all_plots --help
```
