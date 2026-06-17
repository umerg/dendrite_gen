# Visualization

Lightweight plotting utilities for inspecting GT/predicted dendrite tree runs.
The current everyday workflows are:

- `run_all_plots.py` for the full validation figure set
- `run_cylinder_trees.py` for interactive 3D tree renderings
- `run_unconditional.py` for population-level unconditioned diagnostics

Run the commands in this README from the parent project directory that contains
the `dendrite_gen/` package, not from inside `dendrite_gen/` itself. In this
kind of checkout that means a folder like `/path/to/project-root/`, where
`/path/to/project-root/dendrite_gen/` exists.

## Inputs

The visualization runners assume:

- GT trees come from a directory of SWC files
- predicted trees come from a pickle containing `pred_graphs`
- optional EMA selection can be provided with `--ema-key`
- GT and predicted graphs are paired by index order

## Run All Plots

Generate the current validation figure set:

```bash
python -m dendrite_gen.visualization.run_all_plots \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/predictions.pkl \
  --ema-key ema_1 \
  --max-pairs 6 \
  --include-unconditional \
  --out-dir dendrite_gen/outputs/visualization
```

This writes qualitative 2D views, tree-level metric plots, within-tree
distribution plots, and TMD figures. Dataset-level statistics use all paired
trees by default; `--max-pairs` only limits the paired examples selected for
sample-style views.

All visualization runners also write a short `README.md` into the output root
describing how to interpret the generated plot families.

Useful switches:

- `--skip-tmd` for a faster pass without persistence diagrams
- `--projections xy` or `--projections xy xz` to limit qualitative views
- `--include-cylinders` to also render selected 3D cylinder trees
- `--include-unconditional` to also render unconditioned population-level diagnostics

Optional cylinder plots inside `run_all_plots.py` use Plotly by default. Pass
`--cylinder-backend matplotlib` only when you want static PNG cylinders.

## Unconditioned Diagnostics

Generate a population-level PCA for unconditioned models:

```bash
python -m dendrite_gen.visualization.run_unconditional \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/predictions.pkl \
  --ema-key ema_1 \
  --out-dir dendrite_gen/outputs/visualization
```

This writes `unconditional/tree_feature_pca.png` plus one feature-colored PCA
plot per vector component in `unconditional/tree_feature_pca_by_feature/`. GT
and predicted samples are compared as two distributions, not as index-matched
pairs. Each tree is embedded using mean and standard deviation summaries of
branch length, bifurcation angle, path distance, radial distance, and branch
order, plus height, XY span, and bounding-box diagonal.

## 3D Plots

Generate interactive Plotly cylinder trees:

```bash
python -m dendrite_gen.visualization.run_cylinder_trees \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/predictions.pkl \
  --ema-key ema_1 \
  --max-pairs 3 \
  --synthesize-radii \
  --plotly-texture bark \
  --curve-branches \
  --plotly-leaves \
  --angle 20,120 \
  --out-dir dendrite_gen/outputs/visualization
```

Cylinder plots default to Plotly `.html` output. With Plotly, `--plot-mode pair`
writes separate GT and prediction HTML files for each pair instead of a single
side-by-side page, using the same content-based names as `--plot-mode gt` and
`--plot-mode pred`: `{stem}_gt_cylinder_{angle}.html` and
`{stem}_pred{idx}_cylinder_{angle}.html`. The Matplotlib backend still writes a
true side-by-side pair image as `{stem}_cylinder_pair_{angle}.png`.

Cylinder rendering reads `node["radius"]` when available. Ground-truth SWC
files loaded through the repo loader preserve the SWC radius column. Generated
graphs often lack radii, so `--synthesize-radii` is useful for visual thickness.
These synthesized radii are visualization radii, not measured geometry.

Plotly branch meshes use a procedural bark-like color texture by default. Use
`--plotly-texture none` for a flat branch color, or tune contrast with
`--plotly-texture-strength`.

`--curve-branches` inserts smooth, endpoint-preserving random centerline points
before radius synthesis and rendering.

`--plotly-leaves` adds translucent low-poly leaf clumps around distal branch
neighborhoods. Tune them with `--plotly-leaf-count`,
`--plotly-leaf-opacity`, `--plotly-leaf-scale`, and `--plotly-leaf-seed`.

## Help

```bash
python -m dendrite_gen.visualization.run_all_plots --help
python -m dendrite_gen.visualization.run_cylinder_trees --help
python -m dendrite_gen.visualization.run_unconditional --help
```
