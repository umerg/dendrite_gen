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
- optional 3D cylinder-model tree renderings
- tree-level metric histograms and paired GT-vs-pred scatter plots
- within-tree distribution histograms
- TMD persistence-diagram grids, mean persistence-image comparisons, and a joint
  persistence-image embedding scatter

Use `--skip-tmd` when you want a faster pass without persistence diagrams.
Use `--projections xy` or `--projections xy xz` to limit the qualitative views.
Use `--include-cylinders` to add the first-pass cylinder tree model renderings.

## Structure

- `run_all_plots.py`
  - main entrypoint for the current visualization set
- `run_qualitative.py`
  - individual 2D tree views and galleries
- `run_cylinder_trees.py`
  - 3D cylinder-model tree renderings using stored, default, or synthesized visual radii
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

Generate a 3D cylinder-model tree rendering:

```bash
python3 -m dendrite_gen.visualization.run_cylinder_trees \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/validation/step_30000.pkl \
  --ema-key ema_1 \
  --max-pairs 3 \
  --plot-mode pair \
  --out-dir dendrite_gen/outputs/visualization
```

Cylinder rendering reads `node["radius"]` when available. Ground-truth SWC
files loaded through the repo loader now preserve the SWC radius column. Graphs
without radii, such as most generated validation-pickle graphs, use radius
`1.0` for every node. Use `--radius-scale` if you want a quick visual thickness
adjustment.

For bare skeletons you can synthesize rTwig-inspired visual radii before
rendering:

```bash
python3 -m dendrite_gen.visualization.run_cylinder_trees \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/validation/step_30000.pkl \
  --ema-key ema_1 \
  --max-pairs 3 \
  --plot-mode pair \
  --synthesize-radii \
  --out-dir dendrite_gen/outputs/visualization
```

The synthesis roots the graph, estimates downstream branch demand, smooths each
root-to-tip path monotonically, and merges path predictions into a per-node
render radius. These are visualization radii, not measured geometry.

The default renderer is Matplotlib. To try the PyVista/VTK renderer, add
`--backend pyvista`; those images are written to `cylinders_pyvista/` so they
do not overwrite the Matplotlib outputs. PyVista rendering needs an OpenGL or
OSMesa-capable plotting context; in a headless process without one, the runner
will fail before creating a VTK window.

For an interactive browser view that can be rotated, use `--backend plotly`.
This writes self-contained `.html` files to `cylinders_plotly/`. Plotly
renderings add small joint spheres at endpoints and branchpoints by default;
use `--no-joints` to render only the cylinder frustums. Plotly branch meshes
also use a procedural bark-like color texture by default; use
`--plotly-texture none` for a flat branch color, or tune it with
`--plotly-texture-strength`, which controls the contrast of the bark palette.
Strength values less than or equal to zero disable the texture path and render
flat brown branches even when `--plotly-texture bark` is selected.
Bark texture cells are subdivided in proportion to edge length using the
graph's median edge length as the default cell size; lower
`--plotly-texture-target-length-scale` values make shorter cells, which is
useful when the main trunk looks too coarsely segmented.

Curved branch centerlines are available as an explicit visualization-only
option:

```bash
python3 -m dendrite_gen.visualization.run_cylinder_trees \
  --gt-dir /path/to/gt_swc \
  --pred-pkl /path/to/validation/step_30000.pkl \
  --ema-key ema_1 \
  --max-pairs 3 \
  --plot-mode gt \
  --synthesize-radii \
  --curve-branches \
  --backend plotly \
  --out-dir dendrite_gen/outputs/visualization
```

`--curve-branches` inserts smooth, endpoint-preserving random centerline points
before radius synthesis and rendering. It is off by default, and curved outputs
are written to separate folders such as `cylinders_plotly_curved/`.

For coarse translucent foliage in Plotly, add `--plotly-leaves`. This grows
connected neighborhoods across several distal branch segments and draws larger
overlapping low-poly leaf polyhedra around each neighborhood. Terminal tips are
assigned into coverage groups, so `--plotly-leaf-count` controls the requested
number of blobs while still covering every tip when possible. Lower counts make
larger shared clumps; higher counts split the canopy into finer clumps. The
feature is off by default; when enabled, Plotly outputs are written to a
separate folder such as `cylinders_plotly_leaves/` or
`cylinders_plotly_curved_leaves/`. Tune the canopy with
`--plotly-leaf-count`, `--plotly-leaf-opacity`, `--plotly-leaf-scale`, and
`--plotly-leaf-seed`. Leaf-enabled Plotly HTML files include an opacity slider
that adjusts all leaf clumps in the browser.

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

Generate tree-level histogram and paired scatter grids:

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
  --embedding-bins 16 \
  --embedding-sigma 0.05 \
  --embedding-color-attributes height max_path_dist \
  --diagram-distance-attributes height max_path_dist \
  --point-alpha 0.5 \
  --max-pairs 6 \
  --out-dir dendrite_gen/outputs/visualization
```

The TMD runner also writes one joint GT/predicted embedding per selected
filtration, plus mean persistence-image comparison panels such as
`tmd_mean_pi_path.png`. Embedding outputs include `tmd_embedding_path.png` and
`tmd_embedding_path_points.csv`. Colored variants are named like
`tmd_embedding_path_color_height.png`. It also writes one GT/pred pair scatter
per filtration and selected GT attribute, with x as the GT-vs-pred
persistence-diagram Wasserstein distance and y as the GT attribute; outputs are named like
`tmd_diagram_distance_path_by_gt_height.png`. Unlike the per-tree persistence-diagram
grids, these dataset-level plots use all paired trees by default. Use
`--embedding-max-pairs` only when you want a smaller debugging run. Use
`--embedding-combine-filtrations` if you also want the older concatenated
all-filtrations embedding.
The reducer defaults to `auto`, which uses UMAP when `umap-learn` is available
and PCA otherwise. The embedding plot uses a neutral scatter in the main panel
plus GT/predicted marginal density curves below the x axis and left of the y
axis. The standalone TMD runner defaults to the `height` color attribute; pass
`--embedding-color-attributes height max_path_dist` to choose a different list.
The standalone distance scatter uses the same attribute list by default; pass
`--diagram-distance-attributes height max_path_dist` only when you want a
separate y-axis attribute list.
`run_all_plots.py` defaults to all available tree-level scalar color attributes;
use the prefixed `--tmd-embedding-color-attributes` and
`--tmd-diagram-distance-attributes` options there when you want smaller or
separate sets. The reducer is computed once per embedding filtration and then
reused for every color-attribute variant. Use `--embedding-connect-pairs` only
for small subsets where pair lines remain readable.

Show the main CLI:

```bash
python3 -m dendrite_gen.visualization.run_all_plots --help
```
