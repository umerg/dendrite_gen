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

## Paper Figure Drafts

The paper-specific scripts are intentionally small, fixed-layout entry points
for fast iteration. They reuse the graph loaders and canonical metric
implementations without depending on the older all-purpose plotting runners.
The shared semantic palette is defined in `visualization/paper/colors.py`:
method base/light/dark colors, the qualitative tree-depth gradient, and neutral
supporting colors all live there. Figure 1's TOML accepts those semantic names
(for example `semlaflow-dark`) as well as literal Matplotlib colors.

```bash
python -m dendrite_gen.visualization.paper.fig1_unconditional
python -m dendrite_gen.visualization.paper.fig2_unconditional
python -m dendrite_gen.visualization.paper.fig3_conditional
```

They overwrite PNG and PDF drafts in
`dendrite_gen/outputs/paper_figures/`. These previews are not copied into
`iclr27-writeup/Figures/` automatically; copy the selected final files there
explicitly when updating the LaTeX writeup.

- `fig1_unconditional_gallery.{png,pdf}` is a free-layout, configurable
  three-column gallery. `fig1_unconditional.py` is its primary entry point;
  edit the adjacent
  `fig1_unconditional_gallery.toml` to select, position, and rotate samples;
  pass `--show-guides` while arranging them.
- `fig2_unconditional_distributions.{png,pdf}` contains six population-level
  density comparisons. Node/branch observations are weighted so every tree
  contributes equal total mass.
- `fig3_conditional_placeholder.{png,pdf}` contains two target/generated
  pairs, their persistence diagrams, and two within-pair distribution
  comparisons. For an expanded appendix version, increase `--num-examples`;
  the script then uses a separate appendix output name automatically:

  ```bash
  python -m dendrite_gen.visualization.paper.fig3_conditional \
    --num-examples 3
  ```

  Use `--output-stem` only when you want a different filename.

The Figure 1 TOML currently uses the 1,167-tree neuronal test populations:
`neurons_conditional/test`, the canonically sanitised SemlaFlow `epoch209`
pickle, and the extracted `parity_neurons_uncond` generations. Its selected
indices remain deliberately easy to change in the TOML. Figure 3 still uses
index-aligned `neurons_conditional/val` and `step_5500.pkl` data and asserts the
selected filenames before plotting. That checkpoint is cell-type-conditioned
with morphology/TMD conditioning disabled, so Figure 3 is currently a layout
placeholder rather than evidence of morphology-conditional fidelity.

The matching three-way Figure 2 comparison can be regenerated with:

```bash
python -m dendrite_gen.visualization.paper.fig2_unconditional \
  --reference-dir data/neurons_conditional/test \
  --ours-dir data/trees_test_set_generations/parity_neurons_uncond \
  --semla-pkl data/epoch209/neuron_samples_sanitised.pkl \
  --output-stem fig2_unconditional_distributions_epoch209
```

Use `--max-trees` with Figure 2 for a faster style preview. Input paths, EMA
keys, and output directories can be overridden from the command line; inspect
the complete options with `--help`.

## Paper Table Draft

Generate the provisional unconditional-results LaTeX table with:

```bash
python -m dendrite_gen.visualization.paper.table1_unconditional
```

The authoritative output is the copy-pasteable `booktabs` fragment at
`iclr27-writeup/Tables/table1_unconditional.tex`. A JSON sidecar containing the
unrounded metrics, paths, counts, and protocol is written to
`dendrite_gen/outputs/paper_tables/table1_unconditional_metrics.json`.

The defaults intentionally label the output provisional. They compare the only
locally available common legacy-validation artifacts, use 924 valid trees per
distributional row, and leave sampling time blank because no controlled
cross-method timing exists. Replace the input paths with matched held-out-test
runs before using the values as paper claims.

## Convert `.smol` Samples

If you have a raw SemlaFlow `.smol` sample dump, convert it before running the
paper figures. For `epoch209`, the exact command and output consumed by the
Figure 1 TOML and the Figure 2 command above are:

```bash
python -m dendrite_gen.visualization.convert_smol_to_pred_pkl \
  --smol-path data/epoch209/neuron_samples.smol \
  --out-pkl data/epoch209/neuron_samples_sanitised.pkl
```

The general form is:

```bash
python -m dendrite_gen.visualization.convert_smol_to_pred_pkl \
  --smol-path /path/to/samples.smol \
  --out-pkl /path/to/samples_converted.pkl
```

By default the converter writes a validation-style pickle under `ema_1` and
uses SemlaFlow's canonical `validation/sanitise.py:sanitise_graph` policy. It
first builds the raw graph and records raw connectivity, tree validity, and the
full graph-health summary. It then keeps the largest connected component,
relabels it, constructs a Euclidean minimum spanning tree, chooses the root,
binarises non-root multifurcations, contracts non-root degree-2 nodes, and
relabels the resulting critical tree. The figure scripts consume these
sanitised `pred_graphs`; the retained raw metadata is for reporting structural
failures before repair.

The former visualization-only path (largest component, degree root, and BFS
spanning tree, without binarisation or degree-2 contraction) is available only
for reproducing stale drafts:

```bash
python -m dendrite_gen.visualization.convert_smol_to_pred_pkl \
  --smol-path /path/to/samples.smol \
  --out-pkl /path/to/samples_legacy.pkl \
  --preprocessing legacy
```

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
python -m dendrite_gen.visualization.paper.fig1_unconditional --help
python -m dendrite_gen.visualization.paper.fig2_unconditional --help
python -m dendrite_gen.visualization.paper.fig3_conditional --help
python -m dendrite_gen.visualization.paper.table1_unconditional --help
```
