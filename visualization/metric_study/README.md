# Single-pair tree metric study

The first milestone compares exactly two ground-truth SWC trees. It does not
load prediction pickles, model outputs, labels, or dataset-specific pairings.
Metric implementations live in the top-level `metrics/` package; this folder
owns study commands, plans, and eventually analysis/visualization code.

## Run one pair

From the directory containing `dendrite_gen/`:

```bash
source /Users/speltonen/miniconda3/bin/activate trees
python -m dendrite_gen.visualization.metric_study.run_pair \
  --tree-a /absolute/path/to/tree_a.swc \
  --tree-b /absolute/path/to/tree_b.swc \
  --output-json /absolute/path/to/pair_metrics.json
```

From the `dendrite_gen/` repository root, the equivalent module name is:

```bash
python -m visualization.metric_study.run_pair \
  --tree-a /absolute/path/to/tree_a.swc \
  --tree-b /absolute/path/to/tree_b.swc
```

The default command evaluates:

- arc-length-sampled symmetric Chamfer distance
- TMD persistence-diagram Wasserstein distances for path, height, and radial
  (`rho`) filtrations
- 1-Wasserstein distances between seven explicit morphology distributions

Persistence uses the conventional Chebyshev/L-infinity ground norm by default;
`--persistence-ground-norm euclidean` selects the explicitly reported L2-ground
adaptation.

Dense Fused Gromov-Wasserstein is available but intentionally opt-in:

```bash
python -m visualization.metric_study.run_pair \
  --tree-a /absolute/path/to/tree_a.swc \
  --tree-b /absolute/path/to/tree_b.swc \
  --metrics chamfer persistence distributions fgw
```

FGW uses root-centered `xyz` features and the relative SO(2) quotient. Its
default node masses approximate cable length by assigning half of each edge to
each endpoint; raw uniform-node mass remains an explicit ablation. FGW still
uses one shared pairwise scale for structural costs and one for node features;
`--fgw-normalization none` retains physical units. It forms dense node-pair
matrices and repeats the solver over the angle search, so the CLI refuses trees
above 1000 nodes unless the limit is deliberately changed.

Use `--help` to select metric families and expose the preprocessing choices.
The JSON records the settings that affect interpretation.

If a morphology feature exists in only one tree, its distance is JSON `null`
and the adjacent diagnostic records which input was empty. Two empty feature
distributions have value zero but status `both_empty`.

## SO(2) contract

The preferred axis is `z`. Chamfer is minimized over relative rotations about
that axis by default. FGW's default `xyz` features use the same relative SO(2)
minimum. Its optional `--fgw-feature-mode axis` variant uses `(z, rho)` and is
already invariant, but deliberately discards azimuthal information. The
selected TMD filtrations and morphology distributions are intrinsically
SO(2)-invariant. `--no-so2-quotient` is retained only as an explicit diagnostic
variant.

The numerical quotient group never tilts the preferred axis and never includes
reflections or axis flips. This does not mean every summary is sensitive to
those transformations: the current TMD filtrations and morphology
distributions also discard relative vertical-plane reflections, and several
scalar distributions are invariant to still larger rigid groups. The JSON
marks these deliberately over-invariant baselines separately from the actual
SO(2) search group.

## External Elastic SRVFT code

For the initial audit, a local clone can go under
`metrics/external/<repository-name>/`; checkout contents there are ignored by
Git. Keep project-facing calls behind `metrics/adapters/elastic_srvft.py`.
Before enabling the adapter, pin a revision, check its license and API, and
replace or constrain any internal full-SO(3) alignment so the comparison uses
the required SO(2)-only quotient.

See `plan.md` for the broader class-aware comparison study.
