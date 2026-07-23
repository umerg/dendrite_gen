# Ground-truth tree metric study

The first milestone compares exactly two ground-truth SWC trees. It does not
load prediction pickles, model outputs, labels, or dataset-specific pairings.
Metric implementations live in the top-level `metrics/` package; this folder
owns study commands, dataset selection, and analysis plots.

## Run a balanced class comparison

The class-comparison runner reads `cell_class` and `cell_type` from each SWC
header, selects the same number of trees from every class, prepares each tree
once, and computes a symmetric distance matrix. The default first metric is
path-filtration TMD Wasserstein. It is inexpensive after its persistence
diagrams are cached and does not require an angular search.

The labelled `neurons_conditional` dataset currently contains seven classes.
Its test split has ten examples in the smallest class, so the default run uses
ten trees per class (70 trees and 2,415 distinct off-diagonal pairs):

```bash
source /Users/speltonen/miniconda3/bin/activate trees
python -m visualization.metric_study.run_class_comparison \
  --dataset-root /absolute/path/to/neurons_conditional \
  --splits test \
  --metric tmd_path_wasserstein \
  --per-class 10
```

The ignored output directory contains the selected-tree table, class counts,
the reusable compressed distance matrix, run metadata, a class-colored MDS
embedding, an individual-tree distance heatmap grouped by class, and a
class-to-class median heatmap.

Metrics are registered as configured scalar variants in `metric_registry.py`.
Each variant implements `prepare(graph)` and `compare(prepared_a, prepared_b)`;
the dataset, matrix, and plotting code does not need to change when a new
variant is added.

## Run resumable distance matrices

`run_distance_matrices` computes one checkpointed symmetric matrix per scalar
metric. It prepares each selected tree once, computes only the strict upper
triangle, and resumes from a fixed saved tree manifest. Elastic SRVFT is not an
accepted metric in this runner.

The selection modes are:

- `balanced --per-class N`: choose up to `N` trees from every selected class;
  a smaller class contributes all of its available trees (the default cap is
  ten).
- `random --count N`: choose `N` trees from the pooled selected classes.
- `all`: use every tree remaining after the split and class filters.
- `manifest --selection-manifest FILE`: use the listed `tree_id` values in order.

For example, this runs Chamfer, all three persistence variants, and all seven
distribution-Wasserstein variants on ten test trees per class:

```bash
source /Users/speltonen/miniconda3/bin/activate trees
python -u -m visualization.metric_study.run_distance_matrices \
  --splits test \
  --selection balanced \
  --per-class 10 \
  --metrics chamfer persistence distributions \
  --so2-grid-size 72 \
  --so2-refinement-tolerance 1e-8 \
  --output-dir outputs/metric_study/matrices/balanced_test_10_seed0
```

Add `morphometrics` for the reference-z-scored 16-component descriptor and `fgw`
when the dense FGW cost is acceptable, or use `--metrics all` for every current
non-Elastic scalar output. The morphometric reference is fitted once on the
selected ground-truth cohort; its shared Sholl radii, means, and population
standard deviations are stored in `run.json`. It is intrinsically SO(2)-invariant
and therefore uses no angular search. Because its normalization is cohort-fitted,
matrices from different selected cohorts are not numerically interchangeable.
`--no-so2-refine` disables local refinement after the grid search. The angular
settings affect Chamfer and `xyz`-feature FGW; the current persistence and
distribution variants and the morphometric descriptor are already invariant to
the specified SO(2) action.
The runner first applies the proper coordinate change
`(x, y, z) -> (x, -z, y)`, so its internal z-axis search implements the
scientific y-axis quotient used in the submission.

To detach the same run from the terminal, put `nohup` output outside the new
run directory (the runner requires that directory to be empty):

```bash
mkdir -p outputs/metric_study/background
nohup /bin/zsh -lc '
source /Users/speltonen/miniconda3/bin/activate trees
exec python -u -m visualization.metric_study.run_distance_matrices \
  --splits test \
  --selection balanced \
  --per-class 10 \
  --metrics chamfer persistence distributions \
  --so2-grid-size 72 \
  --output-dir outputs/metric_study/matrices/balanced_test_10_seed0
' > outputs/metric_study/background/balanced_test_10_seed0.log 2>&1 &
```

### Slurm with multiple CPUs

The Slurm folder contains one reusable job and a small launcher. Running the
launcher without arguments submits five independent jobs: Chamfer, the three
barcode distances, the seven distribution distances, the morphometric-vector
distance, and FGW.

```bash
bash visualization/metric_study/slurm/submit_metric_families.sh
```

By default Chamfer requests 16 CPUs, barcode and distribution jobs request 8
each, morphometrics requests 4, and FGW requests 16. These can be changed at
submission time, and a
subset of jobs can be named explicitly:

```bash
CHAMFER_CPUS=96 FGW_CPUS=24 \
  bash visualization/metric_study/slurm/submit_metric_families.sh chamfer fgw
```

The defaults follow the storage and conda layout of the older GraphPE Slurm
template. Override `PROJECT_ROOT`, `CONDA_ROOT`, `DATASET_ROOT`, or
`OUTPUT_ROOT` in the same way if the cluster checkout differs. Each job uses a
balanced sample capped at 50 neurons per class and resumes automatically when
its output directory already contains a run.

After an interruption, repeat the exact scientific configuration with
`--resume`; already terminal pairs are skipped. Add `--retry-errors` only when
failed pairs should be attempted again. Resume also verifies the SWC contents,
coordinate frame, metric source code, and relevant package versions, so it
will refuse to mix results from different inputs or implementations. A lock
prevents two processes from writing the same run. `progress.json` summarizes
the run, `selected_trees.csv` fixes matrix row order, and each
`metrics/<metric-name>/` directory contains `distances.npy`, `status.npy`, and
recorded configuration. Distances are mirrored, while the separate status
matrix distinguishes pending, undefined, failed, and successfully computed
pairs. A completed run containing failed cells exits nonzero and reports
`complete_with_errors`.

### Elastic SRVFT Slurm array

Elastic SRVFT has a separate sharded runner because every relative angle runs
the slow external alignment twice for mean symmetrization. The launcher first
screens the complete test split, rejects trees that do not fit the backend's
four-layer representation, and deterministically selects up to 20 compatible
trees per class. Each array job requests eight CPUs and runs eight independent
eight-pair shards concurrently, followed by an `afterany` merge job:

```bash
bash visualization/metric_study/slurm/submit_elastic_srvft.sh
```

The default metric is the mean-symmetrized SO(2) quotient with a 36-angle grid,
local refinement at `1e-3` radians, and `depth_policy="raise"`. With the current
test split the expected plan is 113 trees, 6,328 pairs, and 791 eight-pair shards.
The current preflight finds 470 compatible trees, 690 trees that would be
truncated, and seven zero-edge failures among 1,167 test trees. For 23P, 4P,
5P-ET, 5P-IT, 5P-NP, 6P-CT, and 6P-IT, the compatible counts are 26, 51, 3, 72,
10, 130, and 178; the cap selects 20, 20, 3, 20, 10, 20, and 20. Filtering the
earlier 248-tree cohort alone would retain 111 trees, with the substantially
less even counts 1, 9, 3, 19, 10, 35, and 34.

The 791 shards are grouped into 99 Slurm array jobs. By default at most 12 jobs
run concurrently with eight CPUs each, for at most 96 CPUs in use. The rough
estimate remains about 190 total CPU-hours and two hours of compute time,
excluding queue delay and slow-tail effects. Each job requests two hours and 4
GB per CPU. Override `CPUS_PER_JOB`, `MAX_CONCURRENT_JOBS`,
`MAX_TREES_PER_CLASS`, `DATASET_ROOT`, `OUTPUT_ROOT`, or `RUN_NAME` through
environment variables when needed.

Preparation is skipped once `run.json` exists, and every pair is checkpointed
inside its independently writable shard. Re-running the launcher therefore
resumes unfinished work. Do not submit two arrays for the same output directory
concurrently. This 113-tree manifest differs from the existing matrix runs; to
compare metrics with `matrix_report`, rerun the faster metrics using Elastic's
`selected_trees.csv` as their selection manifest so the row order matches.

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
default node masses approximate neurite length by assigning half of each edge to
each endpoint; raw uniform-node mass remains an explicit ablation. FGW still
uses one shared pairwise scale for structural costs and one for node features;
`--fgw-normalization none` retains physical units. It forms dense node-pair
matrices and repeats the solver over the angle search, so the CLI refuses trees
above 1000 nodes unless the limit is deliberately changed.

Elastic SRVFT alignment energy is also opt-in. With the clone at the expected
location, run:

```bash
source /Users/speltonen/miniconda3/bin/activate trees
python -m pip install -r metrics/external/elastic_srvft/python_distance/requirements.txt
python -m visualization.metric_study.run_pair \
  --tree-a /absolute/path/to/tree_a.swc \
  --tree-b /absolute/path/to/tree_b.swc \
  --metrics elastic_srvft \
  --elastic-so2-grid-size 8
```

The family has a separate coarse angular grid because every angle runs the
external elastic alignment. Local refinement is opt-in with
`--elastic-so2-refine`; `--elastic-symmetrization mean` roughly doubles the
work. Use `--help` for weights, checkout, depth policy, and radius fallback.

Use `--help` to select metric families and expose the preprocessing choices.
The JSON records the settings that affect interpretation.

If a morphology feature exists in only one tree, its distance is JSON `null`
and the adjacent diagnostic records which input was empty. Two empty feature
distributions have value zero but status `both_empty`.

## SO(2) contract

The low-level metric APIs use `z` as their internal preferred axis. Dataset
study commands map the submission's scientific `y` axis to it with the proper
coordinate change stated above. Chamfer and FGW's default `xyz` features then
use the same relative SO(2) minimum. The optional FGW
`--fgw-feature-mode axis` variant uses `(z, rho)` and is already invariant, but
deliberately discards azimuthal information. The
selected TMD filtrations, morphology distributions, and the morphometric vector
are intrinsically SO(2)-invariant. `--no-so2-quotient` is retained only as an
explicit diagnostic variant.

Elastic SRVFT is likewise minimized externally over `R_z(theta)`. The audited
backend itself performs no rotation optimization, and the adapter never
introduces full-SO(3) rotations. The default eight-point grid is a reported
numerical approximation; increase it or enable refinement when accuracy rather
than a first runtime check is the priority.

The numerical quotient group never tilts the preferred axis and never includes
reflections or axis flips. This does not mean every summary is sensitive to
those transformations: the current TMD filtrations and morphology
distributions also discard relative vertical-plane reflections, and several
scalar distributions are invariant to still larger rigid groups. The JSON
marks these deliberately over-invariant baselines separately from the actual
SO(2) search group.

## External Elastic SRVFT code

The adapter is enabled and expects the ignored clone at
`metrics/external/elastic_srvft/`. A missing checkout or Numba dependency gives
an actionable error. The output is the upstream alignment energy `E`, which is
a dissimilarity and has not been established here as a metric. It is
directional unless explicit mean symmetrization is selected.

The upstream data model is limited to four branch layers. The adapter raises
before silent truncation by default, rejects zero-length edges that can hang
the upstream decomposition, and records unresolved canonical branch-order
ties. Radius is carried through the representation but does not affect the
audited energy. A quotient can take minutes for full neurons, so benchmark a
representative pair before a class-wide run.

The audited checkout was revision
`903a82c8ae9ec8692fea85ea57803ba727b438a1`; each result records the actual
local revision because the clone is ignored. Its README declares CC BY-NC 4.0
and requests contact for commercial uses or derivatives; it has no standalone
license file at that revision.

See `plan.md` for the broader class-aware comparison study.
