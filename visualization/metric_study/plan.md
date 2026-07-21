# Ground-Truth Neuron Metric Study

## Purpose

Build a reproducible study for comparing tree dissimilarities on labelled,
ground-truth neuron SWC files. The first version is intentionally independent
of generated-tree prediction pickles.

The study should answer four separate questions:

1. Which dissimilarities distinguish known neuron classes?
2. What morphological changes is each dissimilarity sensitive or invariant to?
3. Which dissimilarities are redundant, and which capture complementary signals?
4. What are the computational and data-quality costs of each method?

Throughout the code and report, use **dissimilarity** as the generic term.
Reserve **metric** for methods that actually satisfy the mathematical metric
properties under the selected preprocessing and normalization.

## Proposed ownership and folder layout

Keep the experiment, plots, and report generation here:

```text
visualization/
  metric_study/
    README.md
    plan.md
    run_pair.py
    dataset.py
    pairs.py
    perturbations.py
    compute.py
    analysis.py
    plots.py
    report.py
  run_metric_study.py
```

Put reusable metric implementations in the standalone package:

```text
metrics/
  pair.py
  so2.py
  chamfer.py
  persistence.py
  fused_gw.py
  distributions.py
  topology.py
  adapters/
  external/
```

Use fully qualified imports (`dendrite_gen.metrics`) where package context is
available so this package remains distinct from the existing training-time
`graph_generation/metrics.py`. The study code owns study-specific sampling,
statistics, plots, and reports; `metrics/` owns reusable pairwise computation.

Do not move existing implementations during the first phase. Wrap them behind
the common interface, test the wrappers, and refactor only after the study is
working.

## Current milestone

The initial single-pair foundation is implemented:

| Family | Current implementation | SO(2) treatment |
| --- | --- | --- |
| Chamfer | uniform arc-length cable samples in `metrics/chamfer.py` | relative rotation minimum around `z` |
| TMD barcode Wasserstein | wrapper in `metrics/persistence.py`; canonical diagram distance remains in `visualization/tmd/distances.py` | path, height, and `rho` filtrations are intrinsically invariant |
| Distribution Wasserstein | seven named morphology distributions in `metrics/distributions.py` | all current distributions are intrinsically invariant |
| Fused Gromov-Wasserstein | opt-in POT-backed implementation with cable-length node mass in `metrics/fused_gw.py` | `xyz` features use the relative rotation minimum by default; `(z, rho)` is a cheaper information-discarding ablation |
| Elastic SRVFT | explicit placeholder in `metrics/adapters/elastic_srvft.py` | external implementation still needs an SO(3)-to-SO(2) alignment audit |

`metrics/pair.py` provides the programmatic one-pair entry point, and
`visualization/metric_study/run_pair.py` loads two SWCs and emits structured
JSON. This milestone has no dependency on prediction pickles or validation
scripts.

## Symmetry contract

The neuron coordinate system has a preferred `z` axis. The allowed group is
SO(2): rotations around `z`. It does not include arbitrary SO(3) rotations,
axis tilts, axis flips, or reflections.

For a dissimilarity that retains azimuth, the shape-quotient variant is

```text
d_shape(T1, T2) = min_{theta in [0, 2*pi)} d(T1, R_z(theta) T2).
```

A scalar comparison must be invariant to a joint allowed rotation. Metrics
built only from SO(2)-invariant quantities such as `z`, radial distance `rho`,
path length, cable length, or relative angles do not need the numerical
minimum. Every result must state whether it is intrinsically invariant, uses
the quotient minimum, or deliberately retains absolute azimuth as a diagnostic
variant.

The optimization group and the effective invariance of a summary are distinct.
TMD scalar filtrations and the current morphology distributions also discard
relative vertical-plane reflections; path lengths, cable lengths, angles, and
branch order discard even more orientation information. Treat these as useful
over-invariant controls, and record their extra invariances rather than implying
that the effective quotient is exactly SO(2).

## Preliminary inventory of existing code

The first implementation task is a complete inventory. Known starting points
include:

- `validation/chamfer.py`
  - uniform point sampling along graph edges
  - directional and symmetric Chamfer distance
  - a large existing GT/pred evaluation workflow
- `validation/geometric_metric.py`
  - radius-based geometric precision, recall, and F1
  - height, XY span, and bounding-box summaries
- `validation/structural_metrics.py`
  - branch-length and bifurcation-angle summaries
  - persistence-diagram bottleneck distance
  - topology-only tree edit distance
- `visualization/tmd/embedding.py`
  - persistence images and embeddings
  - persistence-diagram Wasserstein distance
- `visualization/stats/`
  - tree-level morphology summaries
  - branch length, bifurcation angle, path distance, radial distance, and branch
    order distributions
- `graph_generation/metrics.py` and `utils/eval_helper.py`
  - graph population metrics and MMD-based comparisons
  - degree, clustering, orbit, spectral, and wavelet-related quantities
- `utils/dist_helper.py`
  - histogram EMD and MMD helpers; these are not automatically geometric
    tree-to-tree EMD and must be named carefully

For every candidate, record:

- mathematical definition and citation
- input representation
- whether it is pairwise, population-level, or a scalar feature
- symmetry and normalization behavior
- invariances
- tunable parameters
- expected complexity
- dependency and license requirements
- existing tests and known failure modes

## Data contract

Use an explicit metadata manifest rather than inferring classes implicitly from
file order. A minimal CSV should contain:

```text
tree_id,swc_path,neuron_class
```

Optional columns should include specimen, donor, acquisition batch, brain
region, reconstruction method, or any other grouping variable that may cause
leakage. Preserve a snapshot of the input manifest in every result directory.

Loading should:

- accept a directory plus metadata manifest
- load SWCs through the repository's standard loader
- validate that each graph is connected and acyclic, or record why it is not
- resolve and record the root explicitly
- record node count, edge count, total cable length, bounding-box size, and raw
  sampling density as potential confounders
- never silently discard a tree; write exclusions and errors to a table

## Preprocessing variants

Preprocessing is part of a metric definition and must be included in its
configuration and output name. At minimum, compare:

1. Root-centred coordinates with physical scale and orientation retained.
2. Root-centred, scale-normalized coordinates.
3. Root-centred coordinates with a documented rotation/alignment procedure.
4. Both raw SWC nodes and uniform arc-length samples where meaningful.

For point-set distances, the primary representation should be points sampled
uniformly along tree edges by arc length. Raw SWC nodes reflect tracing density
and can strongly confound Chamfer and transport distances.

Never silently align or normalize trees. Orientation and absolute size may be
biologically meaningful, so aligned and normalized results should be reported
as separate variants.

## Common metric interface

Each pairwise dissimilarity should expose the same conceptual operations:

```python
prepared = metric.prepare(graph, preprocessing_config)
result = metric.compare(prepared_a, prepared_b)
```

The result should contain at least:

- scalar value
- status (`ok`, `invalid`, `timeout`, or `error`)
- runtime
- metric name and version
- complete serialized configuration
- useful diagnostics, including directional components for asymmetric methods

Prepared representations should be cacheable because point samples,
persistence diagrams, spectra, and rooted-tree encodings are expensive and can
be reused across many pairs.

The registry should distinguish:

- pairwise tree dissimilarities
- distances between per-tree feature distributions
- scalar morphology features
- population-level two-sample metrics

Do not force all four categories into one misleading scalar API.

## Initial candidate families

### Point geometry

- symmetric and directional Chamfer
- Hausdorff and robust percentile Hausdorff
- geometric precision/recall/F1 over several physical radii
- optional alignment-aware variants

### Optimal transport

- exact Wasserstein/earth mover's distance on uniformly sampled tree mass
- entropic Sinkhorn approximation
- sliced Wasserstein for a cheaper baseline
- tree- or branch-aware transport methods from external papers
- Fused Gromov-Wasserstein with tree-path structural costs and explicit node
  features

The mass model must be explicit: equal mass per sampled point, equal mass per
unit cable length, radius-weighted cable mass, or another documented choice.
Raw-node FGW is dense and remains discretization-sensitive even with
cable-length masses. Keep it opt-in, record node counts, and compare it with a
future uniform arc-length or critical-tree representation before using it for a
large all-pairs study.

### Elastic tree shape

- extended-SRVF elastic distance for tree-like 3D objects
- branch correspondence and reparameterization diagnostics, when exposed by
  the external implementation
- an explicitly constrained SO(2)-only shape quotient in place of the paper
  implementation's full SO(3) alignment

Treat the external Python implementation as optional until its revision,
license, dependencies, representation assumptions, and internal alignment code
have been audited.

### Topology and rooted structure

- normalized and unnormalized topology-only tree edit distance
- branch-length-aware edit distance if available
- spectral graph dissimilarities
- subtree-signature or path-based distances

### Topological morphology

- TMD persistence-diagram Wasserstein distance
- bottleneck distance
- persistence-image vector distances

### Morphology distributions

- Wasserstein or other distances between branch-length distributions
- bifurcation-angle distributions
- root-path and radial-distance distributions
- branch-order distributions
- distances between vectors of scalar morphology summaries

## Pair construction

If the dataset is small enough, compute the full symmetric pairwise matrix for
each metric variant. Otherwise, create one deterministic, class-stratified pair
table and reuse it for every metric.

The pair table should include:

- both tree IDs and classes
- same-class versus different-class indicator
- any shared specimen/donor/batch indicators
- deterministic pair ID
- sampling stratum and weight

Avoid letting large classes dominate the study. Either balance classes during
pair sampling or use class-aware weights during analysis. Include self-pairs
for sanity checks but exclude them from class-separation summaries.

## Evaluation protocol

### 1. Mathematical and implementation sanity checks

- identity/self-distance
- symmetry where claimed
- non-negativity and finite outputs
- invariance to node relabelling
- triangle inequality checks on sampled triples only for methods advertised as
  true metrics
- agreement between exact and approximate implementations on small trees
- deterministic results under a fixed seed

### 2. Class structure

- within-class and between-class distance distributions
- effect sizes with bootstrap confidence intervals
- ROC-AUC for predicting whether a pair shares a class
- nearest-neighbour retrieval precision at several values of `k`
- balanced k-nearest-neighbour class accuracy using only training-fold
  neighbours
- silhouette score computed from precomputed distances
- confusion and retrieval breakdowns per class

Cross-validation must group by donor, specimen, or acquisition batch when those
identifiers exist. Otherwise a metric may appear to detect neuron class while
actually detecting a shared source or processing pipeline.

### 3. Controlled perturbations

Apply deterministic perturbations at several strengths to each source tree:

- translation
- rigid rotation
- uniform scale
- node relabelling
- arc-length resampling
- coordinate noise
- local branch bending with topology preserved
- branch shortening or elongation
- terminal-branch pruning
- subtree removal
- topology-changing branch reattachment, if it can be defined safely

Plot dissimilarity against perturbation strength. Separate expected invariance
tests from expected sensitivity tests rather than assuming that larger response
is always better.

### 4. Redundancy and complementarity

- Spearman correlation between condensed distance matrices
- rank agreement of nearest neighbours
- clustered metric-correlation heatmap
- low-dimensional summary of metric outputs
- disagreements: example pairs ranked close by one metric and far by another

The disagreement examples should link back to qualitative tree renderings so
that differences can be interpreted morphologically.

### 5. Confounds and computational cost

Measure association with:

- node-count difference
- total cable-length difference
- physical size difference
- sampling-density difference
- class and acquisition metadata

Record runtime, peak memory where practical, timeout rate, invalid-result rate,
and scaling with tree size. Report class discrimination both before and after
accounting for major size and sampling confounders.

## Outputs and reproducibility

Separate expensive computation from analysis and plotting. A future CLI should
support at least:

```bash
python -m dendrite_gen.visualization.run_metric_study compute ...
python -m dendrite_gen.visualization.run_metric_study analyze ...
python -m dendrite_gen.visualization.run_metric_study report ...
```

Each run should write:

```text
metric_study_output/
  config.json
  manifest.csv
  exclusions.csv
  pairs.csv
  distances.csv          # optionally Parquet when available
  distance_matrices/
  summaries/
  figures/
  report.md
```

Each distance row should include tree IDs, pair ID, class relationship, metric
and preprocessing variant, value, runtime, status, diagnostics, configuration
hash, code revision, and random seed. Long-form raw results should remain the
source of truth; plots should be reproducible without recomputing distances.

## External paper repositories

Do **not** place an ordinary nested `git clone` directly inside the metric
implementation package. That produces an embedded repository whose contents
are not reliably versioned by this project and mixes third-party ownership with
our API.

Use this order of preference:

1. **Pinned package dependency.** If the repository is installable, reference a
   release or exact Git commit in the environment requirements.
2. **Ignored local checkout under `metrics/external/<project>`** for the initial
   API and license audit only. Install the checkout into the environment rather
   than modifying `sys.path`.
3. **Pinned Git submodule** when the code is not packaged and must remain largely
   intact after the initial audit.
4. **Isolated command-line adapter** when the external project has conflicting
   dependencies. Exchange explicit files rather than importing its environment
   into this one.
5. **Small, attributed reimplementation** only when licensing permits it and
   the algorithm is compact enough to validate independently.

In every case, add a thin adapter under
`metrics/adapters/`. The adapter should translate our prepared
tree representation into the external method's input, normalize its result,
capture version information, and provide actionable errors when the optional
dependency is absent.

Before integrating an external repository, record:

- repository URL and exact revision
- paper citation
- software license and redistribution restrictions
- install method and dependency conflicts
- expected input/output formats
- CPU/GPU requirements
- a tiny reproducible example and expected output

External metrics should remain optional so the core study and tests run without
installing every research repository.

## Testing strategy

- tiny hand-authored trees with known geometric and topological relationships
- contract tests shared by every registered metric
- preprocessing invariance tests
- deterministic perturbation tests
- cached versus uncached equality
- small end-to-end study fixture with at least two classes
- optional adapter tests skipped with a clear reason when dependencies are absent

## Implementation phases

### Phase 0: Inventory and decisions

- catalogue all current metric-like code and external candidates
- obtain a metadata manifest example and class definitions
- decide which orientation and scale variants answer the biological question
- decide the initial metric shortlist and acceptable runtime budget

### Phase 1: Study foundation

- add the standalone `metrics` interfaces and registry
- add GT-only manifest loading and validation
- implement deterministic pair generation
- define configuration and long-form result schemas

### Phase 2: Existing metric adapters

- wrap Chamfer and uniform edge sampling
- wrap structural edit and persistence distances
- expose morphology distribution distances
- add synthetic-tree and contract tests

### Phase 3: Perturbation benchmark

- implement invariant and morphology-changing perturbations
- generate sensitivity curves and sanity summaries

### Phase 4: Class and redundancy analysis

- add within/between-class statistics and retrieval evaluation
- add confound analysis and metric correlation comparisons
- generate qualitative disagreement examples

### Phase 5: External methods

- evaluate each repository's packaging and license
- integrate one method at a time through an optional adapter
- pin versions and add minimal reproducibility tests

### Phase 6: Report

- generate a concise comparison table
- document metric interpretations and failure modes
- recommend a small complementary metric set rather than a single universal
  winner

## Immediate next step

Run the single-pair command on two representative ground-truth neurons and
inspect the raw values and alignment diagnostics. In parallel, place the
Elastic SRVFT checkout under `metrics/external/` for an API/license/alignment
audit. After that, provide one example class manifest (or the current class
metadata source) so the deterministic pair table and class-aware study schema
can be finalized.
