# Tree metrics

This package contains standalone tree-to-tree dissimilarities. It does not load
prediction pickles, generate plots, or infer dataset pairing. The initial API is
designed to compare one pair of rooted SWC trees.

## Symmetry contract

Trees are root-centered and have a preferred axis, normally `z`. Scalar
dissimilarities must be invariant when both inputs are rotated together around
that axis. Metrics that retain absolute azimuth can additionally form a shape
quotient by minimizing over a relative SO(2) rotation of the second tree.

The quotient does not include rotations that tilt the preferred axis, axis
flips, or reflections.

## Ownership

- `pair.py`: small programmatic orchestrator for one tree pair.
- `so2.py`: shared rotations and quotient minimization.
- `chamfer.py`: standalone arc-length sampling and Chamfer dissimilarity.
- `persistence.py`: temporary metric wrapper around the existing TMD code; the
  reusable persistence distance remains under `visualization/tmd/`.
- `distributions.py`: standalone morphology distributions and 1D Wasserstein
  distances.
- `fused_gw.py`: standalone Fused Gromov-Wasserstein comparison using POT.
- `morphometrics.py`: the 16-component tree descriptor and distances in a
  reference-cohort-standardized descriptor space. Its extractor intentionally
  duplicates the training-time validation implementation so this package does
  not depend on `validation/`; parity is covered by tests.
- `adapters/elastic_srvft.py`: optional project-facing wrapper for the external
  Elastic SRVFT implementation.
- `external/elastic_srvft/`: expected ignored local checkout for that backend.

The large legacy scripts in `validation/` are not dependencies of this package.

The morphometric descriptor combines counts, axial/radial/total extents,
Strahler order, partition asymmetry, branch/root geometry, and three Sholl
summaries. Raw Euclidean distance is not used because these components mix units
and scales. Fit a `MorphometricReference` once on a fixed ground-truth cohort,
then use z-score Euclidean distance. The reference stores shared Sholl radii,
feature means, and population standard deviations. The result is intrinsically
SO(2)-invariant and needs no angular minimization, but it is a pseudometric on
trees and remains sensitive to tracing density and physical scale.

FGW loads POT only when requested. It uses cable-length node mass by default;
the raw uniform-node mode is retained for sensitivity analysis because it is
confounded by SWC tracing density.

## Environment

Use the project environment:

```bash
source /Users/speltonen/miniconda3/bin/activate trees
```

The single-pair command is documented in
`visualization/metric_study/README.md`.

## Elastic SRVFT (optional)

The adapter expects
`metrics/external/elastic_srvft/python_distance/`. The audited upstream is
`martinalex000/complexTrees_distanceMetric` at revision
`903a82c8ae9ec8692fea85ea57803ba727b438a1`. Because the checkout is ignored,
this project does not pin it; every result records the actual local revision
and marks a dirty checkout.

The upstream repository is not an installable Python package. Activate
`trees`, install its `python_distance/requirements.txt`, and leave the checkout
in place. The adapter loads that exact package without changing `sys.path` and
does not require MATLAB or its MEX files. Numba is the only dependency from
that requirements file that is not otherwise needed by the core metric study.

The returned scalar is upstream `compute_distance_energy(...)[0]["E"]`: an
alignment energy/dissimilarity, not a proven metric, and no square root is
silently applied. Defaults are `lam_m=0.2`, `lam_s=1.0`, and `lam_p=0.2`. The
value is directional by default; `symmetrization="mean"` explicitly averages
the two directions.

The audited Python backend has no rotation optimizer. This adapter computes an
external minimum of `E(T1, R_z(theta) T2)`, never full SO(3), tilts, flips, or
reflections. Its default is an eight-angle grid with refinement disabled
because one full-neuron energy evaluation can take tens of seconds. The angle,
zero-angle value, grid result, evaluation counts, and runtime are recorded.

The upstream representation is fixed at four branch layers. Deeper structure
would otherwise be silently omitted, so the adapter raises by default; `warn`
or `allow` is only for a deliberately labelled truncation study. It also
rejects zero-length edges, which can make the upstream decomposition fail to
terminate. Radius values are parsed and carried but do not enter `E` in the
audited revision. Canonical-order ties are reported because tied, geometrically
indistinguishable branch signatures retain their stable SWC order.

The upstream README declares CC BY-NC 4.0, asks users to contact the authors
for commercial uses or derivatives, and provides no standalone `LICENSE` in
the audited checkout. Resolve those terms before redistribution, derivative,
or commercial use.
