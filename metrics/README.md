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
- `adapters/`: optional boundaries to external research implementations.
- `external/`: ignored local checkouts for preliminary inspection only.

The large legacy scripts in `validation/` are not dependencies of this package.

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
