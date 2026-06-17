# Manifold structure of GT neuron embeddings

**Question.** Do the per-neuron embeddings of the ground-truth dataset lie on a
meaningful, low-dimensional, *learnable* manifold — or are they essentially a
high-dimensional blob with no exploitable structure? This is a property of the GT
data alone (no generated samples involved), and it bounds what any generative model
could hope to capture.

We answer it for the two embeddings used by the in-loop validation metrics
(`validation/dist_metrics.py`):

- **`morpho`** — the 16-d hand-engineered morphometric vector (`assemble_morpho_vector`):
  counts, extents, Strahler, partition asymmetry, mean branch length / angle /
  path-to-root / radial-to-root / contraction, and three Sholl summaries.
- **`tmd`** — the 256-d Euclidean-from-root (`radial_root`) TMD persistence image
  (`compute_tmd_embedding`); 201 of 256 bins are non-empty across the dataset.

Reproduce with `data_analysis/morpho_manifold.py` (see end).

---

## Method

"Lies on a meaningful learnable manifold" decomposes into three measurable claims,
each with a concrete test:

1. **Low intrinsic dimension** (a manifold exists). Linear: PCA **effective rank**
   (participation ratio) and #components for 90/95% variance. Nonlinear:
   **TwoNN** (Facco et al. 2017) and **MLE / Levina–Bickel** intrinsic-dimension
   estimators from nearest-neighbour distances.
2. **Structure beyond independent marginals** (the manifold is *genuine*, not just
   narrow per-feature ranges). Compare against a **feature-shuffled null**: permute
   each feature/column independently across samples — this destroys all
   cross-feature correlation while preserving every marginal exactly. If
   `real ≪ null`, the low dimensionality comes from real correlations (a manifold),
   not from narrow marginals.
3. **Reconstructability** (it can be learned). PCA reconstruction RMSE vs #components
   — the *linear lower bound* on how well a low-dim latent can reconstruct the data.
   A nonlinear model can only do better.

Preprocessing: impute any nan to the column median, drop (near-)constant columns,
z-score the rest. Datasets: `neurons_final/{train, val_extended}` (independent
splits) to confirm the structure is a property of the data, not one sample.

---

## Results

| Embedding | Split | n | ambient (active) | eff. rank | TwoNN ID | MLE ID | null TwoNN | #PC 90% | #PC 95% |
|-----------|-------|---|------------------|-----------|----------|--------|------------|---------|---------|
| morpho | train         | 4000 | 16 (16)   | **8.3** | **7.7** | 6.7 | 12.9 | 6  | 8  |
| morpho | val_extended  | 1848 | 16 (16)   | 8.3 | 7.3 | 6.4 | 12.8 | 6  | 8  |
| tmd    | train         | 2500 | 256 (201) | **61.3** | **14.5** | 11.4 | 42.0 | 36 | 51 |
| tmd    | val_extended  | 1848 | 256 (201) | 59.9 | 13.8 | 10.5 | 40.4 | 35 | 50 |

PCA reconstruction RMSE (standardized units; lower = better):

| Embedding | k=1 | k=2 | k=3 | k=5 | k=8 | k=16 |
|-----------|-----|-----|-----|-----|-----|------|
| morpho (train) | 0.78 (39%) | 0.59 (66%) | 0.49 (76%) | 0.38 (86%) | 0.21 (96%) | 0.00 (100%) |
| tmd (train)    | 0.92 (16%) | 0.85 (28%) | 0.81 (34%) | 0.75 (44%) | 0.67 (55%) | 0.52 (73%) |

(Percentages = cumulative variance explained. `val_extended` numbers are within ~1%
of `train` in every cell.)

---

## Findings

### 1. Yes — both embeddings lie on low-dimensional manifolds
- **morpho**: intrinsic dimension ≈ **7** (TwoNN 7.7, MLE 6.7) inside a 16-d ambient
  space; effective rank 8.3; 90% of variance in just **6** components.
- **tmd**: intrinsic dimension ≈ **12** (TwoNN 14.5, MLE 11.4) inside a 201-d active
  ambient space; effective rank ~60.

Both sit far below their ambient dimension — there is real, exploitable structure.

### 2. The structure is genuine (not an artefact of narrow marginals)
Against the feature-shuffled null, intrinsic dimension drops by **~1.7×** (morpho:
7.7 vs 12.9) to **~3×** (tmd: 14.5 vs 42.0), and effective rank by ~2× (morpho 8.3 vs
16) to ~3× (tmd 61 vs 197). The low dimensionality therefore comes from
**cross-feature correlations**, i.e. an actual manifold — exactly the thing a joint
metric (MMD / Density-Coverage) can see but per-feature marginals cannot.

### 3. It is stable across independent splits
`train` (4000) and `val_extended` (1848) give near-identical numbers for every
statistic (intrinsic dim within ±0.7, eff. rank within ±1.4, identical #PC-for-90%).
The manifold is a property of the data, not of a particular sample.

### 4. morpho is near-*linear*; tmd is *nonlinear*
- morpho: intrinsic dim ≈ 7 and **6 linear PCs already explain 90%** → the manifold
  is close to a linear subspace; PCA reconstructs it almost perfectly (k=8 → 96%).
- tmd: intrinsic dim ≈ 12 but it takes **36 linear PCs to reach 90%** and k=16 only
  reaches 73% → the ~12-dim manifold is **curved / nonlinear**, spread across many
  linear directions. A linear method reconstructs it poorly; a nonlinear model
  (diffusion / EGNN) is the right tool.

### 5. It is a connected continuum, not discrete clusters
For both embeddings the PCA scatter is a single connected cloud with a smooth
gradient (size / `total_extent` for morpho, `log #nodes` for tmd); t-SNE breaks it
into patches but the colour flows smoothly across them — characteristic of a
continuum, not separated cell-type clusters. (No cell-type labels were available, so
this is a visual judgement; see the saved plots.)

### What the morpho manifold's axes mean
Three interpretable directions carry ~75% of the morpho variance:
- **PC1 (39%) — size / complexity**: node/leaf/bifurcation counts + Sholl AUC.
- **PC2 (26%) — spatial reach**: mean branch length, radial/path-to-root, total
  extent, radial span — largely *independent* of count.
- **PC3 (10%) — branching shape**: contraction, partition asymmetry, bifurcation
  angle (straight-vs-wandering, symmetric-vs-asymmetric).

Several morpho features are **redundant**, which is *why* 16 features collapse to ~7
degrees of freedom: `node_count ≈ leaf_count ≈ bifurcation_count` (corr **1.00** —
expected for binary trees, leaves = bifurcations + 1), `radial_span ≈ total_extent`
(0.99), `mean_path_to_root ≈ mean_radial_to_root` (0.98). Mean |off-diagonal
correlation| = 0.34.

---

## Implications

- **It is learnable.** A generative model with a modest latent / conditioning
  bottleneck (order ~7–15 effective dimensions) is sufficient to express the GT
  morphometric variation; the bottleneck is not a fundamental limitation.
- **Use a nonlinear model for the TMD structure.** Its manifold is curved, so linear
  summaries undersell it; the diffusion/EGNN generator is appropriate.
- **Validates the eval embedding choice.** The in-loop TMD metric PCA-reduces the
  256-d image to `tmd_pca_ncomp` (64 for neurons). With 90% variance at 36 PCs and
  95% at ~51, **64 components retain essentially all the manifold structure** while
  cutting noise/empty-bin dimensions — consistent with the `neuron_dataset_run*`
  configs.
- **Justifies the joint metrics.** Genuine cross-feature structure (real ≪ null) is
  precisely what `mmd_morpho` / `mmd_tmd` and Density/Coverage are built to measure;
  per-feature W1/KS marginals are blind to it.
- **The two embeddings are complementary.** morpho is a compact, near-linear summary
  of size/reach/shape (~7-d); tmd is a richer, nonlinear branching-topology
  descriptor (~12-d). Reporting both covers different failure modes.

## Caveats

- Intrinsic-dimension estimators are approximate (absolute values ±1–2); the
  *conclusions* are robust because the real-vs-ambient and real-vs-null gaps are large
  and consistent across splits and estimators.
- The morpho manifold dimension reflects *this 16-feature representation*, not the
  full morphology's degrees of freedom; redundant features deflate it.
- The tmd active ambient dimension (201) depends on the persistence-image grid
  (`n_bins=16` → 256 bins); a finer grid raises the ambient dimension but should not
  materially change the intrinsic dimension.
- "Continuum vs clusters" is a visual call (no cell-type labels); a labelled analysis
  could refine it.

## Reproduce

```bash
conda run -n NEURO2 python data_analysis/morpho_manifold.py <SWC_DIR> <N_SAMPLE> <morpho|tmd>
# examples
conda run -n NEURO2 python data_analysis/morpho_manifold.py /path/neurons_final/train 4000 morpho
conda run -n NEURO2 python data_analysis/morpho_manifold.py /path/neurons_final/val_extended 2000 tmd
```

Prints the table rows above and saves plots to `data_analysis/morpho_manifold_out/`:
`<split>_<embedding>_pca.png` (spectrum + PC1/PC2 scatter) and `_tsne.png`.

Estimators: TwoNN (Facco et al. 2017), MLE/Levina–Bickel; feature-shuffled null for
the marginal-vs-joint contrast.
