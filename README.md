# dendrite_gen

Generate synthetic neuron dendrite (binary tree) structures using SO(2)-equivariant graph neural networks with diffusion-based denoising.

## Overview

The core approach iteratively expands a root node into a full binary tree by predicting child positions and expansion labels at each reduction level. During training, trees are contracted into multi-level reduction sequences; the model learns to reverse this process by denoising leaf positions and predicting branching decisions.

### Pipeline at a Glance

```
SWC Files ─── load_swc_graphs_from_dir() ──► NetworkX Graphs
                                                    │
                                     DepthReductionFactory()
                                                    │
                                                    ▼
                                     PrecomputedRedDataset
                                   (all reduction levels, shuffled)
                                                    │
                                          PyG Batch collation
                                                    │
                                                    ▼
                                        Expansion.get_loss()
                                                    │
                                    ┌───────────────┼───────────────┐
                                    │               │               │
                              decode parents   precompute SO(2)   assemble
                              + build edges    geometry on P_0    node feats
                                    │               │               │
                                    └───────────────┼───────────────┘
                                                    │
                                                    ▼
                                    DenoisingDiffusionModel.forward()
                                                    │
                                    ┌───────────────┼───────────────┐
                                    │               │               │
                              noise leaves    patch geometry    model forward
                              (C_0 → C_t)    for noised P_t   SO2_EGNN_Network
                                    │               │               │
                                    └───────────────┼───────────────┘
                                                    │
                                                    ▼
                                          MSE Loss (position + expansion)
```

## Setup

### Environment

All commands require the **NEURO2** conda environment:

```bash
conda activate NEURO2
# or prefix all commands with:
conda run -n NEURO2 python ...
```

### Data Directory Structure

The default configuration expects SWC neuron morphology files organized as:

```
data_dir/
├── train/    # Training SWC files
├── val/      # Validation SWC files
└── test/     # Test SWC files
```

Each `.swc` file is parsed into an undirected NetworkX graph with 3D `pos` attributes per node. The root is auto-detected (preferring a node with >= 2 children). Positions are recentered so the root sits at the origin.

Configure the data path in `config/dataset/trees_test.yaml`:
```yaml
name: tree_dataset
data_dir: /path/to/your/swc/data
load: True
```

## Usage

### Training

```bash
# Default training run (depth reduction + diffusion)
conda run -n NEURO2 python main.py -cn small_trees_run

# Override any config value via CLI
conda run -n NEURO2 python main.py -cn small_trees_run training.num_steps=5000 training.lr=1e-3

# Debugging mode (CPU, verbose)
conda run -n NEURO2 python main.py -cn small_trees_run debugging=True
```

### Testing

```bash
# Run full test suite
conda run -n NEURO2 python -m pytest tests/ -v

# Single test file
conda run -n NEURO2 python -m pytest tests/test_training_smoke.py -v
```

### Resuming from Checkpoint

```bash
# Resume from latest checkpoint
conda run -n NEURO2 python main.py -cn small_trees_run training.resume=True

# Resume from specific step
conda run -n NEURO2 python main.py -cn small_trees_run training.resume=5000

# Resume from specific file
conda run -n NEURO2 python main.py -cn small_trees_run training.resume=/path/to/step_5000.pt
```

## Configuration

Hydra-based YAML configs live in `config/`. The primary training config is `config/small_trees_run.yaml`, which composes:

| Section | Default | File |
|---------|---------|------|
| Dataset | `trees_test` | `config/dataset/trees_test.yaml` |
| Diffusion | `basic` | `config/diffusion/basic.yaml` |
| Method | `expansion_oneshot` | `config/method/expansion_oneshot.yaml` |

### Key Config Parameters

```yaml
reduction:
  type: "depth"           # "depth" (deterministic) or "cherry" (stochastic)
  mode: "stochastic"      # cherry selection mode
  cherry_p: 1.0           # probability of selecting each cherry
  contract_root: False    # if False, smallest graph retains root + children

training:
  batch_size: 128
  lr: 5e-4
  num_steps: 30000
  lr_scheduler: "cosine_annealing"

model:
  name: egnn              # SO2_EGNN_Network
  num_layers: 6
  feats_dim: 64
  m_dim: 64               # message dimension

diffusion:
  name: "basic"           # DenoisingDiffusionModel
  num_steps: 64           # denoising steps during sampling
```

## Architecture Deep Dive

### 1. Data Loading (`utils/data_loading.py`)

SWC files are parsed into NetworkX graphs. Each node gets a 3D `pos` attribute. Root detection follows priority rules: prefer node 1 if it has >= 2 children, else try node 2, else fallback to the first node with `parent_id == 0`. Positions are recentered to place the root at the origin.

### 2. Tree Reduction (`graph_generation/depth_reduction.py`)

Trees are contracted into multi-level reduction sequences. The **DepthCherryReducer** removes leaf cherries (parents whose ALL children are leaves) at the deepest depth level first, working upward:

```
Full Tree (N nodes)
    │ remove deepest-level cherries
    ▼
Level 1 (fewer nodes, parents become new leaves)
    │ remove next deepest cherries
    ▼
Level 2 ...
    │
    ▼
Minimal graph (root + children)
```

Each reduction step records:
- **Survivor mask**: which nodes remain
- **Leaf indices and expansion labels**: `{1: terminal, 2: will branch}`
- **New leaves from next level**: nodes that became leaves in this contraction step
- **Parent indices**: 1-based for safe PyG batching (0 = root sentinel)

The reduction type determines the dataset class:
- **`type: "depth"`** (default): Uses `PrecomputedRedDataset` -- all reduction sequences are generated once at startup and reshuffled each epoch. Deterministic and reproducible.
- **`type: "cherry"`**: Uses `InfiniteRandRedDataset` -- infinite streaming with stochastic cherry selection. Caches one sequence per graph, resamples when depleted.

### 3. Dataset (`graph_generation/data/`)

`ReducedGraphData` (PyG `Data` subclass) wraps each reduction level as a training sample:

| Field | Shape | Description |
|-------|-------|-------------|
| `pos` | `[N, 3]` | 3D node positions |
| `parent_idx_1b` | `[N]` | 1-based parent indices (0 = root) |
| `leaf_idx` | `[L]` | Indices of leaf nodes |
| `leaf_mask` | `[N]` | Boolean leaf mask |
| `leaf_expansion` | `[L]` | Expansion labels in {1, 2} |
| `new_leaf_idx_from_next` | `[L_new]` | "New" leaves from next reduction level |
| `total_tree_size` | scalar | Original full tree node count |
| `tmd` | `[1, D]` | Topological Morphology Descriptor |

PyG batching (`Batch.from_data_list`) concatenates node tensors and auto-offsets index fields (`leaf_idx`, `parent_idx_1b`, `new_leaf_idx_from_next`) by cumulative node counts.

### 4. Training Method (`graph_generation/method/expansion.py`)

`Expansion.get_loss()` is the training entry point. It:

1. **Decodes parent indices**: Converts 1-based `parent_idx_1b` to 0-based with -1 for roots
2. **Builds directed edges**: Creates bidirectional parent-child edge pairs with type labels (0 = parent->child, 1 = child->parent)
3. **Selects training leaves**: Uses `new_leaf_idx_from_next` -- the leaves that appeared when contracting from the next finer level
4. **Computes relative position targets**: `C_0 = leaf_pos - parent_pos` (in local SO(2) frame)
5. **Precomputes full geometry on clean P_0**: SO(2) decomposition, branch angles, left/right sibling labels -- computed once and patched for noised positions later
6. **Assembles node features**: `[is_leaf, geo_lr, new_leaf_flag, size_ratio, padding]`
7. **Calls diffusion forward**: Delegates to `DenoisingDiffusionModel`

### 5. Diffusion (`graph_generation/diffusion/basic.py`)

`DenoisingDiffusionModel` implements sigma-conditioned denoising:

**Training** (`forward`):
- Samples noise level sigma per graph from a log-normal distribution
- Noises leaf relative offsets: `C_t = C_0 + sigma * epsilon`
- Places noised leaves: `P_t[leaf] = parent_pos + local_to_global(C_t)`
- Patches precomputed geometry for changed leaf positions
- Model predicts clean `(C_0, e_0)` from noised `(P_t, e_t, sigma)`
- Loss: `MSE(C_pred, C_0) + weight * MSE(e_pred, e_0)`

**Sampling** (`sample`):
- Follows a sigma schedule from `sigma_max` down to 0
- At each step, model predicts clean state, then re-noises at next (lower) sigma
- Final prediction is the denoised output

### 6. Model (`graph_generation/model/egnn_so2.py`)

`SO2_EGNN_Network` is an SO(2)-equivariant message-passing neural network:

- **Input**: `[positions | node_features | diffusion_conditioning | TMD_embedding]`
- **Edge features**: SO(2) invariants `(rho, du)` + branch angles `(cospsi, sinpsi, cos_theta)` + edge direction type
- **Layers**: Alternating EGNN message-passing and global linear attention (ISAB)
- **Output head**: Predicts `[rel_offset(3) | expansion_signal(1)]` per node; only leaf predictions are used for loss

### 7. Geometry (`graph_generation/method/helpers.py`)

Key geometric computations:

- **`geo_lr_mask`**: Assigns left/right labels to sibling nodes based on their angular relationship relative to the parent's incoming direction. Used as a node feature to break sibling symmetry.
- **Branch angles**: Per-node `(cospsi, sinpsi, cos_theta)` describing the in-plane rotation and axial tilt of each branch relative to its parent's incoming direction.
- **SO(2) decomposition**: Each edge vector is decomposed into components parallel (`du`) and perpendicular (`rho`, `r_perp`) to the SO(2) axis (`uhat = [0,0,1]`).
- **Compute-once optimization**: Full geometry is computed once on clean positions P_0. During diffusion training, only leaf-affected quantities are patched for noised positions P_t (O(L) instead of O(N+E)).
- **Local basis frames**: Each leaf gets a local coordinate frame (forward, sideways, uhat) for SO(2)-equivariant prediction. Targets and predictions are expressed in this local frame.

### 8. Sampling / Inference (`Expansion.expand()`)

At inference time, trees are grown iteratively:

1. Start with root nodes (one per graph in batch)
2. Each step: predict child positions and expansion labels for current leaves
3. Leaves with `expansion = 2` spawn two children; roots spawn one
4. Repeat until target sizes are reached or no more leaves to expand
5. Convert to NetworkX graphs with 3D positions

### 9. Trainer (`graph_generation/training.py`)

Orchestrates the training loop:
- Adam optimizer with optional cosine annealing LR schedule
- EMA model tracking (configurable beta)
- Periodic validation with graph generation and optional metric computation
- Checkpoint saving/resuming
- Optional Weights & Biases logging

## Project Structure

```
dendrite_gen/
├── main.py                          # Entry point (Hydra)
├── config/
│   ├── small_trees_run.yaml         # Primary training config
│   ├── dataset/                     # Dataset configs
│   ├── diffusion/                   # Diffusion configs (basic, edm)
│   └── method/                      # Method configs
├── graph_generation/
│   ├── __init__.py
│   ├── training.py                  # Trainer class
│   ├── depth_reduction.py           # Depth-based tree contraction
│   ├── reduction.py                 # Cherry-based tree contraction
│   ├── metrics.py                   # Evaluation metrics
│   ├── model/
│   │   ├── egnn_so2.py              # SO2_EGNN_Network (main model)
│   │   └── ...
│   ├── method/
│   │   ├── expansion.py             # Diffusion-wrapped expansion
│   │   ├── expansion_oneshot.py     # One-shot expansion (no diffusion)
│   │   └── helpers.py               # Geometry helpers
│   ├── diffusion/
│   │   ├── basic.py                 # DenoisingDiffusionModel
│   │   └── ...
│   └── data/
│       ├── data.py                  # ReducedGraphData (PyG Data subclass)
│       └── reduction_dataset.py     # Dataset classes (Precomputed, Infinite, Finite)
├── utils/
│   ├── data_loading.py              # SWC file parsing
│   └── tmd.py                       # Topological Morphology Descriptor
├── validation/
│   ├── chamfer.py                   # Main evaluation script (Chamfer + full metric suite)
│   ├── geometric_metric.py          # Point-set metrics (F1, height, span, bbox)
│   ├── structural_metrics.py        # Tree metrics (branch length, angles, TMD, TED)
│   ├── plot.py                      # Visualization helpers
│   └── plot_sequence.py             # Reduction sequence visualization
├── tests/                           # Test suite
├── TRAINING_FLOW_TRACE.md           # Detailed training forward-pass trace
├── SAMPLING_FLOW_TRACE.md           # Detailed inference/sampling trace
└── CLAUDE.md                        # Claude Code project instructions
```

## Validation & Evaluation (`validation/`)

Post-training evaluation compares generated trees against ground-truth SWC morphologies. The main entry point is `validation/chamfer.py`, which runs a comprehensive metric suite.

### Running Validation

```bash
# Basic evaluation (Chamfer distance + structural metrics)
conda run -n NEURO2 python validation/chamfer.py \
    --gt-dir /path/to/swc/test \
    --pred-pkl outputs/validation/step_30000.pkl \
    --ema-key ema_1 \
    --output-json results.json

# With plots and topology edit distance
conda run -n NEURO2 python validation/chamfer.py \
    --gt-dir /path/to/swc/test \
    --pred-pkl outputs/validation/step_30000.pkl \
    --ema-key ema_1 \
    --plot-dir plots/ \
    --plot-max 12 \
    --plot-pairs \
    --ged \
    --output-json results.json
```

The `--pred-pkl` file is produced by the trainer during validation (saved to `outputs/validation/step_XXXX.pkl`). It contains a dict keyed by EMA beta (e.g., `ema_1`) with a `pred_graphs` list of NetworkX graphs.

### Metric Suite

The evaluation pipeline matches GT and predicted graphs by node count (exact match first, then closest-size fallback) and computes per-pair metrics:

#### Geometric Metrics (`geometric_metric.py`)

| Metric | Description |
|--------|-------------|
| **Chamfer Distance** | Symmetric nearest-neighbor distance between edge-sampled point clouds. Points are sampled at fixed spacing along edges plus all node positions. |
| **Precision/Recall/F1** | Radius-based: a predicted point is a TP if within `--f1-radius` of any GT point. Computed on both sampled point clouds and raw node positions. |
| **Height (z-range)** | Vertical extent: `max(z) - min(z)` |
| **Span (XY diameter)** | Maximum pairwise distance in the XY plane |
| **Bounding Box Diagonal** | 3D bounding box diagonal length |

#### Structural Metrics (`structural_metrics.py`)

| Metric | Description |
|--------|-------------|
| **Mean Branch Length** | Mean Euclidean edge length across all edges |
| **Mean Branch Amplitude** | Mean pairwise angle between sibling branches at each bifurcation node |
| **Bifurcation Angles** | Distribution of all sibling-branch angles (degrees) at branching points |
| **TMD Bottleneck Distance** | Bottleneck distance between persistence diagrams (path-length filtration from root, weighted by Euclidean edge lengths, simplified to critical tree) |
| **Tree Edit Distance** | Topology-only TED via `zss` library (optional, `--ged` flag). Unlabeled: insert/delete cost = 1, substitution cost = 0. Child order canonicalized by subtree signatures for approximate unordered TED. |

#### Visualization (when `--plot-dir` is set)

For each matched pair (up to `--plot-max`):
- Multi-angle graph views (GT and predicted, separate and overlay)
- Point cloud views at multiple azimuths
- Skeleton views (edges only, no nodes)
- Persistence diagram overlay (GT vs predicted)
- Tornado histograms for branch length and bifurcation angle distributions

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--gt-dir` | required | Directory containing GT SWC files |
| `--pred-pkl` | required | Pickle with predicted graphs |
| `--ema-key` | `None` | EMA key inside pickle (e.g., `ema_1`) |
| `--spacing` | `1.0` | Point sampling spacing along edges |
| `--squared` | `False` | Use squared distances for Chamfer |
| `--tmd-normalize` | `minmax` | TMD filtration normalization (`minmax`, `max`, `none`) |
| `--f1-radius` | `0.2` | Neighborhood radius for precision/recall/F1 |
| `--ged` | `False` | Enable topology tree edit distance |
| `--ged-mode` | `raw` | TED reporting: `raw`, `normalized`, or `both` |
| `--plot-dir` | `None` | Directory to save comparison plots |
| `--plot-max` | `12` | Max graph pairs to plot |
| `--plot-pairs` | `False` | Also save side-by-side GT/pred pair plots |
| `--hist-bins` | `32` | Histogram bin count for distribution plots |
| `--output-json` | `None` | Path to save full JSON results |

### Output Format

The JSON output contains:
- **`summary`**: Aggregate Chamfer stats (count, mean, std, min, max, median)
- **`per_size_summary`**: Chamfer stats grouped by node count
- **`per_tree`**: Full metrics per matched pair (Chamfer, F1, branch lengths, TMD, spatial extents, etc.)
- **`unmatched`**: Size groups where GT/pred counts didn't match
- **`ged_summary`** / **`ged_norm_summary`**: TED stats (if `--ged` enabled)

## Detailed Documentation

For line-by-line traces of the forward pass:
- **`TRAINING_FLOW_TRACE.md`** -- Complete trace from batch construction through geometry precomputation, diffusion noising, model forward pass, and loss computation.
- **`SAMPLING_FLOW_TRACE.md`** -- Complete trace of inference/sampling flow.
