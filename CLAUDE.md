# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**dendrite_gen** generates synthetic neuron dendrite (binary tree) structures using SO(2)-equivariant graph neural networks (EGNN) with optional diffusion-based denoising. The core approach: iteratively expand a root node into a full tree by predicting child positions and expansion labels at each reduction level.

## Environment

All Python commands must be run inside the **NEURO2** conda environment:
```bash
conda run -n NEURO2 python ...
conda run -n NEURO2 python -m pytest ...
```

## Commands

### Training
```bash
conda run -n NEURO2 python main.py -cn small_trees_run             # Real SWC data
conda run -n NEURO2 python main.py training.num_steps=1000         # Override any config value
```

### Tests
```bash
conda run -n NEURO2 python -m pytest tests/ -v                     # Full suite
conda run -n NEURO2 python -m pytest tests/test_training_smoke.py -v  # Single test file
conda run -n NEURO2 python -m pytest tests/test_forward_pass.py -v -k "test_name"  # Single test
```

## Architecture

### Pipeline Flow
1. **Data**: Load SWC files or generate synthetic trees → NetworkX graphs
2. **Reduction**: Contract trees into multi-level sequences (CherryReducer or DepthReducer)
3. **Dataset**: Wrap reduction sequences as PyG `Data` objects (InfiniteRandRedDataset / PrecomputedRedDataset)
4. **Model**: SO2_EGNN_Network processes graph → predicts position offsets + expansion labels
5. **Method**: Expansion (diffusion-wrapped) computes loss
6. **Trainer**: Orchestrates training loop with EMA, LR scheduling, validation, checkpointing

### Key Modules
- `graph_generation/model/egnn_so2.py` — Main model with SO(2) geometry, global linear attention
- `graph_generation/method/expansion.py` — Diffusion-wrapped expansion; delegates denoising to `diffusion/` (basic, edm, flow, flow_v)
- `graph_generation/method/helpers.py` — Geometry helpers (branch angles, L/R sibling assignment)
- `graph_generation/reduction.py` — CherryReducer: stochastic graph contraction
- `graph_generation/depth_reduction.py` — DepthReducer: deterministic depth-based contraction
- `graph_generation/training.py` — Trainer class
- `graph_generation/diffusion/` — DenoisingDiffusionModel and EDM variants
- `utils/data_loading.py` — SWC file parsing, graph construction
- `utils/tmd.py` — Topological Morphology Descriptor computation

### Configuration
Hydra-based YAML configs in `config/`. Top-level configs (e.g., `smoke.yaml`, `small_trees_run.yaml`) compose defaults from `config/dataset/`, `config/method/`, and `config/diffusion/`. Override any value via CLI: `python main.py key=value`.

## Detailed Documentation
Might not be updated, but relevant to identify overall flow and logic and relevant files.
- `TRAINING_FLOW_TRACE.md` — Line-by-line trace of the training forward pass 
- `SAMPLING_FLOW_TRACE.md` — Line-by-line trace of inference/sampling
- `README_EGNN.md` — EGNN library reference
