# Neuron Dataset Stats & Loss Accounting — `neurons_conditional`

**Living document.** Tracks the structure of the trainable neuron dataset and, importantly,
**every neuron and node we lose in preprocessing** — needed for honest reporting in the paper.

- **Analysis date:** 2026-07-06
- **Raw source:** `~/Downloads/swc_simplified.tar.zstd` (26,490 neurons; degree-2 collapsed, tip-rooted, un-binarized) → `/Users/umer/Documents/neurons_simplified/swc_simplified/{train,val,test}`
- **Cell-type labels:** `~/Downloads/targets_cell_type.csv` (100% join coverage)
- **Trainable output:** `/Users/umer/Documents/neurons_conditional/{train,val,test}/<id>.swc` — soma-rooted, strictly binary, radii/types kept, cell-class integer embedded (local staging; rsync to `/scratch/guptau/neurons_conditional`, config `config/dataset/neurons_conditional.yaml`).
- **Reproduce:**
  - Clean: `conda run -n NEURO2 python preprocessing/prepare_conditional_dataset.py`
  - Losses on raw: `conda run -n NEURO2 python data_analysis/dataset_loss_accounting.py --max-children 16`
  - Verify output: `... dataset_loss_accounting.py --root /Users/umer/Documents/neurons_conditional --drop-classes ""`

---

## 1. Preprocessing pipeline & decisions

Raw `swc_simplified` had degree-2 nodes collapsed but was **tip-rooted** (soma = interior node, always id 1 / type 1) and **not binarized** (~3.4% of non-root nodes multifurcated). `preprocessing/prepare_conditional_dataset.py` (reusing `clean_trees.clean_swc_tree`, `root_mode="index"`) produces the trainable set per split (**splits preserved, no re-split**):

1. **Re-root at the soma** (node id 1) → soma-rooted, radiating outward.
2. **Binarize** non-root nodes: a 3-child node (undirected deg 4) gets one inserted node (lossless split); a ≥4-child node (deg ≥5) keeps its **2 thickest children** and deletes the rest (lossy prune).
3. **Keep radii/types** (no `--drop-attrs`) — so the "thickest" prune is real, not node-order arbitrary. Harmless downstream: `nx_graph_to_adj_pos` drops radius/type; only positions + adjacency reach the model.
4. **Drop rare classes** WM-P, MC, BPC (test-only, 21 neurons).
5. **Drop somata with > 16 children** (`MAX_CHILDREN=16`, the one-hot ordinal width in `expansion.py`; raised from 10).
6. **No depth cap.**
7. **Embed cell class** as a `# cell_class N` header; `load_swc_graph` parses it to `G.graph['cell_class']`.

### Class-integer mapping (7 kept pyramidal types, cortical-layer order)
| id | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| type | 23P | 4P | 5P-IT | 5P-ET | 5P-NP | 6P-IT | 6P-CT |

Dropped entirely (no id): WM-P, MC, BPC.

---

## 2. At a glance (trainable `neurons_conditional`)

| | train | val | test | total |
|---|---|---|---|---|
| kept neurons | 22,750 | 2,528 | 1,167 | **26,445** |
| nodes/file: mean / median / p95 / max | 61 / 53 / 116 / 537 | 60 / 52 / 113 / 495 | 83 / 77 / 157 / 397 | — |
| soma-rooted max depth: mean / p95 / max | 8.2 / 13 / 35 | 8.1 / 13 / 30 | 8.8 / 14 / 25 | — |
| soma degree (#primary dendrites): mean / median / max | 7.7 / 8 / 16 | 7.7 / 8 / 16 | 8.0 / 8 / 16 | — |

Structural verification (on the output): **0 multifurcations, 0 degree-2 nodes, 0 broken/disconnected, all soma-rooted, root degree ≤ 16, splits preserved.** The **test** split holds systematically larger neurons (mean 83 vs 61 nodes).

### Cell-type composition (kept)
| type | id | train | val | test | total |
|---|---|---|---|---|---|
| 23P   | 0 | 7776 | 866 | 339 | 8981 |
| 4P    | 1 | 5783 | 642 | 281 | 6706 |
| 5P-IT | 2 | 3021 | 336 | 139 | 3496 |
| 5P-ET | 3 |  581 |  64 |  38 |  683 |
| 5P-NP | 4 |  253 |  28 |  10 |  291 |
| 6P-IT | 5 | 3181 | 353 | 211 | 3745 |
| 6P-CT | 6 | 2155 | 239 | 149 | 2543 |
| **all** | | **22,750** | **2,528** | **1,167** | **26,445** |

Pyramidal types are stratified ~86/10/4 across splits. (The raw dataset had held WM-P/MC/BPC 100% in test as an OOD group; per the drop decision these are now excluded entirely.)

---

## 3. LOSS ACCOUNTING (report these in the paper)

All numbers computed analytically on the raw dataset with `dataset_loss_accounting.py --max-children 16` (drop-classes WM-P,MC,BPC), matching what the cleaning pipeline does.

### 3.1 SAMPLE loss — whole neurons dropped: **45 / 26,490 = 0.17%** (kept 26,445 = 99.83%)

| reason | count | which |
|---|---|---|
| rare class (WM-P/MC/BPC) | 21 | 17 / 3 / 1, all in test |
| soma degree > 16 | 24 | 23 train + 1 val |
| broken / disconnected | 0 | — |

By cell type (soma > 16 cap): **23P 20 (0.22%)**, **5P-ET 3 (0.44%)**, 5P-IT 1 (0.03%), all others 0.

> **Paper note — the cap choice matters and is disclosed.** At the *old* cap of 10 this filter dropped **2,476 neurons (9.35%)** and was heavily cell-type-biased (5P-ET 31%, 23P 18%). Raising `MAX_CHILDREN` to **16** cuts that to **24 neurons (0.09%)** and removes the bias (max 0.44%). Cost: the model one-hot is 16-wide (bits 10–15 were previously always-zero padding, so this is a fresh-training change; `feats_dim` unchanged — neuron configs 128/256 have ample budget). Somata with 17–23 children (24 neurons) remain unrepresentable and are dropped.

**Soma-degree distribution (all raw files):**
```
deg 3:  90     deg 8: 4957    deg 13:  299     deg 18:   4 *
deg 4: 721     deg 9: 3763    deg 14:  146     deg 19:   3 *
deg 5:2333     deg10: 2384    deg 15:   48     deg 20:   2 *
deg 6:4411     deg11: 1300    deg 16:   15     deg 23:   1 *
deg 7:5335     deg12:  643    deg 17:  14 *                       (* dropped, deg>16)
```

### 3.2 NODE loss — multifurcation pruning (within kept neurons)

Non-soma branch points with **≥4 children (deg ≥5)** can't be binarized by insertion, so the 2 thickest children are kept and the rest deleted. 3-child nodes (deg 4) are split losslessly.

- Of all non-soma multifurcations: **~77% lossless 3-child split, ~23% lossy prune.**
- Prune-node child-count distribution (kept): `{4:3339, 5:1471, 6:688, 7:415, 8:190, 9:130, 10:69, 11:36, 12:9, 13:9, 14:4, 15:3, 16:1, 17:1, 18:1}`.
- **Branches (subtrees) deleted: ~19,165.**
- **Nodes deleted: ~20,611 ≈ 1.25% of kept nodes** (train 1.28%, val 1.22%, test 1.04%).
- Nodes **inserted** by lossless split: 21,267 (net node count ≈ +656, essentially unchanged).

> Because radii are kept, the deleted branches are genuinely the **thinnest** at each ≥5-way junction (a defensible morphological choice), unlike the old `--drop-attrs` pipeline where the surviving pair was arbitrary node order.

### 3.3 OPTIONAL loss — depth cap (NONE applied)

For reference, nodes beyond a hypothetical cap (soma-rooted depth): cap 12 → 1.16%, cap 16 → 0.50%, cap 20 → 0.19%.

### 3.4 Loss summary

| stage | unit | lost | % |
|---|---|---|---|
| rare-class drop | neurons | 21 | 0.08% of all |
| soma degree > 16 | neurons | 24 | 0.09% of all |
| multifurcation prune | nodes | ~20,611 | ~1.25% of kept nodes |
| multifurcation prune | branches | ~19,165 | — |
| depth cap | nodes | 0 (not applied) | — |
| **Net trainable corpus** | **neurons** | **26,445 kept** | **99.83%** |

---

## 4. C₀ offset distribution — parameters transfer unchanged

`tests/analyse_c0_distribution.py --axis y --pos-scale 45.1` on old `neurons_final` vs a
preprocessed sample of this dataset: local-frame child offsets are nearly identical, so
**`pos_scale_factor = 45.1` and the flow prior std `[0.74, 0.61, 0.83]` carry over with no retuning.**

| axis | old mean / std | new mean / std |
|---|---|---|
| forward | 0.480 / 0.739 | 0.504 / 0.729 |
| sideways | 0.000 / 0.605 | 0.006 / 0.610 |
| axial (y) | −0.089 / 0.833 | −0.097 / 0.773 |
| offset-norm \|C\| | 1.009 | 1.002 |
| expand-label fraction | 0.428 | 0.437 |

---

## 5. Reproducibility & touch points

- Cleaning: `preprocessing/prepare_conditional_dataset.py` (class map + drops + cap live here; `MAX_CHILDREN=16`).
- Loss/structure: `data_analysis/dataset_loss_accounting.py` (`--max-children`, `--drop-classes`, `--root`).
- Cap constant: `graph_generation/method/expansion.py::MAX_CHILDREN = 16` (module-level, single source; also `data_analysis/prepare_neurons_final.py`).
- Class read: `utils/data_loading.py::load_swc_graph` → `G.graph['cell_class']`.
- Config: `config/dataset/neurons_conditional.yaml`; use via `python main.py -cn neuron_dataset_run_3 dataset=neurons_conditional`.
- **Deferred (not yet wired):** feeding `cell_class` into the model as a conditioning signal — mirror the `tmds` channel (`main.py` classes list → dataset → `ReducedGraphData` → `nn.Embedding` in `egnn_so2.py`, budget subtracted from `avail_feats_dim`). This is the "conditional runs" follow-up.
- Verified end-to-end: SO(2) invariance for k=11–16 (`tests/test_so2_invariance.py`), forward/dataset tests, and a 20-step smoke train on the cleaned data (load → train → generation, no errors).
