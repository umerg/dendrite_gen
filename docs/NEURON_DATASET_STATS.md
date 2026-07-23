# Neuron Dataset Stats & Loss Accounting — `neurons_conditional_full`

**Living document.** Tracks the structure of the trainable neuron dataset and, importantly,
**every neuron and node we lose in preprocessing** — needed for honest reporting in the paper.

> **Update 2026-07-23 — root-degree cap removed.** The earlier `neurons_conditional` set dropped
> somata with > 16 primary dendrites (`MAX_CHILDREN=16`). That cap was hard to defend, so we raised
> `MAX_CHILDREN` to **23 = the maximum primary-dendrite count observed in the corpus** and regenerated
> to `neurons_conditional_full`. No neuron is now filtered by soma degree (the previously-dropped 24
> are back). This section's numbers describe the cap-free `neurons_conditional_full`; the old cap=16
> `neurons_conditional` set is still on disk for A/B. Raising `MAX_CHILDREN` widens the root-child
> ordinal one-hot (an input feature) — a fresh-training change; checkpoints trained at 16 are stale.

- **Analysis date:** 2026-07-23 (cap-free); original cap=16 analysis 2026-07-06
- **Raw source:** `~/Downloads/swc_simplified.tar.zstd` (26,490 neurons; degree-2 collapsed, tip-rooted, un-binarized) → `/Users/umer/Documents/neurons_simplified/swc_simplified/{train,val,test}`
- **Cell-type labels:** `~/Downloads/targets_cell_type.csv` (100% join coverage)
- **Trainable output:** `/Users/umer/Documents/neurons_conditional_full/{train,val,test}/<id>.swc` — soma-rooted, strictly binary, radii/types kept, cell-class integer embedded, **no root-degree cap** (local staging; rsync to `/scratch/guptau/neurons_conditional_full`, config `config/dataset/neurons_conditional_full.yaml`).
- **Reproduce:**
  - Clean: `conda run -n NEURO2 python preprocessing/prepare_conditional_dataset.py --out-root /Users/umer/Documents/neurons_conditional_full`
  - Losses on raw: `conda run -n NEURO2 python data_analysis/dataset_loss_accounting.py --max-children 23`
  - Verify output: `... dataset_loss_accounting.py --root /Users/umer/Documents/neurons_conditional_full --drop-classes "" --max-children 23`

---

## 1. Preprocessing pipeline & decisions

Raw `swc_simplified` had degree-2 nodes collapsed but was **tip-rooted** (soma = interior node, always id 1 / type 1) and **not binarized** (~3.4% of non-root nodes multifurcated). `preprocessing/prepare_conditional_dataset.py` (reusing `clean_trees.clean_swc_tree`, `root_mode="index"`) produces the trainable set per split (**splits preserved, no re-split**):

1. **Re-root at the soma** (node id 1) → soma-rooted, radiating outward.
2. **Binarize** non-root nodes: a 3-child node (undirected deg 4) gets one inserted node (lossless split); a ≥4-child node (deg ≥5) keeps its **2 thickest children** and deletes the rest (lossy prune).
3. **Keep radii/types** (no `--drop-attrs`) — so the "thickest" prune is real, not node-order arbitrary. Harmless downstream: `nx_graph_to_adj_pos` drops radius/type; only positions + adjacency reach the model.
4. **Drop rare classes** WM-P, MC, BPC (test-only, 21 neurons).
5. **No soma-degree cap.** `MAX_CHILDREN=23` (the one-hot ordinal width in `expansion.py`) = the max primary-dendrite count in the corpus, so the `rdeg > MAX_CHILDREN` guard drops nothing. (Was 16, which dropped 24 neurons; before that 10, which dropped 2,476.)
6. **No depth cap.**
7. **Embed cell class** as a `# cell_class N` header; `load_swc_graph` parses it to `G.graph['cell_class']`.

### Class-integer mapping (7 kept pyramidal types, cortical-layer order)
| id | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| type | 23P | 4P | 5P-IT | 5P-ET | 5P-NP | 6P-IT | 6P-CT |

Dropped entirely (no id): WM-P, MC, BPC.

---

## 2. At a glance (trainable `neurons_conditional_full`)

| | train | val | test | total |
|---|---|---|---|---|
| kept neurons | 22,773 | 2,529 | 1,167 | **26,469** |
| nodes/file: mean / median / p95 / max | 61 / 53 / 116 / 537 | 60 / 52 / 113 / 495 | 83 / 77 / 157 / 397 | — |
| soma-rooted max depth: mean / p95 / max | 8.2 / 13 / 35 | 8.1 / 13 / 30 | 8.8 / 14 / 25 | — |
| soma degree (#primary dendrites): mean / median / max | 7.7 / 8 / **23** | 7.7 / 8 / 17 | 8.0 / 8 / 16 | — |

Structural verification (on the output): **0 multifurcations, 0 degree-2 nodes, 0 broken/disconnected, all soma-rooted, root degree ≤ 23, splits preserved.** The **test** split holds systematically larger neurons (mean 83 vs 61 nodes). (6 files — 5 train, 1 test — have no explicit type-1 soma but are re-rooted at node id 1 regardless; this is a raw-data property, unchanged by the cap removal.)

### Cell-type composition (kept) — cap-free `neurons_conditional_full`
| type | id | total | (vs old cap=16) |
|---|---|---|---|
| 23P   | 0 | 9001 | +20 |
| 4P    | 1 | 6706 | — |
| 5P-IT | 2 | 3497 | +1 |
| 5P-ET | 3 |  686 | +3 |
| 5P-NP | 4 |  291 | — |
| 6P-IT | 5 | 3745 | — |
| 6P-CT | 6 | 2543 | — |
| **all** | | **26,469** | **+24** |

The +24 are the neurons previously dropped by the `>16` soma-degree cap (23P 20, 5P-ET 3, 5P-IT 1);
every class is now kept 100% of its labelled neurons. Splits: 22,773 train / 2,529 val / 1,167 test.
(The raw dataset held WM-P/MC/BPC 100% in test as an OOD group; per the drop decision these 21 are
excluded entirely — the only sample loss that remains.)

---

## 3. LOSS ACCOUNTING (report these in the paper)

All numbers computed analytically on the raw dataset with `dataset_loss_accounting.py --max-children 23` (drop-classes WM-P,MC,BPC), matching what the cleaning pipeline does.

### 3.1 SAMPLE loss — whole neurons dropped: **21 / 26,490 = 0.08%** (kept 26,469 = 99.92%)

| reason | count | which |
|---|---|---|
| rare class (WM-P/MC/BPC) | 21 | 17 / 3 / 1, all in test |
| soma degree > 23 | 0 | none — max observed degree is 23 |
| broken / disconnected | 0 | — |

The only remaining sample loss is the 21 rare-class neurons (§ class-map). **No neuron is dropped for soma degree.**

> **Paper note — the degree cap is removed.** Earlier versions filtered somata by primary-dendrite
> count because the model encodes each root child's sibling rank as a one-hot of width `MAX_CHILDREN`
> (an *input* feature; ranks ≥ width would collide). At cap **10** this dropped **2,476 neurons
> (9.35%)**, heavily cell-type-biased (5P-ET 31%, 23P 18%); at cap **16**, 24 neurons (0.09%). We now
> set `MAX_CHILDREN = 23 = the maximum primary-dendrite count observed in the corpus`, so the filter
> drops **0** neurons — the corpus is complete up to its natural maximum, with no arbitrary threshold
> to defend. Cost: the ordinal one-hot is 23-wide (7 more input bits than at 16; a fresh-training
> change — checkpoints trained narrower are stale). `feats_dim` unchanged: neuron configs (128/256)
> need only `≥ MAX_CHILDREN + 4 + conditioning dims` and have ample budget.

**Soma-degree distribution (all raw files; nothing dropped — max is 23):**
```
deg 3:  90     deg 8: 4957    deg 13:  299     deg 18:   4
deg 4: 721     deg 9: 3763    deg 14:  146     deg 19:   3
deg 5:2333     deg10: 2384    deg 15:   48     deg 20:   2
deg 6:4411     deg11: 1300    deg 16:   15     deg 23:   1
deg 7:5335     deg12:  643    deg 17:  14      (all kept; MAX_CHILDREN=23)
```

### 3.2 NODE loss — multifurcation pruning (within kept neurons)

Non-soma branch points with **≥4 children (deg ≥5)** can't be binarized by insertion, so the 2 thickest children are kept and the rest deleted. 3-child nodes (deg 4) are split losslessly.

- Of all non-soma multifurcations (27,684 total): **~77% lossless 3-child split, ~23% lossy prune.**
- **Branches (subtrees) deleted: ~19,213.**
- **Nodes deleted: ~20,663 ≈ 1.25% of kept nodes** (train 1.28%, val 1.22%, test 1.04%).
- Nodes **inserted** by lossless split: 21,301 (net node count ≈ +638, essentially unchanged).
- (Numbers are marginally higher than the old cap=16 set — the 24 re-added neurons contribute a few
  more multifurcations — but the per-node prune fraction is unchanged.)

> Because radii are kept, the deleted branches are genuinely the **thinnest** at each ≥5-way junction (a defensible morphological choice), unlike the old `--drop-attrs` pipeline where the surviving pair was arbitrary node order.

### 3.3 OPTIONAL loss — depth cap (NONE applied)

For reference, nodes beyond a hypothetical cap (soma-rooted depth): cap 12 → 1.16%, cap 16 → 0.50%, cap 20 → 0.19%.

### 3.4 Loss summary

| stage | unit | lost | % |
|---|---|---|---|
| rare-class drop | neurons | 21 | 0.08% of all |
| soma degree cap | neurons | 0 (removed; MAX_CHILDREN=23=max) | 0% |
| multifurcation prune | nodes | ~20,663 | ~1.25% of kept nodes |
| multifurcation prune | branches | ~19,213 | — |
| depth cap | nodes | 0 (not applied) | — |
| **Net trainable corpus** | **neurons** | **26,469 kept** | **99.92%** |

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

- Cleaning: `preprocessing/prepare_conditional_dataset.py` (class map + rare-class drop live here; `MAX_CHILDREN=23` guard drops nothing). Run with `--out-root /Users/umer/Documents/neurons_conditional_full`.
- Loss/structure: `data_analysis/dataset_loss_accounting.py` (`--max-children`, `--drop-classes`, `--root`).
- Cap constant: `graph_generation/method/expansion.py::MAX_CHILDREN = 23` — the root-child ordinal one-hot width. Mirrored as hardcoded copies in `preprocessing/prepare_conditional_dataset.py` and `data_analysis/prepare_neurons_final.py` (keep in lockstep).
- Class read: `utils/data_loading.py::load_swc_graph` → `G.graph['cell_class']`.
- Config: `config/dataset/neurons_conditional_full.yaml`; used by `config/neuron_type_conditional_run.yaml`. Run via `python main.py -cn neuron_type_conditional_run` (override `dataset=neurons_conditional` for the old cap=16 set).
- Cell-class conditioning is wired end-to-end (one-hot → Linear in `egnn_so2.py`, budget subtracted from `avail_feats_dim`); see the model config's `num_classes`/`class_hidden_dim` and `docs/` conditional-generation notes.
- Verified: full pytest suite green at `MAX_CHILDREN=23` (`tests/test_so2_invariance.py`, forward/dataset/validation tests), cap-free regeneration (26,469 written, 0 soma-degree drops), and structural re-check on the output (0 multifurcations, root degree ≤ 23, splits preserved).
