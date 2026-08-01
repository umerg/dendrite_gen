# Tree Dataset Stats & Loss Accounting — `trees_genus_d{10,15,20}`

**Living document.** Tracks the structure of the trainable biological-tree datasets and,
importantly, **every tree and node we lose in preprocessing** — needed for honest reporting in
the paper. Companion to `docs/NEURON_DATASET_STATS.md`; same sections, same purpose.

Trees are the second exemplar of the general claim: the method targets **binary branching
morphologies in nature**, not just neurons.

- **Analysis date:** 2026-07-31
- **Raw source:** `/Users/umer/Documents/trees/{Graphs.tar.zst,labels.csv}` — the "Graph" subset of
  **BioDiv-3DTrees** (Forest Gap Experiment, Biodiversity Exploratories; CC BY 4.0). Identical
  payload to `~/Downloads/tree_dataset_raw.zip`, which additionally ships `README.md` +
  `MANIFEST.TXT`. 3,386 `Graph/<treeID>_cor_graph.graphml` QSM graphs + 4,952 label rows.
- **Genus labels & splits:** `labels.csv` (`species` → genus; official `split` column). 100% join
  coverage — all 3,386 graph treeIDs are present in `labels.csv`.
- **Trainable output:** `/Users/umer/Documents/trees_genus_d{10,15,20}/{train,val,test}/<treeID>.swc`
  — base-rooted, strictly binary away from the root, radii kept, genus integer embedded (local
  staging; rsync to `/scratch/guptau/trees_genus_d{10,15,20}`; configs
  `config/dataset/trees.yaml` (=d10), `trees_genus_d15.yaml`, `trees_genus_d20.yaml`).
- **Reproduce:**
  - Build all three: `conda run -n NEURO2 python preprocessing/prepare_tree_dataset.py --max-depth 10 15 20`
  - Verify output: `... prepare_tree_dataset.py --verify /Users/umer/Documents/trees_genus_d10`
  - C₀ scale: `N_GRAPHS=1000 conda run -n NEURO2 python tests/analyse_c0_distribution.py --data-dir /Users/umer/Documents/trees_genus_d10/train --pos-scale 1.0 --axis z`

---

## 1. Preprocessing pipeline & decisions

The raw graphs are **full cylinder resolution**, not branching skeletons: mean **10,496 nodes/tree**
(median 7,311, max 73,538; 35.35 M nodes total), of which **~66% are degree-2 chain nodes**. They
are **undirected with no parent pointers** (`pos_x/pos_y/pos_z/radius` only), so rooting and
orientation are ours to choose. `preprocessing/prepare_tree_dataset.py` (reusing
`clean_trees.clean_swc_tree`, `root_mode="parent"`) produces each trainable set
(**splits preserved from `labels.csv`, no re-split**):

1. **Root at the QSM base cylinder** = graphml node `'0'`, then BFS-orient. The dataset README
   defines cylinder id 1 as the base of the tree. Present in 3,386/3,386 graphs.
2. **Collapse degree-2 chains** → the branching skeleton (mean 3,544 nodes, depth mean 71.7).
3. **Trim to `max_depth`** (10 / 15 / 20) in collapsed edge hops.
4. **Binarize** non-root nodes: a 3-child node gets one inserted node (lossless split); a
   ≥4-child node keeps its **2 thickest children** and deletes the rest (lossy prune).
5. **Keep radii/types** (`keep_attrs=True`) — so the "thickest" prune is real, not node-order
   arbitrary. Harmless downstream: `nx_graph_to_adj_pos` drops radius/type; only positions +
   adjacency reach the model.
6. **Drop rare tail genera** Tilia, Prunus, Ulmus (16 trees) and trees with a blank `split` (5).
7. **Embed genus** as a `# cell_class N` header; `load_swc_graph` parses it to
   `G.graph['cell_class']`. `# cell_type`, `# species`, `# tree_id` are also written as inert
   provenance comments.

### Genus-integer mapping (6 kept genera, frequency order)
| id | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| genus | Fagus | Quercus | Acer | Carpinus | Fraxinus | Betula |

Dropped entirely (no id): Tilia, Prunus, Ulmus. Source of truth:
`utils.data_loading.TREE_GENUS_NAMES`, surfaced to the trainer via `dataset.class_names`.

> **Why not broadleaf/conifer?** `labels.csv` has a `type` column, but **QSMs were only
> reconstructed for broadleaf trees** — all 3,386 graphs are broadleaf, and the 1,543 conifers have
> no graph at all. The `type` axis therefore carries zero signal, and `species`→genus is the only
> usable class axis.

> **Rooting decision — node `'0'`, not lowest-z.** A lowest-z heuristic disagrees with node `'0'`
> for **24 of 3,386** graphs, and in those cases the heuristic is wrong: the lowest-z node is an
> interior branch node (degree 2 or 3) in 17 of them, and sits up to **85.3 m** below the trunk
> base on stray QSM cylinders (also 11.2 m, 4.6 m). Rooting at an interior node re-orients every
> edge on the path to the real base, inverting the topology of a large part of the tree. Switching
> from lowest-z to node `'0'` removed all 11 root-degree-3 trees and reduced max root degree from
> 3 to 2 — independent confirmation that node `'0'` is the true base.

---

## 2. At a glance (all three depths, 3,368 trees each)

| | d10 | d15 | d20 |
|---|---|---|---|
| kept trees (train / val / test) | 2,695 / 337 / 336 | same | same |
| nodes/file: mean / median / p95 / max | 84 / 72 / 182 / 378 | 224 / 176 / 561 / 1,456 | 445 / 335 / 1,203 / 3,056 |
| max depth: mean / p95 / max | 10.8 / 12 / 14 | 16.1 / 18 / 20 | 21.2 / 23 / 26 |
| total nodes | 283,651 | 754,175 | 1,499,019 |
| size vs d10 | 1.0× | 2.7× | 5.3× |
| `pos_scale_factor` | 0.45 | 0.34 | 0.29 |

Realized depth exceeds the cap by up to 4 hops because `normalize_high_degree` inserts split
nodes *after* trimming (a 3-child node at depth `cap-1` gains a child at `cap`, whose own children
sit deeper). This is expected, not a cap violation.

Per-split node counts are near-identical across splits (d10 train/val/test means 84/84/83), i.e.
the stratified split did not concentrate large trees — unlike the neuron corpus, where the test
split holds systematically larger neurons.

**Root degree:** 1 for **3,363** trees, 2 for **5**. The root is exempt from binarization, exactly
as a neuron soma is (`normalize_high_degree` filters on `parent > root_parent_value`), so a tree
that forks at the trunk base keeps degree 2. Far below `MAX_CHILDREN = 23`, so the root-child
ordinal one-hot drops nothing.

Structural verification on the written output (`--verify`, all three sets): **0 non-root
multifurcations, 0 degree-2 nodes, 0 disconnected, single root everywhere, root degree ≤ 2,
`cell_class` present and in 0..5, splits preserved.**

### Genus composition (kept, identical for all three depths)
| genus | id | train | val | test | total | % |
|---|---|---|---|---|---|---|
| Fagus | 0 | 2,377 | 298 | 295 | **2,970** | 88.18 |
| Quercus | 1 | 142 | 18 | 18 | **178** | 5.28 |
| Acer | 2 | 57 | 7 | 8 | **72** | 2.14 |
| Carpinus | 3 | 52 | 6 | 7 | **65** | 1.93 |
| Fraxinus | 4 | 47 | 6 | 6 | **59** | 1.75 |
| Betula | 5 | 20 | 2 | 2 | **24** | 0.71 |
| **all** | | **2,695** | **337** | **336** | **3,368** | 100 |

---

## 3. LOSS ACCOUNTING (report these in the paper)

### 3.0 Upstream availability (not our loss)

| stage | trees | note |
|---|---|---|
| labelled in `labels.csv` | 4,952 | full BioDiv-3DTrees corpus |
| `selected4QSM` | 3,409 | broadleaf only — **conifers get no QSM** |
| `QSMsuccess` | 3,386 | 23 QSM failures |
| graphs in the archive | **3,386** | our actual input |

The 1,543 conifers and 23 QSM failures are unavailable upstream, not discarded by us.

### 3.1 SAMPLE loss — whole trees dropped: **18 / 3,386 = 0.53%** (kept 3,368 = 99.47%)

| reason | count | which |
|---|---|---|
| rare tail genus | 16 | Tilia 13, Prunus 2, Ulmus 1 |
| blank `split` in `labels.csv` | 5 | 3 of which are also tail genus |
| overlap correction | −3 | counted once |
| unlabelled / unknown genus | 0 | 100% join coverage |
| disconnected / broken | 0 | every graph is a single connected tree |
| errors | 0 | — |

Net: 16 + 5 − 3 = **18 dropped**. The 2 non-tail blank-split trees are `AEW42_G_240`
(Acer_campestre) and `AEW47_G_192` (Quercus_rubra).

> **Paper note — the tail-genus drop.** Tilia (13), Prunus (2) and Ulmus (1) cannot support a
> conditional class: they are 0.47% of the corpus and would contribute ≤2 validation trees each.
> They are dropped rather than pooled into an "Other" class, which would be a morphologically
> incoherent mix. Alternative considered: keep all 9 genera (zero sample loss, 3 extra one-hot
> bits) — cheap to revisit, since only `TREE_GENUS_NAMES` and `DROP_GENERA` would change.

### 3.2 NODE loss — depth cap (the dominant loss)

The collapsed branching skeleton holds **11,934,559 nodes**. The cap is what discards most of it:

| cap | skeleton kept | discarded | % of skeleton lost |
|---|---|---|---|
| 10 | 288,619 | 11,645,940 | **97.58%** |
| 15 | 770,081 | 11,164,478 | **93.55%** |
| 20 | 1,533,731 | 10,400,828 | **87.15%** |

This is a deliberate, reported trade-off, not an artifact: even d20 keeps only ~13% of the
available branching structure, and an uncapped set would average 3,544 nodes/tree at depth ~72
(max 27,830 nodes / depth 326) — far beyond what the reduction-sequence pipeline handles.

### 3.3 NODE loss — multifurcation pruning (within kept trees)

Non-root branch points with **≥4 children** can't be binarized by insertion, so the 2 thickest
children are kept and the rest deleted. 3-child nodes are split losslessly.

| cap | lossless 3-child splits | lossy ≥4-child junctions | % lossless | branches deleted | nodes deleted | % of kept nodes |
|---|---|---|---|---|---|---|
| 10 | 7,364 | 859 | 89.6% | 1,880 | 12,332 | **4.35%** |
| 15 | 17,610 | 1,578 | 91.8% | 3,404 | 33,516 | **4.44%** |
| 20 | 33,516 | 2,484 | 93.1% | 5,293 | 68,228 | **4.55%** |

These counters are **exact, not estimated**: they are measured by replaying
`normalize_high_degree`'s prune pass on the actual trimmed tree (`measure_prune`), including its
dynamic `len(kids) <= 3` recheck, so multifurcations sitting inside an already-deleted subtree are
not counted. Validated by the identity
`capped_skeleton − deleted + inserted == final node count`, which held for **all 3,368 trees × 3
caps with 0 mismatches**.

The prune fraction (~4.4%) is ~3.5× the neuron corpus's 1.25%, because tree QSMs multifurcate
more often than dendrites.

> Because radii are kept, the deleted branches are genuinely the **thinnest** at each ≥4-way
> junction (a defensible morphological choice), unlike the legacy `--drop-attrs` pipeline where the
> surviving pair was arbitrary node order. **103 raw cylinders across 12 trees have `radius = 0`**;
> they only matter as tie-break material in that ranking.

### 3.4 Loss summary

| stage | unit | lost | % |
|---|---|---|---|
| conifers + QSM failures | trees | 1,566 | upstream, unavailable |
| rare-genus + blank-split drop | trees | 18 | 0.53% of graphs |
| disconnected / errors | trees | 0 | 0% |
| depth cap (d10 / d15 / d20) | nodes | 11.65 M / 11.16 M / 10.40 M | 97.6% / 93.6% / 87.2% of skeleton |
| multifurcation prune (d10/d15/d20) | nodes | 12,332 / 33,516 / 68,228 | 4.35% / 4.44% / 4.55% of kept |
| **Net trainable corpus** | **trees** | **3,368 kept** | **99.47%** |

---

## 4. C₀ offset distribution — per-depth scaling (each dataset needs its own)

`tests/analyse_c0_distribution.py --axis z` on each train split (the real pipeline: load → scale →
depth reduction → `precompute_full_geometry` → `global_to_local`). Rule:
`pos_scale_factor` = mean of the three **raw** per-axis stds (2 s.f.); `prior_std_pos` = raw ÷ scale.

| depth | raw per-axis std | `pos_scale_factor` | `prior_std_pos` | mean axis std | \|C\| mean | anisotropy |
|---|---|---|---|---|---|---|
| d10 | [0.278, 0.244, 0.833] | **0.45** | **[0.62, 0.54, 1.85]** | 1.005 | 1.001 | 3.41 |
| d15 | [0.257, 0.226, 0.550] | **0.34** | **[0.76, 0.66, 1.62]** | 1.013 | 1.046 | 2.44 |
| d20 | [0.243, 0.214, 0.424] | **0.29** | **[0.84, 0.74, 1.46]** | 1.012 | 1.095 | 1.99 |
| *(legacy `small_trees`)* | [0.285, 0.256, 0.851] | *0.5* | *[0.57, 0.51, 1.70]* | *0.928* | — | — |

All three pass the script's own check (`prior_std_pos OK`, per-axis |Δ|/std ≤ 0.005).
N_GRAPHS = 1000 for d10/d15, 400 for d20 (~5× per-tree cost).

**The parameters do NOT transfer between depths** — unlike the neuron sets, where 45.1 carried
over unchanged. Deeper caps add short, more isotropic twig offsets, so the axial std falls
(0.833 → 0.550 → 0.424) and anisotropy drops from 3.4 to 2.0. A shallow cap keeps only the long
trunk/major-branch jumps.

Two properties worth noting for the flow prior:
- **Trees are far more anisotropic than neurons** (neuron ratio ≈1.4, tree d10 ≈3.4). An isotropic
  `prior_std` is a poor fit; `prior_std_pos` is doing real work here.
- **The axial axis is heavy-tailed** (kurtosis 8.97 / 12.18 / 13.94 for d10/d15/d20 vs ~1.4 for
  forward), and the offsets are **not mean-zero** (data mean [0.268, 0.001, 0.460] at d10). The
  script's `RECOMMENDED prior_mean_pos` output quantifies the DC offset the flow must transport on
  every sample; unused so far, same as for neurons.
- **Expansion-label balance is near-even** (expand fraction 0.494 / 0.498 / 0.499) — healthier
  than the neuron corpus's 0.428.

---

## 5. Reproducibility & touch points

- Build: `preprocessing/prepare_tree_dataset.py` (genus map + tail drop + rooting rule live here;
  `--max-depth` takes several caps and builds them in ONE pass over the archive, since graphml
  parsing dominates). Streams `Graphs.tar.zst` via `zstandard` — never extracts the ~15 GB.
- Cleaning primitives: reused from `preprocessing/clean_trees.py` (`clean_swc_tree`,
  `collapse_degree2`, `trim_to_max_depth`, `SWCTree`, `write_swc`) — no new cleaning logic.
- Structure/loss re-check: `prepare_tree_dataset.py --verify <root>`; per-tree stats are written to
  `<root>/dataset_stats.csv` (one row per tree per cap).
- Genus names: `utils/data_loading.py::TREE_GENUS_NAMES` → `dataset.class_names` in the dataset
  config → `graph_generation/training.py::Trainer.class_names` (falls back to `CELL_CLASS_NAMES`
  when a dataset declares no `class_names`, so neuron runs are untouched).
- Class read: `utils/data_loading.py::load_swc_graph` → `G.graph['cell_class']`; `main.py` raises if
  conditioning is on and any graph lacks it.
- Configs: `config/dataset/trees.yaml` (=d10), `trees_genus_d15.yaml`, `trees_genus_d20.yaml`;
  run config `config/tree_genus_conditional_run.yaml` (`so2_axis: [0,0,1]`, `num_classes: 6`,
  `class_hidden_dim: 16`, `per_cell_class_min_count: 15`). Its defaults list puts `_self_` **last**
  so its `diffusion.prior_std_pos` override beats `flow_v10.yaml`'s neuron default.
- Run: `python main.py -cn tree_genus_conditional_run` (d10). For d15/d20 override the dataset,
  `prior_std_pos` **and** batch size together — see the header comment in that config.
- `MAX_CHILDREN = 23` (`graph_generation/method/expansion.py`) is imported, not copied, by the
  build script's root-degree guard; it drops nothing here (max root degree 2).

---

## 6. Known limitations

- **Severe genus imbalance: 88.2% Fagus.** Conditional generation will be dominated by one class.
  This is a property of the corpus (a beech-dominated forest experiment) and should be stated
  rather than engineered around.
- **Per-genus metrics are only viable for Fagus and Quercus.** Validation-split counts are Fagus
  298, Quercus 18, Acer 7, Carpinus 6, Fraxinus 6, Betula 2. At the neuron default
  `per_cell_class_min_count: 20` only Fagus would qualify; the tree run config uses **15** so
  Quercus is included. The other four genera cannot support stratified distribution metrics
  (PCA/kNN degenerate) at this corpus size.
- **The legacy `small_trees` set is not a like-for-like baseline.** Same cap 10, but: no class
  labels; an unusable split (train 3,656 / val 406 / **test 1**); 893 of its 4,062 treeIDs (22%)
  absent from `labels.csv` entirely (it came from an earlier release with different tree
  numbering); and `--drop-attrs` made its ≥4-way prune arbitrary. The d10 set here is
  deliberately **not** bit-identical to it.
- **Only broadleaf trees exist in this corpus**, so no conifer-vs-broadleaf generalisation claim
  can be made from it.
- **The C₀ scale factors are depth-specific** and must be re-measured if the cap changes.
