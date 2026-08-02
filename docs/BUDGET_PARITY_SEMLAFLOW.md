# Training-Budget Parity — iterative expansion vs SemlaFlow

**Living document.** Defines what "an epoch" means for each method, measures the exact
per-dataset accounting, and fixes the protocol used to train both methods on a comparable
budget. Companion to `docs/NEURON_DATASET_STATS.md` and `docs/TREE_DATASET_STATS.md`
(corpora) — this doc is about *how long each method trains on them*.

- **Analysis date:** 2026-08-02
- **Our side:** `main.py` + `PrecomputedRedDataset`, deterministic depth reduction
  (`reduction.type: depth`, `mode: deterministic`, `contract_root: False`), flow matching
  (`config/diffusion/flow_v10.yaml`).
- **Baseline:** SemlaFlow (`/Users/umer/Documents/semla-flow`), `semlaflow.train`, neuron
  path (`--dataset neurons_conditional`), cost-bucketed batching, equivariant OT, EMA,
  self-conditioning.
- **Reproduce every number here:**
  `conda run -n NEURO2 python data_analysis/budget_accounting.py --dataset {neurons,trees_d10,trees_d15,trees_d20}`
  (add `--sample N` to subsample the reducer pass; the SemlaFlow-side numbers are exact regardless).

---

## 1. "Epoch" is bookkeeping, not statistics — on both sides

Both methods are flow matching, so **neither has a finite dataset in the (x, t, ε) sense**.

- **SemlaFlow** draws `t ~ Beta(2, 1)` *per molecule, inside the collate function*, every time
  the molecule is fetched (`semlaflow/data/interpolate.py:229`, distribution at `:186`),
  together with a fresh prior sample and a fresh equivariant-OT alignment
  (`interpolate.py:277-279`). So it is **one noise level per molecule per epoch** — not several —
  and molecule *i* at epoch 7 and epoch 8 are different training examples. Over 300 epochs each
  neuron is seen with 300 independent `(t, ε, OT-pairing)` draws. Validation uses a fixed
  `t = 0.9` (`train.py:359`).
- **Ours** does the same thing one level down: `_sample_time(num_graphs)`
  (`graph_generation/diffusion/flow.py:69`, called at `:123`) draws one **uniform** `t` per graph
  in the batch, fresh noise each visit.

So "epoch" on either side means only: *one pass over the items in the dataset* — and "item"
means different things on the two sides (§2). The quantities that actually compare are
**(a) denoising events per node, (b) gradient steps, (c) wall-clock**, in that order of
priority (§5.0).

## 2. What one item is, and why one pass is the same supervision on both sides

| | SemlaFlow | Ours |
|---|---|---|
| item | one graph | one **(graph, reduction level)** pair |
| items per source graph | 1 | `L` = levels in its reduction sequence (= its max root-rooted depth) |
| targets per item | all `N` coordinates (+ types, bonds) | the leaves created by that level's contraction |
| targets per source-graph pass | `N` | `N − 1` (measured, §3) |

The dataset that our trainer actually iterates is the **flattened list of levels**
(`PrecomputedRedDataset.samples`), reshuffled on every pass (`reduction_dataset.py:234-240`) —
so a well-defined epoch already exists in the code, it is just never counted.

The supervision identity matters: `select_training_leaf_indices`
(`graph_generation/method/helpers.py:1311`) keys the loss off `new_leaf_idx_from_next`, so
**each non-root node's offset is a denoising target exactly once per reduction sequence**
(measured: 0.984 / 0.988 / 0.996 / 0.998 targets per node across the four corpora — the deficit
is the root). SemlaFlow denoises every node once per epoch. Therefore:

> **One epoch = every node position denoised once, on both sides.** That is the definition
> used throughout this doc and the one to state in the paper.

What differs is *context cost*: we re-encode every surviving node at every coarser level, so
one pass costs 4.5–7.8× more node forward-passes than the baseline's single pass.

## 3. Our side — measured reduction-sequence accounting

Full train splits, real reducer (not an estimate). `items/epoch = Σᵢ Lᵢ`.

| | neurons | trees d10 | trees d15 | trees d20 |
|---|---|---|---|---|
| train graphs | 22,773 | 2,695 | 2,695 | 2,695 |
| nodes/graph (mean) | 60.98 | 84.34 | 224.86 | 446.58 |
| **levels/graph** mean (med / p95 / max) | **8.19** (8 / 13 / 35) | **10.79** (11 / 12 / 14) | **16.08** (16 / 18 / 20) | **21.23** (21 / 23 / 25) |
| node-visits/graph (× graph size) | 379.1 (**6.22×**) | 375.5 (**4.45×**) | 1,373.9 (**6.11×**) | 3,498.6 (**7.83×**) |
| supervised targets/graph | 59.98 (= N−1) | 83.34 (= N−1) | 223.86 (= N−1) | 445.58 (= N−1) |
| **items / epoch** | **186,435** | **29,080** | **43,324** | **57,213** |
| node-visits / epoch | 8.63 M | 1.01 M | 3.70 M | 9.43 M |
| mean nodes / item | 46.31 | 34.80 | 85.47 | 164.80 |
| steps/epoch at the *current* `training.batch_size` | 728 (B=256) | 114 (B=256) | 338 (B=128) | 894 (B=64) |

Levels per graph equal the root-rooted max depth exactly: deterministic depth reduction strips
*all* deepest cherries each level, so depth drops by exactly 1 per level. (Neuron mean 8.19
matches the 8.2 in `NEURON_DATASET_STATS.md` §2; tree means match `TREE_DATASET_STATS.md` §2.)

## 4. SemlaFlow side — measured epoch accounting

Bucket limits as proposed in §7, `bucket_cost_scale: quadratic`, `batch_cost` raised for d15/d20
(§7.4), **with both baseline-config fixes applied: the dead-bucket clamp (§7.5) and
`drop_last=False` (§7.8)**. Those two are what make "graphs seen / epoch" 100% and therefore make
a nominal epoch an actual pass.

| | neurons | trees d10 | trees d15 | trees d20 |
|---|---|---|---|---|
| `--batch_cost` | 1024 | 1024 | 4096 | 8192 |
| **steps / epoch** | **588** | **202** | **492** | **796** |
| graphs seen / epoch | 22,773 (**100%**) | 2,695 (**100%**) | 2,695 (**100%**) | 2,695 (**100%**) |
| graphs / step | 38.7 | 13.3 | 5.5 | 3.4 |
| nodes / step | 2,362 | 1,125 | 1,232 | 1,511 |
| node-visits / epoch | 1.39 M | 0.23 M | 0.61 M | 1.20 M |
| dense pair-interactions / epoch | 104.2 M | 26.1 M | 213.8 M | 922.5 M |
| peak dense-bond tensor / batch | 6 MB | 6 MB | 43 MB | **187 MB** |
| 300 epochs = | **176,400 steps** | **60,600 steps** | **147,600 steps** | **238,800 steps** |

Stock SemlaFlow (`drop_last=True`, no clamp) instead gives 580 / 195 / 485 / 788 steps/epoch and
sees only **98.4% / 88.6% / 90.0% / 89.0%** of the corpus per epoch, including the permanently
dead buckets of §7.5. Every number in this doc uses the fixed config.

## 5. The parity protocol (adopted)

### 5.0 Why steps are not the invariant

A gradient step is not a fixed amount of training: its content is `batch_size` items. Matching
steps *alone* would hand one side a 6× data-exposure advantage, because our batch is 256 items
(≈31 graphs × 8.2 levels) while SemlaFlow's cost-bucketed batch averages 39 graphs of which each
contributes one item. Steps are an **optimisation** control (how many updates the optimiser
gets), not a budget.

The quantity that *is* an invariant — and the one the N−1 identity of §2 lets us state exactly —
is **coordinate-denoising events per node**:

> per epoch, **every node's position is a flow-matching target exactly once, on both sides**.

So the budget hierarchy is:

| rank | axis | why | how it is set |
|---|---|---|---|
| 1 | **denoising events per node** (= epochs `E`) | the statistical invariant; the amount of supervision each method receives | fixed to `E = 300`, SemlaFlow's default |
| 2 | **gradient steps** `S` | optimisation advantage; more updates at equal exposure is a real edge | equalised via our `batch_size` (5.1), or reported as a ratio (5.2) |
| 3 | **GPU-hours** | the "comparable training time" claim | measured, never derived (5.4) |

### 5.1 Option B (recommended) — equalise 1 and 2 together

Set `E = 300` on both sides and pick our batch size so steps/epoch agree; then matched epochs
**and** matched steps hold simultaneously, and wall-clock is the only free variable.
`B* = items_per_epoch / steps_per_epoch(SemlaFlow)`:

| | neurons | trees d10 | trees d15 | trees d20 |
|---|---|---|---|---|
| **`training.batch_size`** | **317** | **144** | **88** | **72** |
| **`training.num_steps`** | **176,400** | **60,600** | **147,600** | **238,800** |
| epochs, both sides | 300 | 300 | 300 | 300 |
| coordinate-denoising events (train total, **both sides**) | 416.6 M | 68.2 M | 181.8 M | 361.1 M |

The event counts are identical on both sides only because §7.5 + §7.8 close the baseline's
per-epoch leakage; with stock SemlaFlow they would be 410 M / 60.4 M / 163.6 M / 321.4 M.

### 5.2 Option C — keep our current batch sizes, match epochs only

If `B*` is impractical (memory, throughput, an already-tuned LR/batch pair), match axis 1 and
*report* the step ratio. Only `num_steps` changes. **This is the option we adopted**, and it is
what the shipped configs implement:

| | neurons | trees d10 | trees d10 | trees d15 | trees d20 |
|---|---|---|---|---|---|
| config family | `parity_neurons_*` | `parity_trees_d10_*` | `depth_trees_d10_*` | `depth_trees_d15_*` | `depth_trees_d20_*` |
| `training.batch_size` | 256 | 256 | 128 | 128 | 128 |
| `training.num_steps` (E = 300) | 218,500 | 34,100 | 68,200 | 101,500 | 134,100 |
| our steps ÷ SemlaFlow steps | 1.24× | **0.56×** | 1.13× | **0.69×** | **0.56×** |

Where we take *fewer* updates at equal exposure (0.56–0.69×) the comparison is conservative for
us and can be stated as such; where we take more (1.13–1.24×) it must be disclosed.

The `depth_*` family holds `batch_size: 128` and `validation.batch_size: 32` fixed across all
three depths so the d10-vs-d15-vs-d20 comparison varies depth only — which is why d10 appears
twice (at 256 paired with the neuron runs, at 128 inside the sweep) and why d20 runs at 128
rather than the 64 its per-depth guidance suggests. The d20 configs carry the E-preserving
fallback (`batch_size: 64`, `num_steps: 268200`) in a comment in case it OOMs.

**Option A — match steps at the current batch size — is the one to avoid.** It silently varies
exposure: 242 / 534 / 436 / 267 epochs for neurons / d10 / d15 / d20, i.e. up to 1.8× the
baseline's supervision on the small tree corpora, purely as an artifact of their corpus size and
SemlaFlow's cost-bucketed batching.

**Where we are today:** the shipped configs use `num_steps: 20000` — 27 epochs vs 300 on
neurons, and 33% / 14% / 8% of the matched step budget on d10 / d15 / d20. Every current result
is on the *undertrained* side of parity.

### 5.3 What "one epoch" does and does not equalise

Equal (by construction, measured in §3):

- **coordinate-denoising events per node: 1 per epoch on both sides.** Over a 300-epoch run that
  is 417 M / 68 M / 182 M / 361 M targets for neurons / d10 / d15 / d20.

Not equal — state these, do not try to engineer them away:

| | SemlaFlow | ours |
|---|---|---|
| context encodings per node/epoch | 1 | **4.5–7.8** (surviving nodes re-encoded at every coarser level, §3) |
| topology supervision per graph/epoch | `N²` dense bond entries | `N−1` binary expansion labels (adjacency comes free from the reduction) |
| loss reduction | **per molecule**: `(mse*mask).mean(dim=(1,2))` then batch-mean (`fm.py:748-755`), so a node in a large graph carries less gradient, and padding to the batch max scales it further | **per target**: `F.mse_loss(v_pred, v_target)` over all leaves in the batch (`flow_v.py:198`), so every node carries equal gradient |
| effective epochs | **300 exactly, once §7.5 + §7.8 are applied.** Stock SemlaFlow gives 295.2 / 265.9 / 270.0 / 267.0, and **size-dependent**: its worst-populated bucket is seen only 70–87% of epochs | exactly 300; nothing is dropped |

That last row is why §7.8 is not optional. Left stock, the baseline's nominal 300 epochs is
~266–295 effective, unevenly by graph size — so "equal epochs" would be false by 2–12% in the
baseline's disfavour, and the residual would be a *size-biased* deficit that no single scalar
correction can undo. Turning `drop_last` off costs 1–4% more steps per epoch and makes the
invariant exact.

### 5.4 Compute disclosure — the two currencies disagree, so measure GPU-hours

Per matched epoch:

| currency | neurons | d10 | d15 | d20 | who it favours |
|---|---|---|---|---|---|
| node forward-passes | 6.2× | 4.5× | 6.1× | 7.8× | SemlaFlow cheaper |
| pair/edge interactions | 12.1× | 25.8× | 57.8× | 97.8× | ours cheaper |

Ours is `O(n)` per level (tree edges + linear attention with 8 global tokens); SemlaFlow is
`O(n²)` (dense attention + all-pairs bonds). One accounting says we use ~6× more compute, the
other says ~50× less. Neither is a FLOP count — **report measured GPU-hours** and cite both
columns so no reviewer can claim the budget was chosen to flatter either side.

### 5.5 Practical knobs when stretching to the matched budget

- `scheduler_T_max: ${training.num_steps}` already tracks `num_steps` — nothing to change.
  SemlaFlow uses constant LR + 10k warm-up; keep each method's own tuned schedule and say so.
- **Rescale validation.** `validation.interval: 200` with `eval_mode: rollout` means ~880
  validations over a 176k-step run, which would dominate wall-clock and corrupt the GPU-hours
  comparison. Use `interval ≈ num_steps / 100` (SemlaFlow validates every 20 epochs = 11.8k
  steps at neurons) and **exclude validation time from the reported GPU-hours on both sides**.
- Log `training/step_time` (already logged) and SemlaFlow's per-epoch time; report
  hours-to-best-checkpoint as well as hours-to-end.

## 6. What to put in the paper

One table, both methods, per dataset, in budget-hierarchy order:
`epochs (= denoising events per node) · total denoising events · gradient steps · items seen ·
GPU-hours (train only) · GPU-hours to best checkpoint`.

The claim sentence: *"We define an epoch as every node position being denoised once. Our
iterative expansion emits `L` training items per graph (one per reduction level, mean 8.2–21.2
by dataset), but the levels partition the graph's nodes: each non-root node is a flow-matching
target exactly once per pass, as it is for the baseline. Both methods are trained for 300 such
epochs, and our batch size is set so that both also take the same number of gradient steps."*

Footnote the asymmetries from §5.3 (context encodings 4.5–7.8×, `N²` bond targets vs `N−1`
expansion labels, per-molecule vs per-node loss reduction) and the two compute currencies from
§5.4. If Option C was used instead, replace the last clause with the step ratio and its
direction.

## 7. Running the SemlaFlow baseline on larger corpora (trees d15/d20)

The SWC adapter is already generic — `semlaflow/data/swc.py` reads tree SWCs and the
`# cell_class N` header — so no new data code is needed. What must change is the **size cap**
and the **buckets**, plus the items below.

### 7.1 The preprocessing cap (drops graphs)

`semlaflow/preprocess_neurons.py:25` `MAX_ATOMS = 256`, filter at `:43`: anything larger is
**silently dropped** (it prints a count). Pass `--max_atoms` explicitly:

```bash
python -m semlaflow.preprocess_neurons \
    --input_dir  /path/to/trees_genus_d20 \
    --output_dir /path/to/trees_genus_d20/smol \
    --max_atoms 3100          # > corpus max (3056); default 256 would drop 62% of the trees
```

It prints `train size: min=…, max=…` and the measured `coord_std` — both are needed below.
What the default cap would cost on each corpus (graphs dropped / nodes dropped):

| cap | neurons | d10 | d15 | d20 |
|---|---|---|---|---|
| 256 | 10 (0.04%) / 0.27% | 28 (1.04%) / 3.62% | 852 (31.6%) / 59.0% | 1,673 (62.1%) / 87.0% |
| 512 | 1 / 0.04% | 0 | 173 (6.4%) / 19.6% | 827 (30.7%) / 61.0% |
| 1024 | 0 | 0 | 10 (0.37%) / 1.90% | 200 (7.4%) / 23.7% |
| 2048 | 0 | 0 | 0 | 19 (0.71%) / 3.82% |

**This is a corpus-equality issue, not just a config knob.** Our method trains on all graphs;
if the baseline drops any, the two are not evaluated on the same data. Either preprocess
uncapped, or cap **both** methods identically and report it.

### 7.2 The size-embedding cap (crashes if too small)

`train.py:31` `DEFAULT_MAX_ATOMS = 256` → `SemlaGenerator(max_atoms=…)` →
`torch.nn.Embedding(max_atoms, size_emb)` (`models/semla.py:799`), indexed by the *real* node
count `n_atoms = atom_mask.sum(-1)` (`semla.py:858-860`). There is **no bounds check** (there
is even a `TODO` on `:859`), so any graph with `n_atoms ≥ max_atoms` is an out-of-range
embedding index → device-side assert, not a friendly error.

```bash
python -m semlaflow.train --dataset trees_d20 --max_atoms 3100 ...
```

Rule: `--max_atoms > corpus max node count`, and it must match what preprocessing kept. Cost is
negligible (`max_atoms × size_emb` = 3100 × 64 ≈ 0.2 M params).

### 7.3 The buckets ("bins")

Bucket limits live in `semlaflow/scriptutil.py` (`NEURON_BUCKET_LIMITS:33`,
`NEURON_CONDITIONAL_BUCKET_LIMITS:43`) and are selected in `train.py:build_dm` (`:242-262`),
then passed as `bucket_limits` to `GeometricInterpolantDM`. Rules:

1. Sorted ascending; **top bucket ≥ corpus max**, else `SmolDM.__init__` raises
   (`data/datamodules.py:34-41`).
2. Roughly geometric spacing — padding waste inside a bucket is quadratic in the gap.
3. Every bucket should hold **≥ 2× its own batch size** (see 7.5).

Measured-corpus proposals (used for §4; add to `scriptutil.py` alongside the neuron ones):

```python
# max 537 (neurons_conditional_full — the current top bucket of 256 drops 10 train graphs)
NEURON_CONDITIONAL_FULL_BUCKET_LIMITS = [24, 40, 56, 72, 96, 128, 160, 200, 256, 320, 400, 544]
TREE_D10_BUCKET_LIMITS = [24, 40, 56, 72, 96, 128, 160, 200, 256, 320, 384]          # max 378
TREE_D15_BUCKET_LIMITS = [32, 48, 72, 104, 144, 200, 280, 384, 528, 728, 1000, 1472]  # max 1456
TREE_D20_BUCKET_LIMITS = [48, 72, 104, 152, 216, 304, 424, 592, 824, 1152, 1600, 2216, 3056]  # max 3056
```

### 7.4 `--batch_cost` must scale with the corpus

Bucket batch size is `round_to_8(batch_cost / ((limit² / 256) + 1))`
(`data/util.py:44`, `datamodules.py:137-149`). At the default `batch_cost = 1024`, **every
bucket above ~256 nodes collapses to batch size 1**, so d15/d20 become almost entirely
batch-size-1 steps: 1,288 and 1,937 steps/epoch at 2.0 and 1.4 graphs/step. Measured trade-off:

| batch_cost | d15 steps/epoch (graphs/step, seen) | d20 steps/epoch (graphs/step, seen) |
|---|---|---|
| 1024 | 1,290 (2.0, 98.0%) | 1,938 (1.4, 98.7%) |
| 4096 | **487 (5.2, 94.2%)** | 1,177 (2.2, 96.0%) |
| 8192 | 244 (9.8, 88.7%) | **791 (3.3, 97.6%)** |
| 16384 | 68 (35.7, 90.2%) | 487 (5.2, 93.3%) |

(With the 7.5 clamp applied; without it the high-`batch_cost` rows also silently drop whole
buckets.) Higher `batch_cost` means fewer, fatter steps but more `drop_last` waste in sparse
buckets. Recommended: **1024** for neurons/d10, **4096** for d15, **8192** for d20 — the knee
where graphs/step is workable and ≥94% of the corpus is still seen each epoch.

### 7.5 Bug to patch: buckets smaller than their batch size are never trained

`BucketBatchSampler` computes `n_batches = len(bucket) // batch_size` with `drop_last=True`
(`data/util.py:46-52`). If a bucket holds fewer graphs than its batch size, `n_batches = 0` and
**those graphs are never sampled in any epoch** — silently, and always the *smallest* graphs, so
the baseline is biased, not merely reduced. Measured victims: d10 89 trees (3.3%), d15 113,
d20 232; at `batch_cost ≥ 2048` it hits neurons too (448 graphs).

```python
# semlaflow/data/util.py, after line 44
bucket_batch_sizes = [self._round_batch_size(batch_cost / cost) for cost in bucket_costs]
bucket_batch_sizes = [
    min(bs, len(bucket)) if len(bucket) > 0 else bs
    for bs, bucket in zip(bucket_batch_sizes, buckets)
]
```

All §4 numbers assume this patch **and** §7.8.

### 7.6 New dataset keys (five touch points)

`--dataset` gates the whole neuron path. To add e.g. `trees_d20`:

| file:line | change |
|---|---|
| `scriptutil.py:51` | add the key to `NEURON_DATASETS` (routes vocab / `neuron_mol_transform` / `NeuronCFM` / loss-based checkpointing) |
| `scriptutil.py:21-43` | add `TREE_D20_COORDS_STD_DEV` (printed by preprocessing) + the bucket list |
| `train.py:154-163` | `coord_scale` branch in `build_model` |
| `train.py:242-262` | `coord_std` / `padded_sizes` branch in `build_dm` |
| `train.py:98` | `n_classes = util.NEURON_NUM_CLASSES` is **hardcoded to 7** — trees have **6** genera. Make it per-dataset, or class conditioning trains a 7-wide one-hot with a dead slot and the per-class validation metrics get neuron names (`NEURON_CELL_CLASS_NAMES:47`) for tree genera. |

### 7.7 Scaling walls above ~1k nodes (real, measured)

All three are `O(n²)`/`O(n³)` per **molecule**, so they bite even at batch size 1:

- **Dense bonds.** `GeometricMolBatch.adjacency` (`util/molrepr.py:680-683`) materialises
  `(B, N, N, 5)` float32: 187 MB per batch at N = 3056 (§4), before self-conditioning copies.
- **Fully-connected prior.** `GeometricNoiseSampler.sample_molecule` builds
  `torch.ones((n, n)).nonzero()` (`data/interpolate.py:109`, again at `:333`) — 9.3 M × 2 int64
  ≈ 149 MB **per molecule, in the dataloader worker**, at N = 3056.
- **Equivariant OT.** `linear_sum_assignment` on an n×n cost matrix per molecule
  (`interpolate.py:277-279`) is `O(n³)` worst case; at n ≈ 3000 it dominates data loading.
  Consider `--optimal_transport none` for d20 and report the deviation.
- Semla's attention is dense over N: `(B, heads, N, N)` = 1.2 GB per map at N = 3056, ×12 layers
  of stored activations.

**Recommendation for d20 (and d15 to a lesser degree):** run the headline head-to-head on a
node-capped corpus (`--max_atoms 1024`: −7.4% of trees, −23.7% of nodes) with *both* methods
trained on the same capped set, and report our uncapped d20 run separately as a scalability
result. A capped baseline against an uncapped proposed method is not a comparison.

### 7.8 Turn `drop_last` off — and note it is a *different* loss channel from `max_atoms`

The baseline can end up training on fewer graphs than we do through **two independent channels**,
and they need different fixes:

| channel | what it drops | lever | closed by |
|---|---|---|---|
| **(a) corpus membership** | graphs that never enter the dataset at all — permanently, for every epoch | `--max_atoms` (preprocessing filter `preprocess_neurons.py:43`, size-embedding width `train.py:31`) and the top bucket limit | §7.1–7.3: set `--max_atoms` above the corpus max and extend the buckets |
| **(b) per-epoch remainder** | each bucket's `len(bucket) % batch_size` graphs, re-rolled every epoch | `drop_last=True` in the train sampler, plus bucket population vs batch size | this section + the §7.5 clamp |

**Raising `max_atoms` closes (a) only — it has no effect on (b).** Channel (b) is set here:

```python
# semlaflow/data/datamodules.py:79  (train_dataloader)
sampler = self._sampler(self.train_dataset, drop_last=True)   # -> drop_last=False
```

`BucketBatchSampler` already implements the partial batch — it adds one extra batch per bucket
(`data/util.py:49-51`) and sizes it from the leftover items (`:74-77`) — and the *val* loader
already passes `drop_last=False` (`datamodules.py:95`). Partial batches are strictly smaller than
the nominal cost-budgeted batch, so there is no OOM risk from this change.

Measured effect (with the §7.5 clamp also applied):

| | neurons | d10 | d15 | d20 |
|---|---|---|---|---|
| graphs seen/epoch, stock | 98.4% | 88.6% | 90.0% | 89.0% |
| graphs seen/epoch, `drop_last=False` | **100%** | **100%** | **100%** | **100%** |
| steps/epoch | 580 → 588 | 195 → 202 | 485 → 492 | 788 → 796 |
| effective epochs at nominal 300 | 295.2 → **300** | 265.9 → **300** | 270.0 → **300** | 267.0 → **300** |

Why this matters more than the 2–12% headline: the remainder is **not** drawn uniformly over the
corpus. It is per bucket, so the loss concentrates in whichever size class is worst-matched to
its batch size — the smallest bucket is seen only 70–87% of epochs (§5.3), i.e. the baseline
would be systematically under-trained on small graphs. That is a *distributional* handicap, not
a scalar one, and it lands exactly where morphology metrics are sensitive. Since our side drops
nothing, leaving it stock would make "equal denoising events per node" false in a way no single
correction factor can repair.

Cost: 1.0–3.6% more steps per epoch, which is already reflected in §4 and the §5 budgets. Set it
once, before any baseline run — changing it later changes steps/epoch and therefore `S` and `B*`.

## 8. Known asymmetries (state them; do not engineer them away)

- **Batch composition.** SemlaFlow's batches are size-stratified (312 graphs for the smallest
  bucket, 1 for the largest); ours are uniform over levels. Same nominal step, different
  gradient-noise structure.
- **Time distribution.** Ours is `uniform`, SemlaFlow's is `Beta(2, 1)` (mass near t = 1).
  Both are defaults of their own method; changing either to match is a different experiment.
- **`drop_last` remainders** are *closed*, not tolerated: §7.8 turns them off so both sides see
  100% of the corpus per epoch. Left stock they would cost the baseline 2–12% of its nominal
  epochs, unevenly across size buckets.
- **Per-item difficulty is not equal.** Our item denoises ~7–21 offsets given a coarse tree;
  theirs denoises N coordinates + types + bonds from pure noise. Equal counts of items are not
  equal counts of "problems solved" — which is exactly why §2's node-level definition, not the
  item count, is the parity claim.
- **Per-node gradient weight is not equal** (per-molecule vs per-target loss reduction, §5.3).
  Equal *exposure* per node does not mean equal *gradient* per node; both reductions are their
  own method's default and changing either is a different experiment.

## 9. Reproduce

```bash
# our side + baseline accounting + the parity numbers, per dataset
conda run -n NEURO2 python data_analysis/budget_accounting.py --dataset neurons
conda run -n NEURO2 python data_analysis/budget_accounting.py --dataset trees_d10
conda run -n NEURO2 python data_analysis/budget_accounting.py --dataset trees_d15
conda run -n NEURO2 python data_analysis/budget_accounting.py --dataset trees_d20

# stock (unpatched) SemlaFlow sampler: dead buckets (7.5) + per-epoch remainders (7.8)
conda run -n NEURO2 python data_analysis/budget_accounting.py --dataset trees_d20 --no-clamp --stock-drop-last

# a different baseline batch_cost / epoch budget
conda run -n NEURO2 python data_analysis/budget_accounting.py --dataset trees_d15 \
    --semla-batch-cost 8192 --semla-epochs 300 --batch-size 128
```

Matched-budget training runs — **Option C** (§5.2), one config per arm, no overrides needed.
Every one is E = 300; batch size, `num_steps`, `validation.interval` and the conditioning
switches are baked in, and each config's header states its own step ratio vs SemlaFlow:

```bash
# conditioning matrix (neurons and trees d10), B=256
python main.py -cn parity_neurons_uncond          # 218,500 steps
python main.py -cn parity_neurons_class           # + cell-type conditioning (class_hidden_dim 16)
python main.py -cn parity_neurons_tmd             # + TMD conditioning (tmd_hidden_dim 128)
python main.py -cn parity_trees_d10_uncond        #  34,100 steps
python main.py -cn parity_trees_d10_class         # + genus conditioning
python main.py -cn parity_trees_d10_tmd           # + TMD conditioning (tmd_hidden_dim 128)

# depth sweep: batch fixed at 128, val batch 32, only depth varies (tmd_hidden_dim 64)
python main.py -cn depth_trees_d10_uncond         #  68,200 steps
python main.py -cn depth_trees_d10_tmd
python main.py -cn depth_trees_d15_uncond         # 101,500 steps
python main.py -cn depth_trees_d15_tmd
python main.py -cn depth_trees_d20_uncond         # 134,100 steps
python main.py -cn depth_trees_d20_tmd
```

All 12 share one model block — **21.642 M** params unconditional, +0.115 M with TMD at 128,
+0.053 M at 64, +0.0001 M with class conditioning — so arms differ only in the conditioning
switches and the per-dataset geometry fields (`so2_axis`, `num_classes`, `prior_std_pos`).

For **Option B** (§5.1: equal epochs *and* equal steps) instead, override per run:
`training.batch_size=317 training.num_steps=176400` (neurons) / `144, 60600` (d10) /
`88, 147600` (d15) / `72, 238800` (d20).

The SemlaFlow step counts these ratios are quoted against assume §7.5 + §7.8 are applied.
Against a stock baseline the matched budgets would be 174,000 / 58,500 / 145,500 / 236,400 —
but then its epochs are only ~266–295 effective, so the parity claim weakens; patch it rather
than matching the smaller number.
