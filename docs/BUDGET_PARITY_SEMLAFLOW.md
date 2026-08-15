# Training-Budget Parity — iterative expansion vs SemlaFlow

**Living document.** Defines what "an epoch" means for each method, measures the exact
per-dataset accounting, and fixes the protocol used to train both methods on a comparable
budget. Companion to `docs/NEURON_DATASET_STATS.md` and `docs/TREE_DATASET_STATS.md`
(corpora) — this doc is about *how long each method trains on them*.

- **Analysis date:** 2026-08-02. **Revised 2026-08-15**, twice: first for the node-capped tree
  corpora (`trees_genus_d{10,15,20}_capped`, `TREE_DATASET_STATS.md` §7) that the depth sweep now
  trains on, then to reconcile the whole baseline side with `semla-flow/RUN.md`. Every tree number
  moved in the first pass and every SemlaFlow number moved in the second. Superseded figures are
  kept only where marked.
- **Our side:** `main.py` + `PrecomputedRedDataset`, deterministic depth reduction
  (`reduction.type: depth`, `mode: deterministic`, `contract_root: False`), flow matching
  (`config/diffusion/flow_v10.yaml`).
- **Baseline:** SemlaFlow (`/Users/umer/Documents/semla-flow`), `semlaflow.train`. **`RUN.md` §0
  and §3a there are the source of truth for its flags** — dataset keys, bucket ladders,
  `--batch_cost`, `--max_atoms`, precision, checkpointing. This doc measures the consequences of
  those choices for the training budget; it does not choose them. Cost-bucketed batching,
  equivariant OT, EMA and self-conditioning are all on at their repo defaults.
- **Adopted protocol:** epochs are matched (`E = 300` both sides), gradient steps are **reported,
  not matched** — §4.1 is the table, §5.2 the rationale.
- **Reproduce every number here:** §9. The `PRESETS` in `data_analysis/budget_accounting.py` now
  carry the registered ladders and RUN.md's `--batch_cost`, so `--dataset
  {neurons,trees_d10,trees_d10_capped,trees_d15_capped,trees_d20_capped} --no-clamp
  --stock-drop-last` reproduces §4 exactly (add `--sample N` to subsample the reducer pass; the
  SemlaFlow-side numbers are exact regardless).

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
(measured: 0.984 neurons, 0.988 / 0.996 / 0.998 uncapped d10 / d15 / d20, 0.987 / 0.995 / 0.997
capped — the deficit is the root). SemlaFlow denoises every node once per epoch. Therefore:

> **One epoch = every node position denoised once, on both sides.** That is the definition
> used throughout this doc and the one to state in the paper.

What differs is *context cost*: we re-encode every surviving node at every coarser level, so
one pass costs 4.5–7.9× more node forward-passes than the baseline's single pass.

## 3. Our side — measured reduction-sequence accounting

Full train splits, real reducer (not an estimate). `items/epoch = Σᵢ Lᵢ`. **Five arms**, because
the depth sweep runs on the node-capped corpora while `parity_trees_d10_*` stays on the uncapped
depth-10 set (§3.1).

| | neurons | trees d10 | d10 capped | d15 capped | d20 capped |
|---|---|---|---|---|---|
| dataset config | `neurons_conditional_full` | `trees` | `trees_genus_d10` | `trees_genus_d15` | `trees_genus_d20` |
| train graphs | 22,773 | 2,695 | 2,538 | 2,538 | 2,538 |
| nodes/graph (mean) | 60.98 | 84.34 | 77.62 | 196.69 | 379.84 |
| **levels/graph** mean (med / p95 / max) | **8.19** (8 / 13 / 35) | **10.79** (11 / 12 / 14) | **10.74** (11 / 12 / 14) | **16.01** (16 / 17 / 20) | **21.15** (21 / 23 / 25) |
| node-visits/graph (× graph size) | 379.1 (**6.22×**) | 375.5 (**4.45×**) | 347.1 (**4.47×**) | 1,210.1 (**6.15×**) | 2,990.0 (**7.87×**) |
| supervised targets/graph | 59.98 (= N−1) | 83.34 (= N−1) | 76.62 (= N−1) | 195.69 (= N−1) | 378.84 (= N−1) |
| **items / epoch** | **186,435** | **29,080** | **27,263** | **40,637** | **53,690** |
| node-visits / epoch | 8.63 M | 1.01 M | 0.88 M | 3.07 M | 7.59 M |
| mean nodes / item | 46.31 | 34.80 | 32.31 | 75.58 | 141.34 |
| steps/epoch at the shipped `training.batch_size` | 728 (B=256) | 114 (B=256) | 213 (B=128) | 317 (B=128) | 419 (B=128) |

Levels per graph equal the root-rooted max depth exactly: deterministic depth reduction strips
*all* deepest cherries each level, so depth drops by exactly 1 per level. (Neuron mean 8.19
matches the 8.2 in `NEURON_DATASET_STATS.md` §2; tree means match `TREE_DATASET_STATS.md` §2/§7.)

*Superseded (uncapped d15 / d20, 2,695 train graphs): 224.86 / 446.58 nodes, 16.08 / 21.23 levels,
1,373.9 (6.11×) / 3,498.6 (7.83×) visits, **43,324 / 57,213** items per epoch.*

### 3.1 What the node cap did to the epoch

`TREE_DATASET_STATS.md` §7 drops the 199 trees whose **d20** node count exceeds 1,110 from all
three depths, so d10/d15/d20 hold the same 3,169 trees (2,538 train) and the depth sweep is a
depth ablation rather than a depth-plus-composition one. The `depth_trees_*` configs read those
corpora; `parity_trees_d10_*` keeps the uncapped depth-10 set, since d10 never had the O(N²)
memory problem the cap exists to solve.

| | d10 | d15 | d20 |
|---|---|---|---|
| train graphs | 2,695 → **2,538** (−5.8%) | ← same 157 trees | ← same |
| levels/graph | 10.79 → **10.74** | 16.08 → **16.01** | 21.23 → **21.15** |
| **items / epoch** | 29,080 → **27,263** (−6.2%) | 43,324 → **40,637** (−6.2%) | 57,213 → **53,690** (−6.2%) |
| mean nodes / item | 34.80 → **32.31** | 85.47 → **75.58** | 164.80 → **141.34** |

Items fall slightly *more* than graphs (6.2% vs 5.8%) because the dropped trees are marginally
deeper as well as bigger — but only marginally: mean levels move by 0.05–0.08, confirming
`TREE_DATASET_STATS.md` §7's "the tail is wide, not deep" (corr(nodes, depth) = 0.37 at d20).
**The definition of an epoch is unchanged** — it is scale-free (§2). What changes is every count
derived from it, and therefore `num_steps` (§5.2): a config left at the uncapped budget would run
**320 epochs, not 300**, silently.

## 4. SemlaFlow side — measured epoch accounting

**The source of truth for this side is `semla-flow/RUN.md`** (§0 summary, §3a for the capped
corpora) together with the ladders registered in `semlaflow/scriptutil.py`. Earlier revisions of
this section invented its own ladders and `batch_cost` values; those are superseded, and the
numbers moved a lot when they were replaced — see the note below the table.

Measured with the sampler **as the code stands today**: `drop_last=True` (`datamodules.py:79`) and
no dead-bucket clamp. `--epochs 300`, `--acc_batches 1`, so one batch is one optimiser update.

| | neurons | trees d10 | d10 capped | d15 capped | d20 capped |
|---|---|---|---|---|---|
| `--dataset` | `neurons_conditional_full` | `trees_genus_d10` | `trees_genus_d10_capped` | `trees_genus_d15_capped` | `trees_genus_d20_capped` |
| bucket ladder | `[40 … 256, 537]` | `[96 … 320, 384]` | `[96 … 240, 268]` | `[128 … 592, 666]` | `[160 … 928, 1110]` |
| `--batch_cost` | 1024 | 1024 | 1024 | 2048 | **16384** |
| `--max_atoms` (train) | 538 | 385 | 269 | 667 | 1111 |
| **steps / epoch** | **582** | **237** | **139** | **493** | **218** |
| — of which are batch-size-1 | 24 | 92 | 2 | 343 | 109 |
| graphs seen / epoch | 22,528 (98.9%) | 2,684 (99.6%) | 2,514 (99.1%) | 2,503 (98.6%) | 2,365 (**93.2%**) |
| effective epochs at nominal 300 | 296.8 | 298.8 | 297.2 | 295.9 | **279.6** |
| graphs / step | 38.7 | 11.3 | 18.1 | 5.1 | 10.8 |
| nodes / step | 2,361 | 955 | 1,404 | 999 | 4,121 |
| node-visits / epoch | 1.37 M | 0.23 M | 0.20 M | 0.49 M | 0.90 M |
| dense pair-interactions / epoch | 103.1 M | 26.0 M | 19.5 M | 133.9 M | 494.6 M |
| peak dense-bond tensor / batch | 6 MB | 6 MB | 9 MB | 18 MB | 138 MB |
| peak GPU (RUN.md §0 estimate) | ~18 GB | ~18 GB | ~26 GB | ~26 GB | ~28 GB |
| **300 epochs =** | **174,600 steps** | **71,100** | **41,700** | **147,900** | **65,400** |

**`--batch_cost` is chosen for memory, but it sets the parity denominator.** RUN.md picks it from
what fits a 40 GB card given `--precision` and `--grad_checkpointing`; nothing in that decision is
about training budget. Yet it is the single largest lever on steps/epoch, so the baseline's step
count — the denominator of every ratio in §5.2 — is effectively a by-product of a memory decision.
The clearest symptom is d20c: at `batch_cost 16384` (checkpointing frees the headroom) it takes
**218** steps/epoch, *fewer than d15c's 493*, despite costing ~4× more per epoch (RUN.md §0 rates
the epochs at 20× vs 5× of d10). Its two lowest buckets run at batch 160 and 56.

Where the steps come from is uneven: 343 of d15c's 493 steps and 109 of d20c's 218 are
**batch-size-1** — one graph, one gradient update. At d20c that is the top bucket alone
contributing half the epoch's updates.

The peak dense-bond row is an upper bound — padding is to the batch's largest molecule, not the
bucket limit (`pad_to_bucket=False`, `train.py:465`; `_get_padded_size`, `datamodules.py:225-233`)
— and it is a tensor, not the whole footprint; RUN.md's sampler-aware GPU estimate is the number
to plan against.

*Superseded — the ladders and costs this section used before 2026-08-15 (1024 / 1024 / 1024 /
4096 / 8192 with hand-rolled lists) gave 588 / 202 / 129 / 335 / 639 steps per epoch and, with the
§7.5 + §7.8 patches assumed, 100% of the corpus seen. Those numbers do not describe any run that
will actually be launched.*

### 4.1 The budget table to report

Everything the paper needs on the budget axis, both sides, at the settings above and the shipped
configs of §5.2. **Axis 1 (epochs = denoising events per node) is the matched quantity; axis 2
(gradient steps) is reported, not matched** — that is the decision, see §5.2.

| | neurons | trees d10 | d10 capped | d15 capped | d20 capped |
|---|---|---|---|---|---|
| corpus | uncapped | uncapped | capped | capped | capped |
| **epochs, both sides** | **300** | **300** | **300** | **300** | **300** |
| our items / epoch | 186,435 | 29,080 | 27,263 | 40,637 | 53,690 |
| our `batch_size` | 256 | 256 | 128 | 128 | 128 |
| our steps / epoch | 728.3 | 113.6 | 213.0 | 317.5 | 419.5 |
| **our gradient steps** | **218,500** | **34,100** | **63,900** | **95,200** | **125,800** |
| SemlaFlow steps / epoch | 582 | 237 | 139 | 493 | 218 |
| **SemlaFlow gradient steps** | **174,600** | **71,100** | **41,700** | **147,900** | **65,400** |
| **ours ÷ SemlaFlow steps** | **1.25×** | **0.48×** | **1.53×** | **0.64×** | **1.92×** |
| denoising events, ours | 416.6 M | 68.2 M | 59.1 M | 149.8 M | 289.2 M |
| denoising events, SemlaFlow | 412.1 M | 67.9 M | 58.5 M | 147.7 M | 269.5 M |
| — its effective epochs (§7.8) | 296.8 | 298.8 | 297.2 | 295.9 | 279.6 |
| SemlaFlow epochs for **equal steps** | 375 | — | 460 | — | 577 |

The last row is the alternative to changing our batch size: since steps scale linearly with
epochs, running the baseline for `ratio × 300` epochs equalises axis 2 while *raising* its axis-1
exposure by the same factor. That is a legitimate **second baseline arm** — "the baseline given as
many updates as us" — but it is not the parity arm, and it should be labelled as giving the
baseline more supervision, not less. On d10 and d15c the ratio is already below 1, so there is
nothing to add there; those two arms are conservative for us as they stand.

Denoising events differ by 1–7% only because of the `drop_last` remainder (§7.8), not because of
anything in the definition. d20c is the outlier at −6.8%.

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

### 5.1 Option B — equalise 1 and 2 together (**not adopted**)

Set `E = 300` on both sides and pick our batch size so steps/epoch agree; then matched epochs
**and** matched steps hold simultaneously, and wall-clock is the only free variable.
`B* = items_per_epoch / steps_per_epoch(SemlaFlow)`, at the §4 settings:

| | neurons | trees d10 | d10 capped | d15 capped | d20 capped |
|---|---|---|---|---|---|
| **`training.batch_size`** | **320** | **123** | **196** | **82** | **246** |
| **`training.num_steps`** | **174,600** | **71,100** | **41,700** | **147,900** | **65,400** |
| epochs, both sides | 300 | 300 | 300 | 300 | 300 |
| coordinate-denoising events, ours | 416.6 M | 68.2 M | 59.1 M | 149.8 M | 289.2 M |

**Not adopted** — epoch parity is the decision (§5.2), and `B*` is a moving target: it is
`items/epoch ÷ steps/epoch`, and the denominator is set by the baseline's memory-driven
`batch_cost` (§4). Retuning `--batch_cost` on the SemlaFlow side would silently invalidate our
batch size, which is a worse coupling than reporting a ratio. Kept here as the reference for what
matched steps would cost.

The event counts above are ours; SemlaFlow's are 1–7% lower because of the `drop_last` remainder
(§7.8) — 412.1 / 67.9 / 58.5 / 147.7 / 269.5 M.

### 5.2 Option C — keep our current batch sizes, match epochs only

If `B*` is impractical (memory, throughput, an already-tuned LR/batch pair), match axis 1 and
*report* the step ratio. Only `num_steps` changes. **This is the option we adopted**, and it is
what the shipped configs implement:

| | neurons | trees d10 | d10 capped | d15 capped | d20 capped |
|---|---|---|---|---|---|
| config family | `parity_neurons_*` | `parity_trees_d10_*` | `depth_trees_d10_*` | `depth_trees_d15_*` | `depth_trees_d20_*` |
| corpus | uncapped | uncapped | capped | capped | capped |
| `training.batch_size` | 256 | 256 | 128 | 128 | 128 |
| `training.num_steps` (E = 300) | 218,500 | 34,100 | 63,900 | 95,200 | 125,800 |
| our steps ÷ SemlaFlow steps | **1.25×** | 0.48× | **1.53×** | 0.64× | **1.92×** |

Where we take *fewer* updates at equal exposure (0.48× / 0.64×) the comparison is conservative for
us and can be stated as such; where we take more (**1.25× / 1.53× / 1.92×**) it must be disclosed.

**Read the ratios as a property of the baseline's `batch_cost`, not of our method.** They are
`(items/epoch ÷ our batch) ÷ (its steps/epoch)`, and its steps/epoch is whatever the memory budget
allows (§4). d20c is the extreme case: at `batch_cost 16384` the baseline takes 218 steps/epoch,
so we take 1.92× as many updates on the same exposure. Two honest ways to present it, both
compatible with keeping epoch parity:

1. **Report the ratio** alongside the matched epochs — the table in §4.1 has both columns.
2. **Add a second baseline arm at matched steps**, by running SemlaFlow for `ratio × 300` epochs
   (375 / 460 / 577 for neurons / d10c / d20c). That arm gets *more* exposure than us, so it is a
   generous-baseline check rather than a parity run — say so if it is reported.

For the record, these ratios have moved twice: 1.24× / 0.56× / 1.13× / 0.69× / 0.56× against the
pre-cap corpora and this doc's own invented ladders, then 1.24× / 0.56× / 1.65× / 0.95× / 0.66×
after the cap, and now the values above once the baseline side was reconciled with RUN.md. Only
the last row describes a run that will be launched.

The `depth_*` family holds `batch_size: 128` and `validation.batch_size: 32` fixed across all
three depths so the d10-vs-d15-vs-d20 comparison varies depth only — which is why d10 appears
twice (at 256 paired with the neuron runs, at 128 inside the sweep) and why d20 runs at 128
rather than the 64 its per-depth guidance suggests. The d20 configs carry the E-preserving
fallback (`batch_size: 64`, `num_steps: 251600`) in a comment in case it OOMs.

Note the two d10 arms now differ by **corpus as well as batch size** (uncapped 2,695 trees vs
capped 2,538), so `parity_trees_d10_*` and `depth_trees_d10_*` are not directly comparable to each
other — only within their own family.

**Option A — match steps at the current batch size — is the one to avoid.** It silently varies
exposure: 240 / 626 / 196 / 466 / 156 epochs for neurons / d10 / d10c / d15c / d20c, i.e. anywhere
from 0.5× to 2.1× the baseline's supervision, purely as an artifact of corpus size and of the
`batch_cost` its memory budget happens to allow. d20c would train for 156 epochs against the
baseline's 300 — the same knob that hands us 1.92× the steps under Option C would cost us half the
supervision under Option A. That sensitivity is the argument for anchoring on epochs.

**Where we are today:** the shipped configs carry the matched Option-C budgets above — 218,500 /
34,100 / 63,900 / 95,200 / 125,800 — with `validation.interval` pinned to a 5-epoch cadence
(§5.5). The tree budgets were recomputed for the capped corpora on 2026-08-15; a config left at
the pre-cap `num_steps` would have run E = 320.

### 5.3 What "one epoch" does and does not equalise

Equal (by construction, measured in §3):

- **coordinate-denoising events per node: 1 per epoch on both sides.** Over a 300-epoch run that
  is 417 M / 68 M / 59 M / 150 M / 289 M targets for neurons / d10 / d10c / d15c / d20c.

Not equal — state these, do not try to engineer them away:

| | SemlaFlow | ours |
|---|---|---|
| context encodings per node/epoch | 1 | **4.5–7.9** (surviving nodes re-encoded at every coarser level, §3) |
| topology supervision per graph/epoch | `N²` dense bond entries | `N−1` binary expansion labels (adjacency comes free from the reduction) |
| loss reduction | **per molecule**: `(mse*mask).mean(dim=(1,2))` then batch-mean (`fm.py:748-755`), so a node in a large graph carries less gradient, and padding to the batch max scales it further | **per target**: `F.mse_loss(v_pred, v_target)` over all leaves in the batch (`flow_v.py:198`), so every node carries equal gradient |
| effective epochs | **296.8 / 298.8 / 297.2 / 295.9 / 279.6** as the sampler stands today (§4), and **size-dependent**: the remainder is per bucket, so it lands on whichever size class is worst-matched to its batch size | exactly 300; nothing is dropped |

That last row is §7.8, still open. At the settings RUN.md prescribes the leak is small on four
arms (1.1–1.4%) and material on **d20c (6.8%)**, where `batch_cost 16384` makes the low buckets'
remainders large: bucket 160 discards 81 of its 561 graphs every epoch, bucket 264 discards 52 of
276. So the deficit is concentrated on the *smallest* trees — a distributional handicap in the
baseline's disfavour, not a scalar one, landing exactly where morphology metrics are sensitive.
Since our side drops nothing, "equal denoising events per node" is true to 1.1% on four arms and
to 6.8% on d20c.

### 5.4 Compute disclosure — the two currencies disagree, so measure GPU-hours

Per matched epoch:

| currency | neurons | d10 | d10c | d15c | d20c | who it favours |
|---|---|---|---|---|---|---|
| node forward-passes | 6.3× | 4.5× | 4.5× | 6.2× | 8.5× | SemlaFlow cheaper |
| pair/edge interactions | 11.9× | 25.7× | 22.1× | 43.6× | 65.2× | ours cheaper |

Ours is `O(n)` per level (tree edges + linear attention with 8 global tokens); SemlaFlow is
`O(n²)` (dense attention + all-pairs bonds). One accounting says we use ~6× more compute, the
other says ~20–65× less. Neither is a FLOP count — **report measured GPU-hours** and cite both
columns so no reviewer can claim the budget was chosen to flatter either side.

Both columns are measured against the graphs the baseline actually *sees* per epoch, so d20c's
8.5× / 65.2× carry its 6.8% `drop_last` deficit (§7.8) inside them; on a full pass they would be
7.9× / 69.9×. Quote whichever matches the run being reported, and say which.

The cap also trims our best column: the dropped 5.9% of trees carried ~39% of the corpus `N²`
(`TREE_DATASET_STATS.md` §7), so the pair-interaction advantage falls from 57.8× / 97.8× to
44.2× / 69.9× at d15 / d20. That is the honest consequence of running the baseline on a corpus it
can fit — quote the capped figures for the capped runs, not the uncapped ones.

### 5.5 Practical knobs when stretching to the matched budget

- `scheduler_T_max: ${training.num_steps}` already tracks `num_steps` — nothing to change.
  SemlaFlow uses constant LR + 10k warm-up; keep each method's own tuned schedule and say so.
- **Pin validation in epochs, not steps** (adopted 2026-08-15). `validation.interval` is a step
  count, but the cadence it should express is an epoch count — the same currency as the budget.
  Set `interval = 5 × steps_per_epoch`, which gives **exactly 60 validations on every arm** at
  E = 300, and match it on the baseline with an explicit `--val_check_epochs 5`.

  | | `parity_neurons_*` | `parity_trees_d10_*` | `depth_trees_d10_*` | `depth_trees_d15_*` | `depth_trees_d20_*` |
  |---|---|---|---|---|---|
  | steps / epoch | 728.3 | 113.6 | 213.0 | 317.5 | 419.5 |
  | **`validation.interval`** | **3,640** | **570** | **1,065** | **1,590** | **2,100** |
  | validations over the run | 60.0 | 59.8 | 60.0 | 59.9 | 59.9 |
  | *was* | 2,200 (99) | 350 (97) | 700 (91) | 1,000 (95) | 1,350 (93) |

  The old `interval ≈ num_steps / 100` rule was *already* an epoch rule — at E = 300,
  `num_steps/100 = 3 × steps_per_epoch` — but rounded unevenly, so the arms sat at 3.02–3.29
  epochs. This pins them and halves the count.

- **Open: the two sides validate at different cadences.** `RUN.md` does not pass
  `--val_check_epochs`, so the baseline runs the code default **10** (`train.py:47`) = **30
  validations** over 300 epochs, against our 60. (`NEURONS.md`'s flag table still says 20 and a
  disk-cost example in it assumes 5; the code is the authority.) This is not only a wall-clock
  question: §6 reports *hours-to-best-checkpoint*, and both sides select on a validated metric
  (`val-morpho-selection` / `val-loss` there, ours here), so validating twice as often gives us
  twice the checkpoint-selection resolution. Close it either way before the headline runs — add
  `--val_check_epochs 5` on the baseline, or double our intervals to a 10-epoch cadence
  (7,280 / 1,140 / 2,130 / 3,175 / 4,195) — and state which in the paper.
- **Exclude validation time from the reported GPU-hours on both sides.** With `eval_mode: rollout`
  a validation is a full ODE rollout over the val set; it is not training.
- Disk follows the cadence: a full `step_N.pt` (model + EMA + Adam state) is ~310 MB and is written
  at **every** validation with no pruning — ~19 GB per run at 60, down from ~29 GB at 93.
- Log `training/step_time` (already logged) and SemlaFlow's per-epoch time; report
  hours-to-best-checkpoint as well as hours-to-end.

## 6. What to put in the paper

**Use §4.1 as the table** — it is already in budget-hierarchy order (epochs → denoising events →
gradient steps), per dataset, both methods. Add the two measured columns it cannot supply:
`GPU-hours (train only)` and `GPU-hours to best checkpoint`.

The claim sentence: *"We define an epoch as every node position being denoised once. Our
iterative expansion emits `L` training items per graph (one per reduction level, mean 8.2–21.2
by dataset), but the levels partition the graph's nodes: each non-root node is a flow-matching
target exactly once per pass, as it is for the baseline. Both methods are trained for 300 such
epochs. Because the two sides batch differently — the baseline's batch size is set by a
memory-driven cost budget — matched epochs do not imply matched gradient steps; we report the
ratio (0.48–1.92× by dataset) rather than tuning our batch size to hide it."*

Footnote the asymmetries from §5.3 (context encodings 4.5–7.9×, `N²` bond targets vs `N−1`
expansion labels, per-molecule vs per-node loss reduction) and the two compute currencies from
§5.4. If Option C was used instead, replace the last clause with the step ratio and its
direction.

## 7. Running the SemlaFlow baseline on larger corpora (trees d15/d20)

> **Read `semla-flow/RUN.md` first — it is the runbook and it supersedes the recommendations this
> section originally made.** Everything below is now either (a) mechanism, explaining *why* RUN.md
> chose what it chose, or (b) a marked open item. Where the two disagree, RUN.md wins; this
> section has been corrected to match it rather than the other way round.

The SWC adapter is already generic — `semlaflow/data/swc.py` reads tree SWCs and the
`# cell_class N` header — so no new data code is needed. What must change is the **size cap**
and the **buckets**, plus the items below.

### 7.1 The preprocessing cap (drops graphs)

`semlaflow/preprocess_neurons.py:25` `MAX_ATOMS = 256`, filter at `:43`: anything larger is
**silently dropped** (it prints a count). Pass `--max_atoms` explicitly:

```bash
python -m semlaflow.preprocess_neurons \
    --input_dir  /path/to/trees_genus_d20_capped \
    --output_dir /path/to/trees_genus_d20_capped/smol \
    --max_atoms 1120          # > capped corpus max (1110); uncapped d20 needed 3100
```

**RUN.md's values are the ones to use**, and they are tighter than the round numbers this section
used to suggest: preprocess at the corpus max (268 / 666 / 1110), train at **max + 1**
(269 / 667 / 1111). Headroom 1 is deliberate — any future corpus change that grows a graph by one
node then fails loudly at `models/semla.py:880` instead of training on a silently different
corpus. Nothing is dropped at those values (`dropped 0 graphs` on every split). Measured
`coord_std` for the capped corpora: **1.6346 / 1.7935 / 1.9658** (vs 1.6417 / 1.8062 / 1.9999
uncapped).

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

**This is now settled for the depth sweep** (2026-08-14): the `*_capped` corpora apply one shared
cap — drop the 199 trees whose d20 node count exceeds 1,110, from all three depths — *before*
either method sees the data, and the `depth_trees_*` configs read them. So the cap is a property
of the corpus, not of one method's preprocessing, and `--max_atoms` above the capped max drops
nothing further on the baseline side. Report it as a 5.91% sample loss and do not compare extent
metrics against uncapped runs (`TREE_DATASET_STATS.md` §7).

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

**These are already registered — do not invent new ones.** `DATASET_CONFIGS` in `scriptutil.py`
carries a `bucket_limits` per dataset, and the tree entries were rebuilt on 2026-08-15 around
rule 3. What is live:

```python
neurons_conditional_full  [40, 56, 72, 96, 128, 160, 200, 256, 537]                    # max 537
trees_genus_d10           [96, 128, 160, 200, 256, 320, 384]                           # max 378
trees_genus_d10_capped    [96, 128, 160, 200, 240, 268]                                # max 268
trees_genus_d15_capped    [128, 200, 264, 336, 416, 512, 592, 666]                     # max 666
trees_genus_d20_capped    [160, 200, 264, 336, 424, 528, 648, 784, 928, 1110]          # max 1110
# superseded corpora keep the fine-grained _SWC_BUCKET_PREFIX = [24, 40, 56, 72, 96, 128, 160, 200]
# plus a tail: trees_genus_d15 (… 1536), trees_genus_d20 (… 3072). Both are lossy — see 7.5.
```

**The design rule is rule 3, and it is what makes the low end coarse.** Batch size is inversely
quadratic in the bucket limit, so the *smallest* bucket gets the *largest* batch: at
`batch_cost 1024` a 24-node bucket is given batch 312. A ~2,700-graph tree corpus cannot fill
that, and an underfull bucket trains on nothing at all (§7.5). The neuron corpora spread 22,773
graphs over the same fine prefix and never starve, which is why one ladder cannot serve both.
Starting the tree ladders at 96 / 128 / 160 makes every bucket clear its own batch size, and
RUN.md §3a measures the result: **0.00% of the train split permanently excluded, on every capped
corpus at every `batch_cost` from 1024 to 16384.**

Two corollaries worth keeping:

- **The top limit sets the top batch size, and `_round_batch_size` is discontinuous.** At
  `batch_cost 8192`, a top limit of 728 gives `8192/2071 = 3.95 → 8*round(0.49) = 0 → 1`, while
  672 gives `8192/1765 = 4.64 → 8*round(0.58) = 8`. Same corpus, same cost, **8× the batch**.
  Whenever `batch_cost` is retuned, the top bucket silently changes the answer.
- **An empty bucket costs nothing.** `n_batches = len(bucket) // batch_size = 0`, so buckets above
  the corpus max form no batches and allocate nothing. An earlier revision of this section claimed
  a stale 3056 entry would hold a 187 MB allocation; it does not. That figure came from
  `budget_accounting.py`'s peak column, which maxes over every *listed* bucket rather than every
  populated one.

The §4 peak row is likewise an **upper bound**: `pad_to_bucket=False` everywhere
(`train.py:465`), so `_get_padded_size` pads to the largest molecule *in the batch*
(`datamodules.py:225-233`), not to the bucket limit.

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

Higher `batch_cost` means fewer, fatter steps. That table used this doc's own ladders and is kept
only as the shape of the trade-off; **the values that will actually run come from RUN.md §0/§3a,
and they are chosen for memory, not for budget:**

| | neurons | d10 | d10c | d15c | d20c |
|---|---|---|---|---|---|
| `--batch_cost` | 1024 | 1024 | 1024 | 2048 | **16384** |
| `--precision` | 32 | 32 | 32 | bf16-mixed | bf16-mixed |
| `--grad_checkpointing` | no | no | no | no | **yes** |
| resulting steps / epoch | 582 | 237 | 139 | 493 | 218 |

d20c's 16384 is the consequential one: gradient checkpointing frees enough activation memory that
a far larger cost budget fits (~27.6 GB at 16384 with checkpointing, against 35.1 GB pinned at
≤2048 without), so throughput comes back. The side effect is that **d20c takes fewer gradient
steps per epoch than d15c — 218 against 493 — on a corpus ~2× larger per graph.** Every entry in
§5.2's ratio column follows from that one flag.

**Do not raise `batch_cost` past RUN.md's value for a corpus without re-checking §7.5.** RUN.md
§3a tabulates the permanently-excluded train fraction per corpus per cost: the capped corpora are
at 0.00% everywhere, `trees_genus_d10` reaches 1.04% at 8192, and the uncapped d15/d20 are lossy
at every setting (2.2–22.3%).

### 7.5 Buckets smaller than their batch size are never trained — **closed by the ladders**

`BucketBatchSampler` computes `n_batches = len(bucket) // batch_size` with `drop_last=True`
(`data/util.py:46-52`). If a bucket holds fewer graphs than its batch size, `n_batches = 0`, and
`__iter__` draws buckets with `random.choices(weights=remaining_batches)`, so a zero-weight bucket
is never drawn: **those graphs are never sampled in any epoch** — silently, and always the
*smallest* graphs, so the baseline is biased, not merely reduced. Distinct from the ordinary
`drop_last` remainder (§7.8), which re-randomises every epoch.

**Resolved on the SemlaFlow side on 2026-08-15, by ladder design rather than by the sampler
patch this section used to propose.** Coarsening the low end of the tree ladders (§7.3) makes
every bucket clear its own batch size; RUN.md §3a measures **0.00% permanently excluded on every
capped corpus at every `batch_cost`**. That is the better fix: `BucketBatchSampler` is shared with
the molecular datasets, so changing its semantics belongs in its own change.

Two things to carry forward:

- **The node cap is not what fixed it.** Under the old `_SWC_BUCKET_PREFIX` ladders the victims
  were identical on capped and uncapped corpora (89 / 113 / 232 trees) — the cap removes the
  largest graphs, this bug ate the smallest. Only the ladder change closed it.
- **Runs on `trees_genus_d10` before 2026-08-15 trained on 2,606 of 2,695 graphs**, having
  silently dropped the 89 trees of ≤24 nodes. Do not compare them with runs after it. The uncapped
  `trees_genus_d15` / `trees_genus_d20` still carry the prefix ladder and are still lossy; they are
  superseded by the capped corpora.

The startup line `items per bucket / bucket batch sizes / batches per bucket` is the check: any
bucket with a non-zero item count and 0 batches is a silent loss.

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
node-capped corpus with *both* methods trained on the same capped set, and report our uncapped d20
run separately as a scalability result. A capped baseline against an uncapped proposed method is
not a comparison.

**Taken, 2026-08-14 — but with a different rule than the `--max_atoms 1024` sketched above.** The
`*_capped` corpora drop every tree above **1,110 d20 nodes** (199 trees, 5.91%) from all three
depths at once, so the depth sweep stays sample-matched; see `TREE_DATASET_STATS.md` §7. Against
the three walls above, `N_max` falls 3,056 → 1,110, so every per-graph `O(N²)` term falls **7.6×**
and the `O(N³)` OT term ~21×: the fully-connected prior drops to ~20 MB per molecule and
`linear_sum_assignment` becomes affordable — so **`--optimal_transport` can stay on for d20**,
removing that deviation from the baseline's own method. RUN.md then spends the freed headroom on
throughput rather than banking it (`--batch_cost 16384` with `--grad_checkpointing`), which is why
§4's peak *batch* bond tensor is 138 MB at d20c and not the 56 MB the cap alone would give.

### 7.8 Turn `drop_last` off — and note it is a *different* loss channel from `max_atoms`

The baseline can end up training on fewer graphs than we do through **two independent channels**,
and they need different fixes:

| channel | what it drops | lever | closed by |
|---|---|---|---|
| **(a) corpus membership** | graphs that never enter the dataset at all — permanently, for every epoch | `--max_atoms` (preprocessing filter `preprocess_neurons.py:43`, size-embedding width `train.py:31`) and the top bucket limit | §7.1–7.3: set `--max_atoms` above the corpus max and extend the buckets |
| **(b) per-epoch remainder** | each bucket's `len(bucket) % batch_size` graphs, re-rolled every epoch | `drop_last=True` in the train sampler, plus bucket population vs batch size | **open** — this section. (§7.5's zero-batch channel is separately closed by the ladders.) |

**Raising `max_atoms` closes (a) only — it has no effect on (b).** Channel (b) is set here:

```python
# semlaflow/data/datamodules.py:79  (train_dataloader)
sampler = self._sampler(self.train_dataset, drop_last=True)   # -> drop_last=False
```

`BucketBatchSampler` already implements the partial batch — it adds one extra batch per bucket
(`data/util.py:49-51`) and sizes it from the leftover items (`:74-77`) — and the *val* loader
already passes `drop_last=False` (`datamodules.py:95`). Partial batches are strictly smaller than
the nominal cost-budgeted batch, so there is no OOM risk from this change.

**This is the one item still open** — the sampler is unpatched, and §4/§4.1 are measured with it
as it stands. Measured effect at the RUN.md settings (§7.3 ladders, §7.4 costs):

| | neurons | d10 | d10c | d15c | d20c |
|---|---|---|---|---|---|
| graphs seen/epoch, today | 98.9% | 99.6% | 99.1% | 98.6% | **93.2%** |
| graphs seen/epoch, `drop_last=False` | **100%** | **100%** | **100%** | **100%** | **100%** |
| steps/epoch | 582 → 589 | 237 → 240 | 139 → 143 | 493 → 497 | 218 → 227 |
| effective epochs at nominal 300 | 296.8 → **300** | 298.8 → **300** | 297.2 → **300** | 295.9 → **300** | **279.6** → **300** |
| cost in steps | +1.2% | +1.3% | +2.9% | +0.8% | +4.1% |

The coarse ladders of §7.3 shrank this a lot — four arms are now within 1.4% of a full pass, where
the old fine ladders leaked 2–12%. **d20c is the exception and the reason this still matters:** at
`batch_cost 16384` the low buckets carry large batch sizes, so their remainders are large.
Per epoch it discards **81 of the 561 graphs in bucket 160** and **52 of the 276 in bucket 264**,
against 0 in the batch-size-1 top bucket.

So the deficit is **not** uniform — it is per bucket, and at d20c it falls almost entirely on the
smallest trees: a distributional handicap in the baseline's disfavour, not a scalar one, landing
exactly where morphology metrics are sensitive. Since our side drops nothing, "equal denoising
events per node" currently holds to 1.1–1.4% on four arms and to 6.8% on d20c.

Whichever way this is resolved on the SemlaFlow side, decide it **before** the headline runs:
flipping it later changes steps/epoch and therefore every ratio in §4.1.

## 8. Known asymmetries (state them; do not engineer them away)

- **Batch composition.** SemlaFlow's batches are size-stratified (144 graphs in the smallest
  neuron bucket, 160 in d20c's smallest, 1 in every top bucket); ours are uniform over levels.
  Same nominal step, very different gradient-noise structure — and at d15c/d20c half or more of
  its steps are single-graph (§4).
- **Time distribution.** Ours is `uniform`, SemlaFlow's is `Beta(2, 1)` (mass near t = 1).
  Both are defaults of their own method; changing either to match is a different experiment.
- **`drop_last` remainders** are **open** (§7.8): the baseline sees 98.6–99.6% of the corpus per
  epoch on four arms and 93.2% on d20c, unevenly across size buckets. State the effective epoch
  count (§4.1) rather than claiming 300 flat.
- **Per-item difficulty is not equal.** Our item denoises ~7–21 offsets given a coarse tree;
  theirs denoises N coordinates + types + bonds from pure noise. Equal counts of items are not
  equal counts of "problems solved" — which is exactly why §2's node-level definition, not the
  item count, is the parity claim.
- **Per-node gradient weight is not equal** (per-molecule vs per-target loss reduction, §5.3).
  Equal *exposure* per node does not mean equal *gradient* per node; both reductions are their
  own method's default and changing either is a different experiment.

## 9. Reproduce

```bash
# Every number in sections 3, 4 and 4.1. PRESETS now carries the bucket ladders registered in
# semlaflow/scriptutil.py and the --batch_cost RUN.md prescribes; --no-clamp --stock-drop-last
# matches the sampler as it actually stands today (7.5 / 7.8), which is what section 4 reports.
S="--no-clamp --stock-drop-last"
conda run -n NEURO2 python data_analysis/budget_accounting.py --dataset neurons          $S
conda run -n NEURO2 python data_analysis/budget_accounting.py --dataset trees_d10        $S
conda run -n NEURO2 python data_analysis/budget_accounting.py --dataset trees_d10_capped $S
conda run -n NEURO2 python data_analysis/budget_accounting.py --dataset trees_d15_capped $S
conda run -n NEURO2 python data_analysis/budget_accounting.py --dataset trees_d20_capped $S

# drop the flags to see what the drop_last=False patch of 7.8 would give
conda run -n NEURO2 python data_analysis/budget_accounting.py --dataset trees_d20_capped

# what a different baseline batch_cost would do to the ratio
conda run -n NEURO2 python data_analysis/budget_accounting.py --dataset trees_d20_capped \
    --semla-batch-cost 2048 $S
```

Matched-budget training runs — **Option C** (§5.2), one config per arm, no overrides needed.
Every one is E = 300; batch size, `num_steps`, `validation.interval` and the conditioning
switches are baked in, and each config's header states its own step ratio vs SemlaFlow:

```bash
# conditioning matrix (neurons and UNCAPPED trees d10), B=256
python main.py -cn parity_neurons_uncond          # 218,500 steps, val every 3,640
python main.py -cn parity_neurons_class           # + cell-type conditioning (class_hidden_dim 16)
python main.py -cn parity_neurons_tmd             # + TMD conditioning (tmd_hidden_dim 128)
python main.py -cn parity_trees_d10_uncond        #  34,100 steps, val every 570
python main.py -cn parity_trees_d10_class         # + genus conditioning
python main.py -cn parity_trees_d10_tmd           # + TMD conditioning (tmd_hidden_dim 128)

# depth sweep on the CAPPED corpora: batch fixed at 128, val batch 32, only depth varies
# (tmd_hidden_dim 64)
python main.py -cn depth_trees_d10_uncond         #  63,900 steps, val every 1,065
python main.py -cn depth_trees_d10_tmd
python main.py -cn depth_trees_d15_uncond         #  95,200 steps, val every 1,590
python main.py -cn depth_trees_d15_tmd
python main.py -cn depth_trees_d20_uncond         # 125,800 steps, val every 2,100
python main.py -cn depth_trees_d20_tmd
```

Every arm validates **60 times** (every 5 epochs, §5.5). RUN.md does not pass
`--val_check_epochs`, so the baseline currently validates **30** times — unresolved, see §5.5.
`parity_trees_d10_*` reads the uncapped `trees` dataset, `depth_trees_*` the capped
`trees_genus_d{10,15,20}` ones — the two d10 families are not comparable to each other.

All 12 share one model block — **21.642 M** params unconditional, +0.115 M with TMD at 128,
+0.053 M at 64, +0.0001 M with class conditioning — so arms differ only in the conditioning
switches and the per-dataset geometry fields (`so2_axis`, `num_classes`, `prior_std_pos`).

For **Option B** (§5.1: equal epochs *and* equal steps — **not adopted**) the overrides would be
`training.batch_size=320 training.num_steps=174600` (neurons) / `123, 71100` (uncapped d10) /
`196, 41700` (d10c) / `82, 147900` (d15c) / `246, 65400` (d20c).

The baseline step counts these ratios are quoted against are the sampler **as it stands**
(`drop_last=True`, §7.8). If that is turned off, its steps/epoch rise to 589 / 240 / 143 / 497 /
227, its budgets to 176,700 / 72,000 / 42,900 / 149,100 / 68,100, and the §4.1 ratios fall
slightly to 1.24× / 0.47× / 1.49× / 0.64× / 1.85×. Fix that decision before the headline runs.
