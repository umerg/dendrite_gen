# What does the dendrite generator represent? — Probe results

> **Status (2026-10-20).** Internal results report. Unconditional arm (`parity_neurons_uncond`, step 218 500,
> EMA weights), validation split (2 529 neurons), flow time t = 0. Sources: run v3 for P2, P3, P5 and the
> random-init controls; run v2 for P6 (unchanged since). **P1 v3 on trained weights is pending** — claim C7 is
> a placeholder. The TMD-conditioned arm is in Appendix A. Method details: `docs/INTERP_METHODOLOGY.md`.
> This report supersedes `docs/INTERP_PROBES_REPORT.md` (pilot).

---

## 0. Summary

| # | Claim | Key number (unconditional, val) | Strength |
|---|---|---|---|
| C1 | Big jumps in pooled probe scores right after a global-attention step are mostly the network knowing **which tree** a node is in, not anything about the node | P2 descendants at A1: pooled ρ 0.07–0.29, within-tree ρ ≈ 0 (−0.01 to 0.03); the random-init network shows the same pooled jumps | strong (method finding) |
| C2 | **Node-level look-ahead.** A frontier node's representation ranks how much it will grow, compared with other frontier nodes of the *same dendrite* | within-dendrite ρ at L11: descendants 0.17–0.37, subtree path length 0.26–0.43; input and random-init ≈ 0; basal dendrites only 0.12–0.24 | strong |
| C3 | The first global read computes **tree-level geometry** (tallest reach, the apical's relative reach) | tallest reach R²: input 0.45, random-init best 0.76, trained G1 0.92 | moderate |
| C4 | **Cell type and final shape** are decodable from shallow partial trees, near the level of the finished tree | cell type at visible depth 4: input 0.28, random-init best 0.44, trained 0.54, finished-tree TMD 0.55 (balanced acc.) | strong |
| C5 | A weak signal of **how much tree is left** | levels remaining R²: input 0.34, random-init best 0.40, trained 0.51 | weak |
| C6 | "In the apical dendrite" is **routed down the lineage** from the slot-0 input, not inferred | AUROC at L11: 0.99 (depth 2) → 0.83 (depth 12+); input 0.50–0.62 | strong, but by design |
| C7 | Local geometry: a one-hop parent cue, no integration of the path to the soma | pending P1 v3 (audit: §8b.5 of the methodology) | — |

What did **not** survive is listed in §4 — several earlier "results" were shortcuts.

---

## 1. Setup

- **Generation order.** A neuron is grown depth layer by depth layer. At each step the network sees every
  node shallower than the current *frontier* at its true position and places the frontier nodes. Every
  node therefore has a known future: its descendants in the real neuron.
- **Flow time t = 0.** The frontier's positions and branch signal are pure noise in the input, so anything
  a probe finds about a frontier node's future was computed from the placed tree.
- **Sites.** `in` (input features), `L0`–`L11` (after each of 12 message-passing blocks), `A1, A3, …, A11`
  (right after the global-attention step of the odd blocks), `G1…G11` (the 8 global tokens), `pool_<site>`
  (mean over the nodes of a tree).
- **Rows.** One row = one node of one *snapshot* (a neuron at one reduction level). Graph-level analyses
  (P3, P5) use one row per snapshot.

## 2. How to read the numbers

**Probe.** Linear: ridge for continuous targets, logistic for classes. Activations are only centred.
Grouped 5-fold cross-validation by neuron — a neuron is never in both the fitting and the scoring part —
and every score is computed on out-of-fold predictions. 95 % intervals: neuron bootstrap.

**References on every result** (as in Future Lens, Othello-GPT and chess-GPT probing work):
1. **Input layer** (`in`, `pool_in`): what is given to the network.
2. **Random-init network** (same architecture and pipeline, untrained): what the architecture alone
   produces. Its signal fades with depth (its `pool_L11` is often ≈ 0), so the fair comparison is the
   **random network's best site**, usually `pool_L0` or `G1`.

No model of the visible tree is used as a baseline. Claims C3–C5 are therefore "decodable beyond the input
and beyond an untrained network", not "beyond what the visible tree determines".

**Levels of variation (the key method).** Every target is attached to one of three levels:
tree (constant within a snapshot), dendrite (constant within a primary dendrite), node. A node-level probe
pools rows from many trees, and its score mixes *which tree is this node in?* with *how does this node
differ within its tree?*

> Toy case: two trees with parent soma distances {10, 11, 12} and {50, 51, 52} µm. Predicting each tree's
> mean gives pooled Spearman ρ = 0.88 while knowing nothing about any node.

So each depth bin is scored three ways:

| view | what it removes | reads as |
|---|---|---|
| pooled ρ | nothing | tree + dendrite + node |
| within-tree ρ (centre target and prediction on the snapshot) | everything shared by the tree | dendrite + node |
| within-dendrite ρ (centre on the primary dendrite) | everything shared by the dendrite | node only |

`share_between` = the fraction of a bin's target variance that lies between snapshots. **Rule:** a node
claim needs a within-dendrite score; a tree claim needs a graph-level probe.

---

## 3. Claims

### C1 — Pooled jumps after a global step are tree-level broadcasts

**Evidence** (P2, log descendants of frontier nodes, Spearman ρ by depth bin) — Figure F1:

| depth bin | 3–4 | 5–6 | 7–8 | 9–12 | 13+ |
|---|---|---|---|---|---|
| share of variance between snapshots | 0.14 | 0.34 | 0.47 | 0.48 | 0.63 |
| pooled ρ at A1, trained | 0.07 | 0.18 | 0.16 | 0.05 | 0.29 |
| within-tree ρ at A1, trained | −0.00 | −0.01 | 0.00 | −0.01 | 0.03 |
| pooled ρ at L1, random-init | 0.10 | 0.10 | 0.07 | 0.07 | 0.21 |
| within-tree ρ at L1, random-init | 0.08 | 0.08 | 0.04 | −0.03 | 0.03 |
| pooled ρ at L11, trained | 0.52 | 0.52 | 0.48 | 0.42 | 0.60 |
| within-tree ρ at L11, trained | 0.52 | 0.40 | 0.25 | 0.17 | 0.19 |

- The jump at A1 disappears completely within tree, for every P2 target: one global read delivers tree
  averages, the same value to every node.
- An untrained network reproduces pooled jumps of this kind.
- From depth 5 on, half or more of the pooled score at L11 is between-tree.

**Consequence.** Pooled node-level scores overstate what nodes know; every node claim below uses
within-dendrite scores. The audit (methodology §8b.5) found the same for visible tree size, tallest reach,
own reach and parent soma distance at A1.

**Caveat.** Depth bin 1–2 (soma children) is excluded: within-tree ρ there is ≈ 0.2 already at the input,
because the slot-0 (apical) flag is an input.

### C2 — Node-level look-ahead within a dendrite

**Question.** The training loss asks only whether a frontier node branches at the *next* step. Does its
representation say how far beyond that its subtree will grow, compared with its neighbours in the same
dendrite?

**Rows.** Frontier nodes that will branch at the next step. **Targets.** log descendants, subtree path
length, height, Strahler order, "grows ≥ 2 more generations", all in the finished neuron.

**Evidence** (within-dendrite Spearman ρ) — Figure F2:

| target | site | 3–4 | 5–6 | 7–8 | 9–12 | 13+ |
|---|---|---|---|---|---|---|
| log descendants | input | −0.01 | −0.02 | 0.01 | 0.00 | 0.05 |
| | random-init L1 / L11 | 0.02 / −0.02 | 0.07 / −0.03 | 0.05 / −0.01 | −0.03 / −0.01 | 0.03 / 0.08 |
| | trained L1 | 0.28 | 0.22 | 0.12 | 0.09 | 0.15 |
| | **trained L11** | **0.37** | **0.32** | **0.23** | **0.17** | **0.19** |
| log subtree path length | trained L1 | 0.33 | 0.29 | 0.15 | 0.14 | 0.16 |
| | **trained L11** | **0.43** | **0.40** | **0.33** | **0.26** | **0.27** |
| | random-init L11 | 0.00 | −0.02 | 0.02 | 0.03 | 0.01 |
| log descendants, **basal dendrites only** | trained L11 | 0.12 | 0.14 | 0.24 | 0.16 | — |
| | random-init L11 | 0.00 | 0.08 | −0.06 | 0.09 | — |

Height and "grows twice" show the same pattern (within-dendrite ρ at L11 0.12–0.35 and 0.18–0.32).

**Reading.**
- The signal is absent at the input and in the untrained network, so it is computed.
- It appears at L1, after the first message from the parent, and keeps growing through the layers for
  shallow frontiers (3–8).
- It survives on basal-only rows, so it is not just the apical label carried down the lineage.
- It weakens for deep frontiers. Part of that is a ceiling: a frontier node at the last levels can grow
  only a little, so the target's range is small.

**Caveats.** Presence, not use: the steering experiment is the causal test. Rows with fewer than 2
frontier nodes in their dendrite drop out of the within-dendrite score.

### C3 — Tree-level geometry from the first global read

**Rows.** One per snapshot (graph-level). **Targets.** Defined on placed nodes only (frontier positions
are noise): tallest dendrite reach along the axis, the apical's reach relative to the tallest, "the apical
is the tallest", log visible tree size.

**Evidence** — Figure F3:

| target (metric) | input `pool_in` | random-init best (site) | trained `pool_L0` | trained **G1** | trained G11 |
|---|---|---|---|---|---|
| tallest reach (R²) | 0.45 | 0.76 (G1) | 0.85 | **0.92** | 0.88 |
| apical reach / tallest (R²) | 0.33 | 0.51 (G1) | 0.56 | **0.67** | 0.64 |
| apical is tallest (AUROC) | 0.80 | 0.86 (pool_L0) | 0.92 | **0.94** | 0.94 |
| log visible size (R²) | 0.86 | 0.996 (G1) | 0.98 | 0.99 | 0.96 |

**Reading.**
- Reach-type summaries peak at the **first** global block (G1) and stay flat or fall slightly afterwards:
  the network builds a picture of the tree's overall shape in its first global read.
- The trained network beats the untrained one by 0.08–0.16 at matched early sites. The untrained network
  is already strong here because one random message layer plus a tree average carries a lot of geometry.
- Visible tree size is not a finding: the untrained network decodes it just as well.

**Caveat.** These are the facts the earlier node-level "awareness of the apical" probes (P6b) were
measuring; at node level they were tree-level all along (§4).

### C4 — Cell type and final shape from shallow partial trees

**Rows.** One per neuron and visible depth v (the depth the partial tree has reached). **Targets.** cell
class (balanced accuracy; chance ≈ 0.14–0.2 for 5–7 classes) and the first principal components of the
finished neuron's TMD (R²). **Ceiling.** a probe on the finished neuron's own TMD vector — how well cell
class can be read from the complete tree's topology.

**Evidence** (cell class, balanced accuracy) — Figure F4:

| visible depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 9+ |
|---|---|---|---|---|---|---|---|---|
| input `pool_in` | 0.21 | 0.26 | 0.28 | 0.28 | 0.29 | 0.29 | 0.35 | 0.31 |
| random-init best (`pool_L0`/`G1`) | 0.21 | 0.28 | 0.36 | 0.44 | 0.41 | 0.43 | 0.47 | 0.47 |
| trained `pool_L11` | 0.22 | 0.37 | 0.48 | **0.54** | **0.54** | **0.56** | 0.64 | 0.63 |
| finished-tree TMD (ceiling) | 0.57 | 0.57 | 0.58 | 0.55 | 0.56 | 0.54 | 0.64 | 0.60 |

Final-TMD PC1 (R²) at `pool_L11`: 0.42 (v = 2) → 0.63 (v = 4) → 0.71 (v = 9+), against random-init best
0.30–0.50 and input 0.26–0.43.

**Reading.**
- From visible depth 4, the network's pooled state decodes cell type about as well as the finished
  neuron's TMD does.
- The trained network is 0.09–0.17 above the untrained one at every depth from 2, and it gains over its own
  first layer (`pool_L0` 0.36–0.59 → `pool_L11` 0.37–0.64): the type signal is built up through depth.
- At v = 1 (soma and its children only) nothing is decodable, as expected.

**Caveats.** Without a model of the visible tree as a baseline, this does not claim the network knows more
than the partial tree determines. In the TMD arm the type is given by the conditioning (Appendix A).
Logistic fits for this probe often hit their iteration cap (Open items).

### C5 — How much tree is left (weak)

**Evidence** (graph-level) — Figure F5:

| target (metric) | input | random-init best | trained `pool_L0` | trained G11 | trained `pool_L11` |
|---|---|---|---|---|---|
| levels remaining (R²) | 0.34 | 0.40 (G1) | 0.43 | 0.50 | **0.51** |
| final depth (R²) | 0.08 | 0.32 (G1) | 0.31 | 0.42 | **0.46** |
| last step (AUROC) | 0.84 | 0.85 (G1) | 0.90 | 0.92 | **0.93** |

**Reading.** The trained network adds about 0.1 over the untrained one, mostly in the deeper layers. The
model has no size input, so this is the only place a stop signal can come from; it is present but modest.

### C6 — Apical identity is routed, not inferred

The apical dendrite is fixed in the input: soma child slot 0 is the apical, chosen on the finished neuron
(by design; sampling also declares slot 0 the apical). The question is how far down the lineage that label
travels.

**Evidence** (P6a, nodes at depth ≥ 2, AUROC for "in the apical dendrite") — Figure F6:

| node depth | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13+ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| input | 0.62 | 0.60 | 0.60 | 0.58 | 0.56 | 0.52 | 0.52 | 0.54 | 0.50 | 0.53 | 0.55 | 0.59 |
| L1 | 0.98 | 0.83 | 0.85 | 0.86 | 0.84 | 0.83 | 0.79 | 0.81 | 0.76 | 0.74 | 0.70 | 0.74 |
| L7 | 0.99 | 0.94 | 0.92 | 0.90 | 0.87 | 0.85 | 0.83 | 0.83 | 0.81 | 0.82 | 0.80 | 0.79 |
| L11 | 0.99 | 0.96 | 0.94 | 0.93 | 0.91 | 0.88 | 0.87 | 0.88 | 0.87 | 0.86 | 0.83 | 0.83 |

**Reading.** At depth 2 (one hop from the slot-0 soma child) the label is fully present after one message
(L1 0.98). Deeper nodes reach high AUROC only in later layers and less completely — consistent with a label
passed down the lineage hop by hop.

**Caveats.** No random-init control was run for P6 (v2). Part of the AUROC is geometry (the apical runs
along −û). This is routing of an input, so it is not reported as inference.

### C7 — Local geometry and soma distance (placeholder: P1 v3 pending)

Planned evidence: within-tree ρ for parent edge length and angle, grandparent edge, parent soma distance,
tortuosity and radial alignment, against local relative-coordinate cues at 1, 2 and 3 hops. The audit
(300 neurons, methodology §8b.5) suggests the network uses about a one-hop cue at L1 and does not integrate
the path to the soma (network best layer ρ 0.66 / 0.62 / 0.46 vs three-hop cue 0.94 / 0.89 / 0.72 at depth
5–6 / 7–8 / 9–12). Dendrite-level targets (own reach, own reach vs the apical's) also come from P1.

---

## 4. What did not survive

| earlier result | why it does not count |
|---|---|
| Visible tree size decodable at nodes | tree-level; untrained network equal (C3) |
| Visible Strahler / descendants of placed nodes | given by the input leaf flag |
| Apical's final reach decodable at basal nodes | given by the slot-0 apical flag (chosen on the finished neuron) |
| Basal nodes "know the apical's reach", fading with hop distance (P6b) | tree-level for basal rows; a tree average of edge geometry shows the same fade |
| Large jumps in pooled scores after A1 | tree broadcast (C1) |
| Pooled scores in the TMD arm | inflated by the conditioning, which describes the finished tree (Appendix A) |
| Depth decodable at nodes | sanity only: no node separates depth d from d + 1 beyond d ≈ 6 |

## 5. Open items

1. **P1 v3 on trained weights** → C7 and the dendrite-level targets.
2. **P3 logistic convergence**: many cell-class fits stop at their iteration cap; rerun P3 with a higher
   `max_iter` before quoting C4 externally.
3. **Random-init for P6**, so C6 has the same control as the other claims.
4. **Steering** (causal tests): push along a look-ahead direction (C2) and a tree-shape direction (C3) during
   generation and measure the grown neurons.
5. **P6c** (global-attention ablation) and probes on the model's own rollouts.

---

## Appendix A — TMD-conditioned arm

The conditioning vector describes the finished tree, so every tree-level target is partly given.
- **Look-ahead (C2) replicates within dendrite:** log descendants within-dendrite ρ at L11 0.38 / 0.37 /
  0.27 / 0.21 / 0.14 (depth 3–4 … 13+), input and random-init ≈ 0. Pooled scores at the input are already
  0.1–0.3, which within-tree scoring removes.
- **C3–C5:** pooled input scores are higher (e.g. levels remaining `pool_in` 0.62 vs 0.34 unconditional);
  not interpretable as computed.
- **C4:** cell type is effectively an input; not reported.

## Appendix B — Preprocessing robustness

Probes were refit after switching from winsorise + z-score to centring only. P1 checks moved by ≤ 0.005
(e.g. frontier depth at L11 0.852 → 0.854; parent edge at L1 0.981 → 0.986) and P2 / P3 by ≤ 0.01. No
conclusion depends on the preprocessing.

## Appendix C — Full tables

Text reports: `~/Documents/generations/interp/{uncond,tmd}_reportv3.txt` and `*_control_reportv3.txt`
(v3); `{uncond,tmd}_v2.txt` (P6). CSVs: see Appendix D.

## Appendix D — Data provenance

| claim | analysis | run | cluster path (under `/itet-stor/guptau/net_scratch/interp/`) |
|---|---|---|---|
| C1, C2 | `p2_bins_val_t0.csv` | v3 | `parity_neurons_uncond/step_218500/ema/results_v3/`, control `parity_neurons_uncond/nockpt/random0/results_v3/` |
| C3, C5 | `p5_val_t0.csv` | v3 | same |
| C4 | `p3_val_t0.csv` | v3 | same |
| C6 | `p6_val_t0.csv` | v2 | `parity_neurons_uncond/step_218500/ema/results_v2/` |

Figures: `python -m interp.analyses.figures --runs ~/Documents/generations/interp/runs --arm uncond`
(writes `docs/interp_figs/F1…F6`).
