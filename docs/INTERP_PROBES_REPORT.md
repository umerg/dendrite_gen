# What does the generator know? — Interpretability probes, first results

> **Superseded (2026-10-20)** by [`INTERP_RESULTS.md`](INTERP_RESULTS.md), which reports the full runs with
> within-tree scoring and without Δ. Kept as the record of the pilot.

> **Update (2026-10-11).** Everything below is **superseded**. The first full cluster run (both
> arms, val, t = 0, with random-init controls) used winsorised, z-scored probes. Probes now only
> centre activations (commit 0b18bb0), and P1 gained the ctx / path / morpho groups
> (`docs/INTERP_METHODOLOGY.md` §9.1). Every analysis is being rerun on the trained checkpoints:
> batch 1 is P1–P3, batch 2 is P5–P6. Random-init controls will be rerun once, before write-up.
> This report will be rewritten from those runs.

> **Status (2026-10-06).** Pilot results on the **TMD-conditioned parity arm**
> (`parity_neurons_tmd`, step 218 500, EMA weights), **400 validation neurons**, against a
> **random-initialised copy of the same network** as control. Full val + test runs and the
> unconditional arm are pending on the cluster. P3 (cell type / TMD from partial trees), P4
> (amnesic erasure), P6c (global-attention ablation) and steering are built but **not yet
> run** on trained weights.
>
> **Audit note (2026-10-06, after the pilot).** A three-part code audit fixed issues that affect
> these pilot numbers in both directions: (i) Δ is now measured against the *same* second-stage
> probe fit on the baseline alone (previously the raw baseline — a small upward bias, up to
> ~+0.03 for multiclass, ≈0 for AUROC); (ii) the weighted ridge now has a proper intercept;
> (iii) observables and "currently visible" targets now use only *already-placed* positions
> (frontier positions are noise at t=0 — previously the baseline saw them, a downward bias);
> (iv) P2 gains a **strict** variant that also gives the baseline the number of levels remaining,
> separating node-specific look-ahead from global progress; (v) per-bin regression scores in P6 are
> now Spearman ρ; (vi) all analyses default to t = 0. **Treat the numbers below as provisional; the
> cluster runs supersede them.**
>
> Code lives on branch `interp` (worktree `../dendrite_gen_interp`), package `interp/`.
> Plan: `~/.claude/plans/i-think-let-s-plan-piped-river.md`.

---

## 0. TL;DR

| Question | Answer so far | Confidence |
|---|---|---|
| **P2 — Does the model plan ahead?** Does a node being generated "know" how big the subtree it will eventually grow is? | **Yes.** Beyond everything visible, its own expansion output, its 12-hop neighbourhood *and* the TMD it is conditioned on, activations add **+0.05 to +0.10 R²** about the future subtree (size, height, Strahler, path length) and **+0.04 AUROC** for "grows ≥2 more generations". Builds up layer by layer, peaks at the last layers. Random-init: ≈ 0. | **High** (tight CIs, clean control) — pilot size |
| **P5 — Does it know how much tree is left?** (no size input) | **Partly.** Layer-averaged node features predict *levels remaining* **+0.13 R²** beyond visible-tree observables (random: +0.05, from the input layer only). Final depth: small, not significant. | Medium |
| **P6a — Does a deep node know it is in the apical dendrite?** | Trained net keeps apical identity at **every depth** (AUROC 0.89–0.98); random net loses it with depth (→ 0.54–0.65). But geometry alone (k-hop baseline) already gives ~0.90–0.96, so the *extra* the network adds is significant only at depths 2–5. | Medium |
| **P6b — Does a basal-dendrite node know about the apical?** | **Yes for "is the apical currently the tallest arm?"**: +0.02 to +0.07 AUROC beyond local information, **including 9–12 hops away** (+0.066 [0.037, 0.109]); random ≈ 0. Two other targets turned out not to be interpretable (§5.5). | Medium |
| **P1 — Local structure** | Local geometry (edge length, branch angle) is strongly and specifically encoded. **Absolute depth is *not*** encoded beyond ~3 — neither net separates depth d from d+1 for d ≥ 6. | High |

The headline story worth pursuing for the paper: **the generator is trained only on one-step
decisions, yet builds a forward-looking representation of the subtree each node will become,
and carries "apical is already tall" information across dendrites.** The causal test of that
claim — steering along the look-ahead direction — is next.

---

## 1. Why these questions

The generator grows a neuron **one depth layer at a time**. At each step it sees the partial
tree (true geometry for everything already placed), places the new frontier nodes by flow
matching, and decides for each whether it branches again. The training loss only ever asks
about **this step**: where does this node go, and does it branch next. Nothing asks about the
future.

That makes it a nice object for interpretability, in the spirit of
*"emergent world models"* in game-playing transformers (Othello-GPT; Li et al. 2023, Nanda et
al. 2023) and *learned look-ahead* in chess networks (Jenner et al. 2024): **if the network
represents things the loss never asked for — the eventual size of a subtree, whether the apical
is already dominant, how much tree is left — it has learned structure about neurons, not just a
step-wise rule.**

Each probe asks one such question:

| ID | Question | Why it matters for this model |
|---|---|---|
| P1 | Is local structure (depth, edge length, branch angle) encoded, and how does it build up across layers? | Sanity check + calibration of what message passing gives "for free". |
| P2 | **Look-ahead:** does a frontier node encode its *future* subtree beyond the one-step label? | Loss only asks one step ahead → any look-ahead is emergent. |
| P3 | Can the *unconditional* model's global state decode cell type / final TMD from a partial tree? | Does it infer latent "what kind of neuron" on its own? (built, not run yet) |
| P5 | Without a size input, does it represent how much tree remains? | The only stop signal is its own expansion head. |
| P6 | Does a node know it is in the apical subtree; does a basal node know about the apical? | Mechanistic angle on the known "multiple weak apicals" failure mode (`docs/APICAL_AXIAL_ERROR_MODE.md`). |

---

## 2. What we built (the pipeline)

```
real neuron ──► same depth-reduction sequence the model trains on
                (level ℓ = all nodes with hop depth ≤ D−ℓ; frontier = the depth-(D−ℓ) layer)
            ──► run the TRAINED network teacher-forced at a FIXED flow time t
            ──► record activations at every layer (+ the 8 global tokens)
            ──► join each row with what that node BECOMES in the real neuron
            ──► linear probes, layer by layer, against strong baselines and a random-init control
```

### 2.1 Activations ("sites")

The network has 12 message-passing blocks (256-d node features); blocks 1, 3, 5, 7, 9, 11 are
preceded by a global-attention step through 8 learned "global tokens".

| Site | What it is |
|---|---|
| `in` | the raw input features (positions excluded; includes the TMD embedding in the TMD arm) |
| `L0 … L11` | node features after each message-passing block |
| `A1, A3, … A11` | node features right after a global-attention step |
| `G1, G3, … G11` | the 8 global tokens after that step (a per-tree summary) |
| `pool_L*` | node features averaged over the visible tree (another per-tree summary) |

### 2.2 Flow time

The model is a flow-matching model: the frontier positions and the "will it branch" signal are
**noised** with flow time t (t = 0: pure noise, t = 1: clean). We pin t:

- **t = 0** for P2: the noisy branch signal `e_t` is pure noise, so the model cannot just read
  its own one-step label off the input.
- **t = 0.6** for the others *in this pilot* (a representative mid-trajectory point on the sampler
  grid). **From the cluster runs on, every analysis defaults to t = 0**: at t = 0 the frontier
  inputs (positions and branch signal) are pure noise, so any information found must have been
  computed from the tree rather than read off partly-formed new nodes. At t = 0.6 the apical's
  own new nodes are already 60 % formed, which would mix "inferred" with "seen" for P6b.
  t = 0.6 / 0.9 remain available as a t-sweep (the extraction stores all three).

### 2.3 Every row knows its future

Because the depth reduction keeps original nodes (it just peels off the deepest layer each
step), every node at every level maps to a node of the real neuron — verified exactly against
the reducer on 50 neurons. So for each row we know its final descendants, subtree height,
Strahler order, path length, which primary dendrite it belongs to, whether that is the apical,
and how tall every dendrite currently is / will be.

### 2.4 Code map (`interp/` on branch `interp`)

| File | Role |
|---|---|
| `load.py` | builds the model through `main.get_expansion_items` (architecture-exact) and loads EMA / raw / random weights |
| `reduction_ids.py` | reduction sequences tagged with original node ids |
| `targets.py` | per-node final-tree, per-level visible/hidden, cross-dendrite targets |
| `hooks.py` | activation recording, global-attention ablation, layer editing (for steering / erasure), fixed-t, per-node-keyed flow noise |
| `extract.py` / `store.py` | teacher-forced extraction → on-disk store |
| `baselines.py` | observables, k-hop neighbourhood profile, partial-tree morphometrics |
| `probe.py` | grouped-CV probes, Δ-over-observables, controls, bootstrap, mass-mean directions, LEACE |
| `analyses/p1_local.py … p6_apical.py`, `report.py` | the experiments and a text summary |
| `scripts/interp_extract.sbatch`, `scripts/interp_analyze.sbatch` | cluster jobs |

33 tests (`tests/test_interp_*.py`) cover the id mapping, hooks, extraction determinism, the
probe library on synthetic data with known answers, and every analysis end to end.

---

## 3. How to read the numbers

### 3.1 A probe

A **linear probe** is a linear/logistic regression from a layer's 256-d activation to a target.
If it works, the information is *linearly readable* at that layer — which is what the next layer
(or a steering vector) can use. Linear on purpose: a powerful nonlinear probe can compute the
answer itself and tells you little.

Scores: **R²** for continuous targets (0 = no better than predicting the mean, 1 = perfect),
**AUROC** for yes/no targets (0.5 = chance, 1 = perfect).

### 3.2 The key statistic: Δ over observables

A raw probe score is misleading: much of a node's future is predictable from what is *already
visible* (a node near the soma in a big tree will have a big subtree). So the main statistic is

> **Δ = score( observables + activation ) − score( observables alone )**

- **Observables** = everything the model can see at that step without learning: depth, child
  slot, visible tree size, frontier size, its own dendrite's visible size/reach/height, its
  parent's edge length and angle, the noisy branch input `e_t`, **the model's own predicted
  branch decision**, a **12-hop neighbourhood profile** (how many visible nodes / frontier nodes
  / mean axial distance at each hop distance 1–12), and — for the TMD arm — **the 16 principal
  components of the TMD vector it is conditioned on**. Never the level, the final depth, or
  anything from the hidden future.
- The observables model is a non-linear model (gradient boosting or ridge, picked by
  cross-validation), so Δ > 0 means the activation adds **linearly readable information beyond a
  non-linear summary of everything visible**.

**Δ > 0 with a CI excluding 0 = the network computed something new.**

### 3.3 Controls

- **Random-initialised network** (same architecture, same data, no training). Message passing
  alone produces some structure; anything the random net also shows is architecture, not
  learning. The most important control here.
- **Shuffle within neuron:** permute labels among one neuron's rows (keeps neuron-level info,
  destroys node-level) → ~0.52–0.55 AUROC, as it should.
- **Grouped cross-validation by neuron:** a neuron is never in both fit and test.
- **95% CIs** by resampling whole neurons (1000× bootstrap; 200× in this pilot).

### 3.4 Methodological issues found and fixed during the pilot

Worth knowing because several gave spurious "results" before they were fixed:

1. **Ridge regularisation chosen by GCV** blew up on heavy-tailed random-init features (R² down
   to −40) → now chosen by grouped inner CV. (Features were also winsorised at 0.5/99.5 % and
   z-scored at the time; that was later dropped in favour of centring only — see
   `docs/INTERP_METHODOLOGY.md` §5.2. The pilot numbers here predate that change.)
2. **Observables model overfit** neuron-constant targets (memorising neurons from row features)
   → would *inflate* Δ in the network's favour. Now chosen among several complexities by grouped
   inner CV; regression tests ensure Δ ≈ 0 for pure-noise activations.
3. **Baseline column was penalised** in the stacked probe → Δ could go spuriously negative. It now
   enters unpenalised.
4. **TMD conditioning leaks the answer**: in the TMD arm every node's input contains a summary of
   the *final* tree. The TMD components are now part of the observables for this arm (this
   roughly halved the P2 effect sizes — what's left is beyond the conditioning).
5. **Flow noise depended on batch packing** → the same node got different noise in different
   extractions. Now seeded per (seed, t, neuron, level, node).
6. The **Hewitt–Liang control task** keyed on (depth, slot) is trivially solvable here (those are
   input features) → opt-in only; not informative for this model.

---

## 4. Setup of this pilot

| | |
|---|---|
| Model | `parity_neurons_tmd`, step 218 500, **EMA** weights (EMA and raw losses agree: 0.6035 / 0.6034 position, 1.264 / 1.266 expansion) |
| Control | same architecture, random init (seed 0), stored in float32 |
| Data | first 400 validation neurons (3 251 reduction steps; 23 818 frontier rows) |
| Flow time | t = 0 (P2), t = 0.6 (P1, P5, P6) |
| Activation scale | trained net max \|x\| = 346 → float16 storage is safe; random net overflows fp16 in deep layers |

---

## 5. Results

### 5.1 P2 — look-ahead ✅ the main result

Rows: frontier nodes that **will** branch next step (10 390 rows). The question is whether the
activation encodes how far beyond that one step the subtree will go.

| Target | Observables only | Δ at `in` | Δ at L5 | **Δ at L11** | Random-init Δ at L11 |
|---|---|---|---|---|---|
| grows ≥2 more generations (AUROC) | 0.738 | +0.005 | +0.018 | **+0.040** [0.033, 0.046] | +0.001 |
| log descendants (R²) | 0.378 | +0.013 | +0.042 | **+0.080** [0.069, 0.092] | −0.001 |
| subtree height (R²) | 0.397 | +0.013 | +0.037 | **+0.065** [0.048, 0.083] | −0.007 |
| Strahler order (R²) | 0.257 | +0.003 | +0.026 | **+0.049** [0.040, 0.061] | −0.002 |
| log path length (R²) | 0.457 | +0.007 | +0.039 | **+0.088** [0.075, 0.100] | −0.001 |
| *already-placed nodes:* hidden descendants (R²) | 0.560 | +0.034 | +0.085 | **+0.102** [0.082, 0.123] | −0.001 |
| *already-placed nodes:* hidden height (R²) | 0.517 | +0.035 | +0.062 | **+0.067** [0.041, 0.098] | −0.006 |

**What this says.**

- The network represents **how big the subtree under a node will become**, beyond the one-step
  decision it is trained on and beyond everything it can see — including the TMD summary of the
  whole final tree it is given.
- The information **accumulates with depth** (Δ grows monotonically from `in` → L11). In the
  random net the plain probe score *falls* with depth, in the trained net it *rises*: this is
  computed, not inherited from the input.
- It holds both for nodes being placed now (frontier) and for already-placed nodes whose
  descendants are not yet generated ("hidden" rows).
- The small Δ at `in` (+0.003 to +0.035) is most likely TMD detail beyond its first 16
  principal components; the *growth* across layers is the result.

**What it does not yet say.** That the model *uses* this information. A probe shows presence;
the steering experiment (add the "tall future" direction to one dendrite's frontier during
generation and see whether that dendrite grows taller) is the causal test. The directions are
already saved per layer (`directions_p2_val_t0.npz`: `v_grow`, `v_label`, and `v_orth` =
`v_grow` with the "branch next step" and depth directions removed — the cleanest test of
look-ahead beyond the one-step decision).

### 5.2 P5 — does it know how much tree is left?

One row per reduction step (3 251). Targets: levels remaining ℓ, final depth D, "last step".

| Target | Visible depth only | Observables | Best trained Δ | Best random Δ |
|---|---|---|---|---|
| levels remaining (R²) | 0.189 | 0.671 | **+0.132** [0.094, 0.162] (`pool_L10`, total 0.79) | +0.047 (`pool_L0`, input-level) |
| final depth (R²) | 0.265 | 0.706 | +0.027 [−0.006, 0.060] n.s. | −0.034 |
| is last step (AUROC) | 0.819 | 0.982 | +0.012 [0.006, 0.020] | +0.003 |

**What this says.** Late-layer, tree-averaged node features carry information about how many
generations remain beyond the visible-tree observables (which already include the TMD
components). The random net's gain is smaller and only at the input layer (TMD detail again).
The final-depth result being weaker than levels-remaining is partly a variance effect (ℓ = D −
visible depth, and visible depth is an observable), so read this as **"some global progress
information, not a strong depth budget"**. Interestingly, the averaged node features carry it
better than the 8 global tokens.

### 5.3 P6a — does a node know it is in the apical subtree?

Rows: all stored nodes at depth ≥ 2 (soma children excluded — they are *given* the apical flag
as input).

| | AUROC overall | depth 2 | depth 5 | depth 8 | depth 9 | depth 12 |
|---|---|---|---|---|---|---|
| **trained L11** | **0.962** | 0.98 | 0.93 | 0.89 | 0.89 | 0.88 |
| random L11 | 0.883 | 0.91 | 0.74 | 0.60 | 0.54 | 0.76 |
| k-hop neighbourhood baseline (no network) | 0.954 | 0.92 | 0.96 | 0.89 | 0.90 | 0.93 |

Δ over the k-hop baseline (trained L11): +0.066 [0.058, 0.078] at depth 2, +0.049 at 3,
+0.026 at 4, +0.020 [0.008, 0.037] at 5; from depth 6 on the CIs include 0. Random: ≈ 0 beyond
depth 3.

**What this says.** The trained network holds on to "I am in the apical dendrite" at every
depth, while a random network's identity signal washes out after ~4 hops. But the apical
subtree is also **geometrically distinctive** (it runs along −y), so a non-learned summary of
the neighbourhood already gets ~0.9 at deep nodes. The network adds significant information
beyond geometry only at depths 2–5. **We cannot yet claim the global tokens carry apical
identity to deep nodes** — that needs the ablation (P6c).

### 5.4 P6b — does a basal node know about the apical?

Rows: frontier nodes in **non-apical** dendrites (14 518). Binned by **hop distance through the
soma** to the apical's farthest-reaching node (dAB). Beyond ~12 hops, message passing alone
cannot carry the information — only the global tokens can. Baseline: local-only observables
(nothing summarising other dendrites).

| Target | dAB 2–4 | 5–8 | 9–12 | 13–16 |
|---|---|---|---|---|
| **is the apical currently the tallest arm?** (Δ AUROC, trained L11) | +0.029 [0.003, 0.062] | +0.022 [0.006, 0.039] | **+0.066** [0.037, 0.109] | +0.055 [−0.023, 0.134] |
| same, random-init | +0.011 (n.s.) | +0.008 (n.s.) | +0.007 (n.s.) | +0.072 (n.s.) |
| apical's current reach (Δ R², trained L11) | +0.011 (n.s.) | **+0.036** [0.017, 0.058] | **+0.111** [0.053, 0.200] | (unreliable) |
| same, random-init | +0.004 | 0.000 | −0.013 | (unreliable) |
| another arm is tall / own height rank | ≈ 0 — local observables already near ceiling (0.98 / 0.87) | | | |

**What this says.** A frontier node in a basal dendrite carries information about **whether the
apical is currently dominant** and **how far it reaches**, beyond its own local view, and this
is present **9–12 hops away**. A random network carries none of it. This is exactly the
information you would need to *not* grow a second tall arm — relevant to the
multiple-weak-apicals failure mode.

### 5.5 What turned out not to be interpretable (honest list)

- **"Apical's final relative reach" (P6b) and the 13+ hop bins:** the observables baseline has
  negative R² there and the **random net shows the same "gain" (+0.3 to +1.0)** as the trained
  net. These are baseline artefacts (calibration of a near-constant, neuron-level target inside
  narrow bins), not findings. Excluded from conclusions; will be re-scored with a rank metric.
- **Depth diagonal (P1):** the plan was to see how far depth information spreads per layer. In
  practice **neither network linearly encodes absolute depth beyond ~3–5** (depth d vs d+1
  AUROC ≈ 0.5 for d ≥ 6 in both), so depth is the wrong variable to detect routing. P6 is the
  better tool for that question.

### 5.6 P1 — local structure (sanity)

| Target (R²) | trained best | random best | k-hop baseline |
|---|---|---|---|
| parent edge length (placed nodes) | 0.97 (L1) | 0.59 | 0.45 |
| branch angle | 0.69 (L5–L7) | 0.22 | 0.06 |
| own edge length (frontier, noised) | 0.83 (L10) | 0.51 | 0.03 |
| number of visible children | 0.95 | 0.995 | 0.98 |
| depth (placed nodes) | 0.65 | 0.62 | **0.76** |

**What this says.** The trained network encodes local **geometry** strongly and keeps it through
depth (in the random net it decays). It does **not** keep an explicit depth coordinate — the
non-learned neighbourhood profile is a better depth readout than any layer. Plausibly the model
reasons in relative, geometric terms rather than "I am at depth 7".

---

## 6. Caveats

- **Pilot scale:** 400 val neurons, one arm, one seed. Effect sizes are consistent and CIs tight,
  but the paper numbers should come from full val (selection) + test (reporting).
- **Teacher-forced only:** the model sees true partial trees. Whether the same representations
  appear in its own rollouts (P6c-rollout) is not yet tested.
- **Presence ≠ use.** Everything here is probing. Causal claims need steering (S-growth,
  S-apical) and ablation (P6c, P4).
- **Linear readability only** — information present non-linearly is not counted.

---

## 7. Next steps

1. **Cluster runs** (both arms; trained + random-init; val + test): extraction, then all probes.
2. **Unconditional arm**: P2/P5 without the TMD confound; P3 (does it infer cell type / TMD from
   a partial tree on its own?).
3. **P6c ablation**: re-extract 500 test neurons with the global read zeroed / mean-replaced →
   does apical/cross-dendrite information at 9+ hops disappear, and does the model need it?
4. **P4 amnesic erasure** (TMD arm): remove the TMD subspace at layers 1/5/9 and measure how much
   conditioning fidelity is lost.
5. **Steering**: S-growth with `v_grow` / `v_orth` on one basal dendrite (does it grow taller?
   dose–response, random-direction control); S-apical to suppress a second tall arm.

### Commands

```bash
# 1. extraction (GPU)
sbatch --export=ALL,CKPT=<ckpt>,CONFIG=parity_neurons_tmd scripts/interp_extract.sbatch
sbatch --export=ALL,CONFIG=parity_neurons_tmd,WHICH=random:0,T=0,0.6,DTYPE=float32 scripts/interp_extract.sbatch
# 2. probes (CPU)
sbatch --export=ALL,STORE=<store>/ema scripts/interp_analyze.sbatch
sbatch --export=ALL,STORE=<store>/random0 scripts/interp_analyze.sbatch
# 3. summary
python -m interp.analyses.report <store>/ema/results --control <store>/random0/results
```

---

## Glossary

- **Frontier** — the layer of nodes being generated at this step.
- **Reduction level ℓ** — how many depth layers are still to be generated after this step (ℓ = 0
  is the last step).
- **Visible depth v** — deepest layer present at this step; final depth D = v + ℓ.
- **Teacher-forced** — the model is run on the true partial tree rather than its own samples.
- **EMA weights** — exponential moving average of training weights; what evaluation and sampling use.
- **Strahler order** — branching-hierarchy order of a subtree (1 = tip; increases when two
  equal-order branches merge).
- **TMD** — topological morphology descriptor (persistent-homology summary of the tree); the TMD
  arm is conditioned on its persistence-image vector.
- **AUROC / R²** — see §3.1. **Δ** — see §3.2.
- **Mass-mean direction** — difference of mean activations between two groups (e.g. nodes with a
  tall vs short future), stratified by depth; used as a steering vector.
