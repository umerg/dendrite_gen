# Apical / Axial Error Mode — the "missing or duplicated axiom"

**Living document / parked investigation.** Records a confirmed generator failure mode
and its mechanism so it can be picked up later. Nothing here is fixed yet.

- **Diagnosed:** 2026-07-19
- **Evidence checkpoint:** `~/Downloads/step_5500.pkl` (class-conditional run, 2528 generated vs GT val)
- **Data:** `/Users/umer/Documents/neurons_conditional/{train,val}` (soma-rooted, binarized, `swc_type` + `# cell_class` kept)
- **Repro scripts (session scratchpad, copy into `data_analysis/` if keeping):**
  `apical_stats.py`, `apical_geom.py`, `gen_vs_gt.py`, `ordinal_fidelity.py`

---

## 1. Symptom

`axial_extent_w1` is consistently the **worst** distribution metric, and generated
neurons often have no clearly-identifiable apical dendrite — sometimes two or three
comparable tall trunks instead of one. "Axiom" here = the **apical dendrite** (SWC
`type == 4`): the single dominant trunk running along the model axis (`so2_axis = y`;
apicals point **−y**). It is not the axon — axons (`type == 2`) are present in only ~0.5%
of this dataset.

Per-class `axial_extent_w1` (ema_1) tracks apical prominence — worst for the
long/thick-apical classes, best for the short-apical L2/3 class:

| class | 23P | 4P | 5P-IT | 5P-ET | 5P-NP | 6P-CT | 6P-IT |
|---|---|---|---|---|---|---|---|
| axial_extent_w1 | 38.6 | 78.9 | 117.7 | **195.8** | (high) | — | — |

(For scale, `branch_length_w1 ≈ 4`, `radial_span_w1 ≈ 20`.)

## 2. Ground truth — the apical is real, single, and directional

From `neurons_conditional` (22.7k train / 2.5k val):

- Apical present (any `type==4`): **95%** of neurons (class-dependent: 4P 98.7%, but 6P-IT 84.6%, 5P-NP 88.5%).
- Exactly **one** primary apical trunk (soma child of type 4): **93%**; two: 2.6%; zero: 4.7%.
- The apical is the **most-axial** primary trunk in ~92–96% (median reach 2.5× the tallest basal).
- Dataset is globally oriented: the tallest arm points **−y in ~97%**.

So a good generator must reproduce a sharply peaked "exactly one dominant −y trunk" mode.

## 3. Confirmed gap — generated vs GT (val, n = 2528)

Diagnostic per neuron: for each primary (soma-child) subtree, axial reach = max |y − y_root|;
"tall arm" = reaches ≥ 60% of the tallest; dominance = tallest / 2nd-tallest.

| Metric | GT | Generated (ema_1) | Meaning |
|---|---|---|---|
| Root degree (soma #children) | 7.74 | **7.74 (identical)** | count is teacher-forced → **not** a count problem |
| Axial extent (y-range), median | 354 µm | **271 µm** (−23%) | apical under-reaches |
| Exactly **one** tall arm | **80.3%** | **64.2%** | fewer clean single apicals |
| **≥2** tall arms | 19.7% | **35.8%** | ~2× the GT rate of "multiple apicals" |
| Dominance (tallest / 2nd), median | 2.57 | **2.15** | leading arm less dominant |
| Tallest arm points −y | 96.8% | **86.8%** | ~13% point the wrong way (vs 3%) |

Nobody generates *zero* arms — the failure is not a literal missing apical but a
**weak, short, non-dominant** apical competing with 2–3 comparable siblings. That is
exactly "couldn't make out the axiom." Because the count is matched, the failure is
purely **placement / identity**, not number of neurites.

## 4. Mechanism (root cause)

The soma spawns all `k` children in one step (`method/expansion.py`, `num_root_children`).
The **only** per-child identity signal the model receives is the `geo_ordinal` one-hot
(`MAX_CHILDREN`-wide). For root children this ordinal comes from
`method/helpers.py::_order_root_children_by_uhat` (via `compute_geo_order`, feeding the
one-hot at `expansion.py:~410/686`):

- **ordinal 0** = child whose **first edge** has the lowest `uhat` component (most −y),
  tiebreak largest perp distance;
- ordinals 1..k−1 = remaining children ordered **azimuthally** (clockwise) about `uhat`.

So there *is* an intended "ordinal 0 = apical" signal. (Note: the reducer's random
`sibling_order`, `depth_reduction.py:391-399`, is **not** what conditions root-child
offsets — `geo_ordinal` is. An earlier note calling the ordinal "random" was wrong.)

**The problem: that label is only ~60% accurate.** child_0 is chosen by *first-edge
direction*, but the apical is defined by *extent*. On GT val (`ordinal_fidelity.py`):

- ordinal-0 arm is apical-typed: **59.5%**
- ordinal-0 arm == the tallest-extent arm: **59.3%**
- (sanity) tallest-extent arm is apical-typed: 91.9%

So in ~40% of neurons ordinal 0 is actually a **basal** that happened to have the
steepest-downward first segment, and the true long apical sits at some other (azimuthal)
ordinal. The model is therefore trained on a **blurry** "which slot is the long trunk"
signal, never commits one slot to the dominant −y apical, and at generation smears
apical-ness across siblings → the measured ≥2-tall-arms inflation, reduced dominance,
under-reach, and noisier orientation.

**Compounding factors:**
- The apical's great length is accumulated over many expansion/reduction levels, so
  errors compound → systematic under-reach (271 vs 354 µm) even when identity is right.
- The clean signal (`swc_type`) is **discarded**: `utils/data_loading.py::nx_graph_to_adj_pos`
  keeps only positions + adjacency; type never reaches the model.

## 5. Making ordinal 0 a stronger apical signal

**Which criterion picks the true apical** (`ordinal_fidelity.py` / `label_criteria.py`, GT val,
over the 2419 neurons that have an apical; all criteria are computable at data-prep from the
full GT tree and are SO(2)-invariant about `uhat`):

| Rule for choosing ordinal 0 | picks true apical |
|---|---|
| **current** — min first-edge −y component | 62.2% |
| **deepest −y SUBTREE extent** | **96.7%** |
| max subtree radius | 79.0% |
| thickest trunk (child-node radius) | 71.8% |

→ Switching the criterion from *first-edge direction* to *subtree axial extent* raises
fidelity **62% → 97%**, positions-only (no `swc_type`/radius plumbing needed for the label).

**Structural note — ordinal 0 currently does two jobs, but the frame role turns out not to
matter.** child_0 is both the apical-identity signal AND the shared azimuthal frame reference
(`fwd0`/`v_in` for all root children, `_compute_tree_directions:182-192`, built from child_0's
*perp* direction). Two rounds of verification retired the concern that coupling them is costly:

- *Degenerate branch never fires* (`verify_degenerate.py`, 2528 val): `fwd0_norm <= eps=1e-8`
  hit 0/2528; min observed 0.003 (~3e5× eps). No frame collapse. (An earlier claim that it
  "often hits the degenerate branch" was WRONG.)
- *The anchor is frame-independent for everything we care about.* Radial reach (`|perp|`) and
  axial reach (`offset·uhat`) do not depend on `fwd0`; only the azimuthal **angle** does, and
  that is SO(2) freedom (and a *random* shared value at sampling, since root children spawn as
  placeholders → `helpers.py:528-546`). The only real azimuthal signal is basal-fan spacing,
  and it is weak: gap CV 0.62 vs 0.80 for an i.i.d.-uniform null (`fan_isotropy.py`; basals
  repel mildly, 72% more regular than random). That spacing is a pairwise/joint property the
  model's cross-child attention can learn regardless of the anchor.

⇒ Anchor conditioning is **second-order at most** and should NOT drive the fix choice. (An
earlier claim that Option 2 "taxes in-plane fidelity" via anchor noise was overstated.)

### Candidate fixes

The real trade-off is not the frame but (a) the 0-apical (5%) / 2-apical (2.6%) tails and
(b) code cost.

1. **Re-point ordinal 0 by subtree extent (pragmatic default).** In
   `_order_root_children_by_uhat`, choose ordinal 0 = deepest-−y-subtree child (pass subtree
   extent in — `_compute_tree_directions` has full-tree `pos`+`parent_idx`). 62%→97% identity,
   positions-only, no new feature channel, no frame edit needed (leave `fwd0` as-is — it doesn't
   matter). Forces ordinal 0 onto *someone*, so for apical-less neurons it lands on the deepest
   basal; lean on **class conditioning** (already plumbed) to absorb the apical-presence
   variation. Cheapest change that targets the mechanism.
2. **Dedicated `is_apical` node bit.** Add one feature channel (room after `is_leaf` + 16-wide
   one-hot + `new_flag`, `expansion.py:424-462`); flag the deepest-−y-subtree child (97%), and
   flag 0 or 2 when GT has 0/2 apicals — the only reason to prefer this over #1: it models the
   tails faithfully instead of forcing exactly one. Plumb like `sibling_order`
   (reduction → `data.py` → features); set in `expand()` at sampling. SO(2)-safe scalar.
3. **Add conditioning already plumbed:** TMD (`tmd_hidden_dim > 0`, height filtration on `y`;
   currently 0). Orthogonal, stacks with #1/#2.

Recommendation: start with **#1 + class conditioning**; escalate to **#2** only if per-class
metrics show the 0/2-apical tails are modeled wrong. Alternative/orthogonal: **feed `swc_type`**
as a per-node feature/target (most direct, but a data+model change). All fixes validate only
after a retrain → pair with the §6 diagnostic.

## 6. Diagnostics to add

Wire into `validation/dist_metrics.py` so the failure is tracked every eval instead of
eyeballed: the **≥2-tall-arms rate**, **dominance median**, and **per-class axial_extent_w1**
(gen vs GT). Target: pull the ≥2-tall-arms rate from 35.8% toward the ~20% GT level.

## 7. Key code references

- `method/helpers.py:236` `_order_root_children_by_uhat` — root-child ordinal (the label)
- `method/helpers.py:724` `compute_geo_order` — assembles `geo_ordinal`
- `method/expansion.py` — root spawn (`num_root_children`), ordinal one-hot (`~408-438`, `~686`)
- `depth_reduction.py:391-399` — random `sibling_order` (NOT used for root geo_ordinal)
- `utils/data_loading.py` `nx_graph_to_adj_pos` — drops `swc_type`
- `validation/dist_metrics.py:168` `_size_extent` — `axial_extent = pos·uhat` range
