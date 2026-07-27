# Group structure and the grokking clock

> **Status.** This is a pilot, not a settled result. The central claim is
> supported at p = 0.018 (exact, one-sided, n = 8 groups). An earlier version
> of this repo claimed that quaternionic-type groups "never grok"; that claim
> was **false** and is corrected below -- they grok at higher train fraction.
> A second claim about order-16 groups at high train fraction was **retracted**
> because the validation set was as small as 8 examples. Both are documented in
> CORRECTIONS.md rather than quietly removed. Everything in the Mechanism
> section is measured at train fraction 0.70, which is below the critical
> dataset size for the quaternionic groups and above it for the controls.

Does the representation theory of a finite group determine when a neural
network groks its composition table? Recent delay laws predict grokking time
from optimisation quantities (weight decay, learning rate, norm ratios) but
never vary the algebraic structure of the task. The circuit literature
(Chughtai 2023, Stander 2024, Wu 2025) studies which circuits form, not when;
Wu et al. leave complex/quaternionic-type irreps as preliminary (their App.
K.2). Notsawo et al. 2026 vary algebraic structure (groups to general
algebras); to our reading they do not isolate representation type at fixed
group order, but that comparison rests on the abstract and should be checked
against the full paper before submission.
This repo varies the group itself at fixed order, dataset size and output
cardinality. For three of the comparisons the character table is held fixed
too, which is the only sense in which "everything else" is controlled.

## Setup

Groups of order exactly 24, so the composition task has the same dataset size
(576 pairs) and the same 24 output classes in every condition. The core grid
uses six groups at 3 seeds; the dose-response adds Dic24 and Z2 x A4 for eight
groups at 2 seeds (see extra_groups.py). One hidden layer MLP (width 512, no
bias), one-hot inputs, full-batch AdamW (lr 2e-3, wd 1.0), cross-entropy. Delay is
T_gen (sustained val >= 0.90) minus T_mem (sustained train >= 0.995),
both on a 50-epoch eval grid (window 5). Budgets: runs early-stop once val
holds >= 0.99; otherwise the real grid is capped at 80k epochs (two runs
extended to 150k with no change in outcome) and the algebra sweep at 60k.
Censored entries are lower bounds at those budgets. NOTE: censoring at a
single train fraction conflates 'does not grok' with 'is below its critical
dataset size' -- see the correction below.
Cayley tables are built from scratch and checked programmatically: identity,
inverses, EXHAUSTIVE associativity (all 24^3 = 13824 triples), conjugacy class
count against irrep data, and the
Frobenius-Schur identity sum(eps_rho * d_rho) = #{g : g^2 = e}.

## Results

Train fraction 0.70, budget up to 80k epochs, 3 seeds. **The final-accuracy
column here is NOT comparable with the dose-response table below**, which is
capped at 9k epochs; e.g. SL(2,3) seed 0 reads 0.936 at 80k and 0.688 at 9k.

| group | k(G) | d_max | quaternionic mass | delays (3 seeds) | final val acc |
|-------|------|-------|-------------------|------------------|---------------|
| Z24     | 24 | 1 | 0    | 1100, 1200, 1150      | 1.000 x3 |
| D12     | 9  | 2 | 0    | 700, 950, 850         | 1.000 x3 |
| SL(2,3) | 7  | 3 | 1/6  | 30950, >79900, >79900 | 0.936, 0.757, 0.838 |
| S4      | 5  | 3 | 0    | 4100, 4200, 3400      | 0.994 x3 |
| D4 x Z3 | 15 | 2 | 0    | 2700, 3250, 3850      | 0.994, 0.983, 0.983 |
| Q8 x Z3 | 15 | 2 | 1/6  | >79900 x3 at frac 0.70 | 0.821, 0.751, 0.769 |

### Correction: this is a data-requirement effect, not a failure to grok

The table above is measured at one train fraction (0.70) and the binary it
implies is wrong. **Q8 x Z3 does grok** -- at frac 0.78 (T_gen 7200, val 0.945)
and 0.80 (T_gen 5550, val 0.974), verified in this repo's own train.py. What
representation type shifts is the *critical dataset size* (Varma et al. 2023,
Zhu et al. 2024), not whether grokking is possible:

| group | groks at | does not grok at | seeds |
|-------|----------|------------------|-------|
| D4 x Z3 | frac 0.65 (T_gen 5850 / 7200) | frac 0.60 (0.844 / 0.779) | 3 / 2 |
| Q8 x Z3 | frac 0.78 (T_gen 5000 / 6550 / 6950) | frac 0.75 (peak 0.88 at 60k) | 3 / 2 |

A separation of at least 13 points of train fraction (75 pairs) between two
groups with identical character tables.

The honest statistic is a dose-response curve at fixed budget rather than a
threshold crossing (`dose_response.py`, fig 10). Final val accuracy after a
FIXED 9k epochs at every fraction, 2 seeds, all eight groups. The 9k cap makes
the columns comparable to each other; it makes them lower than the 80k numbers
in the main table above.

| group | q-mass | frac 0.50 | frac 0.60 | frac 0.70 |
|-------|--------|-----------|-----------|-----------|
| D12       | 0   | 0.948 / 0.882 | 1.000 / 0.996 | 1.000 / 1.000 |
| Z24       | 0   | 0.764 / 0.646 | 0.991 / 0.974 | 1.000 / 1.000 |
| D4 x Z3   | 0   | 0.417 / 0.260 | 0.818 / 0.749 | 0.983 / 0.965 |
| S4        | 0   | 0.125 / 0.177 | 0.645 / 0.675 | 0.983 / 0.977 |
| Z2 x A4   | 0   | 0.184 / 0.174 | 0.636 / 0.675 | 0.948 / 0.960 |
| Q8 x Z3   | 1/6 | 0.042 / 0.042 | 0.242 / 0.255 | 0.671 / 0.642 |
| Dic24     | 1/2 | 0.031 / 0.042 | 0.251 / 0.195 | 0.642 / 0.584 |
| SL(2,3)   | 1/6 | 0.000 / 0.003 | 0.030 / 0.091 | 0.688 / 0.538 |

The three quaternionic-type groups are the bottom three at every data budget,
with no overlap. Chance is 1/24 = 0.042.

**Statistics, stated correctly.** The unit of replication is the GROUP, not
the run: seeds and train fractions resample the same eight tasks, so quoting
"30 runs versus 18" would be pseudoreplication. The honest test is exact and
one-sided:

    P(the 3 quaternionic groups are the bottom 3 of 8 by chance)
      = 1 / C(8,3) = 1/56 = 0.018

With the original six groups this was 1/C(6,2) = 0.067, i.e. NOT significant;
Dic24 and Z2 x A4 were added specifically to fix that, not to pad the table.
p = 0.018 is significant at 0.05 and nowhere near strong enough to be called
settled. Eight tasks is eight tasks.

THREE matched character-table twin pairs now carry the claim:

| pair | identical | frac | non-quaternionic | quaternionic |
|------|-----------|------|------------------|--------------|
| D4 x Z3 / Q8 x Z3 | character table, k=15, d_max=2 | 0.60 | 0.784 | 0.249 |
| D12 / Dic24       | character table, k=9,  d_max=2 | 0.50 | 0.915 | 0.036 |
| D12 / Dic24       | character table, k=9,  d_max=2 | 0.60 | 0.998 | 0.223 |
| D16 / Q16 (order 16) | character table               | 0.80 | 1.000 (grokked) | 0.115 |

D12 and Dic24 have identical character tables and D12 is the fastest-grokking
group in the whole study; its quaternionic twin sits at chance with half the
table available.

k(G) and d_max are each non-monotone in the outcome: D12 (k=9) beats
D4 x Z3 (k=15); Z24 (d_max=1) and S4 (d_max=3) both sit above every
quaternionic group. Only quaternionic mass separates cleanly.

Four controls were run against this claim; all four passed: censoring (Q8 x Z3 at frac 0.75
still fails at 60k on 2 seeds); seeds (both boundaries hold 3/3); model
capacity (separation survives width 256 / 512 / 1024 at frac 0.70; model size
is known to matter for grokking -- see arXiv 2605.09724 -- so this is the
relevant confound to rule out); implementation
(replicated in an independent NumPy rewrite sharing no training code).

This is in tension with arXiv 2607.05104 (Ootani, July 2026), whose first
finding is that the grokking coverage threshold "tracks the output cardinality
(the modulus) more than composition structure". That study is a ~12K-parameter
Llama-style transformer on modular arithmetic: the tasks live on Z_m, where
every irrep is one-dimensional and Frobenius-Schur type is constant, so
representation type cannot vary within their design. Here output cardinality is
held exactly fixed (24 classes, 576 pairs, every condition) and representation
type is the variable. The two results are not strictly contradictory -- theirs
is a transformer, this is an MLP, and no experiment here varies cardinality --
but they point in opposite directions on whether task structure matters.

Delay spans more than an order of magnitude at fixed group order, and is
non-monotone in both the number of conjugacy classes and the largest irrep
dimension. The groups that fail to grok at frac 0.70 are exactly those
containing a quaternionic irrep (Frobenius-Schur indicator -1) -- two in the
core grid, three once Dic24 is included.

The pair Q8 x Z3 / D4 x Z3 is one of three controlled tests (see the twin
table above). Q8 and D4 have identical
character tables, so their products with Z3 do too (verified: same class-size
multiset, same irrep dimensions). They differ only in second-power structure (element-order
statistics), which representation theory packages as the FS type
of the 2-dim irreps. At frac 0.70 one groks in ~3k epochs and the other fails
on every seed; the second groks at frac 0.78 (see the correction above).
No quantity computable from the character table can account for that.

## Weight-algebra sweep

Schur: the endomorphism algebra of a real irrep is R, C or H. If the failure
were a capacity mismatch, moving the weight algebra should fix it. It does
not. Parameter-matched complex (width 256) and quaternion (width 128)
networks were trained with the same recipe. NOTE: the real rows are censored
at 80k and the algebra rows at 60k, and the sweep is 1 seed per cell (2 for
complex SL(2,3)) against 3 for the real grid -- these rows are indicative, not
matched.

| arch | D4 x Z3 (control) | SL(2,3) | Q8 x Z3 |
|------|-------------------|---------|---------|
| real       | 2700-3850, ~0.99 | 30950 / cens@80k / cens@80k | cens@80k x3, 0.75-0.82 |
| complex    | 3750, 0.977      | cens@60k 0.879 / cens@60k 0.798 | cens@60k 0.827 |
| quaternion | 3750, 0.994      | 41550, 0.896 | cens@60k 0.838 |

At frac 0.70 the control groks under every algebra while the quaternionic-type
groups fail or barely scrape the threshold under every algebra. The effect is not
representational capacity. It tracks the group's FS type and appears to be a
property of the training dynamics on self-conjugate isotypic components,
which is the case the general-group theory of He et al. (2026) leaves open.

## Mechanism

The obvious guess is starvation: the quaternionic sector never fills. Tracking
isotypic energy fractions during training (track_isotypic.py) falsifies this.
Aggregate fractions stay near the random-init baseline in every sector, for
both twins, so the structure is functional rather than energetic. ablate.py
therefore removes one sector at a time from the trained weights (projecting
embeddings and unembedding onto the orthogonal complement) and measures
validation accuracy:

| condition | D4 x Z3 (grokked) | Q8 x Z3 seeds 0/1/2 (stuck) |
|-----------|-------------------|------------------------------|
| full             | 0.994 | 0.821 / 0.751 / 0.769 |
| minus pair       | 1.000 | 0.965 / 0.919 / 0.971 |
| minus twin       | 0.723 | 0.624 / 0.653 / 0.659 |
| minus 1-dims     | 0.751 | 0.618 / 0.543 / 0.572 |
| minus both 2-dim | 0.509 | 0.509 / 0.503 / 0.526 |

Two results. First, a theory check: the twelve 1-dim irreps factor through the
order-12 abelianisation, predicting a 50% ceiling when both 2-dim sectors are
removed; measured 0.50-0.53 everywhere. Second: removing the FS = 0
conjugate-pair sector from the stuck Q8 x Z3 networks raises accuracy to
0.92-0.97 on all three seeds (and 0.994 to 1.000 on the grokked twin), while
removing the twin or the 1-dims hurts. The generalizing solution was already
present, masked by one uncleaned sector.

SL(2,3) replicates the repair with the roles reversed. isotypic_general.py
computes isotypic sectors for any group straight from its Cayley table
(spectral decomposition of the group-algebra centre; it recovers the
hand-built twin projectors exactly). There the complex pair and the 3-dim
irrep are load-bearing, and ablating the quaternionic block itself repairs
the network: 0.942 -> 0.983 and 0.757 -> 0.896 on the two seeds tested.
Removing all 2-dim sectors lands on a second predicted bound: the remaining
irreps factor through A4, capping accuracy at 50%; measured 0.497-0.549.
Unified statement, five of five stuck runs across two groups at frac 0.70: quaternionic-
type groups stall with exactly one uncleaned 2-dim sector masking an
otherwise-formed solution, one projection repairs it, and the real-type
control has nothing to repair. Which sector interferes differs by group.

Timing (fig7): evaluating the repaired network at every step of Q8 x Z3
training shows the masked solution crossing 90% at epoch 8,000, within a factor of
about 3 of the 2,800-3,900 at which the character-table twin crosses the same
threshold, while observed accuracy never gets there in 80k epochs. The network
groks internally on a comparable timescale; only cleanup of the interfering
sector fails. Note this does NOT rule out a sample-complexity account -- the dose-response
above shows frac 0.70 sits below Q8 x Z3's critical dataset size, so the data
was NOT sufficient. What the timing shows is narrower: at a fixed sub-critical
budget the masked solution is present and reachable long before anything
surfaces. Everything in this Mechanism section is measured at frac 0.70, i.e.
below Q8 x Z3's D_crit and above D4 x Z3's; read it as a characterisation of
the sub-critical regime, not as evidence that the group cannot be learned.

## Files

- groups.py: Cayley table constructions and verification, invariants
- train.py, run_experiments.py: real-weight training and the locked grid
- complex_train.py, quaternion_train.py: parameter-matched algebra variants
- track_isotypic.py: isotypic-sector energy fractions during training
- ablate.py: sector ablation on trained networks
- isotypic_general.py: isotypic projectors for any group from its Cayley table
- analysis.py: thresholds, censoring, tables, figures, budget audit
- dose_response.py: final-val-vs-train-fraction sweep, all eight groups (fig 10)
- extra_groups.py: Dic24 and Z2 x A4, the two groups added to fix the sample size
- diagnose.py, diagnose_ckpt.py: equilibrium diagnostics (sector x {train,val}
  ablation matrix, per-sector gradient balance, neuron coupling, regrowth)
- diagnose_multi.py: the 3-seed diagnostic suite incl. margin decomposition
- regrow_carryover.py: regrowth with AdamW state carryover (fig8)
- sweep_wd.py: chunked weight-decay dose-response
- antirepair.py: sector-penalised training (the intervention, fig9)
- chunk_train.py: resumable trainer with model+optimizer checkpoints
- results/: full accuracy and weight-norm histories per run (JSON)
- figs/: figures

## Reproduce

pip install -r requirements.txt, then:

    python groups.py            # verify the six core Cayley tables
    python extra_groups.py      # verify Dic24 and Z2 x A4
    python run_experiments.py Z24 0 1 2      # real baseline (repeat per group)
    python complex_train.py SL23 0           # complex-weight variant
    python quaternion_train.py SL23 0        # quaternion-weight variant
    python ablate.py                         # twin-group sector ablations
    python isotypic_general.py               # general sectors + SL(2,3) ablation
    python track_isotypic.py Q8xZ3 0         # isotypic energy during training
    python analysis.py                       # tables and figures
    python dose_response.py                  # data-requirement curve (fig 10)
    python chunk_train.py Q8xZ3 0 15000      # resumable checkpointed training
    python diagnose_ckpt.py                  # equilibrium diagnostics
    python diagnose.py                       # same, incl. regrowth, from scratch
    python diagnose_multi.py                 # 3-seed suite + margin decomposition
    python regrow_carryover.py 0 6000        # attractor test, optimizer carried
    python sweep_wd.py Q8xZ3 0 3.0 10000     # weight-decay dose-response
    python antirepair.py 0 10.0 pair 12000   # the intervention (fig9)

Runs are CPU-only; the full grid takes roughly an hour on a laptop.

## Equilibrium diagnostics (3 seeds per twin, 80k budget, frac 0.70)

**Regime.** Everything below is at train fraction 0.70, which the dose-response
places BELOW Q8 x Z3's critical dataset size (0.75-0.78) and ABOVE D4 x Z3's
(0.60-0.65). So this is a characterisation of what a sub-critical network looks
like, with a super-critical control alongside it. It is not evidence that
Q8 x Z3 cannot learn the group -- it can, given more data.

The three stuck runs were re-trained from scratch with checkpointing
(chunk_train.py); full-batch training is deterministic, and the 80k states
reproduce the headline table exactly (val 0.821/0.751/0.769). The grokked
D4 x Z3 controls sit at 0.994/0.977/0.988 -- these differ from the main table's
0.994/0.983/0.983 because the main grid early-stops at val >= 0.99 while these
runs go the full 80k. Diagnostics in diagnose_multi.py:

1. Repair replicates and never costs the training set: ablating the pair
   sector lifts stuck val to 0.965/0.919/0.971 (train stays >= 0.998),
   matching ablation.json exactly, and takes the control to 1.000 x3.
2. Every sector of every network sits at an AdamW equilibrium: raw loss
   gradients ~0 with cos(P grad L, P W) from -0.931 to -0.911 across all
   18 sector/seed/group measurements. Sector energy fractions stay at the
   random-init baseline in all six networks (ones 0.475-0.491,
   twin 0.187-0.192, pair 0.322-0.337). Nothing is cleaned up in either group.
3. Margin decomposition. The decision margin l[y] - l[best wrong] splits
   exactly across sectors -- the per-sector contributions sum to the total to
   floating-point precision, verified here, so this is a decomposition and not
   an attribution heuristic. On stuck validation errors the pair sector
   contributes -3.027/-2.228/-2.802 against total margins of
   -1.198/-1.336/-1.442: remove its contribution and the margin is positive on
   every seed (+1.829/+0.892/+1.360). On correct examples it stays live
   (+0.571/+0.756/+0.770). In the grokked control the same never-cleaned
   sector contributes -0.030/-0.120/+0.004 on correct examples -- functionally
   inert. One number separates the twins.
4. The stuck state is an attractor (fig8): projecting the pair sector out of
   the 80k states and resuming with AdamW moments and step counts carried
   (regrow_carryover.py) crashes val from 0.965/0.919/0.971 to
   0.422/0.405/0.422 within 500-750 epochs, regrows the sector's energy
   monotonically (0 -> 0.223/0.230/0.226 by 6k, toward its ~1/3 equilibrium),
   and relaxes back toward the stuck plateau (0.855/0.844/0.792 at 6k). Train
   accuracy stays >= 0.998 throughout. Carrying the optimizer state is what
   makes this an attractor claim rather than a re-initialisation artifact.
5. Weight decay does not select the outcome (single seed, preliminary):
   wd = 0.3 and 3 leave Q8 x Z3 on the same stuck plateau (0.69 and 0.64 at
   20k, peaked) while the D4 x Z3 control groks at wd = 3; wd = 10 is a
   degenerate regime (train 1.0, val below chance) and is uninformative.

**Reading, stated at the right scope.** In the sub-critical regime, failure is
not a failed cleanup: both twins hold the same sector energies at the same
gradient equilibria indefinitely, and FS type determines whether the
equilibrium *content* of the never-cleaned sector is inert or interfering.
The interfering configuration is dynamically stable. What this does NOT show
is that cleanup is irrelevant to grokking in general, or that Q8 x Z3's
difficulty is anything other than a higher data requirement -- only that at a
fixed sub-critical budget the difference between the twins is functional
rather than energetic, and localised to four dimensions.

## Anti-repair: the intervention (3 seeds + specificity control)

**What this now means.** At frac 0.70 Q8 x Z3 sits below its critical dataset
size, so unconstrained training cannot reach 0.90 there at any budget tested.
Penalising one 4-dimensional sector from initialisation takes it to 1.000. The
constraint therefore pushes Q8 x Z3's D_crit from above 0.75 to below 0.70 --
a shift of at least 8 points of train fraction, in the opposite direction to
the one FS type produces. That is the sharpest form of the headline claim: the
same phase boundary that representation type raises, a rank-4 constraint
lowers, and both are measured in the same units.

Train Q8 x Z3 with loss = CE + lambda * (pair-sector weight energy), same
recipe otherwise (antirepair.py, fig9). At lambda = 10, T90 = 1100/1200/1500
and val reaches 1.000 by epoch 1500/2200/2500 on all three seeds -- a perfect
score no unconstrained network in this project reaches at any fraction (the
healthy control tops out at 0.994; repaired stuck nets at 0.971). Specificity:
the same penalty on the load-bearing twin sector gives val 0.468 and no
grokking, so this is not "constraining anything helps".

Three controls sharpen it (AR2_*.json). First, the kill-control: pair
suppression also accelerates the healthy twin (D4 x Z3 + penalty: T90 = 900,
1.000 by 2200, against 2700-3850 unconstrained), so speed alone is not
FS-specific -- the pair sector is a net liability in both groups. What is
FS-specific is the categorical change: for the quaternionic twin the penalty
moves the run across a phase boundary; for the real-type twin it speeds up a
run already on the right side of one. Second, lambda sensitivity: the cure
works at lambda = 1 and 10 (T90 1000-1500, final 1.000) and degrades by
lambda = 30 (T90 3800, 0.960 at the 12k cap). Third, generalisation:
SL(2,3) with its quaternionic block penalised is partially rescued at
lambda = 10 (T90 3800, final 0.960) and fully cured at lambda = 3 (T90 2400,
final 0.994, against an unconstrained best of 0.936 reached only after 31k
epochs) -- one seed each, so preliminary. The order-16 replication of all of
this is in PREDICTIONS.md, where lambda = 3 also beat lambda = 10, consistent
with the useful lambda scaling inversely with sector trace.

The penalty cannot be annealed off. Resuming from the perfect penalised state
with lambda = 0 and optimizer state carried, the pair sector regrows from 0.000
toward its ~0.33 equilibrium and validation collapses 1.000 -> 0.890 within 500
epochs, bottoms at 0.653, and settles at 0.723 by 8k (AR2_anneal_off_s0.json).
The malign basin recaptures even a perfect solution. This is the strongest
statement available here: at a sub-critical data budget the generalising
solution is not merely hard to find, it is not a stable equilibrium of the
unconstrained dynamics, so training could not have held it had it found it.
Whether that is a distinct phenomenon or the sector-localised version of
Varma et al.'s ungrokking is not settled by anything in this repo.

## References and what has actually been checked

Verified by reading the source during this audit:

- arXiv 2606.02993 -- Neural Networks Provably Learn Spectral Representations
  for Group Composition. Trains on the COMPLETE composition table (no
  train/test split) and states in its conclusion that the train-test split,
  and grokking specifically, remains open. Its Thm 4.3 is built on
  Orb(rho) = {rho, rho-dual}, which degenerates to a singleton when rho is
  self-dual, and it defers the self-conjugate case to its App. F.2. Both real
  (FS +1) and quaternionic (FS -1) irreps are self-dual, so the twin pairs here
  sit inside the deferred case and separate the FS sign within it.
- arXiv 2607.05104 -- Ootani, Grokking Is Conditional and Fragile. Abstract
  read; see the tension noted above.
- arXiv 2605.09724 -- Model Capacity Determines Grokking. Abstract read;
  motivates the width sweep.
- arXiv 2309.02390 -- Varma et al., Explaining Grokking Through Circuit
  Efficiency. The ungrokking / critical-dataset-size result the correction
  above is responding to.

NOT verified from full text -- CHECK BEFORE SUBMITTING ANYWHERE:

- "He et al. (2026)" is cited second-hand via 2606.02993, and there are TWO
  He et al. 2026 papers in this area which an earlier draft disambiguated by
  arXiv ID. Restore the arXiv ID here before submitting.
- Wu et al. App. K.2 on complex/quaternionic irreps being preliminary.
- Notsawo et al. 2026, and whether they isolate representation type.
- Chughtai 2023 and Stander 2024 are cited for framing only, not for any
  number in this repo.

## Limitations

- Eight groups at order 24, three at order 16. The central test is exact but
  n = 8: p = 1/C(8,3) = 0.018. Move one group in the ordering and it fails.
- 3 seeds on the core grid, 2 on the dose-response, 1-2 on the algebra sweep.
- One architecture (single-hidden-layer MLP), one optimiser, one recipe. The
  width sweep covers 256/512/1024 at one train fraction only.
- Everything in Mechanism is at frac 0.70, below the quaternionic groups'
  critical dataset size and above the controls'. It characterises a
  sub-critical regime and is not evidence about the super-critical one.
- D_crit boundaries are bracketing intervals from a coarse fraction grid, not
  fitted thresholds, and the grok/no-grok binary is threshold-sensitive near
  the boundary (Q8 x Z3 at frac 0.75 peaks at 0.88, just under 0.90).
- Order-16 results at train fraction >= 0.90 are RETRACTED: the validation set
  is 26 examples or fewer, and 8 at frac 0.97. See CORRECTIONS.md.
- An earlier version of this README claimed quaternionic-type groups never
  grok. That was false; they grok at higher train fraction.
- No transformer replication, no order-48 family, no confidence intervals on
  the dose-response points.
- The isotypic tracking runs cover different budgets (80k vs 16k), so the
  "both twins stay at baseline" comparison is weaker than the rest.
