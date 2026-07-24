# Group structure and the grokking clock

Does the representation theory of a finite group determine when a neural
network groks its composition table? Recent delay laws predict grokking time
from optimisation quantities (weight decay, learning rate, norm ratios) but
never vary the algebraic structure of the task. The circuit literature
(Chughtai 2023, Stander 2024, Wu 2025) studies which circuits form, not when;
Wu et al. explicitly leave complex/quaternionic-type irreps as preliminary
(their App. K.2). Notsawo et al. 2026 vary algebraic structure (groups to
general algebras) but do not isolate representation type at fixed order.
This repo varies the group itself, holding everything else fixed.

## Setup

Six groups of order exactly 24, so the composition task has the same dataset
size (576 pairs) and the same 24 output classes in every condition. One
hidden layer MLP (width 512, no bias), one-hot inputs, full-batch AdamW
(lr 2e-3, wd 1.0), cross-entropy, train fraction 0.7, 3 seeds. Delay is
T_gen (sustained val >= 0.90) minus T_mem (sustained train >= 0.995),
both on a 50-epoch eval grid (window 5). Budgets: runs early-stop once val
holds >= 0.99; otherwise the real grid is capped at 80k epochs (two runs
extended to 150k with no change in outcome) and the algebra sweep at 60k.
Censored entries are lower bounds at those budgets.
Cayley tables are built from scratch and checked programmatically: identity,
inverses, associativity, conjugacy class count against irrep data, and the
Frobenius-Schur identity sum(eps_rho * d_rho) = #{g : g^2 = e}.

## Results

| group | k(G) | d_max | quaternionic mass | delays (3 seeds) | final val acc |
|-------|------|-------|-------------------|------------------|---------------|
| Z24     | 24 | 1 | 0    | 1100, 1200, 1150      | 1.000 x3 |
| D12     | 9  | 2 | 0    | 700, 950, 850         | 1.000 x3 |
| SL(2,3) | 7  | 3 | 1/6  | 30950, >79900, >79900 | 0.936, 0.757, 0.838 |
| S4      | 5  | 3 | 0    | 4100, 4200, 3400      | 0.994 x3 |
| D4 x Z3 | 15 | 2 | 0    | 2700, 3250, 3850      | 0.994, 0.983, 0.983 |
| Q8 x Z3 | 15 | 2 | 1/6  | >79900 x3             | 0.821, 0.751, 0.769 |

Delay spans more than an order of magnitude at fixed group order, and is
non-monotone in both the number of conjugacy classes and the largest irrep
dimension. The two groups that fail to grok are exactly the two containing a
quaternionic irrep (Frobenius-Schur indicator -1).

The pair Q8 x Z3 / D4 x Z3 is the controlled test. Q8 and D4 have identical
character tables, so their products with Z3 do too (verified: same class-size
multiset, same irrep dimensions). They differ only in second-power structure (element-order
statistics), which representation theory packages as the FS type
of the 2-dim irreps. One groks in ~3k epochs; the other fails on every seed.
No quantity computable from the character table can account for that.

## Weight-algebra sweep

Schur: the endomorphism algebra of a real irrep is R, C or H. If the failure
were a capacity mismatch, moving the weight algebra should fix it. It does
not. Parameter-matched complex (width 256) and quaternion (width 128)
networks were trained with the same recipe:

| arch | D4 x Z3 (control) | SL(2,3) | Q8 x Z3 |
|------|-------------------|---------|---------|
| real       | 2700-3850, ~0.99 | 30950 / cens / cens | cens x3, 0.75-0.82 |
| complex    | 3750, 0.977      | cens 0.879 / cens 0.798 | cens 0.827 |
| quaternion | 3750, 0.994      | 41550, 0.896 | cens 0.838 |

The control groks under every algebra; the quaternionic-type groups fail (or
barely scrape the threshold) under every algebra. The effect is not
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
Unified statement, five of five stuck runs across two groups: quaternionic-
type groups stall with exactly one uncleaned 2-dim sector masking an
otherwise-formed solution, one projection repairs it, and the real-type
control has nothing to repair. Which sector interferes differs by group.

Timing (fig7): evaluating the repaired network at every step of Q8 x Z3
training shows the masked solution crossing 90% at epoch ~8,000, the same
timescale at which the character-table twin visibly groks, while observed
accuracy never gets there in 80k epochs. The network groks internally on
schedule; only cleanup of the interfering sector fails. This also rules out
a sample-complexity account: the data was sufficient all along.

## Files

- groups.py: Cayley table constructions and verification, invariants
- train.py, run_experiments.py: real-weight training and the locked grid
- complex_train.py, quaternion_train.py: parameter-matched algebra variants
- track_isotypic.py: isotypic-sector energy fractions during training
- ablate.py: sector ablation on trained networks
- isotypic_general.py: isotypic projectors for any group from its Cayley table
- analysis.py: thresholds, censoring, tables, figures
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

    python groups.py            # verify all six Cayley tables and invariants
    python run_experiments.py Z24 0 1 2      # real baseline (repeat per group)
    python complex_train.py SL23 0           # complex-weight variant
    python quaternion_train.py SL23 0        # quaternion-weight variant
    python ablate.py                         # twin-group sector ablations
    python isotypic_general.py               # general sectors + SL(2,3) ablation
    python track_isotypic.py Q8xZ3 0         # isotypic energy during training
    python analysis.py                       # tables and figures
    python chunk_train.py Q8xZ3 0 15000      # resumable checkpointed training
    python diagnose_ckpt.py                  # equilibrium diagnostics
    python diagnose.py                       # same, incl. regrowth, from scratch
    python diagnose_multi.py                 # 3-seed suite + margin decomposition
    python regrow_carryover.py 0 6000        # attractor test, optimizer carried
    python sweep_wd.py Q8xZ3 0 3.0 10000     # weight-decay dose-response
    python antirepair.py 0 10.0 pair 12000   # the intervention (fig9)

Runs are CPU-only; the full grid takes roughly an hour on a laptop.

## Equilibrium diagnostics (3 seeds per twin, 80k budget)

The three stuck runs were re-trained from scratch with checkpointing
(chunk_train.py); full-batch training is deterministic, and the 80k states
reproduce the headline table bit-for-bit (val 0.821/0.751/0.769). The
grokked D4 x Z3 controls sit at 0.994/0.977/0.988. Diagnostics in
diagnose_multi.py:

1. Repair replicates and never costs the training set: ablating the pair
   sector lifts stuck val to 0.965/0.919/0.971 (train stays >= 0.998),
   matching ablation.json exactly, and takes the control to 1.000 x3.
2. Every sector of every network sits at an AdamW equilibrium: raw loss
   gradients ~0 with cos(P grad L, P W) between -0.93 and -0.91 across all
   18 sector/seed/group measurements. Sector energy fractions stay at the
   random-init baseline in all six networks (ones 0.48-0.49,
   twin 0.19-0.19, pair 0.32-0.34). Nothing is ever cleaned up, in either group.
3. Margin decomposition (the decision margin l[y] - l[best wrong] splits
   exactly across sectors, asserted numerically): on stuck validation
   errors the pair sector contributes -3.03/-2.23/-2.80 against total
   margins of -1.20/-1.34/-1.44 -- remove its contribution and the margin
   is positive on every seed (+1.83/+0.89/+1.36). On correct examples it
   stays live (+0.57/+0.76/+0.77). In the grokked control the same
   never-cleaned sector contributes -0.03/-0.12/+0.00 on correct examples:
   functionally inert. One number separates the twins.
4. The stuck state is an attractor (fig8): projecting the pair sector out
   of the 80k states and resuming with AdamW moments and step counts
   carried (regrow_carryover.py) crashes val from 0.965/0.919/0.971 to
   0.422/0.405/0.422 within 500-750 epochs, regrows the sector's energy
   monotonically (0 -> ~0.23 by 6k, toward its 1/3 equilibrium), and
   relaxes back toward the stuck plateau (0.855/0.844/0.792 at 6k), with
   train accuracy 1.000 throughout.
5. Weight decay does not select the outcome (single seed, preliminary):
   wd = 0.3 and 3 leave Q8 x Z3 on the same stuck plateau (0.69 and 0.64
   at 20k, peaked) while the D4 x Z3 control groks at wd = 3; wd = 10 is a
   degenerate regime (train 1.0, val below chance) and is uninformative.

Reading: grokking here is not a cleanup event. Both twins hold identical
sector energies at identical gradient equilibria forever; FS type
determines whether the equilibrium *content* of the never-cleaned sector
is inert or interfering, and the interfering configuration is dynamically
stable under the full training dynamics.

## Anti-repair: the intervention (3 seeds + specificity control)

If the malign equilibrium is the failure mode, deny it from the start:
train Q8 x Z3 with loss = CE + lambda * (pair-sector weight energy), same
recipe otherwise (antirepair.py, fig9). At lambda = 10, T90 = 1100-1500 and
val reaches 1.000 by epoch 1500-2500 on all three seeds -- a perfect score
no unconstrained network in this project reaches (the healthy control tops
out at 0.994; repaired stuck nets at 0.971). Specificity: the same penalty
on the load-bearing twin sector gives val 0.468 and no grokking.

Three controls sharpen the claim (AR2_*.json). First, the kill-control:
pair suppression also accelerates the healthy twin (D4 x Z3 + penalty:
T90 = 900, 1.000 by 2200, vs 2700-3850 unconstrained), so speed alone is
not FS-specific -- the pair sector is a net liability in both groups. What
is FS-specific is the categorical change: for the quaternionic twin the
penalty converts never-groks into perfect-groks; for the real-type twin it
speeds up a run that already succeeds. Second, lambda sensitivity: the cure
works at lambda = 1 and 10 (T90 1000-1500, final 1.000) and degrades by
lambda = 30 (T90 3800, 0.96 at the 12k cap). Third, generalisation:
SL(2,3) with its quaternionic block penalised is partially rescued at
lambda = 10 (T90 3800, final 0.96) and fully cured at lambda = 3
(T90 = 2400, final 0.994, vs an unconstrained best of 0.94 reached only
after 31k epochs) -- one seed each, so labelled preliminary.

The penalty cannot be annealed off. Resuming from the perfect penalised
state with lambda = 0 (optimizer state carried), the pair sector regrows
from 0.000 toward its 0.33 equilibrium and validation collapses 1.000 ->
0.89 within 500 epochs -> 0.723 by 8k (AR2_anneal_off_s0.json). The malign
basin recaptures even a perfect solution, which retroactively explains why
unconstrained training never groks: it could not hold the solution if it
found it. The constrained subspace is where the grokked solution is stable.

## Limitations

Six groups, one group order, 3 seeds, MLPs only, one hyperparameter setting.
The FS pattern survived a character-table-matched control and an R/C/H capacity
sweep, but this is a pilot. Next: SL(2,3) anti-repair seeds 1-2 and a
per-group lambda rule (lambda ~ inverse sector trace fits the two data
points); annealing schedules or pair-freezing that hold 1.000 without a
live penalty; why the pair sector is a net liability even in the healthy
twin; order-16 (see PREDICTIONS.md); families at |G| in {48, 60, 120};
transformer replication; 5+ seeds with Kaplan-Meier delay medians; a
structural term in the norm-separation delay law.
