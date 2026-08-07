# grokking-structure

Does the algebra of a task change how much data a neural network needs to learn it?

One-hidden-layer MLPs are trained to complete the multiplication table of a
finite group. Group order, dataset size, input dimension and output cardinality
are held fixed across every condition, so the only thing that varies is the
group. The central instrument is the **character-table twin**: two groups whose
ordinary complex character tables are identical, so that number of conjugacy
classes, irreducible dimensions and commuting probability are matched by
construction, but whose real group algebras differ.

For the dihedral and quaternion groups of order 8,

    R[D8] = R^4 + M2(R),        R[Q8] = R^4 + H

Both `M2(R)` and `H` complexify to `M2(C)`, which is why the ordinary character
table cannot separate them. The Frobenius–Schur indicator can, because it needs
the power map `chi(g^2)`, which the character table as a matrix does not
contain.

---

## Results

**All fifteen groups of order 24.** (Accuracy at a fixed data and training
budget, not a measured sample complexity — no dense ρ-by-time sweep has been
run.) The Small Groups Library lists fifteen
isomorphism classes of order 24; `groups24.py` constructs fifteen pairwise
non-isomorphic groups matching that count. Distinct fingerprints establish that
there are at least fifteen; that there are at most fifteen is the external
classification result. Five carry
quaternionic-type irreducible representations: SL(2,3), Dic24, Q8×Z3, Z2×Dic12
and Z3⋊Z8. At training fraction 0.60 with twelve instances per group, ranked by
mean final validation accuracy, those five
occupy the bottom five positions with no inversion — rank sum 15, the minimum
attainable. The preregistered exact one-sided ordering test gives
**p = 1/C(15,5) = 3.3 × 10⁻⁴**; the preregistered secondary permutation test on
group means gives the same, with a difference of +0.558.

The boundary is close: Z2×A4 (0.685, non-quaternionic) sits 0.011 above
Z2×Dic12 (0.674, quaternionic), about a third of the within-group standard
deviation there. Eight of the fifteen groups generated the hypothesis, so this
is a preregistered extension to the complete population, not a wholly
prospective confirmation; the genuinely unseen part is the seven new groups, and
both new quaternionic ones landed in the bottom five as predicted.

Between-group variance exceeds within-group variance by a factor of 92, where
an instance draws an independent random relabelling of the group, an independent
train/validation split and an independent initialisation. Under this protocol,
variation from those three sources is much smaller than variation between
groups. This is a descriptive comparison, not a fitted variance decomposition.

**Both character-table twin pairs separate, in the predicted direction.**

| pair | non-quaternionic | quaternionic | Δ |
|---|---|---|---|
| D8×Z3 / Q8×Z3 | 0.739 | 0.271 | +0.469 |
| D24 / Dic24 | 0.999 | 0.198 | +0.801 |
| *Z2×D12 / Z2×Dic12 (not a twin)* | *0.998* | *0.674* | *+0.324* |

There are exactly two character-table twin pairs at order 24 and both are among
the groups on which the hypothesis was formed, so they are descriptive evidence
rather than a prospective test; no sign test is quoted over them. The third row
is matched only on the multisets of irreducible degrees and conjugacy-class
sizes. It is **not** a twin: Z2×Dic12 has abelianisation Z2×Z4 and four linear
characters taking values ±i, while every linear character of Z2×D12 is real. It
was preregistered and had never been trained, and it separates in the predicted
direction, but it does not replicate the twin design.

**q-mass predicts the sign of the effect, not its size.** The fraction of the
regular representation carried by quaternionic sectors takes four values here.
Mean accuracy by level is 0.869 (q = 0), 0.228 (1/6), 0.674 (1/3), 0.198 (1/2) —
correlation −0.697, but not monotone.

**Weight decay has two roles and they point in opposite directions.** Crossing
end-to-end weight decay with the weight decay used to refit a readout on the
frozen hidden layer (Q8×Z3, ρ = 0.70, refit accuracy):

```
   lam_train \ lam_readout      0    0.01     0.1     0.3       1       3
                       0    0.013   0.012   0.012   0.002   0.002   0.000
                     0.1    0.276   0.266   0.150   0.143   0.123   0.044
                       0.3   0.798   0.796   0.736   0.586   0.539   0.561
                       1    0.879   0.881   0.842   0.761   0.734   0.765
                       3    0.821   0.815   0.769   0.667   0.640   0.672
```

Span down the first column is 0.865; span across the λ_train = 1 row is 0.146.
Higher end-to-end weight decay is associated with substantially greater linear
decodability at this fixed budget — probe accuracy on the frozen layer runs
0.162 → 0.992 as λ_train goes 0 → 1 — while readout weight decay limits how much
of it any linear readout expresses. We have not separated regularisation from
the effective optimisation timescale, the logit scale, or convergence at a
matched epoch budget. At λ_train = 1 the
native readout reaches 0.730; refitting at the training decay recovers +0.004,
refitting at zero recovers +0.148. In the grokking twin the same contrast is
+0.002 and +0.023 from a native 0.975.

**What a probe gap measures depends on the probe.** Thirty readout
configurations are fitted to one unchanged frozen Q8×Z3 representation, all
reaching training accuracy 1.000, each evaluated on a test split held out from
every fitting and selection decision. Eight non-redundant settings:

| recipe | Q8×Z3 gap | D8×Z3 gap |
|---|---|---|
| AdamW, wd 1.0, 200 epochs (Chou et al. 2026, A.1.3) | −0.021 | −0.079 |
| AdamW, wd 1.0, 4000 epochs | +0.008 | +0.000 |
| AdamW, wd 1.0, lr 2e-3 | +0.002 | +0.000 |
| AdamW, wd 0.1 | +0.125 | +0.025 |
| AdamW, wd 0.01 | **+0.181** | +0.027 |
| AdamW, wd 0 | +0.177 | +0.027 |
| ridge, λ chosen inside the training split | **+0.235** | −0.033 |

The reported gap spans 0.256 on a representation that never changes within a
seed (three seeds, three representations, averaged). The endpoints differ in
objective, optimiser, budget and initialisation as well as regularisation, so
this is sensitivity to the probe protocol as a whole; the clean
regularisation-only contrast is the 0.153 span of the factorial above. Under the protocol of Chou et al., who define the sub-optimal-readout signature
this quantity is meant to detect, Q8×Z3 reports −0.021, and at that weight decay
run to convergence, +0.002: no defect, and no difference between the twins
either. The cleanest single contrast is the weight decay alone — +0.002 at
λ_probe = 1.0 against +0.181 at λ_probe = 0.01, with objective, optimiser,
budget and initialisation all fixed. The gap is a
property of (network, probe regulariser), not of the network.

The same run checks the representation-degradation signature that Chou et al.
use to classify a grokking setup as spurious. Probe accuracy rises from near zero to 0.994–1.000 through training and
effective rank rises with it (57–60 → 69–71), so that signature is absent in all
six runs. The sub-optimal-readout signature is present in 3/3 Q8×Z3 seeds and
1/3 D8×Z3 seeds — rarer and much smaller in the grokking twin, not cleanly
absent.

**Sector localisation is partial and not specific.** Writing `W2 = sum_S P_S W2`
against the real isotypic projectors, exactly, and transplanting one sector's
block between the native and refitted readouts with the hidden layer frozen
(Q8×Z3, native 0.738, refitted 0.881, three seeds):

| sector | sufficiency | necessity |
|---|---|---|
| trace-8 complex pair | 50.0% | 37.8% |
| trace-4 Frobenius–Schur-typed | 60.8% | 44.6% |
| group-equivariant rank-8 control | 0.0% | 4.1% |

In the grokking twin the whole repair range is 0.031, too small for the
fractions to mean anything, so no localisation is claimed there.

The two algebraically meaningful sectors carry comparable amounts and neither
dominates. A random rank-8 subspace destroys the readout entirely (0.09–0.10),
which is expected, since a random subspace of R²⁴ is not group-equivariant and
cuts across every isotypic component; it is reported for scale, not as a control.

---

## Layout

```
groups24.py              all 15 groups of order 24, isotypic projectors,
                         Cayley-table invariants, completeness check
core.py                  batched trainer, leak-free probes, readout refitting
e1_wd_factorial.py       training x readout weight decay
e2_full_population.py    all 15 groups, variance decomposition, group-level tests
e3_probe_protocol.py     probe families, Chou signatures, sector transplants
analyse.py               every table and figure, regenerated from results/
make_paper.ps1/.bat      regenerate figures, then build paper/paper.pdf
test_equivalence.py      batched trainer vs a single model on stock torch AdamW
PREREG.md                predictions written before the new groups were trained
paper/paper.tex          the write-up
results/                 raw histories
figs/                    generated

train.py groups.py extra_groups.py ablate.py isotypic_general.py
dose_response.py analysis.py track_isotypic.py antirepair.py order16.py
o16_lib.py diagnose*.py sweep_wd.py regrow_carryover.py chunk_train.py
complex_train.py quaternion_train.py run_experiments.py
                         the original codebase, which produced everything in
                         results/ whose filename is not e1_/e2_/e3_
```

## Running it

```bash
pip install -r requirements.txt

python groups24.py                  # 5s,  verifies the 15 groups and the 3 twin pairs
python e2_full_population.py        # ~30m, all 15 groups at rho = 0.60
python e1_wd_factorial.py           # ~13m, the weight-decay factorial
python e3_probe_protocol.py         # ~11m, probes, signatures, transplants
python analyse.py                   # every table and figure; fails if a result is missing
python test_equivalence.py          # 1m, batched trainer vs stock torch.optim.AdamW
```

To rebuild the paper, run `make_paper.bat` (or `.\make_paper.ps1`) from the
repository root. It regenerates `figs/` from `results/` and then compiles
`paper/paper.pdf`. Note that PowerShell rejects `&&` as a statement separator,
so `cd paper && pdflatex paper.tex` will not work there; use `;` or the script.

`analyse.py` writes `figs/`, which `paper/paper.tex` includes, so run it before
compiling the paper. It exits nonzero if a required result file is absent rather
than quietly reporting on a subset; `--allow-missing` overrides that.

Timings are for eight CPU threads on the default settings. CUDA is used
automatically when available; `--cpu` forces CPU and `GS_THREADS=8` sets the
thread count. `e2_full_population.py --arms` adds two control arms — relabelling
held fixed, and the original design where one seed drives both split and
initialisation — and `--dense` sweeps ρ for an interval-censored critical
training fraction, which takes several hours.

## Setup

Task: for a group `G` with `|G| = n`, the model receives `(g, h)` as two
concatenated one-hot vectors in `R^2n` and predicts `gh` as one of `n` classes.
Every group here has `n = 24`, so the dataset is 576 pairs and the output space
is 24 classes in every condition. A fraction ρ is used for training.

Model: one hidden layer, width 512, ReLU, no biases. Full-batch AdamW,
lr = 2×10⁻³, β = (0.9, 0.98), weight decay 1.0 unless swept, cross-entropy.

Groups: Cayley tables are built from presentations and checked programmatically
— unique identity, existence of inverses, exhaustive associativity over all
24³ = 13,824 triples, and the Frobenius–Schur identity
`sum_rho eps_rho d_rho = #{g : g² = e}`. The fifteen are separated by an
isomorphism fingerprint combining element-order profile, conjugacy-class sizes,
isotypic sector types, centre order and derived-subgroup order.

Projectors: class sums span the centre of `R[G]`. A random symmetric central
element is scalar on each real isotypic component, so its eigenspaces are those
components. The Frobenius–Schur indicator of an irreducible character is
`nu = (1/|G|) sum_g chi(g²)` in {+1, 0, −1}. The quantity attached to a real
isotypic sector is `s = (1/|G|) sum_g chi_S(g²) = d·nu`: `+d` for real type, `0`
for a merged conjugate pair, `−d` for quaternionic type. Idempotency, integer trace,
integer `s` and `sum_S P_S = I` are asserted numerically at construction, so the
logit decomposition is exact to floating point rather than an attribution
heuristic.

Probe protocol: every readout is fitted on the model's **full** training
split — the same data the native readout saw. The training split is also
partitioned into a fit set (80%) and a selection set (20%) purely to choose the
ridge strength, which is then refitted on the whole training split. All other
recipes are fixed in advance and reported separately rather than selected. Validation accuracy is logged through training so
the dynamics figures can be drawn, and drives no selection decision; the split,
initialisation and probe partition are driven by three separate seed streams.

Batching: `core.py` trains many runs in one batched job. The cross-entropy is
multiplied by the number of runs so each run's gradient equals what it would be
if that run were trained alone. Without this the loss averages over `R*N`
examples, every gradient is scaled by `1/R`, and Adam's `eps` becomes an
effective `R * 1e-8` — comparable to `sqrt(v)` once gradients are small, which
is exactly the phase these experiments measure.

## Limitations

The design does not isolate the Frobenius–Schur indicator. Character-table twins
match the ordinary complex character table; they do not match the power map, the
involution count, the subgroup lattice or the element-order profile, all of which
a network can read off the Cayley table. Involution count in particular is
strongly anti-correlated with quaternionic type on these pairs — D24 has 13,
Dic24 has 1. Frobenius–Schur type is a candidate predictor in this population,
not an isolated cause. Separating them needs an out-of-sample predictive
comparison against the competing invariants that `groups24.invariants` computes.

Fifteen groups is fifteen groups, and one order. Three twin pairs is three, and
there are no more at order 24. Strengthening either requires other orders.

Which sector interferes is not predictable in advance. It was identified by
scanning sectors and reporting the one that repairs, and a different sector
repairs SL(2,3). Until a rule exists this is localisation, not explanation, and
no preregistered prediction of the sector identity is possible.

One architecture, one optimiser, one recipe. A TransformerLens port trains but
had not converged on the harder groups at the budgets tested, so no
architecture-transfer claim is made.

Files in `results/` that are not `e1_`, `e2_` or `e3_` come from the original
codebase and use its naming: `D12` for the dihedral group of order 24, `D4xZ3`
for `D8xZ3`. Four quantities that appear in earlier write-ups have no raw
history there — the ρ = 0.78 and 0.80 runs, the D8×Z3 boundary at 0.60 and 0.65,
the width sweep, and the order-16 family.

## Scope

This is a controlled algebraic testbed, not a claim about frontier models.
Interpretability of small models on algorithmic tasks is a mature area and
nothing here should be assumed to transfer. What the setting buys is exactness:
the projectors come from the Cayley table, sum to the identity, and make the
decomposition verifiable to floating point rather than arguable.

The transferable question, if there is one, is what a linear probe measures. A
model can contain a computation that its own output map does not express, and
whether a probe reports that depends on how the probe is regularised — by 0.378
of accuracy here, on one unchanged representation, with a published protocol
landing on the side that reports nothing. Distinguishing capability absent from
capability present but unexpressed matters for evaluation and elicitation, and it
cannot be done with an unspecified probe.

## References

1. Power, Burda, Edwards, Babuschkin, Misra. *Grokking: Generalization Beyond
   Overfitting on Small Algorithmic Datasets.* arXiv:2201.02177, 2022.
2. Nanda, Chan, Lieberum, Smith, Steinhardt. *Progress Measures for Grokking via
   Mechanistic Interpretability.* ICLR 2023. arXiv:2301.05217.
3. Chughtai, Chan, Nanda. *A Toy Model of Universality.* ICML 2023.
4. Stander, Yu, Fan, Biderman. *Grokking Group Multiplication with Cosets.*
   ICML 2024. arXiv:2312.06581.
5. Varma, Shah, Kenton, Kramár, Kumar. *Explaining Grokking Through Circuit
   Efficiency.* arXiv:2309.02390, 2023.
6. Chou, Uzdelewicz, Chiu, Yang, Chung. *Two Speeds of Learning: A
   Representation–Readout Decomposition of Grokking and Double Descent.*
   arXiv:2605.27078, 2026.
7. Tikeng Notsawo, Dumas, Rabusseau. *Grokking Finite-Dimensional Algebra.*
   ICML 2026. arXiv:2602.19533.
8. Truong. *What Does the Weight Norm Control in Grokking? Logit-Scale Mediation
   under Cross-Entropy.* arXiv:2606.18465, 2026.

## Licence

MIT, see `LICENSE`.
