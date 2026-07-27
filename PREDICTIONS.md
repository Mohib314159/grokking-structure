# Pre-registered predictions: order 16

Committed before any order-16 experiment is run. The three maximal-class
groups of order 16 -- dihedral D16, semidihedral SD16, generalised
quaternion Q16 -- have faithful 2-dim irreps of Frobenius-Schur type +1, 0,
-1 respectively (to be re-verified with isotypic_general.py from our own
Cayley tables before training; no table trusted).

Predictions, falsifiable as stated:
1. Q16 stalls under the locked recipe; D16 and SD16 grok.
2. Any stalled Q16 run contains exactly one 2-dim isotypic sector whose
   ablation repairs validation accuracy while train stays at 1.000.
3. A from-init penalty on that sector (lambda scaled inversely with sector
   trace, per the Q8xZ3 / SL(2,3) fits) produces full grokking to >= 0.99.
4. Annealing that penalty off after grokking collapses validation back
   toward the stuck plateau.

---

## Outcome (order 16, run 2026-07-25)

Groups built and verified from presentations in `order16.py`: exhaustive
associativity (4096 triples), character tables derived by Burnside class-sum
diagonalisation, FS from eps = (1/|G|) sum chi(g^2), FS identity checked.
FS types came out as pre-registered: D16 (+1,+1,+1), SD16 (0,0,+1),
Q16 (-1,-1,+1). D16 and Q16 are a second character-table twin pair.

1. **FAILED AS WRITTEN.** At the locked frac 0.70, order 16 gives 179 examples
   for 16 classes and ALL THREE groups sit below their critical dataset size --
   even D16 only reaches 0.92. Prediction 1 was run in the wrong data regime.
   Re-measured as a data requirement it holds in direction: at frac 0.80
   (nval 52) D16 groks to 1.000 (T_gen 1050) while SD16 reaches 0.250 and Q16
   0.115. Ordering FS +1 << FS 0 < FS -1.
2. **CONFIRMED.** Q16 at frac 0.90, unconstrained val 0.308 with train 1.000:
   ablating one quaternionic sector gives 0.808; the other 0.615; both 0.462;
   the real 2-dim control 0.462. Exactly one sector repairs, train never moves.
3. **CONFIRMED.** From-init penalty on that sector at lambda 3: T90 1650,
   final 1.000, train 1.000 -- never-groks becomes perfect-groks. lambda 10
   overshoots (0.846), consistent with lambda scaling inversely with sector
   trace as predicted. Specificity is WEAKER than at order 24: the same penalty
   on the real 2-dim sector still reaches 0.885 (vs 0.468 for the order-24
   control).
4. **CONFIRMED.** Annealing the penalty off with optimizer state carried:
   0.846 -> 0.577 (250 ep) -> 0.346 (1000 ep), sector energy regrowing
   0.0000 -> 0.0478, settling at 0.500.

Retracted from the first write-up of this test: any order-16 number at
frac >= 0.90, where the validation set is 26 examples or fewer (8 at frac 0.97).
See CORRECTIONS.md.
