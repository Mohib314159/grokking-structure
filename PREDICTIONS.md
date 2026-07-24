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
