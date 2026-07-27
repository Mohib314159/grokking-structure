# Replication in the repo's own torch code (not the NumPy rewrite)

torch 2.13.0, 1 CPU thread, `train.run_one` unmodified except for `frac`.

## D_crit boundary — both twins, torch

    Q8xZ3  frac 0.75  no grok, peak 0.861
    Q8xZ3  frac 0.78  GROKS  T_gen 7200, final 0.945
    Q8xZ3  frac 0.80  GROKS  T_gen 5550, final 0.974

    D4xZ3  frac 0.60  seed 0 no grok 0.844 | seed 1 no grok 0.779
    D4xZ3  frac 0.65  seed 0 GROKS 5850    | seed 1 GROKS 7200

Same boundaries as the NumPy rewrite (Q8xZ3 between 0.75 and 0.78; D4xZ3
between 0.60 and 0.65). Two independent implementations, same answer.
The published claim that Q8xZ3 "never groks" is false in the repo's own code.

## The stronger result: all six groups at FIXED data (frac 0.60), 3 seeds

Dose-response at a fixed budget, which avoids the threshold-sensitivity of a
grok / no-grok binary near D_crit.

    group    q-mass  d_max   k   final val acc, seeds 0/1/2
    Z24       0        1     24   0.991  (grokked, T_gen 2250)
    D12       0        2      9   1.000  (grokked, T_gen 1500)
    D4xZ3     0        2     15   0.840  0.758  0.801
    S4        0        3      5   0.762  0.745  0.827
    Q8xZ3    1/6       2     15   0.355  0.303  0.247
    SL23     1/6       3      7   0.078  0.121  0.100

ZERO OVERLAP. Non-quaternionic 0.745-1.000; quaternionic 0.078-0.355.
Chance is 1/24 = 0.042, so SL23 is near chance at this data budget.

Two matched comparisons inside the table:
  * D4xZ3 vs Q8xZ3 — identical character table, k=15, d_max=2.
    0.758-0.840 vs 0.247-0.355, non-overlapping across 3 seeds each.
  * S4 vs SL23 — d_max=3 both.
    0.745-0.827 vs 0.078-0.121, non-overlapping.

k(G) and d_max are both non-monotone in the outcome (D12 with k=9 beats
D4xZ3 with k=15; Z24 with d_max=1 and S4 with d_max=3 both land above every
quaternionic group). Quaternionic mass is the only column that separates
cleanly.

## Honest limits on this table

- One train fraction (0.60), 12-15k epochs. A snapshot of a dose-response
  curve, not the curve. Run 0.50/0.55/0.60/0.65/0.70 for the real figure.
- Effect size varies WITHIN the quaternionic class: SL23 (0.10) is much worse
  than Q8xZ3 (0.30) at equal q-mass, so q-mass alone does not set the size of
  the effect, only its sign.
- Six groups at one order. The order-16 pair supports the direction; its
  high-fraction tail is retracted (see CORRECTIONS.md).
