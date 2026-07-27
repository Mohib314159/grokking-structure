# What I got wrong, found by attacking my own results

Four adversarial checks on the D_crit claim I produced. One killed a result.

## RETRACTED: the order-16 high-fraction tail

At |G| = 16 the validation set shrinks as the train fraction grows:

    frac  0.80  0.85  0.90  0.94  0.97
    nval    52    39    26    16     8

"SD16 reaches 0.875 at frac 0.97" is **7 correct out of 8**. "Q16 0.500" is
**4 out of 8**. The claim "SD16's D_crit is above 0.97" rests on eight
examples and must be withdrawn. Anything at frac >= 0.90 at order 16 is noise.

Usable order-16 evidence is frac <= 0.85 only (nval >= 39):

    frac 0.80 (nval 52)   D16 1.000 (grokked, T_gen 1050) | SD16 0.250 | Q16 0.115
    frac 0.85 (nval 39)   D16 1.000 (grokked, T_gen  850) | SD16 0.410 | Q16 0.103

The FS ordering at order 16 therefore rests on final-accuracy separation at
moderate fraction, NOT on measured D_crit crossings for SD16 and Q16. Say so.

## SURVIVED: censoring artifact

Q8xZ3 at frac 0.75 was called "no grok" from a 20k budget. Re-run to 60k on
two seeds: still no grok, peak 0.882 / 0.889. Not a censoring artifact --
but note it plateaus just under the 0.90 threshold, so the binary at this
boundary is threshold-sensitive. The dose-response is the honest statistic:
final val jumps 0.88 -> 0.96 between frac 0.75 and 0.78.

## SURVIVED: seed robustness at the boundary

    D4xZ3 frac 0.65   seeds 0/1/2 -> T_gen 5800 / 5700 / 10800   ALL GROK
    Q8xZ3 frac 0.78   seeds 0/1/2 -> T_gen 5000 / 6550 /  6950   ALL GROK
    Q8xZ3 frac 0.75   seeds 0/1   -> no grok at 60k, peak 0.88

Claim the SEPARATION, not precise intervals: D4xZ3 groks at frac 0.65 on 3/3
seeds; Q8xZ3 needs 0.78. A gap of >= 13 points of train fraction, >= 75 pairs.

## SURVIVED: model-capacity confound

Two papers (arXiv 2402.15175, 2605.09724) report D_crit falling with model
size. Everything here was width 512. Swept at frac 0.70:

    width   256    D4xZ3 grok T_gen 6050  |  Q8xZ3 no grok, 0.549
    width   512    D4xZ3 grok T_gen ~3850 |  Q8xZ3 no grok, 0.757
    width  1024    D4xZ3 grok T_gen 2400  |  Q8xZ3 no grok, 0.827

Separation survives a 4x width sweep. Q8xZ3 improves monotonically with width
but never crosses. Not a capacity artifact.

## Still unfixed -- state these as limitations

- All D_crit numbers come from an independent NumPy reimplementation, not the
  repo's torch path. Seed -> init mapping differs, so T_gen values are not
  comparable run-for-run with the published table (only distributions are).
- n = 2 twin pairs, 2 group orders. The core claim rests on two experiments.
- The grok/no-grok binary is threshold-dependent near the boundary. Report
  final-val-vs-fraction curves, not threshold crossings.
- SD16 and Q16 never grok at any fraction with a usable validation set, so
  the FS 0 vs FS -1 ordering is weaker evidence than the +1 vs -1 ordering.
- D4xZ3's lower boundary (frac 0.60) is single-seed. Do not quote it.
