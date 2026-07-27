# Roadmap

Planned extensions, roughly in order.

Near term
- More seeds (5-10 per condition) with confidence intervals on delays and
  repair sizes.
- Order-16 family: Q16 vs D8 vs SD16. Q16 has a quaternionic irrep, D8 does
  not, SD16 is the semidihedral control. Same matched-order design, second
  independent test of the FS pattern.
- What selects the interfering sector. (What it *does* is now measured:
  margin decomposition shows the pair sector alone flips the stuck errors,
  3/3 seeds, and is inert in the control. When/why training commits it to
  the malign configuration is the open part; track the per-sector margin
  decomposition through the transition.)
- Anti-repair: DONE + extended (AR2_*). Kill-control run (pair penalty also
  speeds the healthy twin: categorical fail->perfect is the FS-specific
  part); lambda works at 1-10, degrades at 30; SL(2,3) fully cured at
  lambda = 3 (1 seed); anneal-off fails -- the perfect state collapses back
  to the stuck plateau, so the constraint must persist. Open: SL(2,3) seeds,
  lambda rule, anneal schedules / pair freezing.
- Transformer replication: swap the MLP for a one-layer transformer (token
  embeddings for the two arguments, attention + MLP, same six groups) to show
  the effect is not architecture-specific.
- Order-matched families at |G| = 48, 60, 120; the binary octahedral group
  gives a quaternionic case at 48.
- Per-epoch interference dynamics: extend the hidden-solution tracking to all
  stuck runs and both interfering-sector types.
- Repeat the weight-algebra sweep on the new families.

Longer term
- A structural term in the norm-separation delay law.
- A dynamical account of why cleanup fails only for quaternionic-type groups.
- Projection-repair as a practical diagnostic for stalled grokking in tasks
  with known symmetry.
