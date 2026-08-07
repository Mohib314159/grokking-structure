"""Dose-response: final val accuracy vs train fraction, all six groups.

Replaces the original grok / no-grok binary, which was threshold-sensitive and
-- worse -- confounded with the critical dataset size. Q8xZ3 does not "fail to
grok"; it groks at frac >= 0.78. What representation type shifts is the DATA
REQUIREMENT, and a dose-response curve is the right way to show that.

Fixed budget (9k epochs, eval every 100) at every fraction so the points are
comparable. 2 seeds per cell.

    python dose_response.py          # run the sweep -> results/dose_response.json
    python dose_response.py --plot   # figure only, from stored results
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FRACS = (0.50, 0.60, 0.70)
SEEDS = (0, 1)
BUDGET, EVERY = 9000, 100
ORDER = ["Z24", "D12", "D4xZ3", "S4", "Z2xA4", "Q8xZ3", "Dic24", "SL23"]
QMASS = {"Z24": 0.0, "D12": 0.0, "D4xZ3": 0.0, "S4": 0.0, "Z2xA4": 0.0,
         "Q8xZ3": 1/6, "Dic24": 0.5, "SL23": 1/6}
LABEL = {"Z24": "Z$_{24}$", "D12": "D$_{12}$", "S4": "S$_4$",
         "D4xZ3": "D$_4{\\times}$Z$_3$", "Q8xZ3": "Q$_8{\\times}$Z$_3$", "SL23": "SL(2,3)",
         "Z2xA4": "Z$_2{\\times}$A$_4$", "Dic24": "Dic$_{24}$"}
OUT = "results/dose_response.json"


def sweep():
    import extra_groups
    from train import run_one
    import torch
    extra_groups.install()   # ORDER contains Z2xA4 and Dic24, which live here
    torch.set_num_threads(1)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for f in FRACS:
        for g in ORDER:
            for s in SEEDS:
                k = f"{g}|{f}|{s}"
                if k in res:
                    continue
                r = run_one(g, s, frac=f, max_epochs=BUDGET, eval_every=EVERY,
                            verbose_every=10 ** 9)
                res[k] = [r["T_gen"], r["final_val_acc"]]
                print(f"{g:7s} frac {f} seed {s}  final {r['final_val_acc']:.3f}", flush=True)
                json.dump(res, open(OUT, "w"))
    return res


def plot(res):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for g in ORDER:
        ys = [np.mean([res[f"{g}|{f}|{s}"][1] for s in SEEDS]) for f in FRACS]
        lo = [min(res[f"{g}|{f}|{s}"][1] for s in SEEDS) for f in FRACS]
        hi = [max(res[f"{g}|{f}|{s}"][1] for s in SEEDS) for f in FRACS]
        quat = QMASS[g] > 0
        ax.plot(FRACS, ys, "o-" if not quat else "s--",
                color="C3" if quat else "C0", lw=2.0 if quat else 1.4,
                ms=6, label=f"{LABEL[g]}" + ("  (quaternionic)" if quat else ""))
        ax.fill_between(FRACS, lo, hi, color="C3" if quat else "C0", alpha=0.10)
        ax.annotate(LABEL[g], (FRACS[-1], ys[-1]), textcoords="offset points",
                    xytext=(6, -3), fontsize=8,
                    color="C3" if quat else "C0")
    ax.axhline(1 / 24, color="0.5", ls=":", lw=1)
    ax.text(0.505, 1 / 24 + 0.012, "chance (1/24)", fontsize=7.5, color="0.4")
    ax.set_xlabel("train fraction of the 576 composition pairs")
    ax.set_ylabel("final validation accuracy (9k epochs)")
    ax.set_xlim(0.485, 0.735)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(FRACS)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Quaternionic representation type raises the data requirement\n"
                 "solid = FS type real/complex only, dashed = contains a quaternionic irrep",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig("figs/fig10_dose_response.png", dpi=160)
    print("wrote figs/fig10_dose_response.png")

    from math import comb
    print("\n=== separation check ===")
    for f in FRACS:
        nq = [res[f"{g}|{f}|{s}"][1] for g in ORDER if QMASS[g] == 0 for s in SEEDS]
        q = [res[f"{g}|{f}|{s}"][1] for g in ORDER if QMASS[g] > 0 for s in SEEDS]
        print(f"  frac {f}: non-quaternionic [{min(nq):.3f}, {max(nq):.3f}] "
              f"(n={len(nq)})  vs  quaternionic [{min(q):.3f}, {max(q):.3f}] "
              f"(n={len(q)})   "
              f"{'SEPARATED' if min(nq) > max(q) else 'OVERLAP'}")
    nQ = sum(1 for g in ORDER if QMASS[g] > 0)
    p = 1 / comb(len(ORDER), nQ)
    print(f"\n  Unit of replication is the GROUP, not the run: seeds and fractions\n"
          f"  resample the same {len(ORDER)} tasks, so quoting run counts is\n"
          f"  pseudoreplication. Exact one-sided test:\n"
          f"    P(the {nQ} quaternionic groups are the bottom {nQ} of {len(ORDER)} "
          f"by chance) = 1/C({len(ORDER)},{nQ}) = {p:.4f}"
          f"   {'SIGNIFICANT at 0.05' if p < 0.05 else 'NOT significant'}")


if __name__ == "__main__":
    res = json.load(open(OUT)) if "--plot" in sys.argv else sweep()
    plot(res)
