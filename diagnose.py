"""
e1_wd_factorial.py -- the experiment Section 4.3 is titled after but does not run.

THE PROBLEM
  Section 4.3 is called "The obstruction is weight decay". What Table 5 shows
  is that on a representation trained at wd = 1.0, a fresh readout fitted at
  low wd generalises better than one fitted at wd = 1.0. That is a statement
  about the refitting step, conditional on a representation that was itself
  produced under wd = 1.0. It leaves untouched: joint optimisation of W1 and
  W2, path dependence, AdamW transients, the readout tracking an earlier
  representation, interaction between representation scale and decoupled decay,
  and implicit optimiser bias among the many interpolating readouts.

  The paper's own planned-work gate 2 says exactly this:
    "Sweep training weight decay and re-measure the gap. Gate: if the gap
     persists at training weight decay 0, the mechanism is not the regulariser
     and section 4 is mis-titled."
  The section is titled as though the gate had already been passed. This runs it.

THE DESIGN
    lambda_train   in {0, 0.01, 0.1, 0.3, 1.0, 3.0}    end-to-end
    lambda_readout in {0, 0.01, 0.1, 0.3, 1.0, 3.0}    frozen-h refit
  crossed, both twins, 3 seeds. 36 hosts, 216 refits.

OUTCOMES
  (i)   gap large at lambda_train=1.0, ~0 at lambda_train=0
          -> training weight decay implicated. 4.3 survives and finally has
             its experiment.
  (ii)  gap large at EVERY lambda_train
          -> the mechanism is not the regulariser. Gate 2 has fired. Retitle
             to "Readout recovery is controlled by readout weight decay" and
             drop the causal claim.
  (iii) gap tracks lambda_train but probe accuracy falls with it too
          -> weight decay is doing two jobs (making the representation, and
             constraining the readout). Report both curves, not the difference.

WHAT ELSE THIS FIXES
  * Leak-free probe. Three-way split: fit (80% of train) / select (20% of
    train, chooses the ridge strength) / test (the held-out 173, touched once).
    exp5_readout.py selected its headline number on the test set.
  * Prespecified retrain. The headline retrained number is at
    lambda_readout = lambda_train with native init, decided before running.
    The full 6x6 matrix is reported but is not the headline.
  * Diagnostics. Train accuracy is 1.000 in every cell of Table 5, which is
    why it cannot separate the interpolating readouts. Train CE, decision
    margin, logit scale, ||W1||, ||W2||, ||h||, and effective representation
    rank can, and are recorded through training.
  * Learning curves. The paper contains no figure. This writes the histories
    figures.py needs.

RUN
    python e1_wd_factorial.py                 # ~10 min GPU, ~35 min 8-thread CPU
    python e1_wd_factorial.py --epochs 30000  # converged hosts
    python e1_wd_factorial.py --cpu
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import core
import groups24 as G24

LAM_TRAIN = [0.0, 0.01, 0.1, 0.3, 1.0, 3.0]
LAM_READOUT = [0.0, 0.01, 0.1, 0.3, 1.0, 3.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["Q8xZ3", "D8xZ3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=15000)
    ap.add_argument("--frac", type=float, default=0.70)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--readout-steps", type=int, default=4000)
    ap.add_argument("--ckpts", type=int, default=25)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="results/e1_wd_factorial.json")
    a = ap.parse_args()

    dev = core.pick_device(prefer_gpu=not a.cpu)
    # split, initialisation and probe partition use three separate seed
    # streams so none of them is confounded with the others
    runs = [dict(group=g, split_seed=1000 + s, init_seed=2000 + s, rep=s,
                 wd=lt, lam_train=lt)
            for g in a.groups for lt in LAM_TRAIN for s in a.seeds]
    B = core.Batch(runs, frac=a.frac, width=a.width, lr=a.lr, device=dev)
    print(f"device={dev}  R={B.R}  epochs={a.epochs}  "
          f"Ntr={B.Ntr} Nva={B.Nva}", flush=True)

    # three-way split, fixed before training, never redrawn
    cut = int(0.8 * B.Ntr)
    I_FIT, I_SEL = [], []
    for r in runs:                      # a distinct probe partition per run
        perm = np.random.RandomState(3000 + r["rep"]).permutation(B.Ntr)
        I_FIT.append(torch.tensor(perm[:cut], device=dev))
        I_SEL.append(torch.tensor(perm[cut:], device=dev))
    print(f"probe protocol  fit={cut}  select={B.Ntr-cut}  test={B.Nva}. "
          f"Validation is logged for the dynamics figures and drives no "
          f"selection decision.", flush=True)

    ck = set(core.log_grid(a.epochs, a.ckpts))
    hist = {i: [] for i in range(B.R)}
    t0 = time.time()
    for ep in range(a.epochs + 1):
        if ep:                       # log epoch 0 BEFORE any update, then do
            B.step()                 # exactly a.epochs updates in total
        if ep not in ck:
            continue
        with torch.no_grad():
            Htr, Hva = B.h_tr(), B.h_va()
            tr, va = B.accs()
            d = B.diagnostics()
            for i in range(B.R):
                pred, lam, _, _, _ = core.probe(
                    Htr[i][I_FIT[i]], B.yt[i][I_FIT[i]],
                    Htr[i][I_SEL[i]], B.yt[i][I_SEL[i]], Hva[i], B.n,
                    H_all=Htr[i], y_all=B.yt[i])
                pv = float((pred == B.yv[i]).float().mean())
                hist[i].append(dict(
                    epoch=ep, train=float(tr[i]), val=float(va[i]), probe=pv,
                    **{k: float(v[i]) for k, v in d.items()}))
        el = time.time() - t0
        print(f"  ep {ep:6d}/{a.epochs}  [{el:.0f}s, eta "
              f"{el/max(ep,1)*(a.epochs-ep)/60:.1f}m]  "
              f"gap@wd1.0 " + " ".join(
                  f"{hist[i][-1]['probe']-hist[i][-1]['val']:+.3f}"
                  for i in range(B.R)
                  if runs[i]["group"] == a.groups[0] and runs[i]["wd"] == 1.0),
              flush=True)

    print(f"\nhosts done in {time.time()-t0:.0f}s. Refitting readouts.",
          flush=True)
    with torch.no_grad():
        Htr, Hva = B.h_tr(), B.h_va()
    rows = []
    for i, r in enumerate(runs):
        lt = r["lam_train"]
        with torch.no_grad():
            nat = float((core.torch.bmm(Hva[i:i+1], B.W2[i:i+1].transpose(1, 2))
                         .argmax(-1)[0] == B.yv[i]).float().mean())
            nat_tr = float((core.torch.bmm(Htr[i:i+1],
                                           B.W2[i:i+1].transpose(1, 2))
                            .argmax(-1)[0] == B.yt[i]).float().mean())
        pred, lam, insel, ptr, _ = core.probe(
            Htr[i][I_FIT[i]], B.yt[i][I_FIT[i]],
            Htr[i][I_SEL[i]], B.yt[i][I_SEL[i]], Hva[i], B.n,
            H_all=Htr[i], y_all=B.yt[i])
        pv = float((pred == B.yv[i]).float().mean())
        rt = {}
        for lr_wd in LAM_READOUT:
            _, tra, vaa = core.retrain_readout(
                Htr[i], B.yt[i], Hva[i], B.yv[i], B.n, wd=lr_wd,
                steps=a.readout_steps, W0=B.W2[i])
            rt[f"{lr_wd:g}"] = dict(train=tra, val=vaa)
        d = B.diagnostics()
        rows.append(dict(group=r["group"], seed=r["rep"], lam_train=lt,
                         native=nat, native_train=nat_tr, probe=pv,
                         gap=pv - nat, probe_lam=lam, inner_sel=insel,
                         prespecified_retrain=rt[f"{lt:g}"], retrain=rt,
                         **{k: float(v[i]) for k, v in d.items()}))
        print(f"  {r['group']:6s} s{r['rep']}"
              f" lt={lt:<5g} native {nat:.3f} probe {pv:.3f} "
              f"gap {pv-nat:+.3f}  retrain@lt {rt[f'{lt:g}']['val']:.3f} "
              f"retrain@0 {rt['0']['val']:.3f}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(rows=rows, hist={str(k): v for k, v in hist.items()},
                   config=vars(a), lam_train=LAM_TRAIN,
                   lam_readout=LAM_READOUT,
                   wall=round(time.time() - t0, 1)), open(a.out, "w"))

    print("\n" + "=" * 84)
    print("E1 -- probe minus native by TRAINING weight decay (mean over seeds)")
    print("=" * 84)
    hdr = f"{'lam_train':>10s}"
    for g in a.groups:
        hdr += f" | {g[:6]:>6s} nat  probe    gap"
    print(hdr)
    for lt in LAM_TRAIN:
        line = f"{lt:>10g}"
        for g in a.groups:
            sub = [x for x in rows if x["group"] == g and x["lam_train"] == lt]
            line += (f" | {np.mean([s['native'] for s in sub]):10.3f}"
                     f" {np.mean([s['probe'] for s in sub]):6.3f}"
                     f" {np.mean([s['gap'] for s in sub]):+6.3f}")
        print(line)

    print("\nFULL FACTORIAL, group %s: retrained val by (lam_train, lam_readout)"
          % a.groups[0])
    print(f"{'lt \\ lr':>10s} " + " ".join(f"{lr:>7g}" for lr in LAM_READOUT))
    for lt in LAM_TRAIN:
        sub = [x for x in rows if x["group"] == a.groups[0]
               and x["lam_train"] == lt]
        print(f"{lt:>10g} " + " ".join(
            f"{np.mean([s['retrain'][f'{lr:g}']['val'] for s in sub]):7.3f}"
            for lr in LAM_READOUT))

    print("\nGATE 2. If the gap in column 1 does not collapse as lam_train -> 0,")
    print("the mechanism is NOT the regulariser and section 4.3 must be retitled")
    print('to "Readout recovery is controlled by readout weight decay".')
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
