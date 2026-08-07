"""
e3_probe_protocol.py -- Finding B, rebuilt so it survives review, plus the
result that makes it novel again.

=========================================================================
PART 1 -- THE PROBE GAP IS NOT WELL DEFINED, AND THAT IS THE FINDING
=========================================================================
Chou et al. (arXiv 2605.27078, May 2026) decompose a network into encoder
and linear readout and define four diagnostic signatures, of which

    signature 3 (sub-optimal readout): "LP accuracy substantially exceeds
    model accuracy, indicating that the readout is under-performing
    relative to what the encoder affords"

is exactly Finding B. They run it on permutation composition in S_5 on an
MLP -- the nearest published neighbour to this task. So "the representation
generalises but the readout does not express it" cannot be claimed as new.

But their probe (their Appendix A.1.3) is fitted with

    AdamW, lr 1e-3, betas (0.9, 0.98), WEIGHT DECAY 1.0, 200 full-batch
    epochs, W ~ N(0, 1/N), no bias,

justified as: "The strong weight decay ensures that high probe accuracy
reflects genuinely structured representations rather than memorization."

That is the lambda_readout = 1.0 cell of this repo's own sweep, where the
measured recovery is ZERO. So their protocol, applied to Q8xZ3, reports
signature 3 as ABSENT on a network where the defect demonstrably exists.

The claim to make is therefore not "we found a probe gap". It is:

    The linear-probe gap is a property of (network, probe regulariser), not
    of the network. On one frozen representation, unchanged, the reported
    gap runs from 0.00 to about +0.25 as the probe's regularisation is
    relaxed -- including exactly zero under the protocol of the paper that
    introduced the diagnostic.

This script measures that curve directly. It fits, on the SAME frozen
representation, every probe family used in the literature:

    chou_200    AdamW wd=1.0, lr 1e-3, 200 epochs, N(0,1/N) init   [their recipe]
    chou_4000   the same for 4000 epochs                            [is 200 undertrained?]
    adamw_wd    AdamW at wd in {1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0}
    ridge       closed form, lambda in 1e-6 .. 1e6
    ridge_cv    lambda chosen on a split INSIDE train, then refitted on all of it

  ALL of these are fitted on the model's FULL training split, the same data the
  native readout saw. The fit/select partition is used only to choose the ridge
  strength. An earlier version fitted them on the 80% subset, which handed the
  native readout 20% more data than the probe and confounded the comparison.

and reports the gap for each. The spread across rows IS the result.

=========================================================================
PART 2 -- THE LEAK
=========================================================================
exp5_readout.py selects its headline retrained number with

    if va_a > best_va: best_va, best_W2 = va_a, W2_r

over 8 configurations, where va_a is VALIDATION accuracy -- the metric it
then reports. best_W2 also feeds the Table 6 sector transplants. That is
selection on the reported metric, twice.

Here the split is three-way and fixed before training:

    fit     80% of train      fits the probe
    select  20% of train      chooses the hyperparameter
    test    the held-out 173  touched once, at the end

and every reported configuration is named in advance.

=========================================================================
PART 3 -- CHOU'S OWN SPURIOUSNESS TEST, RUN ON OURSELVES
=========================================================================
Their section 4 retires the Omnigrok MNIST grokking example as SPURIOUS,
using signature 2 (representation DEGRADATION: falling LP accuracy with
rising critical dimension) under a recipe that "actively degrades
representation quality". A referee will aim that at wd = 1.0 at rho = 0.70.

We pre-empt it by running their discriminant on ourselves: track probe
accuracy and an effective-rank proxy through training. Signature 2 requires
probe accuracy to FALL. If ours rises while the native readout stalls, the
regime is signature 3, not signature 2, and the twin control at the
identical recipe groks to 0.98. Report the curves, not the assurance.

=========================================================================
PART 4 -- TRANSPLANTS, ON A PRESPECIFIED READOUT
=========================================================================
Sector transplantation W2_mix = P_S W2_donor + (I - P_S) W2_host with W1
frozen. Donor is the PRESPECIFIED repaired readout (lambda_readout = 0,
native init), not an argmax over the test set. Controls: the FS-typed
trace-4 twin sector, a group-equivariant rank-8 subspace assembled from
other sectors, and a random rank-8 subspace (weak, reported for scale only).

RUN
    python e3_probe_protocol.py                 # ~8 min GPU, ~25 min CPU
    python e3_probe_protocol.py --epochs 30000
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import core
import groups24 as G24

RIDGE_GRID = [10.0 ** k for k in range(-6, 7)]
WD_GRID = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.0]


# ------------------------------------------------------------ probe families

def _acc(H, W, y):
    return float(((H @ W.T).argmax(-1) == y).float().mean())


def fit_adamw_probe(H, y, n, wd, steps, lr, init, seed=0):
    """A readout fitted by AdamW on a frozen representation.

    weight_decay goes to torch.optim.AdamW, so this is AdamW by construction
    rather than by hand-rolled decay -- which matters here, because the whole
    point of the comparison is that the recipe is reproduced faithfully.

    init='chou'   -> W ~ N(0, 1/N), their Appendix A.1.3
    init='unif'   -> U(+-1/sqrt(d))
    a tensor      -> start from that readout (e.g. the native W2)
    """
    d = H.shape[1]
    g = torch.Generator().manual_seed(7000 + seed)
    if isinstance(init, torch.Tensor):
        W = init.detach().clone()
    elif init == "chou":
        W = (torch.randn(n, d, generator=g) / math.sqrt(d)).to(H.device, H.dtype)
    else:
        b = 1.0 / math.sqrt(d)
        W = ((torch.rand(n, d, generator=g) * 2 - 1) * b).to(H.device, H.dtype)
    W.requires_grad_(True)
    opt = torch.optim.AdamW([W], lr=lr, betas=(0.9, 0.98), weight_decay=wd)
    for _ in range(steps):
        loss = F.cross_entropy(H @ W.T, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return W.detach()


@torch.no_grad()
def readout_stats(W, H_fit, y_fit, H_eval, y_eval):
    """What separates two readouts that both interpolate the training split.

    A referee will say the weakly regularised readout wins by being large,
    ill-conditioned or overconfident on 173 examples. These are the numbers
    that answer that: norm, margins on both splits, evaluation cross-entropy,
    and mean confidence on the evaluated set.
    """
    def margin(H, y):
        Z = H @ W.T
        rr = torch.arange(len(y), device=H.device)
        true = Z[rr, y]
        Z = Z.clone()
        Z[rr, y] = -1e30
        return float((true - Z.max(-1).values).mean())
    Ze = H_eval @ W.T
    return dict(w_norm=float(W.norm()),
                margin_fit=margin(H_fit, y_fit),
                margin_eval=margin(H_eval, y_eval),
                ce_eval=float(F.cross_entropy(Ze, y_eval)),
                confidence=float(Ze.softmax(-1).max(-1).values.mean()))


def probe_family_sweep(Hf, yf, Hs, ys, Ht, yt, n, W_native, Hall, yall):
    """Every probe recipe in the literature, on ONE frozen representation.
    Hf/Hs are the fit and select splits (both inside train); Ht is the
    untouched test split."""
    res = {}

    # -- Chou et al. 2605.27078, Appendix A.1.3, verbatim
    def record(tag, W, recipe):
        # `fit` is accuracy on the FULL training split, which is what every
        # recipe here is fitted on. Reporting it on the 80% subset would not
        # support the claim that all of them interpolate the training data.
        res[tag] = dict(test=_acc(Ht, W, yt), fit=_acc(Hall, W, yall),
                        fit_subset=_acc(Hf, W, yf), recipe=recipe,
                        **readout_stats(W, Hall, yall, Ht, yt))

    # Chou et al. fit the probe on the probed model's OWN training set, using
    # the same split as the model. So these are fitted on Hall (all 403), not
    # on the 80% fit subset -- otherwise the probe sees 20% less data than the
    # native readout and the comparison is confounded.
    for tag, steps in (("chou_200", 200), ("chou_4000", 4000)):
        W = fit_adamw_probe(Hall, yall, n, wd=1.0, steps=steps, lr=1e-3,
                            init="chou")
        record(tag, W, f"AdamW lr1e-3 wd1.0 N(0,1/N) {steps} epochs "
                       f"[Chou et al. A.1.3]")

    # -- AdamW at the repo's lr, sweeping the readout decay
    for wd in WD_GRID:
        W = fit_adamw_probe(Hall, yall, n, wd=wd, steps=4000, lr=2e-3,
                            init="unif")
        record(f"adamw_wd{wd:g}", W, f"AdamW lr2e-3 wd{wd:g} 4000 ep, "
                                     f"full training split")
    # -- and from the native readout, which is the repo's "retrained" column
    for wd in WD_GRID:
        W = fit_adamw_probe(Hall, yall, n, wd=wd, steps=4000, lr=2e-3,
                            init=W_native)
        record(f"adamw_native_wd{wd:g}", W,
               f"AdamW from native init, wd{wd:g}")

    # -- closed-form ridge across the whole lambda grid
    mk_all = core.ridge_family(Hall, yall, n)
    for lam in RIDGE_GRID:
        W = mk_all(lam).to(Ht.dtype)
        record(f"ridge_{lam:g}", W, f"closed-form ridge lambda={lam:g}, "
                                    f"full training split")
    mk = core.ridge_family(Hf, yf, n)
    # -- ridge with lambda selected INSIDE train (this repo's headline)
    best, blam = -1.0, RIDGE_GRID[0]
    for lam in RIDGE_GRID:
        a = _acc(Hs, mk(lam).to(Hs.dtype), ys)
        if a > best:
            best, blam = a, lam
    # lambda selected on the inner split, then REFITTED on the whole training
    # split so this probe also sees all 403 examples
    W = mk_all(blam).to(Ht.dtype)
    record("ridge_cv", W, f"ridge, lambda={blam:g} selected on the inner split "
                          f"then refitted on the full training split")
    res["ridge_cv"].update(lam=blam, inner_sel=best)
    return res


# ---------------------------------------------------------------- transplants

def transplant(W_host, W_donor, P):
    return P @ W_donor + (torch.eye(P.shape[0], device=P.device,
                                    dtype=P.dtype) - P) @ W_host


def equivariant_rank(secs, exclude, target, device, dtype):
    chosen, tot = [], 0
    for i, s in enumerate(secs):
        if i in exclude or tot >= target:
            continue
        if tot + s["trace"] <= target:
            chosen.append(i)
            tot += s["trace"]
    if tot != target:
        return None, []
    P = sum(secs[i]["P"] for i in chosen)
    return torch.tensor(P, device=device, dtype=dtype), chosen


def random_P(n, rank, seed, device, dtype):
    g = torch.Generator().manual_seed(9000 + seed)
    Q, _ = torch.linalg.qr(torch.randn(n, rank, generator=g))
    return (Q @ Q.T).to(device, dtype)


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["Q8xZ3", "D8xZ3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=15000)
    ap.add_argument("--frac", type=float, default=0.70)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--ckpts", type=int, default=25)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="results/e3_probe_protocol.json")
    a = ap.parse_args()

    dev = core.pick_device(prefer_gpu=not a.cpu)
    # split, initialisation and probe partition are driven by three separate
    # seed streams so none of them is confounded with the others
    runs = [dict(group=g, split_seed=1000 + s, init_seed=2000 + s, rep=s)
            for g in a.groups for s in a.seeds]
    B = core.Batch(runs, frac=a.frac, width=a.width, lr=a.lr, wd=a.wd,
                   device=dev)
    print(f"device={dev}  R={B.R}  epochs={a.epochs}", flush=True)

    cut = int(0.8 * B.Ntr)
    I_FIT, I_SEL = [], []
    for r in runs:                      # a distinct probe partition per run
        perm = np.random.RandomState(3000 + r["rep"]).permutation(B.Ntr)
        I_FIT.append(torch.tensor(perm[:cut], device=dev))
        I_SEL.append(torch.tensor(perm[cut:], device=dev))
    print(f"three-way split: fit={cut} select={B.Ntr-cut} test={B.Nva}. "
          f"Validation is logged through training for the dynamics figures "
          f"and is used for NO selection decision; every hyperparameter is "
          f"chosen on the selection split.", flush=True)

    # ---- train, tracking Chou's discriminant through training
    ck = set(core.log_grid(a.epochs, a.ckpts))
    track = {i: [] for i in range(B.R)}
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
            track[i].append(dict(epoch=ep, train=float(tr[i]),
                                 val=float(va[i]),
                                 probe=float((pred == B.yv[i]).float().mean()),
                                 **{k: float(v[i]) for k, v in d.items()}))
        el = time.time() - t0
        print(f"  ep {ep:6d}/{a.epochs} [{el:.0f}s eta "
              f"{el/max(ep,1)*(a.epochs-ep)/60:.1f}m]  "
              f"{a.groups[0]} probe/native "
              + " ".join(f"{track[i][-1]['probe']:.2f}/{track[i][-1]['val']:.2f}"
                         for i in range(B.R) if runs[i]["group"] == a.groups[0]),
              flush=True)

    with torch.no_grad():
        Htr, Hva = B.h_tr(), B.h_va()
    out = dict(config=vars(a), track={str(k): v for k, v in track.items()},
               rows=[])

    # ---- PART 1+2: the probe-family sweep
    print("\n" + "=" * 92)
    print("PROBE FAMILY SWEEP -- one frozen representation, every recipe")
    print("=" * 92)
    for i, r in enumerate(runs):
        with torch.no_grad():
            nat = float((Hva[i] @ B.W2[i].T).argmax(-1).eq(B.yv[i])
                        .float().mean())
        fam = probe_family_sweep(Htr[i][I_FIT[i]], B.yt[i][I_FIT[i]],
                                 Htr[i][I_SEL[i]], B.yt[i][I_SEL[i]],
                                 Hva[i], B.yv[i], B.n, B.W2[i],
                                 Htr[i], B.yt[i])
        row = dict(group=r["group"], seed=r["rep"], native=nat,
                   family={k: dict(v, gap=v["test"] - nat)
                           for k, v in fam.items()})
        out["rows"].append(row)
        print(f"\n--- {r['group']} s{r['split_seed']}   native = {nat:.3f}")
        print(f"    {'recipe':34s} {'test':>7s} {'gap':>7s}   {'fit':>6s}")
        for k in ("chou_200", "chou_4000", "ridge_cv", "ridge_0.0001",
                  *[f"adamw_wd{w:g}" for w in WD_GRID]):
            if k in fam:
                v = fam[k]
                print(f"    {k:34s} {v['test']:7.3f} {v['test']-nat:+7.3f} "
                      f"  {v['fit']:6.3f}")

    # aggregate: the headline table
    print("\n" + "=" * 92)
    print("THE POINT: probe minus native, mean over seeds, by probe recipe")
    print("=" * 92)
    keys = ["chou_200", "chou_4000"] + [f"adamw_wd{w:g}" for w in WD_GRID] \
        + ["ridge_cv"]
    print(f"{'recipe':22s}" + "".join(f"{g:>14s}" for g in a.groups))
    for k in keys:
        line = f"{k:22s}"
        for g in a.groups:
            sub = [r for r in out["rows"] if r["group"] == g]
            line += f"{np.mean([s['family'][k]['gap'] for s in sub]):+14.3f}"
        print(line)
    print("\nThe SPREAD DOWN THE FIRST COLUMN is the result. The gap is a")
    print("property of (network, probe regulariser), not of the network. Row")
    print("`chou_200` is the protocol of the paper that introduced the")
    print("diagnostic; if it reads ~0.000 while `ridge_cv` reads ~+0.25, the")
    print("diagnostic reports NO defect on a network that has one.")

    # ---- PART 3: Chou's spuriousness discriminant, run on ourselves
    print("\n" + "=" * 92)
    print("CHOU SIGNATURE CHECK (their section 4). Signature 2 = SPURIOUS")
    print("grokking requires probe accuracy to FALL and effective rank to RISE.")
    print("=" * 92)
    for i, r in enumerate(runs):
        t = track[i]
        mid = [x for x in t if x["train"] >= 0.995]
        if not mid:
            continue
        p0, p1 = mid[0]["probe"], t[-1]["probe"]
        k0, k1 = mid[0]["h_rank_eff"], t[-1]["h_rank_eff"]
        sig2 = (p1 < p0) and (k1 > k0)
        print(f"  {r['group']:7s} s{r['split_seed']}  probe "
              f"{p0:.3f} -> {p1:.3f}   eff-rank {k0:.1f} -> {k1:.1f}   "
              f"signature 2 (degradation): {'PRESENT' if sig2 else 'absent'}"
              f"   signature 3 (sub-optimal readout): "
              f"{'PRESENT' if t[-1]['probe'] - t[-1]['val'] > 0.05 else 'absent'}")

    # ---- PART 4: transplants on a prespecified donor
    print("\n" + "=" * 92)
    print("SECTOR TRANSPLANTS (donor prespecified: lambda_readout=0, native init)")
    print("=" * 92)
    tp = {}
    for g in a.groups:
        secs = G24.isotypic_sectors(G24.cayley(g))
        idx = [i for i, r in enumerate(runs) if r["group"] == g]
        i_pair = next((j for j, s in enumerate(secs) if s["trace"] == 8), None)
        i_twin = next((j for j, s in enumerate(secs) if s["trace"] == 4), None)
        if i_pair is None:
            print(f"  {g}: no trace-8 sector -- the readout defect of section 4 "
                  f"cannot operate here. Skipping.")
            continue
        P_pair = torch.tensor(secs[i_pair]["P"], device=dev, dtype=B.W2.dtype)
        P_twin = torch.tensor(secs[i_twin]["P"], device=dev, dtype=B.W2.dtype)
        P_eq, chosen = equivariant_rank(secs, {i_pair, i_twin}, 8, dev,
                                        B.W2.dtype)
        tp[g] = []
        for i in idx:
            W_rep = fit_adamw_probe(Htr[i], B.yt[i], B.n, wd=0.0, steps=4000,
                                    lr=2e-3, init=B.W2[i])
            nat = B.W2[i]
            cell = dict(native=_acc(Hva[i], nat, B.yv[i]),
                        repaired=_acc(Hva[i], W_rep, B.yv[i]))
            conds = [("pair_t8", P_pair), ("twin_t4", P_twin),
                     ("rand8", random_P(B.n, 8, runs[i]["rep"], dev,
                                        B.W2.dtype))]
            if P_eq is not None:
                conds.insert(2, ("equivar_r8", P_eq))
            for lbl, P in conds:
                cell[f"{lbl}_repaired+native_block"] = _acc(
                    Hva[i], transplant(W_rep, nat, P), B.yv[i])
                cell[f"{lbl}_native+repaired_block"] = _acc(
                    Hva[i], transplant(nat, W_rep, P), B.yv[i])
            tp[g].append(cell)
        m = {k: np.mean([c[k] for c in tp[g]]) for k in tp[g][0]}
        print(f"\n  {g}  (equivariant control = sectors {chosen})")
        for k, v in m.items():
            print(f"    {k:34s} {v:.3f}")
    out["transplants"] = tp

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print(f"\n-> {a.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
