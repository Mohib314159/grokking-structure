"""
analyse.py -- every number and figure in the paper, regenerated from results/.

    python analyse.py            all sections, writes figs/
    python analyse.py --tables   numbers only, no plotting

Reads:
    results/e1_wd_factorial.json      weight-decay factorial
    results/e2_full_population.json   all 15 groups of order 24
    results/e3_probe_protocol.json    probe families, signatures, transplants
    results/dose_response.json        legacy dose-response grid
    results/<GROUP>_seed<k>.json      legacy main grid

Writes:
    figs/twins.png  figs/population.png  figs/wd_factorial.png
    figs/two_speeds.png  figs/probe_families.png  figs/transplant.png
"""
import argparse
import itertools
import json
import os
from math import comb

import sys

import numpy as np

R = "results"
F = "figs"
# Verified character-table twins. Z2xD12 / Z2xDic12 is NOT one -- Z2xDic12 has
# abelianisation Z2 x Z4 and four linear characters taking values +-i -- so it is
# reported separately as a degree- and class-size-matched contrast.
TWINS = [("D8xZ3", "Q8xZ3"), ("D24", "Dic24")]
CONTRASTS = [("Z2xD12", "Z2xDic12")]
ALIAS = {"D24": "D12", "D8xZ3": "D4xZ3"}          # names used in the legacy files


MISSING = []


def load(name, required=True):
    """Missing results are an error, not a silent skip. A table regenerated
    from 'whatever happens to be present' is how a paper ends up quoting
    numbers that no longer exist in the repository."""
    p = os.path.join(R, name)
    if os.path.exists(p):
        return json.load(open(p))
    if required:
        MISSING.append(name)
    return None


def hr(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


# ---------------------------------------------------------------- statistics

def rank_sum_test(means, positive):
    """Exact one-sided Mann-Whitney on GROUP means: probability that the
    |positive| named groups have a rank sum this small under exchangeability.

    This is the right test here. The bottom-k ordering statistic is brittle --
    a single inversion of 0.001 destroys it -- while the rank sum uses the
    whole ranking and is still exact. The unit of replication is the group.
    """
    names = sorted(means, key=lambda g: means[g])
    rank = {g: i + 1 for i, g in enumerate(names)}
    k = len(positive)
    obs = sum(rank[g] for g in positive)
    hit = sum(1 for c in itertools.combinations(range(1, len(names) + 1), k)
              if sum(c) <= obs)
    return obs, hit / comb(len(names), k), names


def mean_permutation(means, positive):
    """Exact permutation over all C(n,k) labellings, difference in group means."""
    names = sorted(means)
    m = np.array([means[g] for g in names])
    pos = np.array([g in positive for g in names])
    obs = m[~pos].mean() - m[pos].mean()
    k = int(pos.sum())
    hit = tot = 0
    for idx in itertools.combinations(range(len(names)), k):
        mask = np.zeros(len(names), bool)
        mask[list(idx)] = True
        tot += 1
        hit += (m[~mask].mean() - m[mask].mean()) >= obs
    return obs, hit / tot


# ------------------------------------------------------------------ sections

def section_population(d, plot):
    if d is None:
        print("results/e2_full_population.json absent -- run e2_full_population.py")
        return
    rows = d["main"]["rows"]
    qm = d["qmass"]
    groups = sorted({r["group"] for r in rows})
    by = {g: np.array([r["final_val"] for r in rows if r["group"] == g])
          for g in groups}
    means = {g: float(by[g].mean()) for g in groups}
    quat = {g for g in groups if qm[g] > 0}
    n_inst = len(by[groups[0]])

    hr(f"ALL {len(groups)} GROUPS OF ORDER 24  "
       f"(rho={d['config']['frac']}, {d['config']['epochs']} epochs, "
       f"{n_inst} instances)")
    order = sorted(groups, key=lambda g: means[g])
    print(f"{'rank':>4s} {'group':11s} {'q-mass':>7s} {'mean':>7s} {'sd':>6s} "
          f"{'min':>6s} {'max':>6s}")
    for i, g in enumerate(order):
        print(f"{i+1:>4d} {g:11s} {qm[g]:7.4f} {means[g]:7.3f} "
              f"{by[g].std(ddof=1):6.3f} {by[g].min():6.3f} {by[g].max():6.3f}"
              f"{'   quaternionic' if g in quat else ''}")

    U, p_rank, _ = rank_sum_test(means, quat)
    obs, p_perm = mean_permutation(means, quat)
    print(f"\nrank sum of the {len(quat)} quaternionic-type groups: {U} "
          f"(minimum possible {sum(range(1, len(quat)+1))})")
    print(f"exact one-sided rank-sum test:      p = {p_rank:.3e}   "
          f"[1/C({len(groups)},{len(quat)}) = {1/comb(len(groups),len(quat)):.3e}]")
    print(f"exact permutation on group means:   p = {p_perm:.3e}, "
          f"difference {obs:+.3f}")

    bottom = set(order[:len(quat)])
    if bottom == quat:
        print(f"bottom-{len(quat)} ordering: clean")
    else:
        intr, esc = sorted(bottom - quat), sorted(quat - bottom)
        marg = min(abs(means[a] - means[b]) for a in intr for b in esc)
        print(f"bottom-{len(quat)} ordering: FAILS. {intr} intrude, {esc} "
              f"escape, margin {marg:.3f}. No exact ordering p is quoted.")

    print(f"\n{'pair':30s} {'non-quat':>10s} {'quat':>8s} "
          f"{'delta':>8s} {'Welch t':>9s}")
    ok = 0
    for a, b in TWINS + CONTRASTS:
        if a not in by or b not in by:
            continue
        se = np.sqrt(by[a].var(ddof=1) / len(by[a]) + by[b].var(ddof=1) / len(by[b]))
        dl = means[a] - means[b]
        ok += dl > 0
        tag = "" if (a, b) in TWINS else "  (not a twin)"
        print(f"{a+' / '+b+tag:30s} {means[a]:10.3f} {means[b]:8.3f} "
              f"{dl:+8.3f} {dl/se:9.1f}")
    print(f"{ok}/{len(TWINS) + len(CONTRASTS)} in the predicted direction. There "
          f"are exactly two character-table\ntwin pairs at order 24 and both are "
          f"among the groups that generated the\nhypothesis, so no sign test is "
          f"quoted over them.")

    vb = np.var([means[g] for g in groups], ddof=1)
    vw = np.mean([by[g].var(ddof=1) for g in groups])
    print(f"\nvariance: between-group {vb:.4f}, within-group {vw:.4f}, "
          f"ratio {vb/vw:.1f}x  (instance = independent relabelling, split "
          f"and initialisation)")

    levels = sorted({qm[g] for g in groups})
    r = np.corrcoef([qm[g] for g in groups], [means[g] for g in groups])[0, 1]
    print(f"\nq-mass as a dose: corr = {r:+.3f}")
    for L in levels:
        sel = [g for g in groups if abs(qm[g] - L) < 1e-9]
        print(f"    q = {L:.4f}  n={len(sel)}  mean {np.mean([means[g] for g in sel]):.3f}"
              f"   {sorted(sel)}")
    print("monotone in q-mass: "
          + ("yes" if all(np.mean([means[g] for g in groups if abs(qm[g]-levels[i])<1e-9])
                          >= np.mean([means[g] for g in groups if abs(qm[g]-levels[i+1])<1e-9])
                          for i in range(len(levels)-1)) else "NO"))

    if plot:
        fig_population(means, qm, by, order, quat)
        fig_twins(means, by)


def section_factorial(d, plot):
    if d is None:
        print("results/e1_wd_factorial.json absent -- run e1_wd_factorial.py")
        return
    rows, LT, LR = d["rows"], d["lam_train"], d["lam_readout"]
    groups = sorted({r["group"] for r in rows}, key=lambda g: g != "Q8xZ3")

    hr(f"WEIGHT-DECAY FACTORIAL  (rho={d['config']['frac']}, "
       f"{d['config']['epochs']} epochs, {len(d['config']['seeds'])} seeds)")
    print("Native accuracy and leak-free probe accuracy are reported "
          "separately.\nTheir difference is NOT comparable across lambda_train: "
          "the probe ceiling moves.")
    print(f"\n{'lam_train':>10s}" + "".join(
        f" | {g+' native':>15s} {'probe':>6s}" for g in groups))
    for lt in LT:
        line = f"{lt:>10g}"
        for g in groups:
            s = [r for r in rows if r["group"] == g and r["lam_train"] == lt]
            line += (f" | {np.mean([x['native'] for x in s]):15.3f} "
                     f"{np.mean([x['probe'] for x in s]):6.3f}")
        print(line)

    for g in groups:
        print(f"\nretrained validation accuracy, {g} "
              f"(rows lambda_train, columns lambda_readout)")
        print(f"{'':>8s}" + "".join(f"{lr:>8g}" for lr in LR))
        M = {}
        for lt in LT:
            s = [r for r in rows if r["group"] == g and r["lam_train"] == lt]
            M[lt] = [np.mean([x["retrain"][f"{lr:g}"]["val"] for x in s])
                     for lr in LR]
            print(f"{lt:>8g}" + "".join(f"{v:8.3f}" for v in M[lt]))
        rs = max(M[lt][0] for lt in LT) - min(M[lt][0] for lt in LT)
        cs = max(M[1.0]) - min(M[1.0])
        nat1 = np.mean([r["native"] for r in rows
                        if r["group"] == g and r["lam_train"] == 1.0])
        print(f"  span across lambda_train at lambda_readout=0 : {rs:.3f}")
        print(f"  span across lambda_readout at lambda_train=1 : {cs:.3f}"
              f"   ratio {rs/max(cs,1e-9):.1f}x")
        print(f"  at lambda_train=1: native {nat1:.3f}; refit at the training "
              f"decay {M[1.0][LR.index(1.0)]:.3f} "
              f"({M[1.0][LR.index(1.0)]-nat1:+.3f}); refit at zero "
              f"{M[1.0][0]:.3f} ({M[1.0][0]-nat1:+.3f})")

    if plot:
        fig_factorial(d, rows, LT, LR, groups)


def section_probes(d, plot):
    if d is None:
        print("results/e3_probe_protocol.json absent -- run e3_probe_protocol.py")
        return
    rows = d["rows"]
    groups = sorted({r["group"] for r in rows}, key=lambda g: g != "Q8xZ3")
    keys = ["chou_200", "chou_4000", "adamw_wd1", "adamw_wd0.3", "adamw_wd0.1",
            "adamw_wd0.03", "adamw_wd0.01", "adamw_wd0.003", "adamw_wd0",
            "ridge_cv"]
    keys = [k for k in keys if k in rows[0]["family"]]

    hr("PROBE FAMILIES ON ONE FROZEN REPRESENTATION")
    for g in groups:
        s = [r for r in rows if r["group"] == g]
        print(f"  {g} native accuracy: "
              + " ".join(f"{r['native']:.3f}" for r in s))
    print(f"\n{'probe recipe':16s}" + "".join(f"{g:>12s}" for g in groups)
          + "   fit acc")
    for k in keys:
        line = f"{k:16s}"
        for g in groups:
            s = [r for r in rows if r["group"] == g]
            line += f"{np.mean([x['family'][k]['gap'] for x in s]):+12.3f}"
        fit = np.mean([r["family"][k]["fit"] for r in rows])
        print(line + f"   {fit:.3f}")
    g0 = groups[0]
    s = [r for r in rows if r["group"] == g0]
    gaps = [np.mean([x["family"][k]["gap"] for x in s]) for k in keys]
    print(f"\nspread on {g0}: {max(gaps)-min(gaps):.3f} "
          f"(from {min(gaps):+.3f} to {max(gaps):+.3f}) on ONE unchanged "
          f"representation.\nEvery probe reaches training accuracy "
          f"{np.mean([r['family'][k]['fit'] for r in rows for k in keys]):.3f}: "
          f"they interpolate and\ngeneralise differently.")

    track = d.get("track", {})
    if track:
        print("\nrepresentation-degradation check (Chou et al. section 4):")
        print(f"  {'run':14s} {'probe start':>11s} {'probe end':>10s} "
              f"{'rank start':>11s} {'rank end':>9s}  verdict")
        cfg = d["config"]
        for i, r in enumerate(rows):
            t = track.get(str(i))
            if not t:
                continue
            mid = [x for x in t if x["train"] >= 0.995] or t
            deg = t[-1]["probe"] < mid[0]["probe"] and \
                t[-1]["h_rank_eff"] > mid[0]["h_rank_eff"]
            sub = t[-1]["probe"] - t[-1]["val"] > 0.05
            print(f"  {r['group']+' s'+str(r['seed']):14s} "
                  f"{mid[0]['probe']:11.3f} {t[-1]['probe']:10.3f} "
                  f"{mid[0]['h_rank_eff']:11.1f} {t[-1]['h_rank_eff']:9.1f}  "
                  f"{'DEGRADATION' if deg else 'no degradation'}"
                  f"{', sub-optimal readout' if sub else ''}")

    tp = d.get("transplants", {})
    if tp:
        hr("SECTOR TRANSPLANTS  (donor fixed in advance: "
           "lambda_readout=0, native init)")
        for g, cells in tp.items():
            m = {k: float(np.mean([c[k] for c in cells])) for k in cells[0]}
            rng = m["repaired"] - m["native"]
            print(f"\n  {g}: native {m['native']:.3f}, repaired "
                  f"{m['repaired']:.3f}, range {rng:.3f}")
            print(f"    {'sector':18s} {'native+repaired':>16s} "
                  f"{'sufficiency':>12s} {'repaired+native':>16s} "
                  f"{'necessity':>10s}")
            if rng < 0.05:
                print("    repair range is too small for the fractions to mean "
                      "anything; absolute accuracies only.")
            for lbl in ("pair_t8", "twin_t4", "equivar_r8", "rand8"):
                a, b = f"{lbl}_native+repaired_block", f"{lbl}_repaired+native_block"
                if a not in m:
                    continue
                if rng < 0.05 or lbl == "rand8":
                    # a random rank-8 subspace is not group-equivariant, so it
                    # cuts across isotypic components and destroys any working
                    # readout. Reporting it as a percentage of the repair range
                    # is meaningless; it is here only for scale.
                    print(f"    {lbl:18s} {m[a]:16.3f} {'--':>12s} "
                          f"{m[b]:16.3f} {'--':>10s}"
                          f"{'  (destroys the readout)' if lbl == 'rand8' else ''}")
                    continue
                suf = (m[a] - m["native"]) / rng
                nec = (m["repaired"] - m[b]) / rng
                print(f"    {lbl:18s} {m[a]:16.3f} {suf*100:11.1f}% "
                      f"{m[b]:16.3f} {nec*100:9.1f}%")
    if plot:
        fig_probe_families(rows, groups, keys)
        if track:
            fig_two_speeds(d)
        if tp:
            fig_transplant(tp)


def section_legacy():
    d = load("dose_response.json", required=False)
    if d is None:
        return
    hr("LEGACY DOSE-RESPONSE GRID (results/dose_response.json)")
    gs, fs = [], []
    for k in d:
        g, f, _ = k.split("|")
        if g not in gs:
            gs.append(g)
        if f not in fs:
            fs.append(f)
    print(f"{'group':10s}" + "".join(f"{'rho='+f:>16s}" for f in fs))
    for g in gs:
        line = f"{g:10s}"
        for f in fs:
            # stored value is [T_gen, final_val]; T_gen may be null
            v = [d[k][-1] if isinstance(d[k], list) else d[k]
                 for k in d if k.startswith(f"{g}|{f}|")]
            line += f"{' / '.join(f'{x:.3f}' for x in v):>16s}"
        print(line)


# -------------------------------------------------------------------- figures

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "figure.dpi": 160,
                         "axes.spines.top": False, "axes.spines.right": False})
    os.makedirs(F, exist_ok=True)
    return plt


def fig_population(means, qm, by, order, quat):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(order))
    cols = ["#c1121f" if g in quat else "#8d99ae" for g in order]
    ax.bar(x, [means[g] for g in order], color=cols, width=0.7, zorder=2)
    for i, g in enumerate(order):
        ax.scatter(np.full(len(by[g]), i), by[g], s=4, c="k", alpha=0.5,
                   zorder=3, linewidths=0)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=55, ha="right")
    ax.set_ylabel("final validation accuracy")
    ax.axhline(1 / 24, ls=":", c="k", lw=0.8)
    ax.text(0.2, 1 / 24 + 0.015, "chance", fontsize=7)
    ax.set_title("All 15 groups of order 24 (red = quaternionic-type), "
                 r"$\rho=0.60$, 12 instances")
    fig.tight_layout()
    fig.savefig(f"{F}/population.png")
    print(f"  wrote {F}/population.png")


def fig_twins(means, by):
    plt = _mpl()
    pairs = [(a, b) for a, b in TWINS + CONTRASTS if a in by and b in by]
    fig, axes = plt.subplots(1, len(pairs), figsize=(2.4 * len(pairs), 2.9),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (a, b) in zip(axes, pairs):
        for i, (g, c) in enumerate(((a, "#8d99ae"), (b, "#c1121f"))):
            ax.bar(i, means[g], color=c, width=0.6, zorder=2)
            ax.scatter(np.full(len(by[g]), i), by[g], s=6, c="k", alpha=0.6,
                       zorder=3, linewidths=0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([a, b], rotation=20, ha="right")
        lbl = "" if (a, b) in TWINS else "  (not a twin)"
        ax.set_title(f"$\\Delta$ = {means[a]-means[b]:+.3f}{lbl}", fontsize=8.5)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("final validation accuracy")
    fig.suptitle("The two character-table twin pairs at order 24, and one "
                 "degree- and class-size-matched contrast that is not a twin",
                 fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{F}/twins.png")
    print(f"  wrote {F}/twins.png")


def fig_factorial(d, rows, LT, LR, groups):
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for ax, g in zip(axes[:2], groups):
        nat = [np.mean([r["native"] for r in rows
                        if r["group"] == g and r["lam_train"] == lt]) for lt in LT]
        prb = [np.mean([r["probe"] for r in rows
                        if r["group"] == g and r["lam_train"] == lt]) for lt in LT]
        xs = np.arange(len(LT))
        ax.plot(xs, prb, "o-", c="#023e8a", label="probe (frozen features)")
        ax.plot(xs, nat, "s-", c="#c1121f", label="native readout")
        ax.fill_between(xs, nat, prb, color="#adb5bd", alpha=0.35, lw=0)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{lt:g}" for lt in LT])
        ax.set_xlabel(r"training weight decay $\lambda_{\rm train}$")
        ax.set_title(g)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("validation accuracy")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    g = groups[0]
    M = np.array([[np.mean([r["retrain"][f"{lr:g}"]["val"] for r in rows
                            if r["group"] == g and r["lam_train"] == lt])
                   for lr in LR] for lt in LT])
    im = axes[2].imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axes[2].set_xticks(range(len(LR)))
    axes[2].set_xticklabels([f"{v:g}" for v in LR])
    axes[2].set_yticks(range(len(LT)))
    axes[2].set_yticklabels([f"{v:g}" for v in LT])
    axes[2].set_xlabel(r"readout weight decay $\lambda_{\rm readout}$")
    axes[2].set_ylabel(r"$\lambda_{\rm train}$")
    axes[2].set_title(f"{g}: refit accuracy")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            axes[2].text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                         fontsize=6.5, c="w" if M[i, j] < 0.6 else "k")
    fig.colorbar(im, ax=axes[2], fraction=0.045)
    fig.tight_layout()
    fig.savefig(f"{F}/wd_factorial.png")
    print(f"  wrote {F}/wd_factorial.png")


def fig_two_speeds(d):
    plt = _mpl()
    rows, track = d["rows"], d["track"]
    groups = sorted({r["group"] for r in rows}, key=lambda g: g != "Q8xZ3")
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.0),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, groups):
        for i, r in enumerate(rows):
            if r["group"] != g or str(i) not in track:
                continue
            t = track[str(i)]
            e = [max(x["epoch"], 1) for x in t]
            ax.plot(e, [x["probe"] for x in t], c="#023e8a", lw=1.1, alpha=0.8)
            ax.plot(e, [x["val"] for x in t], c="#c1121f", lw=1.1, alpha=0.8)
            ax.plot(e, [x["train"] for x in t], c="#adb5bd", lw=0.9, alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(g)
        ax.set_ylim(-0.02, 1.05)
    axes[0].set_ylabel("accuracy")
    axes[0].plot([], [], c="#023e8a", label="probe on frozen features")
    axes[0].plot([], [], c="#c1121f", label="native validation")
    axes[0].plot([], [], c="#adb5bd", label="train")
    axes[0].legend(frameon=False, fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(f"{F}/two_speeds.png")
    print(f"  wrote {F}/two_speeds.png")


def fig_probe_families(rows, groups, keys):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = np.arange(len(keys))
    w = 0.38
    for j, (g, c) in enumerate(zip(groups, ["#c1121f", "#8d99ae"])):
        s = [r for r in rows if r["group"] == g]
        v = [np.mean([x["family"][k]["gap"] for x in s]) for k in keys]
        ax.barh(y + (j - 0.5) * w, v, height=w, color=c, label=g, zorder=2)
    ax.axvline(0, c="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(keys)
    ax.invert_yaxis()
    ax.set_xlabel("probe accuracy minus native accuracy")
    ax.set_title("One frozen representation, ten probe recipes")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{F}/probe_families.png")
    print(f"  wrote {F}/probe_families.png")


def fig_transplant(tp):
    plt = _mpl()
    gs = list(tp)
    fig, axes = plt.subplots(1, len(gs), figsize=(4.0 * len(gs), 3.0))
    axes = np.atleast_1d(axes)
    labs = ["pair_t8", "twin_t4", "equivar_r8", "rand8"]
    for ax, g in zip(axes, gs):
        cells = tp[g]
        m = {k: float(np.mean([c[k] for c in cells])) for k in cells[0]}
        ax.axhline(m["native"], c="#c1121f", ls="--", lw=1, label="native")
        ax.axhline(m["repaired"], c="#023e8a", ls="--", lw=1, label="repaired")
        x = np.arange(len(labs))
        ax.bar(x - 0.2, [m.get(f"{l}_native+repaired_block", np.nan) for l in labs],
               0.4, color="#023e8a", label="native + repaired block")
        ax.bar(x + 0.2, [m.get(f"{l}_repaired+native_block", np.nan) for l in labs],
               0.4, color="#8d99ae", label="repaired + native block")
        ax.set_xticks(x)
        ax.set_xticklabels(labs, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(g)
    axes[0].set_ylabel("validation accuracy")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{F}/transplant.png")
    print(f"  wrote {F}/transplant.png")


# ----------------------------------------------------------------------- main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", action="store_true", help="skip the figures")
    ap.add_argument("--allow-missing", action="store_true",
                    help="report on whatever is present instead of failing")
    a = ap.parse_args()
    plot = not a.tables
    section_population(load("e2_full_population.json"), plot)
    section_factorial(load("e1_wd_factorial.json"), plot)
    section_probes(load("e3_probe_protocol.json"), plot)
    section_legacy()
    print()
    for f, need in (("e2_full_population.json", 180),
                    ("e1_wd_factorial.json", 36),
                    ("e3_probe_protocol.json", 6)):
        d = load(f, required=False)
        if d is None:
            continue
        got = (len(d["main"]["rows"]) if "main" in d else len(d["rows"]))
        if got != need:
            print(f"WARNING {f}: {got} rows, expected {need}")
        if f.startswith("e1") and not d.get("hist"):
            print(f"WARNING {f}: no checkpoint histories -- this is not a raw "
                  f"output of e1_wd_factorial.py")
        if f.startswith("e2") and not any(r.get("curve") for r in d["main"]["rows"]):
            print(f"WARNING {f}: no training curves -- this is not a raw "
                  f"output of e2_full_population.py")
        if f.startswith("e3") and len(d["rows"][0]["family"]) < 20:
            print(f"WARNING {f}: {len(d['rows'][0]['family'])} probe families, "
                  f"expected 30 -- this is not a raw output of "
                  f"e3_probe_protocol.py")
    if MISSING and not a.allow_missing:
        print("MISSING RESULT FILES: " + ", ".join(MISSING))
        print("Every table in the paper is regenerated from results/. Run the "
              "corresponding\nexperiment, or pass --allow-missing to continue "
              "with a partial report.")
        sys.exit(1)
