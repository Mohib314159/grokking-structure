"""Aggregate results, recompute delays at a uniform threshold (0.90),
verify Frobenius-Schur bookkeeping, and generate figures."""
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from groups import get_group

THRESH = 0.90          # uniform generalisation threshold
SUSTAIN = 5
ORDER = ["Z24", "D12", "SL23", "S4", "D4xZ3", "Q8xZ3"]
LABEL = {"Z24": "Z$_{24}$", "D12": "D$_{12}$", "SL23": "SL(2,3)", "S4": "S$_4$",
         "D4xZ3": "D$_4{\\times}$Z$_3$", "Q8xZ3": "Q$_8{\\times}$Z$_3$"}

# Frobenius-Schur indicator per irrep (hardcoded, verified below):
#   +1 real, 0 complex, -1 quaternionic
FS = {
    "Z24":  [+1 if k in (0, 12) else 0 for k in range(24)],       # dims all 1
    "D12":  [+1] * 9,                                             # dihedral: all real
    "SL23": [+1, 0, 0, -1, 0, 0, +1],                             # dims 1,1,1,2,2,2,3
    "S4":   [+1] * 5,                                             # symmetric: all real
    "Q8xZ3": [+1, 0, 0] * 4 + [-1, 0, 0],                         # dims 1x12, 2x3
    "D4xZ3": [+1, 0, 0] * 4 + [+1, 0, 0],                         # identical char table to Q8xZ3
}

def sustained(vals, epochs, thresh, window=SUSTAIN):
    v = np.asarray(vals)
    for i in range(len(v) - window + 1):
        if np.all(v[i:i + window] >= thresh):
            return int(epochs[i])
    return None

def fs_verify_and_invariants():
    inv = {}
    for g in ORDER:
        T, base = get_group(g)
        n = base["n"]
        dims = base["irrep_dims"]
        eps = FS[g]
        assert len(eps) == len(dims)
        # classical identity: sum_rho eps_rho * d_rho = #{g in G : g^2 = e}
        e = [x for x in range(n) if all(T[x, y] == y for y in range(n))][0]
        sqrt_count = sum(1 for x in range(n) if T[x, x] == e)
        assert sum(e_ * d for e_, d in zip(eps, dims)) == sqrt_count, g
        base["sqrt_of_identity"] = sqrt_count
        base["quaternionic_mass"] = sum(d * d for e_, d in zip(eps, dims) if e_ == -1) / n
        inv[g] = base
    return inv

def _annotate(r):
    h = r["history"]
    r["T_mem_u"] = sustained(h["train_acc"], h["epoch"], 0.995)
    r["T_gen_u"] = sustained(h["val_acc"], h["epoch"], THRESH)
    r["max_epoch"] = h["epoch"][-1]
    r["censored"] = r["T_gen_u"] is None
    r["delay_u"] = None if r["censored"] else r["T_gen_u"] - r["T_mem_u"]
    return r

def load_arch(prefix):
    runs = {}
    for path in sorted(glob.glob(f"results/{prefix}_*_seed*.json")):
        d = json.load(open(path))
        if "history" not in d:
            continue          # skip diagnostics/ablation files in results/
        r = _annotate(d)
        runs.setdefault(r["group"], []).append(r)
    return runs

def load_runs():
    runs = {g: [] for g in ORDER}
    for path in sorted(glob.glob("results/*_seed*.json")):
        if path.split("/")[-1].startswith(("CX_", "QT_", "ISO_")):
            continue
        d = json.load(open(path))
        if "history" not in d:
            continue          # skip diagnostics/ablation files in results/
        r = _annotate(d)
        runs[r["group"]].append(r)
    return runs

def main():
    inv = fs_verify_and_invariants()
    runs = load_runs()

    print(f"\n=== Grokking delay at matched |G|=24 (threshold: sustained val>={THRESH}) ===")
    print(f"{'group':7s} {'k':>3s} {'d_max':>5s} {'q-mass':>6s} | "
          f"{'delays (epochs)':30s} {'final val acc':>18s}")
    rows = []
    for g in ORDER:
        i = inv[g]
        ds, finals = [], []
        for r in sorted(runs[g], key=lambda x: x["seed"]):
            ds.append(f">{r['max_epoch'] - r['T_mem_u']}" if r["censored"]
                      else str(r["delay_u"]))
            finals.append(f"{r['final_val_acc']:.3f}")
        print(f"{g:7s} {i['k']:3d} {i['d_max']:5d} {i['quaternionic_mass']:6.3f} | "
              f"{', '.join(ds):30s} {', '.join(finals):>18s}")
        rows.append((g, i, runs[g]))

    # ---------------------------------------------------------- figure 1
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), sharey=True)
    for ax, g in zip(axes.flat, ORDER):
        r = sorted(runs[g], key=lambda x: x["seed"])[0]
        h = r["history"]
        ax.plot(h["epoch"], h["train_acc"], "--", color="0.55", lw=1.2, label="train")
        ax.plot(h["epoch"], h["val_acc"], color="C0", lw=1.6, label="val")
        ax.axhline(THRESH, color="C3", lw=0.8, ls=":")
        if r["T_gen_u"]:
            ax.axvline(r["T_gen_u"], color="C3", lw=0.8)
        ax.set_xscale("log"); ax.set_xlim(40, max(h["epoch"]))
        ax.set_title(f"{LABEL[g]}   (k={inv[g]['k']}, d_max={inv[g]['d_max']}"
                     + (", quaternionic" if inv[g]["quaternionic_mass"] > 0 else "")
                     + ")", fontsize=10)
        ax.set_ylim(0, 1.02)
    axes[0, 0].legend(fontsize=8, loc="center left")
    for ax in axes[1]: ax.set_xlabel("epoch (log)")
    for ax in axes[:, 0]: ax.set_ylabel("accuracy")
    fig.suptitle("Grokking on group composition: identical recipe, four groups of order 24",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig("figs/fig1_curves.png", dpi=160)

    # ---------------------------------------------------------- figure 2
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for xi, g in enumerate(ORDER):
        i = inv[g]
        for r in runs[g]:
            if r["censored"]:
                y = r["max_epoch"] - r["T_mem_u"]
                ax.scatter(xi, y, marker="^", s=70, color="C3", zorder=3)
                ax.annotate("censored", (xi, y), textcoords="offset points",
                            xytext=(6, 2), fontsize=7, color="C3")
            else:
                ax.scatter(xi, r["delay_u"], s=45, color="C0", zorder=3)
        obs = [r["delay_u"] for r in runs[g] if not r["censored"]]
        if obs:
            ax.hlines(np.mean(obs), xi - 0.18, xi + 0.18, color="C0", lw=2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([f"{LABEL[g]}\nk={inv[g]['k']}, d_max={inv[g]['d_max']}\n"
                        f"q={inv[g]['quaternionic_mass']:.2f}" for g in ORDER],
                       fontsize=8.5)
    ax.set_ylabel(f"grokking delay  T_gen({THRESH}) − T_mem  (epochs, log)")
    ax.set_title("Delay varies 5–30×+ across six groups of identical order 24\n"
                 "(non-monotone in k and d_max; both quaternionic-type groups are catastrophic)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig("figs/fig2_delay.png", dpi=160)

    # ---------------------------------------------------------- figure 3
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for g, c in zip(ORDER, ["C0", "C1", "C2", "C3"]):
        r = sorted(runs[g], key=lambda x: x["seed"])[0]
        h = r["history"]
        ax.plot(h["epoch"], h["sqnorm"], color=c, lw=1.5, label=LABEL[g])
    ax.set_xscale("log"); ax.set_xlim(40, None)
    ax.set_xlabel("epoch (log)"); ax.set_ylabel(r"$\|\theta\|^2$")
    ax.set_title("Weight-norm trajectories (seed 0): the norm-separation mediator\n"
                 "weakens exactly where delay explodes", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("figs/fig3_norms.png", dpi=160)

    # ------------------------------------------- algebra sweep table + fig 4
    archs = [("real", runs, "o", "C0"),
             ("complex", load_arch("CX"), "s", "C1"),
             ("quaternion", load_arch("QT"), "D", "C2")]
    print("\n=== Weight-algebra sweep (all parameter-matched) ===")
    for name, data, _, _ in archs[1:]:
        for g in ORDER:
            for r in sorted(data.get(g, []), key=lambda x: x["seed"]):
                d = (f">{r['max_epoch'] - r['T_mem_u']}" if r["censored"]
                     else str(r["delay_u"]))
                print(f"{name:10s} {g:7s} seed {r['seed']}: delay {d:>8s}  "
                      f"final {r['final_val_acc']:.3f}")

    pair = ["D4xZ3", "Q8xZ3", "SL23"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    offs = {"real": -0.22, "complex": 0.0, "quaternion": 0.22}
    for name, data, mk, col in archs:
        for xi, g in enumerate(pair):
            for r in data.get(g, []) if name != "real" else runs[g]:
                y = (r["max_epoch"] - r["T_mem_u"]) if r["censored"] else r["delay_u"]
                a1.scatter(xi + offs[name], y,
                           marker="^" if r["censored"] else mk,
                           s=50, color=col, zorder=3)
                a2.scatter(xi + offs[name], r["final_val_acc"],
                           marker=mk, s=50, color=col, zorder=3)
    for ax in (a1, a2):
        ax.set_xticks(range(len(pair)))
        ax.set_xticklabels([LABEL[g] for g in pair], fontsize=9)
    a1.set_yscale("log")
    a1.set_ylabel("delay (epochs, log; ▲ = censored)")
    a2.set_ylabel("final val accuracy"); a2.set_ylim(0.6, 1.02)
    from matplotlib.lines import Line2D
    a1.legend(handles=[Line2D([], [], marker=m, ls="", color=c, label=l)
                       for l, (_, _, m, c) in zip(
                           ["real", "complex", "quaternion"],
                           [(0, 0, "o", "C0"), (0, 0, "s", "C1"), (0, 0, "D", "C2")])],
              fontsize=8)
    fig.suptitle("The failure is invariant under the weight algebra "
                 "$\\mathbb{R} \\to \\mathbb{C} \\to \\mathbb{H}$ "
                 "(parameter-matched; control unaffected)", fontsize=10)
    fig.tight_layout()
    fig.savefig("figs/fig4_algebra_sweep.png", dpi=160)

    print("\nfigures written to figs/")

if __name__ == "__main__":
    main()
