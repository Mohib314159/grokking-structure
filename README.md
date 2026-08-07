"""Multi-seed equilibrium diagnostics + per-sector logit decomposition.

For each checkpoint: sector x {train,val} ablation matrix, per-sector
gradient balance, energy fractions, and the decomposition of the decision
margin  l[y] - l[best-wrong]  into isotypic-sector contributions, averaged
separately over validation errors and validation corrects. The sector sums
are asserted to equal the total margin (exact decomposition).
"""
import json
import sys
import numpy as np
import torch
import torch.nn as nn

from ablate import build_pair_projector
from diagnose import make_data, accs, projected_copy, grad_balance, energy_fracs

torch.set_num_threads(4)


def load_ckpt(g, seed, n):
    st = torch.load(f"ckpt_{g}_s{seed}.pt")
    m = nn.Sequential(nn.Linear(2 * n, 512, bias=False), nn.ReLU(),
                      nn.Linear(512, n, bias=False))
    m.load_state_dict(st["m"])
    return m, st["ep"]


def logit_decomp(model, data, n, sectors):
    Xtr, ytr, Xva, yva = data
    with torch.no_grad():
        L = model(Xva)
    m = len(yva)
    idx = torch.arange(m)
    Lm = L.clone()
    Lm[idx, yva] = -1e9
    kbw = Lm.argmax(1)                       # best wrong class per example
    total = L[idx, yva] - L[idx, kbw]
    err = L.argmax(1) != yva
    out, ssum = {}, torch.zeros(m)
    for name, P in sectors.items():
        Pt = torch.tensor(P, dtype=torch.float32)
        Ls = L @ Pt                          # P symmetric: sector part of logits
        c = Ls[idx, yva] - Ls[idx, kbw]
        ssum += c
        out[name] = dict(
            err=round(c[err].mean().item(), 3) if err.any() else None,
            corr=round(c[~err].mean().item(), 3))
    assert torch.allclose(ssum, total, atol=1e-3), "decomposition broken"
    out["_total"] = dict(
        err=round(total[err].mean().item(), 3) if err.any() else None,
        corr=round(total[~err].mean().item(), 3))
    out["_n_err"] = int(err.sum())
    return out


if __name__ == "__main__":
    report = {}
    for g in ["Q8xZ3", "D4xZ3"]:
        P_twin, P_pair, P_ones = build_pair_projector(g)
        sectors = dict(ones=P_ones, twin=P_twin, pair=P_pair)
        for seed in [0, 1, 2]:
            Xtr, ytr, Xva, yva, n = make_data(g, seed)
            data = (Xtr, ytr, Xva, yva)
            model, ep = load_ckpt(g, seed, n)
            ta, va = accs(model, data)
            row = dict(epoch=ep, train=round(ta, 3), val=round(va, 3))
            mat = {"full": (round(ta, 3), round(va, 3))}
            for name, P in [("ones", P_ones), ("twin", P_twin),
                            ("pair", P_pair), ("both2d", P_twin + P_pair)]:
                mat[f"minus_{name}"] = tuple(
                    round(x, 3) for x in accs(projected_copy(model, n, P), data))
            row["ablation"] = mat
            row["grad_balance"] = grad_balance(model, Xtr, ytr, n, sectors)
            row["energy"] = energy_fracs(model, n, sectors)
            row["margin_decomp"] = logit_decomp(model, data, n, sectors)
            report[f"{g}_s{seed}"] = row
            print(f"{g} s{seed} ep{ep}: train {ta:.3f} val {va:.3f}")
            print("  ablation:", mat)
            print("  cos(PdL,PW):", {k: v["cos_gw"]
                                     for k, v in row["grad_balance"].items()})
            print("  energy:", row["energy"])
            print("  margin decomp:", row["margin_decomp"], flush=True)
    json.dump(report, open("results/diagnostics_multi.json", "w"), indent=1,
              default=str)
    print("saved -> results/diagnostics_multi.json")
