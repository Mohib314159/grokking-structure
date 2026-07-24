"""Load checkpoints, run the mechanism diagnostics (fast, no training)."""
import json
import numpy as np
import torch
import torch.nn as nn

from ablate import build_pair_projector
from diagnose import (make_data, accs, projected_copy, grad_balance,
                      neuron_coupling, energy_fracs)

torch.set_num_threads(4)
report = {}
for g in ["Q8xZ3", "D4xZ3"]:
    Xtr, ytr, Xva, yva, n = make_data(g, 0)
    data = (Xtr, ytr, Xva, yva)
    st = torch.load(f"ckpt_{g}_s0.pt")
    model = nn.Sequential(nn.Linear(2 * n, 512, bias=False), nn.ReLU(),
                          nn.Linear(512, n, bias=False))
    model.load_state_dict(st["m"])
    P_twin, P_pair, P_ones = build_pair_projector(g)
    sectors = dict(ones=P_ones, twin=P_twin, pair=P_pair)

    ta, va = accs(model, data)
    print(f"\n=== {g} seed 0 @ epoch {st['ep']}: train {ta:.3f} val {va:.3f} ===")

    mat = {"full": (round(ta, 3), round(va, 3))}
    for name, P in [("ones", P_ones), ("twin", P_twin), ("pair", P_pair),
                    ("both2d", P_twin + P_pair)]:
        mat[f"minus_{name}"] = tuple(round(x, 3) for x in accs(
            projected_copy(model, n, P), data))
    print("ablation (train, val):")
    for k, v in mat.items():
        print(f"  {k:13s} {v}")

    gb = grad_balance(model, Xtr, ytr, n, sectors)
    print("grad balance cos(P dL, P W) | g_norm | w_norm:")
    for k, v in gb.items():
        print(f"  {k:5s} {v}")

    ef = energy_fracs(model, n, sectors)
    print("energy fractions (init baseline ones .50 twin .17 pair .33):", ef)

    names, M = neuron_coupling(model, n, sectors)
    print("neuron coupling M[s_in, t_out]", names)
    print(M)

    report[g] = dict(epoch=st["ep"], train=ta, val=va, ablation=mat,
                     grad_balance=gb, energy=ef,
                     coupling=dict(names=names, M=M.tolist()))

json.dump(report, open("results/diagnose_seed0.json", "w"), indent=1,
          default=str)
print("\nsaved -> results/diagnose_seed0.json")
