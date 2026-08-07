"""Regrowth test with optimizer-state carryover.

Load a stuck checkpoint (model + AdamW state including step counts and
moment estimates), surgically project the pair sector out of the weights,
change NOTHING else, and resume. This is the clean counterfactual: the
ongoing training dynamics, minus one sector. Logged every 250 epochs:
train acc, val acc, per-sector energy fractions.
"""
import json
import sys
import numpy as np
import torch
import torch.nn as nn

from ablate import build_pair_projector
from diagnose import make_data, accs, energy_fracs

torch.set_num_threads(4)
g, seed, epochs = "Q8xZ3", int(sys.argv[1]), int(sys.argv[2])

Xtr, ytr, Xva, yva, n = make_data(g, seed)
st = torch.load(f"ckpt_{g}_s{seed}.pt")
model = nn.Sequential(nn.Linear(2 * n, 512, bias=False), nn.ReLU(),
                      nn.Linear(512, n, bias=False))
model.load_state_dict(st["m"])
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.98),
                        weight_decay=1.0)
opt.load_state_dict(st["o"])                # moments + step count carried

P_twin, P_pair, P_ones = build_pair_projector(g)
sectors = dict(ones=P_ones, twin=P_twin, pair=P_pair)
Q = torch.tensor(np.eye(n) - P_pair, dtype=torch.float32)
with torch.no_grad():                        # surgical edit, weights only
    W1 = model[0].weight
    W1[:, :n] = W1[:, :n] @ Q
    W1[:, n:] = W1[:, n:] @ Q
    model[2].weight[:] = Q @ model[2].weight

loss_fn = nn.CrossEntropyLoss()
log = []
for epoch in range(epochs + 1):
    if epoch % 250 == 0:
        ta, va = accs(model, (Xtr, ytr, Xva, yva))
        ef = energy_fracs(model, n, sectors)
        log.append(dict(epoch=epoch, train=round(ta, 3), val=round(va, 3),
                        **{k: v for k, v in ef.items()}))
        print(log[-1], flush=True)
    opt.zero_grad()
    loss_fn(model(Xtr), ytr).backward()
    opt.step()
json.dump(log, open(f"results/regrow_carry_{g}_s{seed}.json", "w"))
