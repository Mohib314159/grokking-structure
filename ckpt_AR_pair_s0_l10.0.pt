"""Anti-repair training: suppress one isotypic sector from initialisation.

If the stuck twin's failure is the pair sector settling into a malign
equilibrium, then adding lambda * (pair-sector weight energy) to the loss
from epoch 0 should make Q8 x Z3 grok on the control's schedule -- turning
the projection from a post-hoc diagnostic into an intervention. Penalising
a load-bearing sector (--sector twin) instead is the specificity control.

Usage: python antirepair.py SEED LAMBDA SECTOR CHUNK
Same locked recipe otherwise (frac .7, width 512, AdamW lr 2e-3, wd 1.0).
"""
import os
import sys
import json
import torch
import torch.nn as nn

from ablate import build_pair_projector
from diagnose import make_data

seed, lam, sector, chunk = (int(sys.argv[1]), float(sys.argv[2]),
                            sys.argv[3], int(sys.argv[4]))
g = "Q8xZ3"
tag = f"AR_{sector}_s{seed}_l{lam}"
ckpt = f"ckpt_{tag}.pt"
torch.set_num_threads(4)

Xtr, ytr, Xva, yva, n = make_data(g, seed)
P_twin, P_pair, P_ones = build_pair_projector(g)
P = {"pair": P_pair, "twin": P_twin}[sector]
Pt = torch.tensor(P, dtype=torch.float32)

torch.manual_seed(seed)
model = nn.Sequential(nn.Linear(2 * n, 512, bias=False), nn.ReLU(),
                      nn.Linear(512, n, bias=False))
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.98),
                        weight_decay=1.0)
loss_fn = nn.CrossEntropyLoss()
start, hist = 0, dict(epoch=[], train_acc=[], val_acc=[], sector_energy=[])
if os.path.exists(ckpt):
    st = torch.load(ckpt)
    model.load_state_dict(st["m"]); opt.load_state_dict(st["o"])
    start, hist = st["ep"], st["hist"]

for epoch in range(start, start + chunk):
    opt.zero_grad()
    W1, W2 = model[0].weight, model[2].weight
    pen = ((W1[:, :n] @ Pt) ** 2).sum() + ((W1[:, n:] @ Pt) ** 2).sum() \
        + ((Pt @ W2) ** 2).sum()
    (loss_fn(model(Xtr), ytr) + lam * pen).backward()
    opt.step()
    if epoch % 100 == 0:
        with torch.no_grad():
            ta = (model(Xtr).argmax(1) == ytr).float().mean().item()
            va = (model(Xva).argmax(1) == yva).float().mean().item()
            tot = (W1 ** 2).sum() + (W2 ** 2).sum()
        hist["epoch"].append(epoch); hist["train_acc"].append(ta)
        hist["val_acc"].append(va)
        hist["sector_energy"].append(round((pen / tot).item(), 4))

torch.save(dict(m=model.state_dict(), o=opt.state_dict(),
                ep=start + chunk, hist=hist), ckpt)
json.dump(dict(group=g, seed=seed, lam=lam, sector=sector, history=hist),
          open(f"results/{tag}.json", "w"))
v = hist["val_acc"]
def sust(th):
    for i in range(len(v) - 4):
        if all(x >= th for x in v[i:i+5]):
            return hist["epoch"][i]
    return None
print(f"{tag}: ep {start+chunk} train {hist['train_acc'][-1]:.3f} "
      f"val {v[-1]:.3f} T_gen@.90 {sust(0.90)} sector_frac "
      f"{hist['sector_energy'][-1]}")
