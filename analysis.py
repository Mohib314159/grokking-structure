"""Chunked weight-decay dose-response on one group/seed.

Usage: python sweep_wd.py GROUP SEED WD CHUNK
Checkpoints model+opt+history; resumable. History on a 100-epoch grid.
"""
import os
import sys
import json
import torch
import torch.nn as nn

from diagnose import make_data

g, seed, wd, chunk = (sys.argv[1], int(sys.argv[2]), float(sys.argv[3]),
                      int(sys.argv[4]))
tag = f"{g}_s{seed}_wd{wd}"
ckpt = f"ckpt_{tag}.pt"
torch.set_num_threads(4)

Xtr, ytr, Xva, yva, n = make_data(g, seed)
torch.manual_seed(seed)
model = nn.Sequential(nn.Linear(2 * n, 512, bias=False), nn.ReLU(),
                      nn.Linear(512, n, bias=False))
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.98),
                        weight_decay=wd)
loss_fn = nn.CrossEntropyLoss()
start, hist = 0, dict(epoch=[], train_acc=[], val_acc=[])
if os.path.exists(ckpt):
    st = torch.load(ckpt)
    model.load_state_dict(st["m"]); opt.load_state_dict(st["o"])
    start, hist = st["ep"], st["hist"]

for epoch in range(start, start + chunk):
    opt.zero_grad()
    loss_fn(model(Xtr), ytr).backward()
    opt.step()
    if epoch % 100 == 0:
        with torch.no_grad():
            ta = (model(Xtr).argmax(1) == ytr).float().mean().item()
            va = (model(Xva).argmax(1) == yva).float().mean().item()
        hist["epoch"].append(epoch); hist["train_acc"].append(ta)
        hist["val_acc"].append(va)

torch.save(dict(m=model.state_dict(), o=opt.state_dict(),
                ep=start + chunk, hist=hist), ckpt)
json.dump(dict(group=g, seed=seed, wd=wd, history=hist),
          open(f"results/WD_{tag}.json", "w"))
print(f"{tag}: epoch {start+chunk} train {hist['train_acc'][-1]:.3f} "
      f"val {hist['val_acc'][-1]:.3f}")
