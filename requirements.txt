"""Resumable trainer: run N epochs per invocation, checkpoint model+optimizer."""
import os
import sys
import torch
import torch.nn as nn

from diagnose import make_data

g, seed, chunk = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ckpt = f"ckpt_{g}_s{seed}.pt"
torch.set_num_threads(4)

Xtr, ytr, Xva, yva, n = make_data(g, seed)
torch.manual_seed(seed)
model = nn.Sequential(nn.Linear(2 * n, 512, bias=False), nn.ReLU(),
                      nn.Linear(512, n, bias=False))
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.98),
                        weight_decay=1.0)
loss_fn = nn.CrossEntropyLoss()

start, stop, done = 0, 0, False
if os.path.exists(ckpt):
    st = torch.load(ckpt)
    model.load_state_dict(st["m"])
    opt.load_state_dict(st["o"])
    start, stop, done = st["ep"], st["stop"], st["done"]

if done:
    print(f"{g} s{seed}: already done at epoch {start}")
    sys.exit(0)

for epoch in range(start, start + chunk):
    opt.zero_grad()
    loss_fn(model(Xtr), ytr).backward()
    opt.step()
    if epoch % 100 == 0:
        with torch.no_grad():
            va = (model(Xva).argmax(1) == yva).float().mean().item()
        stop = stop + 1 if va >= 0.99 else 0
        if stop >= 8:
            done = True
            epoch += 1
            break

with torch.no_grad():
    ta = (model(Xtr).argmax(1) == ytr).float().mean().item()
    va = (model(Xva).argmax(1) == yva).float().mean().item()
torch.save(dict(m=model.state_dict(), o=opt.state_dict(),
                ep=epoch + (0 if done else 1), stop=stop, done=done), ckpt)
print(f"{g} s{seed}: epoch {epoch + (0 if done else 1)} "
      f"train {ta:.3f} val {va:.3f} done={done}")
