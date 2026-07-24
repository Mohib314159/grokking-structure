"""Complex-weight MLP, parameter-matched to the real baseline.

Mechanism under test: quaternionic irreps (FS = -1) have no real form in
their own dimension (real form doubles it), but ARE realisable over C in
native dimension. If the real-weight pathology on SL(2,3) and Q8xZ3 is
caused by real-realizability, complex weights should erase it.

Parameter matching: complex width 256 = 2*(48*256) + 2*(256*24) real params
= exactly the real width-512 baseline. Everything else identical.
"""
import json
import time
import numpy as np
import torch
import torch.nn as nn

from groups import get_group
from train import sustained_crossing

class ComplexMLP(nn.Module):
    def __init__(self, n_in, width, n_out):
        super().__init__()
        self.W1r = nn.Parameter(torch.empty(width, n_in))
        self.W1i = nn.Parameter(torch.empty(width, n_in))
        self.W2r = nn.Parameter(torch.empty(n_out, width))
        self.W2i = nn.Parameter(torch.empty(n_out, width))
        for p in self.parameters():
            nn.init.kaiming_uniform_(p, a=5 ** 0.5)

    def forward(self, x):                      # x real, imag part = 0
        hr = x @ self.W1r.T                    # complex matmul with xi = 0
        hi = x @ self.W1i.T
        hr, hi = torch.relu(hr), torch.relu(hi)   # CReLU (split ReLU)
        logits = hr @ self.W2r.T - hi @ self.W2i.T  # Re(W2 h)
        return logits

def run_complex(group_name, seed, frac=0.7, width=256, lr=2e-3, wd=1.0,
                max_epochs=60_000, eval_every=50, verbose_every=20_000):
    T, inv = get_group(group_name)
    n = inv["n"]
    pairs = np.array([(i, j) for i in range(n) for j in range(n)])
    labels = np.array([T[i, j] for i, j in pairs])
    X = np.zeros((n * n, 2 * n), dtype=np.float32)
    X[np.arange(n * n), pairs[:, 0]] = 1.0
    X[np.arange(n * n), n + pairs[:, 1]] = 1.0

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n * n)
    n_train = int(frac * n * n)
    tr, va = perm[:n_train], perm[n_train:]
    Xtr, ytr = torch.tensor(X[tr]), torch.tensor(labels[tr])
    Xva, yva = torch.tensor(X[va]), torch.tensor(labels[va])

    torch.manual_seed(seed)
    model = ComplexMLP(2 * n, width, n)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            betas=(0.9, 0.98), weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    hist = dict(epoch=[], train_acc=[], val_acc=[], sqnorm=[])
    t0, stop = time.time(), 0
    for epoch in range(max_epochs + 1):
        opt.zero_grad()
        loss_fn(model(Xtr), ytr).backward()
        opt.step()
        if epoch % eval_every == 0:
            with torch.no_grad():
                ta = (model(Xtr).argmax(1) == ytr).float().mean().item()
                vaacc = (model(Xva).argmax(1) == yva).float().mean().item()
                sq = sum((p ** 2).sum().item() for p in model.parameters())
            hist["epoch"].append(epoch); hist["train_acc"].append(ta)
            hist["val_acc"].append(vaacc); hist["sqnorm"].append(sq)
            if verbose_every and epoch % verbose_every == 0:
                print(f"  [CX {group_name} s{seed}] ep {epoch:6d} "
                      f"train {ta:.3f} val {vaacc:.3f}", flush=True)
            stop = stop + 1 if vaacc >= 0.99 else 0
            if stop >= 8:
                break
    T_mem = sustained_crossing(hist["train_acc"], hist["epoch"], 0.995)
    T_gen = sustained_crossing(hist["val_acc"], hist["epoch"], 0.90)
    return dict(group=group_name, seed=seed, arch="complex", frac=frac,
                width=width, lr=lr, wd=wd,
                T_mem=T_mem, T_gen=T_gen,
                delay=(T_gen - T_mem) if (T_mem is not None and T_gen is not None) else None,
                final_val_acc=hist["val_acc"][-1],
                wall_seconds=round(time.time() - t0, 1), history=hist)

if __name__ == "__main__":
    import sys
    g, s = sys.argv[1], int(sys.argv[2])
    r = run_complex(g, s)
    json.dump(r, open(f"results/CX_{g}_seed{s}.json", "w"))
    print(f"CX {g} s{s}: delay={r['delay']} final={r['final_val_acc']:.3f} "
          f"({r['wall_seconds']}s)")
