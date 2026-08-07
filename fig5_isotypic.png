"""Quaternion-weight MLP, parameter-matched to the real baseline.

Schur's lemma: the endomorphism algebra of a real irreducible representation
is R, C, or H. The two groups that fail to grok under real weights (SL(2,3)
and Q8 x Z3) are exactly the ones with H-type irreps, and both embed in the
unit quaternions. If the failure is an algebra mismatch, quaternion weights
should fix it. Complex weights already didn't (complex_train.py), so this
is the last capacity-side branch; if it also fails, the effect is dynamical.

Parameter matching: quaternion width 128 gives 4*(48*128 + 128*24) real
params, identical to the real width-512 baseline.
"""
import json
import time
import numpy as np
import torch
import torch.nn as nn

from groups import get_group
from train import sustained_crossing

class QuatMLP(nn.Module):
    def __init__(self, n_in, width, n_out):
        super().__init__()
        def mat(o, i):
            p = nn.Parameter(torch.empty(o, i))
            nn.init.kaiming_uniform_(p, a=5 ** 0.5)
            return p
        self.W1 = nn.ParameterList([mat(width, n_in) for _ in range(4)])
        self.W2 = nn.ParameterList([mat(n_out, width) for _ in range(4)])

    @staticmethod
    def qmul(W, x):
        # Hamilton product, weight on the left; x = (r, i, j, k) components
        wr, wi, wj, wk = W
        xr, xi, xj, xk = x
        yr = xr @ wr.T - xi @ wi.T - xj @ wj.T - xk @ wk.T
        yi = xi @ wr.T + xr @ wi.T + xk @ wj.T - xj @ wk.T
        yj = xj @ wr.T - xk @ wi.T + xr @ wj.T + xi @ wk.T
        yk = xk @ wr.T + xj @ wi.T - xi @ wj.T + xr @ wk.T
        return yr, yi, yj, yk

    def forward(self, x):
        z = torch.zeros_like(x)
        h = self.qmul(self.W1, (x, z, z, z))
        h = tuple(torch.relu(c) for c in h)          # split ReLU
        out = self.qmul(self.W2, h)
        return out[0]                                 # real component as logits

def run_quat(group_name, seed, frac=0.7, width=128, lr=2e-3, wd=1.0,
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
    model = QuatMLP(2 * n, width, n)
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
                print(f"  [QT {group_name} s{seed}] ep {epoch:6d} "
                      f"train {ta:.3f} val {vaacc:.3f}", flush=True)
            stop = stop + 1 if vaacc >= 0.99 else 0
            if stop >= 8:
                break
    T_mem = sustained_crossing(hist["train_acc"], hist["epoch"], 0.995)
    T_gen = sustained_crossing(hist["val_acc"], hist["epoch"], 0.90)
    return dict(group=group_name, seed=seed, arch="quaternion", frac=frac,
                width=width, lr=lr, wd=wd, T_mem=T_mem, T_gen=T_gen,
                delay=(T_gen - T_mem) if (T_mem is not None and T_gen is not None) else None,
                final_val_acc=hist["val_acc"][-1],
                wall_seconds=round(time.time() - t0, 1), history=hist)

if __name__ == "__main__":
    import sys
    g, s = sys.argv[1], int(sys.argv[2])
    r = run_quat(g, s)
    json.dump(r, open(f"results/QT_{g}_seed{s}.json", "w"))
    print(f"QT {g} s{s}: delay={r['delay']} final={r['final_val_acc']:.3f} "
          f"({r['wall_seconds']}s)")
