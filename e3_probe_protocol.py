"""Track how weight energy flows into irrep sectors during training.

The twin groups Q8 x Z3 and D4 x Z3 each have a distinguished 2-dim irrep
with trivial Z3 part ("the twin sector", trace-4 isotypic block in both).
It is quaternionic in one group and real in the other, and it is the only
structural difference between them. This script trains the standard real
MLP on both groups and, at every eval, projects the unembedding rows and
embedding columns onto the twin sector to measure the fraction of weight
energy living there. At random init the expected fraction is 4/24.

Projector: for the m=0 2-dim irrep, chi(g,t) = chi2(g8) with chi2 = 2 on e,
-2 on the central involution z, 0 elsewhere. P[x,y] = (2/24) chi2(p8(inv(x) y)).
The combined projector for all twelve 1-dim irreps is (1/2) on x^{-1}y in
{(e,0),(z,0)} and 0 elsewhere. Both are verified idempotent/symmetric with
the right trace before use.
"""
import json
import time
import numpy as np
import torch
import torch.nn as nn

from groups import get_group, cayley_Q8, cayley_D4
from train import sustained_crossing

def build_projectors(group_name):
    T, inv_data = get_group(group_name)
    n = T.shape[0]                      # 24, indexed a + 8*t
    base = cayley_Q8() if group_name == "Q8xZ3" else cayley_D4()
    z = 4 if group_name == "Q8xZ3" else 2          # central involution index
    # chi2 on the 8-element factor
    chi2 = np.zeros(8)
    chi2[0], chi2[z] = 2.0, -2.0
    # element inverses in the product group
    e = [g for g in range(n) if all(T[g, x] == x for x in range(n))][0]
    inv = np.array([[h for h in range(n) if T[g, h] == e][0] for g in range(n)])

    P_twin = np.zeros((n, n))
    P_ones = np.zeros((n, n))
    for x in range(n):
        for y in range(n):
            g = T[inv[x], y]
            a, t = g % 8, g // 8
            P_twin[x, y] = (2.0 / n) * chi2[a]
            if t == 0 and a in (0, z):
                P_ones[x, y] = 0.5
    for P, tr in ((P_twin, 4.0), (P_ones, 12.0)):
        assert np.allclose(P, P.T)
        assert np.allclose(P @ P, P, atol=1e-10)
        assert abs(np.trace(P) - tr) < 1e-9
    return P_twin, P_ones

def frac_in(P, M):
    """Fraction of Frobenius energy of M (rows or cols indexed by G) in sector P."""
    return float(np.linalg.norm(M @ P) ** 2 / (np.linalg.norm(M) ** 2 + 1e-12))

def run_tracked(group_name, seed=0, frac=0.7, width=512, lr=2e-3, wd=1.0,
                max_epochs=80_000, eval_every=100):
    P_twin, P_ones = build_projectors(group_name)
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
    model = nn.Sequential(nn.Linear(2 * n, width, bias=False), nn.ReLU(),
                          nn.Linear(width, n, bias=False))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98),
                            weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    h = dict(epoch=[], train_acc=[], val_acc=[],
             twin_unemb=[], twin_emb=[], ones_unemb=[])
    t0, stop = time.time(), 0
    for epoch in range(max_epochs + 1):
        opt.zero_grad()
        loss_fn(model(Xtr), ytr).backward()
        opt.step()
        if epoch % eval_every == 0:
            with torch.no_grad():
                ta = (model(Xtr).argmax(1) == ytr).float().mean().item()
                vaacc = (model(Xva).argmax(1) == yva).float().mean().item()
                W1 = model[0].weight.numpy()            # [width, 2n]
                W2 = model[2].weight.numpy()            # [n, width]
                A, B = W1[:, :n], W1[:, n:]
                emb = 0.5 * (frac_in(P_twin, A) + frac_in(P_twin, B))
                unemb = frac_in(P_twin, W2.T)           # rows of W2 index G
                ones = frac_in(P_ones, W2.T)
            h["epoch"].append(epoch); h["train_acc"].append(ta)
            h["val_acc"].append(vaacc)
            h["twin_emb"].append(emb); h["twin_unemb"].append(unemb)
            h["ones_unemb"].append(ones)
            if epoch % 20_000 == 0:
                print(f"  [{group_name}] ep {epoch:6d} val {vaacc:.3f} "
                      f"twin(unemb) {unemb:.3f} twin(emb) {emb:.3f}", flush=True)
            stop = stop + 1 if vaacc >= 0.99 else 0
            if stop >= 8:
                break
    out = dict(group=group_name, seed=seed,
               T_gen=sustained_crossing(h["val_acc"], h["epoch"], 0.90),
               final_val=h["val_acc"][-1], wall=round(time.time() - t0, 1),
               history=h)
    json.dump(out, open(f"results/ISO_{group_name}_seed{seed}.json", "w"))
    return out

if __name__ == "__main__":
    import sys
    run_tracked(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
