"""Sector ablation on trained networks.

Train the standard real MLP on each twin group, then surgically remove
individual irrep sectors from the trained weights (project embeddings and
unembedding onto the orthogonal complement) and measure val accuracy.
Sectors: the twelve 1-dim irreps ("ones", trace 12), the m=0 2-dim irrep
("twin", trace 4; quaternionic in Q8 x Z3, real in D4 x Z3), and the
conjugate pair of 2-dim irreps with nontrivial Z3 part ("pair", trace 8).
The 1-dims factor through the order-12 abelianisation, so they determine
the product only up to the commutator subgroup: ablating both 2-dim
sectors should cap accuracy near 50%. The question is which 2-dim sector
each network actually relies on.
"""
import json
import numpy as np
import torch
import torch.nn as nn

from groups import get_group, cayley_Q8, cayley_D4
from track_isotypic import build_projectors

def build_pair_projector(group_name):
    T, _ = get_group(group_name)
    n = T.shape[0]
    z = 4 if group_name == "Q8xZ3" else 2
    chi2 = np.zeros(8); chi2[0], chi2[z] = 2.0, -2.0
    e = [g for g in range(n) if all(T[g, x] == x for x in range(n))][0]
    inv = np.array([[h for h in range(n) if T[g, h] == e][0] for g in range(n)])
    P_all2 = np.zeros((n, n))
    for x in range(n):
        for y in range(n):
            g = T[inv[x], y]
            a, t = g % 8, g // 8
            if t == 0:
                P_all2[x, y] = (6.0 / n) * chi2[a]
    P_twin, P_ones = build_projectors(group_name)
    P_pair = P_all2 - P_twin
    assert np.allclose(P_pair @ P_pair, P_pair, atol=1e-9)
    assert abs(np.trace(P_pair) - 8.0) < 1e-9
    assert np.allclose(P_ones + P_twin + P_pair, np.eye(n), atol=1e-9)
    return P_twin, P_pair, P_ones

def train_model(group_name, seed=0, frac=0.7, width=512, lr=2e-3, wd=1.0,
                max_epochs=80_000):
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
    stop = 0
    for epoch in range(max_epochs + 1):
        opt.zero_grad()
        loss_fn(model(Xtr), ytr).backward()
        opt.step()
        if epoch % 100 == 0:
            with torch.no_grad():
                vaacc = (model(Xva).argmax(1) == yva).float().mean().item()
            stop = stop + 1 if vaacc >= 0.99 else 0
            if stop >= 8:
                break
    return model, (Xva, yva), n

def val_acc(model, val):
    Xva, yva = val
    with torch.no_grad():
        return (model(Xva).argmax(1) == yva).float().mean().item()

def ablated_acc(model, val, n, P_remove):
    """Project out sector P_remove from embeddings and unembedding."""
    m2 = nn.Sequential(*[nn.Linear(l.in_features, l.out_features, bias=False)
                         if isinstance(l, nn.Linear) else nn.ReLU()
                         for l in model])
    m2.load_state_dict(model.state_dict())
    Q = torch.tensor(np.eye(n) - P_remove, dtype=torch.float32)
    with torch.no_grad():
        W1 = m2[0].weight
        W1[:, :n] = W1[:, :n] @ Q
        W1[:, n:] = W1[:, n:] @ Q
        m2[2].weight[:] = Q @ m2[2].weight
    return val_acc(m2, val)

if __name__ == "__main__":
    results = {}
    for g in ["D4xZ3", "Q8xZ3"]:
        P_twin, P_pair, P_ones = build_pair_projector(g)
        model, val, n = train_model(g, seed=0)
        base = val_acc(model, val)
        row = dict(full=base,
                   minus_twin=ablated_acc(model, val, n, P_twin),
                   minus_pair=ablated_acc(model, val, n, P_pair),
                   minus_both_2dim=ablated_acc(model, val, n, P_twin + P_pair),
                   minus_ones=ablated_acc(model, val, n, P_ones))
        results[g] = row
        print(g, {k: round(v, 3) for k, v in row.items()}, flush=True)
    json.dump(results, open("results/ablation.json", "w"), indent=1)
