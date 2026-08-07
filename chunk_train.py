"""Why does cleanup fail? Mechanism diagnostics on the stuck twin.

Measurements on a trained (stuck) Q8xZ3 net vs the grokked D4xZ3 control:
  1. Full sector x {val,train} ablation matrix (no post-hoc villain choice).
  2. Gradient balance: cosine( P-projected dL_train/dW , P-projected W ).
     Decoupled wd shrinks every direction at rate lr*wd (half-life ~350 ep
     at lr=2e-3, wd=1), so a sector that keeps its norm for 40k epochs must
     be actively regrown by the train-loss gradient. Negative cosine =
     loss-gradient opposes shrinkage = the sector is load-bearing for TRAIN.
  3. Train accuracy of the repaired net: does the villain exist to fix
     residual train examples the structured solution cannot reach?
  4. Neuron-mediated sector coupling matrix M[s,t] = sum_h a_s(h) b_t(h):
     how much villain input-energy feeds load-bearing output-energy through
     shared hidden neurons (why internal cleanup is expensive but external
     projection is free).
  5. Regrowth: project the villain out, resume training, watch its energy.
"""
import json
import time
import numpy as np
import torch
import torch.nn as nn

from groups import get_group
from ablate import build_pair_projector

torch.set_num_threads(4)


def make_data(group_name, seed, frac=0.7):
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
    return (torch.tensor(X[tr]), torch.tensor(labels[tr]),
            torch.tensor(X[va]), torch.tensor(labels[va]), n)


def train(group_name, seed, width=512, lr=2e-3, wd=1.0, max_epochs=40_000):
    Xtr, ytr, Xva, yva, n = make_data(group_name, seed)
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
                va = (model(Xva).argmax(1) == yva).float().mean().item()
            stop = stop + 1 if va >= 0.99 else 0
            if stop >= 8:
                break
    return model, (Xtr, ytr, Xva, yva), n, epoch


def accs(model, data):
    Xtr, ytr, Xva, yva = data
    with torch.no_grad():
        ta = (model(Xtr).argmax(1) == ytr).float().mean().item()
        va = (model(Xva).argmax(1) == yva).float().mean().item()
    return ta, va


def projected_copy(model, n, P_remove):
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
    return m2


def sector_vec(model, n, P):
    """Flattened weight components living in sector P (both arg slots + unembed)."""
    Pt = torch.tensor(P, dtype=torch.float32)
    W1, W2 = model[0].weight, model[2].weight
    return torch.cat([(W1[:, :n] @ Pt).flatten(),
                      (W1[:, n:] @ Pt).flatten(),
                      (Pt @ W2).flatten()])


def grad_balance(model, Xtr, ytr, n, sectors):
    """cosine( P dL/dW , P W ) per sector, plus norm ratios vs the wd force."""
    loss = nn.CrossEntropyLoss()(model(Xtr), ytr)
    model.zero_grad()
    loss.backward()
    W1, W2 = model[0].weight, model[2].weight
    G1, G2 = W1.grad, W2.grad
    out = {}
    for name, P in sectors.items():
        Pt = torch.tensor(P, dtype=torch.float32)
        w = torch.cat([(W1[:, :n] @ Pt).flatten(), (W1[:, n:] @ Pt).flatten(),
                       (Pt @ W2).flatten()]).detach()
        g = torch.cat([(G1[:, :n] @ Pt).flatten(), (G1[:, n:] @ Pt).flatten(),
                       (Pt @ G2).flatten()]).detach()
        cos = (g @ w / (g.norm() * w.norm() + 1e-12)).item()
        out[name] = dict(cos_gw=round(cos, 4),
                         g_norm=round(g.norm().item(), 5),
                         w_norm=round(w.norm().item(), 3))
    model.zero_grad()
    return out


def neuron_coupling(model, n, sectors):
    """M[s,t] = sum_h a_s(h) * b_t(h), normalised. Off-diagonals = entanglement."""
    W1 = model[0].weight.detach()
    W2 = model[2].weight.detach()
    names = list(sectors)
    A = {}   # input energy per neuron per sector
    B = {}   # output energy per neuron per sector
    for name in names:
        Pt = torch.tensor(sectors[name], dtype=torch.float32)
        A[name] = ((W1[:, :n] @ Pt) ** 2).sum(1) + ((W1[:, n:] @ Pt) ** 2).sum(1)
        B[name] = ((Pt @ W2) ** 2).sum(0)
    M = np.zeros((len(names), len(names)))
    for i, s in enumerate(names):
        for j, t in enumerate(names):
            M[i, j] = (A[s] * B[t]).sum().item()
    M /= M.sum()
    return names, np.round(M, 4)


def energy_fracs(model, n, sectors):
    tot = sum(sector_vec(model, n, P).norm().item() ** 2 for P in sectors.values())
    return {k: round(sector_vec(model, n, P).norm().item() ** 2 / tot, 4)
            for k, P in sectors.items()}


def regrow(model, data, n, P_villain, sectors, lr=2e-3, wd=1.0, epochs=6000):
    """Project villain out, resume training, watch its energy and val acc."""
    m2 = projected_copy(model, n, P_villain)
    Xtr, ytr, Xva, yva = data
    opt = torch.optim.AdamW(m2.parameters(), lr=lr, betas=(0.9, 0.98),
                            weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    log = []
    for epoch in range(epochs + 1):
        if epoch % 500 == 0:
            ta, va = accs(m2, data)
            vfrac = energy_fracs(m2, n, sectors)
            log.append((epoch, round(ta, 3), round(va, 3), vfrac))
        opt.zero_grad()
        loss_fn(m2(Xtr), ytr).backward()
        opt.step()
    return log


if __name__ == "__main__":
    report = {}
    for g in ["Q8xZ3", "D4xZ3"]:
        t0 = time.time()
        P_twin, P_pair, P_ones = build_pair_projector(g)
        sectors = dict(ones=P_ones, twin=P_twin, pair=P_pair)
        model, data, n, last_ep = train(g, seed=0)
        Xtr, ytr, Xva, yva = data
        ta, va = accs(model, data)
        print(f"\n=== {g} (seed 0, stopped at epoch {last_ep}, "
              f"train {ta:.3f} / val {va:.3f}, {time.time()-t0:.0f}s) ===")

        # 1. full ablation matrix with TRAIN accuracy
        mat = {}
        for name, P in [("ones", P_ones), ("twin", P_twin), ("pair", P_pair),
                        ("both2d", P_twin + P_pair)]:
            m2 = projected_copy(model, n, P)
            mat[f"minus_{name}"] = tuple(round(x, 3) for x in accs(m2, data))
        mat["full"] = (round(ta, 3), round(va, 3))
        print("ablation (train, val):", mat)

        # 2. gradient balance per sector
        gb = grad_balance(model, Xtr, ytr, n, sectors)
        print("grad balance cos(P dL, P W):", gb)

        # 3. weight-energy fractions (baseline: ones .5, twin 1/6, pair 1/3)
        ef = energy_fracs(model, n, sectors)
        print("energy fractions:", ef)

        # 4. neuron-mediated coupling
        names, M = neuron_coupling(model, n, sectors)
        print("neuron coupling M[s_in, t_out] over", names)
        print(M)

        report[g] = dict(last_epoch=last_ep, train=ta, val=va, ablation=mat,
                         grad_balance=gb, energy=ef,
                         coupling=dict(names=names, M=M.tolist()))

        # 5. regrowth test on the stuck net only
        if g == "Q8xZ3":
            print("regrowth after projecting out `pair` (epoch, train, val, energy):")
            log = regrow(model, data, n, P_pair, sectors)
            for row in log:
                print("  ", row)
            report[g]["regrowth"] = log

    with open("results/diagnose_seed0.json", "w") as f:
        json.dump(report, f, indent=1, default=str)
    print("\nsaved -> results/diagnose_seed0.json")
