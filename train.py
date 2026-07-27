"""Train a 1-hidden-layer MLP on group composition; measure grokking delay.

Standard grokking recipe (Nanda et al. 2023 / Power et al. 2022 style):
one-hot concat inputs, full-batch AdamW, weight decay 1.0, cross-entropy.
Identical architecture and hyperparameters for every group, so any
difference in delay is attributable to group structure alone.
Defaults match the locked grid in run_experiments.py; thresholds match
analysis.py (T_gen: sustained val >= 0.90, window 5 on a 50-epoch grid).
"""
import json
import time
import numpy as np
import torch
import torch.nn as nn

from groups import get_group

DEVICE = "cpu"

def sustained_crossing(vals, epochs, thresh, window=5):
    """First epoch at which `vals` >= thresh and stays there for `window` evals."""
    v = np.asarray(vals)
    for i in range(len(v) - window + 1):
        if np.all(v[i:i + window] >= thresh):
            return int(epochs[i])
    return None

def run_one(group_name, seed, frac=0.7, width=512, lr=2e-3, wd=1.0,
            max_epochs=80_000, eval_every=50, verbose_every=5000):
    T, inv = get_group(group_name)
    n = inv["n"]

    # dataset: all n^2 pairs, one-hot concat encoding
    pairs = np.array([(i, j) for i in range(n) for j in range(n)])
    labels = np.array([T[i, j] for i, j in pairs])
    X = np.zeros((n * n, 2 * n), dtype=np.float32)
    X[np.arange(n * n), pairs[:, 0]] = 1.0
    X[np.arange(n * n), n + pairs[:, 1]] = 1.0

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n * n)
    n_train = int(frac * n * n)
    tr, va = perm[:n_train], perm[n_train:]

    Xtr = torch.tensor(X[tr], device=DEVICE)
    ytr = torch.tensor(labels[tr], device=DEVICE)
    Xva = torch.tensor(X[va], device=DEVICE)
    yva = torch.tensor(labels[va], device=DEVICE)

    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(2 * n, width, bias=False),
        nn.ReLU(),
        nn.Linear(width, n, bias=False),
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            betas=(0.9, 0.98), weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()

    hist = dict(epoch=[], train_acc=[], val_acc=[], sqnorm=[])
    t0 = time.time()
    stop_counter = 0

    for epoch in range(max_epochs + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        opt.step()

        if epoch % eval_every == 0:
            model.eval()
            with torch.no_grad():
                ta = (model(Xtr).argmax(1) == ytr).float().mean().item()
                vaacc = (model(Xva).argmax(1) == yva).float().mean().item()
                sq = sum((p ** 2).sum().item() for p in model.parameters())
            hist["epoch"].append(epoch)
            hist["train_acc"].append(ta)
            hist["val_acc"].append(vaacc)
            hist["sqnorm"].append(sq)
            if verbose_every and epoch % verbose_every == 0:
                print(f"  [{group_name} s{seed}] ep {epoch:6d} "
                      f"train {ta:.3f} val {vaacc:.3f} |θ|² {sq:.1f}", flush=True)
            # early stop: val sustained >= 0.99 for 8 evals
            stop_counter = stop_counter + 1 if vaacc >= 0.99 else 0
            if stop_counter >= 8:
                break

    T_mem = sustained_crossing(hist["train_acc"], hist["epoch"], 0.995)
    T_gen = sustained_crossing(hist["val_acc"], hist["epoch"], 0.90)
    delay = (T_gen - T_mem) if (T_mem is not None and T_gen is not None) else None

    # weight norm at memorisation vs. post-generalisation (delay-law mediator)
    sq_at_mem = sq_final = None
    if T_mem is not None:
        i_mem = hist["epoch"].index(T_mem)
        sq_at_mem = hist["sqnorm"][i_mem]
        sq_final = hist["sqnorm"][-1]

    out = dict(group=group_name, seed=seed, frac=frac, width=width, lr=lr, wd=wd,
               max_epochs=max_epochs, eval_every=eval_every, gen_thresh=0.90,
               invariants={k: v for k, v in inv.items() if k != "irrep_dims"},
               irrep_dims=inv["irrep_dims"],
               T_mem=T_mem, T_gen=T_gen, delay=delay,
               sqnorm_at_mem=sq_at_mem, sqnorm_final=sq_final,
               log_norm_ratio=(float(np.log(sq_at_mem / sq_final))
                               if sq_at_mem and sq_final else None),
               final_val_acc=hist["val_acc"][-1],
               wall_seconds=round(time.time() - t0, 1),
               history=hist)
    return out

if __name__ == "__main__":
    import sys
    g, s = sys.argv[1], int(sys.argv[2])
    r = run_one(g, s)
    path = f"results/{g}_seed{s}.json"
    with open(path, "w") as f:
        json.dump(r, f)
    print(f"{g} seed {s}: T_mem={r['T_mem']} T_gen={r['T_gen']} "
          f"delay={r['delay']} final_val={r['final_val_acc']:.3f} "
          f"({r['wall_seconds']}s) -> {path}")
