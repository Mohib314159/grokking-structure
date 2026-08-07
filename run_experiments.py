import os
"""The pre-registered order-16 experiment. Same locked recipe, |G| = 16.

Tests PREDICTIONS.md 1-4:
  1. Q16 stalls; D16 and SD16 grok.
  2. Any stalled Q16 run has exactly one 2-dim isotypic sector whose ablation
     repairs val while train stays at 1.000.
  3. A from-init penalty on that sector produces grokking to >= 0.99.
  4. Annealing that penalty off collapses val back toward the stuck plateau.

Recipe held from order 24: one hidden layer, width 512, no bias, one-hot
concat inputs, full-batch AdamW (lr 2e-3, wd 1.0, betas 0.9/0.98), CE,
train fraction 0.7, T_gen = sustained val >= 0.90 for 5 evals.
NumPy only — own AdamW and PyTorch-equivalent init.
"""
import json, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from order16 import ORDER16, ident, invs, classes, char_table, fs

FRAC, WIDTH, LR, WD = 0.7, 512, 2e-3, 1.0
B1, B2, EPS = 0.9, 0.98, 1e-8
GEN, MEM, SUSTAIN = 0.90, 0.995, 5


# ------------------------------------------------------------ isotypic sectors
def isotypic_sectors(T, seed=0):
    """Real isotypic projectors from the Cayley table (centre of the group
    algebra), labelled by trace and s = (1/|G|) sum_g chi(g^2)."""
    n = T.shape[0]
    E = np.zeros((n, n, n))
    for g in range(n):
        E[g, np.arange(n), T[:, g]] = 1.0
    rng = np.random.RandomState(seed)
    Z = np.zeros((n, n))
    for cls in classes(T):
        M = E[cls].sum(axis=0)
        Z += rng.uniform(1, 2) * (M + M.T)
    w, V = np.linalg.eigh(Z)
    blocks, start = [], 0
    for i in range(1, n + 1):
        if i == n or w[i] - w[i - 1] > 1e-7 * (1 + abs(w[i])):
            blocks.append((start, i)); start = i
    Msq = E[[T[g, g] for g in range(n)]].sum(axis=0)
    secs = []
    for a, b in blocks:
        P = V[:, a:b] @ V[:, a:b].T
        assert np.allclose(P @ P, P, atol=1e-8)
        tr = int(round(P.trace()))
        s = int(round(np.trace(P @ Msq) / n))
        secs.append(dict(P=P, trace=tr, s=s))
    assert np.allclose(sum(x["P"] for x in secs), np.eye(n), atol=1e-8)
    return secs


def label(sec):
    t, s = sec["trace"], sec["s"]
    return (f"real d={s}" if s > 0 else
            f"QUATERNIONIC d={-s}" if s < 0 else "complex pair") + f" (tr {t})"


# ------------------------------------------------------------------- training
def data(T, seed):
    n = T.shape[0]
    pairs = np.array([(i, j) for i in range(n) for j in range(n)])
    y = np.array([T[i, j] for i, j in pairs])
    X = np.zeros((n * n, 2 * n), dtype=np.float32)
    X[np.arange(n * n), pairs[:, 0]] = 1.0
    X[np.arange(n * n), n + pairs[:, 1]] = 1.0
    perm = np.random.RandomState(seed).permutation(n * n)
    k = int(FRAC * n * n)
    return X[perm[:k]], y[perm[:k]], X[perm[k:]], y[perm[k:]]


def init(rng, out_f, in_f):
    b = 1.0 / np.sqrt(in_f)
    return rng.uniform(-b, b, (out_f, in_f)).astype(np.float32)


def train(T, seed=0, max_epochs=80_000, eval_every=50, frac=FRAC,
          penalty_P=None, lam=0.0, W0=None, opt0=None, track_P=None, log=0):
    n = T.shape[0]
    Xtr, ytr, Xva, yva = data(T, seed) if frac == FRAC else _data_frac(T, seed, frac)
    rng = np.random.RandomState(1000 + seed)
    W1, W2 = (init(rng, WIDTH, 2 * n), init(rng, n, WIDTH)) if W0 is None else \
             (W0[0].copy(), W0[1].copy())
    m = [np.zeros_like(W1), np.zeros_like(W2)] if opt0 is None else [x.copy() for x in opt0[0]]
    v = [np.zeros_like(W1), np.zeros_like(W2)] if opt0 is None else [x.copy() for x in opt0[1]]
    t0step = 0 if opt0 is None else opt0[2]
    N = len(ytr)
    oh = np.zeros((N, n), dtype=np.float32); oh[np.arange(N), ytr] = 1.0
    Pp = None if penalty_P is None else penalty_P.astype(np.float32)
    h = dict(epoch=[], train_acc=[], val_acc=[], repaired=[], sector=[])
    t0 = time.time(); step = t0step
    for ep in range(max_epochs + 1):
        Z = Xtr @ W1.T; H = np.maximum(Z, 0); L = H @ W2.T
        L -= L.max(1, keepdims=True); Ex = np.exp(L); S = Ex / Ex.sum(1, keepdims=True)
        dL = (S - oh) / N
        gW2 = dL.T @ H
        dZ = (dL @ W2) * (Z > 0)
        gW1 = dZ.T @ Xtr
        if Pp is not None and lam > 0:
            # penalty = lam * ||P W||^2 on both embedding blocks and unembedding
            gW1[:, :n] += 2 * lam * (W1[:, :n] @ Pp)
            gW1[:, n:] += 2 * lam * (W1[:, n:] @ Pp)
            gW2 += 2 * lam * (Pp @ W2)
        step += 1
        bc1, bc2 = 1 - B1 ** step, 1 - B2 ** step
        for i, (p, g) in enumerate(((W1, gW1), (W2, gW2))):
            p *= (1 - LR * WD)
            m[i] = B1 * m[i] + (1 - B1) * g
            v[i] = B2 * v[i] + (1 - B2) * g * g
            p -= (LR / bc1) * m[i] / (np.sqrt(v[i] / bc2) + EPS)
        if ep % eval_every == 0:
            ta = ((np.maximum(Xtr @ W1.T, 0) @ W2.T).argmax(1) == ytr).mean()
            va = ((np.maximum(Xva @ W1.T, 0) @ W2.T).argmax(1) == yva).mean()
            h["epoch"].append(ep); h["train_acc"].append(float(ta)); h["val_acc"].append(float(va))
            if track_P is not None:
                h["repaired"].append(ablate(W1, W2, Xva, yva, n, track_P))
                E = (np.linalg.norm(W2.T @ track_P) ** 2 + np.linalg.norm(W1[:, :n] @ track_P) ** 2
                     + np.linalg.norm(W1[:, n:] @ track_P) ** 2)
                Etot = (np.linalg.norm(W2) ** 2 + np.linalg.norm(W1) ** 2)
                h["sector"].append(float(E / Etot))
            if log and ep % log == 0:
                print(f"    ep {ep:6d} train {ta:.3f} val {va:.3f} ({time.time()-t0:.0f}s)",
                      flush=True)
    return (W1, W2), (Xva, yva), h, (m, v, step)


def _data_frac(T, seed, frac):
    n = T.shape[0]
    pairs = np.array([(i, j) for i in range(n) for j in range(n)])
    y = np.array([T[i, j] for i, j in pairs])
    X = np.zeros((n * n, 2 * n), dtype=np.float32)
    X[np.arange(n * n), pairs[:, 0]] = 1.0
    X[np.arange(n * n), n + pairs[:, 1]] = 1.0
    perm = np.random.RandomState(seed).permutation(n * n)
    k = int(frac * n * n)
    return X[perm[:k]], y[perm[:k]], X[perm[k:]], y[perm[k:]]


def ablate(W1, W2, Xva, yva, n, P):
    Q = (np.eye(n) - P).astype(np.float32)
    H = np.maximum(Xva[:, :n] @ (W1[:, :n] @ Q).T + Xva[:, n:] @ (W1[:, n:] @ Q).T, 0)
    return float(((H @ (Q @ W2).T).argmax(1) == yva).mean())


def sust(vals, eps_, t, w=SUSTAIN):
    v = np.asarray(vals)
    for i in range(len(v) - w + 1):
        if np.all(v[i:i + w] >= t):
            return int(eps_[i])
    return None
