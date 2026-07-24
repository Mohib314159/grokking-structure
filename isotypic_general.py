"""Isotypic projectors for any finite group, computed from the Cayley table.

Method: class sums span the centre of the group algebra. A random symmetric
central element Z = sum_k r_k (M_k + M_k^T), with M_k the right-regular
matrix of the k-th class sum, is block-scalar on each isotypic component,
so the eigenspaces of Z are exactly the real isotypic sectors (conjugate
pairs of complex-type irreps merge, which is what a real network sees
anyway). Each sector is labelled by its trace (d^2, or 2d^2 for a merged
pair) and by s = (1/|G|) sum_g chi(g^2), the Frobenius-Schur sum on the
block: s = d for real type, 0 for a complex pair, -d for quaternionic.

Everything is verified numerically: idempotency, orthogonality, integer
traces, integer s-values, and agreement with the hand-built projectors for
the twin groups.
"""
import json
import numpy as np

from groups import get_group

def regular_matrices(T):
    n = T.shape[0]
    E = np.zeros((n, n, n))
    for g in range(n):
        E[g, np.arange(n), T[:, g]] = 1.0
    return E

def conjugacy_classes(T):
    n = T.shape[0]
    e = [g for g in range(n) if all(T[g, x] == x for x in range(n))][0]
    inv = np.array([[h for h in range(n) if T[g, h] == e][0] for g in range(n)])
    seen, classes = set(), []
    for g in range(n):
        if g in seen:
            continue
        orb = sorted({T[T[h, g], inv[h]] for h in range(n)})
        seen |= set(orb)
        classes.append(orb)
    return classes, inv

def isotypic_sectors(group_name, seed=0):
    T, _ = get_group(group_name)
    n = T.shape[0]
    E = regular_matrices(T)
    classes, _ = conjugacy_classes(T)
    rng = np.random.RandomState(seed)
    Z = np.zeros((n, n))
    for cls in classes:
        M = E[cls].sum(axis=0)
        Z += rng.uniform(1, 2) * (M + M.T)
    w, V = np.linalg.eigh(Z)
    # cluster eigenvalues into blocks
    order = np.argsort(w)
    w, V = w[order], V[:, order]
    blocks, start = [], 0
    for i in range(1, n + 1):
        if i == n or w[i] - w[i - 1] > 1e-7 * (1 + abs(w[i])):
            blocks.append((start, i)); start = i
    # square map for the FS sum
    sq = np.array([T[g, g] for g in range(n)])
    Msq = E[sq].sum(axis=0)
    sectors = []
    for a, b in blocks:
        P = V[:, a:b] @ V[:, a:b].T
        assert np.allclose(P @ P, P, atol=1e-8)
        tr = P.trace()
        assert abs(tr - round(tr)) < 1e-7
        s = np.trace(P @ Msq) / n
        assert abs(s - round(s)) < 1e-6
        sectors.append(dict(P=P, trace=int(round(tr)), s=int(round(s))))
    total = sum(sec["P"] for sec in sectors)
    assert np.allclose(total, np.eye(n), atol=1e-8)
    return sectors

def describe(sec):
    tr, s = sec["trace"], sec["s"]
    if s > 0:  return f"real, d={s}, trace {tr}"
    if s < 0:  return f"quaternionic, d={-s}, trace {tr}"
    return f"complex pair, trace {tr}"

if __name__ == "__main__":
    # cross-check against the hand-built twin projectors
    from track_isotypic import build_projectors
    for g in ["Q8xZ3", "D4xZ3"]:
        secs = isotypic_sectors(g)
        P_twin_hand, _ = build_projectors(g)
        want = -2 if g == "Q8xZ3" else 2
        auto = [s for s in secs if s["trace"] == 4 and s["s"] == want]
        # the hand-built twin may split from same-label blocks; match by projector
        match = any(np.allclose(s["P"], P_twin_hand, atol=1e-6) for s in secs)
        print(g, "sectors:", [(s['trace'], s['s']) for s in secs],
              "| hand-built twin recovered:", match)

    # SL(2,3): sectors, then ablation
    from ablate import train_model, val_acc, ablated_acc
    secs = isotypic_sectors("SL23")
    print("SL23 sectors:", [describe(s) for s in secs])
    two_dims = [s for s in secs if s["trace"] in (4, 8) and abs(s["s"]) != 3]
    P_quat = [s["P"] for s in secs if s["s"] == -2][0]
    P_pair = [s["P"] for s in secs if s["s"] == 0 and s["trace"] == 8][0]
    P_all2 = P_quat + P_pair
    results = {}
    for sd in [0, 1]:
        model, val, n = train_model("SL23", seed=sd)
        row = dict(full=val_acc(model, val),
                   minus_pair=ablated_acc(model, val, n, P_pair),
                   minus_quat=ablated_acc(model, val, n, P_quat),
                   minus_all_2dim=ablated_acc(model, val, n, P_all2))
        results[f"seed{sd}"] = row
        print("SL23", f"s{sd}", {k: round(v, 3) for k, v in row.items()}, flush=True)
    json.dump(results, open("results/ablation_SL23.json", "w"), indent=1)
