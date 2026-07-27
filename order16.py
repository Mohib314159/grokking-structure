"""Order-16 maximal-class groups: D16, SD16, Q16 — the pre-registered test.

All three are order 16 with the same abelianisation (Z2 x Z2), so 4 one-dim
irreps and 3 two-dim irreps (4*1 + 3*4 = 16), 7 conjugacy classes. They differ
in the Frobenius-Schur type of the faithful 2-dim irreps: +1, 0, -1.

Presentations (elements r^i s^j indexed i + 8j):
  D16   r^8 = s^2 = 1,  s r s^-1 = r^-1
  SD16  r^8 = s^2 = 1,  s r s^-1 = r^3
  Q16   r^8 = 1, s^2 = r^4, s r s^-1 = r^-1

Nothing is hardcoded: character tables and FS indicators are derived from the
Cayley tables by Burnside class-sum diagonalisation, exactly as in the order-24
audit, then checked against sum(eps_rho d_rho) = #{g : g^2 = e}.
"""
import itertools
import numpy as np
from collections import Counter


def cayley_D16():
    T = np.zeros((16, 16), dtype=int)
    for a, b in itertools.product(range(8), range(2)):
        for c, d in itertools.product(range(8), range(2)):
            i = (a + (c if b == 0 else -c)) % 8
            j = (b + d) % 2
            T[a + 8 * b, c + 8 * d] = i + 8 * j
    return T


def cayley_SD16():
    T = np.zeros((16, 16), dtype=int)
    for a, b in itertools.product(range(8), range(2)):
        for c, d in itertools.product(range(8), range(2)):
            i = (a + (c if b == 0 else 3 * c)) % 8
            j = (b + d) % 2
            T[a + 8 * b, c + 8 * d] = i + 8 * j
    return T


def cayley_Q16():
    T = np.zeros((16, 16), dtype=int)
    for a, b in itertools.product(range(8), range(2)):
        for c, d in itertools.product(range(8), range(2)):
            i = (a + (c if b == 0 else -c) + (4 if b + d >= 2 else 0)) % 8
            j = (b + d) % 2
            T[a + 8 * b, c + 8 * d] = i + 8 * j
    return T


ORDER16 = {"D16": cayley_D16, "SD16": cayley_SD16, "Q16": cayley_Q16}


# ------------------------------------------------------------- verification
def ident(T):
    n = T.shape[0]
    return [g for g in range(n) if all(T[g, x] == x and T[x, g] == x for x in range(n))][0]


def invs(T):
    n = T.shape[0]; e = ident(T)
    return np.array([[h for h in range(n) if T[g, h] == e][0] for g in range(n)])


def assoc(T):
    n = T.shape[0]; idx = np.arange(n)
    return bool(np.array_equal(T[T[:, :, None], idx[None, None, :]],
                               T[idx[:, None, None], T[None, :, :]])), n ** 3


def classes(T):
    n = T.shape[0]; e = ident(T); inv = invs(T)
    seen, cls = set(), []
    for g in range(n):
        if g in seen:
            continue
        orb = sorted({T[T[h, g], inv[h]] for h in range(n)})
        seen |= set(orb); cls.append(orb)
    return cls


def char_table(T):
    n = T.shape[0]
    cls = classes(T); k = len(cls)
    cls_of = np.zeros(n, dtype=int)
    for i, c in enumerate(cls):
        for g in c:
            cls_of[g] = i
    sizes = np.array([len(c) for c in cls]); e_cls = cls_of[ident(T)]
    A = np.zeros((k, k, k))
    for a, Ca in enumerate(cls):
        for b, Cb in enumerate(cls):
            cnt = np.zeros(k)
            for x in Ca:
                for y in Cb:
                    cnt[cls_of[T[x, y]]] += 1
            A[a, b] = cnt / sizes
    Ns = [A[a] for a in range(k)]
    rng = np.random.RandomState(12345)
    R = sum(rng.uniform(1, 2) * N for N in Ns)
    w, V = np.linalg.eig(R)
    omegas = []
    for c in range(k):
        v = V[:, c] / V[e_cls, c]
        omegas.append(np.array([(Ns[a] @ v)[e_cls] for a in range(k)]))
    omegas = np.array(omegas)
    dims, chars = [], []
    for om in omegas:
        d = np.sqrt(n / np.sum(om * np.conj(om) / sizes).real)
        dims.append(d); chars.append(om * d / sizes)
    dims = np.array(dims); chars = np.array(chars)
    assert np.allclose(dims, np.round(dims), atol=1e-6), f"non-integer dims {dims}"
    dims = np.round(dims.real).astype(int)
    order = np.lexsort((np.round(chars[:, 0].real, 6), dims))
    return chars[order], dims[order], cls, cls_of, sizes


def fs(T, chars, cls_of):
    n = T.shape[0]
    sq = np.array([T[g, g] for g in range(n)])
    return np.array([np.mean([chi[cls_of[sq[g]]] for g in range(n)]) for chi in chars])


def order_profile(T):
    n = T.shape[0]; e = ident(T); prof = Counter()
    for g in range(n):
        x, k = g, 1
        while x != e:
            x = T[x, g]; k += 1
        prof[k] += 1
    return dict(sorted(prof.items()))


def canon(chars, sizes):
    return sorted((int(sizes[j]), tuple(np.round(np.sort_complex(np.round(chars[:, j], 6)), 6)))
                  for j in range(chars.shape[1]))


if __name__ == "__main__":
    print("=" * 78)
    print("ORDER-16 GROUPS — independent verification (nothing hardcoded)")
    print("=" * 78)
    store = {}
    for name, build in ORDER16.items():
        T = build(); n = T.shape[0]; e = ident(T)
        ok, ntrip = assoc(T)
        chars, dims, cls, cls_of, sizes = char_table(T)
        eps = np.round(fs(T, chars, cls_of).real).astype(int)
        sq_e = sum(1 for g in range(n) if T[g, g] == e)
        lhs = int(sum(d * ep for d, ep in zip(dims, eps)))
        qm = sum(d * d for d, ep in zip(dims, eps) if ep == -1) / n
        print(f"\n--- {name} ---")
        print(f"  exhaustive associativity ({ntrip} triples): {'PASS' if ok else 'FAIL'}")
        print(f"  element orders          : {order_profile(T)}")
        print(f"  class sizes             : {sorted(sizes.tolist())}")
        print(f"  irrep dims (derived)    : {sorted(dims.tolist())}")
        print(f"  FS (dim, eps)           : {sorted(zip(dims.tolist(), eps.tolist()))}")
        print(f"  sum eps*d = {lhs}, #\u007bg:g^2=e\u007d = {sq_e}  "
              f"{'PASS' if lhs == sq_e else 'FAIL'}")
        print(f"  quaternionic mass       : {qm:.4f}")
        store[name] = dict(T=T, chars=chars, dims=dims, eps=eps, sizes=sizes, qm=qm)

    print("\n" + "=" * 78)
    print("IS D16 vs Q16 A SECOND CHARACTER-TABLE TWIN PAIR?")
    print("=" * 78)
    for a, b in (("D16", "Q16"), ("D16", "SD16"), ("SD16", "Q16")):
        A, B = store[a], store[b]
        same_sz = sorted(A["sizes"].tolist()) == sorted(B["sizes"].tolist())
        same_d = sorted(A["dims"].tolist()) == sorted(B["dims"].tolist())
        same_ct = canon(A["chars"], A["sizes"]) == canon(B["chars"], B["sizes"])
        same_fs = sorted(A["eps"].tolist()) == sorted(B["eps"].tolist())
        print(f"  {a:5s} vs {b:5s}: class sizes {same_sz}, dims {same_d}, "
              f"character table {same_ct}, FS {same_fs}"
              + ("   <== TWIN PAIR" if same_ct and not same_fs else ""))
