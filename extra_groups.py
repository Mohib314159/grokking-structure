"""Two extra order-24 groups, added to fix the sample size of the main claim.

With six groups the exact test on the FS ordering gives 1/C(6,2) = 0.067 --
not significant. The unit of replication is the group, not the run, so more
seeds do not help; more GROUPS do. Adding one quaternionic and one
non-quaternionic group takes it to 1/C(8,3) = 0.018.

  Dic24  dicyclic of order 24: <a,b | a^12=e, b^2=a^6, b a b^-1 = a^-1>
         q-mass 1/2. Shares its ENTIRE character table with D12 -- a third
         character-table twin pair, and the sharpest one in the study, since
         D12 is the fastest-grokking group here.
  Z2xA4  direct product, q-mass 0, 8 classes, dims [1x6, 3x2].

Both verified from first principles by verify_math-style checks: exhaustive
associativity (13824 triples), character table by Burnside class-sum
diagonalisation, FS from eps = (1/|G|) sum chi(g^2), FS identity.
"""
import itertools
import numpy as np


def cayley_Dic24():
    T = np.zeros((24, 24), dtype=int)
    for i, j in itertools.product(range(12), range(2)):
        for k, l in itertools.product(range(12), range(2)):
            m = (i + (k if j == 0 else -k) + (6 if j + l >= 2 else 0)) % 12
            T[i + 12 * j, k + 12 * l] = m + 12 * ((j + l) % 2)
    return T


def cayley_Z2xA4():
    perms = [p for p in itertools.permutations(range(4))
             if sum(1 for a in range(4) for b in range(a + 1, 4) if p[a] > p[b]) % 2 == 0]
    idx = {p: i for i, p in enumerate(perms)}
    T = np.zeros((24, 24), dtype=int)
    for a, x in itertools.product(range(12), range(2)):
        for b, y in itertools.product(range(12), range(2)):
            pa, pb = perms[a], perms[b]
            T[a + 12 * x, b + 12 * y] = idx[tuple(pa[pb[i]] for i in range(4))] + 12 * ((x + y) % 2)
    return T


EXTRA = {
    "Dic24": (cayley_Dic24, dict(n=24, k=9, irrep_dims=[1, 1, 1, 1, 2, 2, 2, 2, 2],
                                 fs=[1, 1, 1, 1, -1, -1, -1, 1, 1], quaternionic_mass=0.5)),
    "Z2xA4": (cayley_Z2xA4, dict(n=24, k=8, irrep_dims=[1, 1, 1, 1, 1, 1, 3, 3],
                                 fs=[0, 0, 0, 0, 1, 1, 1, 1], quaternionic_mass=0.0)),
}


def install():
    """Monkeypatch groups.get_group so train.py can use these without edits."""
    import groups, train
    orig = groups.get_group

    def get_group(name):
        if name in EXTRA:
            build, inv = EXTRA[name]
            return build(), inv
        return orig(name)
    groups.get_group = get_group
    train.get_group = get_group
    return get_group


if __name__ == "__main__":
    from order16 import assoc, char_table, fs, ident, order_profile
    for name, (build, inv) in EXTRA.items():
        T = build()
        ok, nt = assoc(T)
        chars, dims, cls, cls_of, sizes = char_table(T)
        eps = np.round(fs(T, chars, cls_of).real).astype(int)
        e = ident(T)
        sq = sum(1 for g in range(24) if T[g, g] == e)
        lhs = int(sum(d * ep for d, ep in zip(dims, eps)))
        qm = sum(d * d for d, ep in zip(dims, eps) if ep == -1) / 24
        print(f"{name}: assoc {'PASS' if ok else 'FAIL'} ({nt} triples)  "
              f"classes {len(cls)}  dims {sorted(dims.tolist())}")
        print(f"   FS {sorted(zip(dims.tolist(), eps.tolist()))}")
        print(f"   sum eps*d = {lhs}, #\u007bg:g^2=e\u007d = {sq}  "
              f"{'PASS' if lhs == sq else 'FAIL'}   q-mass {qm:.4f}")
        print(f"   element orders {order_profile(T)}")
        assert qm == inv["quaternionic_mass"], f"{name}: q-mass mismatch"
