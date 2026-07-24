"""Order-24 group quartet: Z24, D12, SL(2,3), S4.

All four groups have |G| = 24, so the composition task has identical
dataset size (576 pairs) and identical output-class count (24).
Only one thing varies: representation-theoretic structure.

Every Cayley table is verified programmatically (identity, inverses,
associativity sample, conjugacy-class count vs. known irrep data).
"""
import itertools
import numpy as np

# ---------------------------------------------------------------- builders

def cayley_Z24():
    n = 24
    return np.array([[(i + j) % n for j in range(n)] for i in range(n)])

def cayley_D12():
    # element (a, x): rotation a in Z12, flip x in {0,1}; idx = a + 12x
    # (a,x)*(b,y) = ( (a + (-1)^x b) mod 12 , x xor y )
    n = 12
    def idx(a, x): return a + n * x
    T = np.zeros((2 * n, 2 * n), dtype=int)
    for a in range(n):
        for x in (0, 1):
            for b in range(n):
                for y in (0, 1):
                    c = (a + (b if x == 0 else -b)) % n
                    T[idx(a, x), idx(b, y)] = idx(c, x ^ y)
    return T

def cayley_S4():
    perms = list(itertools.permutations(range(4)))
    index = {p: i for i, p in enumerate(perms)}
    T = np.zeros((24, 24), dtype=int)
    for i, p in enumerate(perms):
        for j, q in enumerate(perms):
            comp = tuple(p[q[k]] for k in range(4))   # (p o q)(k) = p(q(k))
            T[i, j] = index[comp]
    return T

def cayley_SL23():
    els = []
    for a, b, c, d in itertools.product(range(3), repeat=4):
        if (a * d - b * c) % 3 == 1:
            els.append((a, b, c, d))
    assert len(els) == 24, f"SL(2,3) enumeration gave {len(els)} elements"
    index = {m: i for i, m in enumerate(els)}
    T = np.zeros((24, 24), dtype=int)
    for i, (a, b, c, d) in enumerate(els):
        for j, (e, f, g, h) in enumerate(els):
            m = ((a * e + b * g) % 3, (a * f + b * h) % 3,
                 (c * e + d * g) % 3, (c * f + d * h) % 3)
            T[i, j] = index[m]
    return T

def cayley_Q8():
    # elements: (sign, unit) with unit in {e,i,j,k}; idx = unit + 4*(sign==-1)
    units = ["e", "i", "j", "k"]
    mul = {("e", u): (1, u) for u in units}
    mul.update({(u, "e"): (1, u) for u in units})
    mul.update({("i", "i"): (-1, "e"), ("j", "j"): (-1, "e"), ("k", "k"): (-1, "e"),
                ("i", "j"): (1, "k"), ("j", "k"): (1, "i"), ("k", "i"): (1, "j"),
                ("j", "i"): (-1, "k"), ("k", "j"): (-1, "i"), ("i", "k"): (-1, "j")})
    def idx(s, u): return units.index(u) + 4 * (s == -1)
    T = np.zeros((8, 8), dtype=int)
    for s1 in (1, -1):
        for u1 in units:
            for s2 in (1, -1):
                for u2 in units:
                    s3, u3 = mul[(u1, u2)]
                    T[idx(s1, u1), idx(s2, u2)] = idx(s1 * s2 * s3, u3)
    return T

def cayley_D4():
    n = 4
    def idx(a, x): return a + n * x
    T = np.zeros((2 * n, 2 * n), dtype=int)
    for a in range(n):
        for x in (0, 1):
            for b in range(n):
                for y in (0, 1):
                    c = (a + (b if x == 0 else -b)) % n
                    T[idx(a, x), idx(b, y)] = idx(c, x ^ y)
    return T

def direct_product(TA, TB):
    nA, nB = TA.shape[0], TB.shape[0]
    T = np.zeros((nA * nB, nA * nB), dtype=int)
    for a1 in range(nA):
        for b1 in range(nB):
            for a2 in range(nA):
                for b2 in range(nB):
                    T[a1 + nA * b1, a2 + nA * b2] = TA[a1, a2] + nA * TB[b1, b2]
    return T

def cayley_Z3():
    return np.array([[(i + j) % 3 for j in range(3)] for i in range(3)])

def cayley_Q8xZ3():
    return direct_product(cayley_Q8(), cayley_Z3())

def cayley_D4xZ3():
    return direct_product(cayley_D4(), cayley_Z3())

# ------------------------------------------------------------ verification

def verify_group(T, name):
    n = T.shape[0]
    # identity
    e = [g for g in range(n) if all(T[g, x] == x and T[x, g] == x for x in range(n))]
    assert len(e) == 1, f"{name}: identity failure"
    e = e[0]
    # inverses
    for g in range(n):
        assert any(T[g, h] == e and T[h, g] == e for h in range(n)), f"{name}: no inverse for {g}"
    # associativity (exhaustive: all n^3 triples; trivial cost at |G| <= 48)
    for a in range(n):
        assert np.array_equal(T[T[a]], T[a][T]), f"{name}: associativity failure at a={a}"
    return e

def conjugacy_class_count(T):
    n = T.shape[0]
    e = [g for g in range(n) if all(T[g, x] == x for x in range(n))][0]
    inv = np.zeros(n, dtype=int)
    for g in range(n):
        inv[g] = [h for h in range(n) if T[g, h] == e][0]
    seen, k = set(), 0
    for g in range(n):
        if g in seen:
            continue
        orbit = {T[T[h, g], inv[h]] for h in range(n)}
        seen |= orbit
        k += 1
    return k

def commuting_probability(T):
    n = T.shape[0]
    return float(np.mean(T == T.T))

# ------------------------------------------------------------- registry

GROUPS = {
    "Z24":   dict(build=cayley_Z24,   irrep_dims=[1] * 24),
    "D12":   dict(build=cayley_D12,   irrep_dims=[1] * 4 + [2] * 5),
    "SL23":  dict(build=cayley_SL23,  irrep_dims=[1] * 3 + [2] * 3 + [3]),
    "S4":    dict(build=cayley_S4,    irrep_dims=[1, 1, 2, 3, 3]),
    # character-table-matched pair (Q8 and D4 share a character table;
    # so do their direct products with Z3) -- differ ONLY in FS type
    "Q8xZ3": dict(build=cayley_Q8xZ3, irrep_dims=[1] * 12 + [2] * 3),
    "D4xZ3": dict(build=cayley_D4xZ3, irrep_dims=[1] * 12 + [2] * 3),
}

def class_size_multiset(T):
    n = T.shape[0]
    e = [g for g in range(n) if all(T[g, x] == x for x in range(n))][0]
    inv = np.zeros(n, dtype=int)
    for g in range(n):
        inv[g] = [h for h in range(n) if T[g, h] == e][0]
    seen, sizes = set(), []
    for g in range(n):
        if g in seen:
            continue
        orbit = {T[T[h, g], inv[h]] for h in range(n)}
        seen |= orbit
        sizes.append(len(orbit))
    return sorted(sizes)

def get_group(name):
    spec = GROUPS[name]
    T = spec["build"]()
    verify_group(T, name)
    dims = spec["irrep_dims"]
    n = T.shape[0]
    k_computed = conjugacy_class_count(T)
    assert sum(d * d for d in dims) == n, f"{name}: sum d^2 != |G|"
    assert len(dims) == k_computed, (
        f"{name}: #irreps {len(dims)} != computed conjugacy classes {k_computed}")
    invariants = dict(
        n=n,
        k=k_computed,                                # number of conjugacy classes = #irreps
        d_max=max(dims),                             # largest irrep dimension
        sum_d3=sum(d ** 3 for d in dims),            # cubic complexity measure
        frac_dim_gt1=sum(d * d for d in dims if d > 1) / n,  # mass in matrix-valued irreps
        commuting_prob=commuting_probability(T),     # = k/|G| (theorem; cross-check)
        irrep_dims=dims,
    )
    assert abs(invariants["commuting_prob"] - k_computed / n) < 1e-12
    return T, invariants

if __name__ == "__main__":
    for name in GROUPS:
        T, inv = get_group(name)
        print(f"{name:5s} |G|={inv['n']}  k={inv['k']:2d}  d_max={inv['d_max']}  "
              f"sum_d^3={inv['sum_d3']:3d}  frac(d>1)={inv['frac_dim_gt1']:.3f}  "
              f"P(commute)={inv['commuting_prob']:.3f}")
