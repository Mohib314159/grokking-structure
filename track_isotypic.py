"""
groups24.py -- ALL FIFTEEN groups of order 24, built from presentations and
deduplicated by isomorphism, plus every cheap Cayley-table invariant.

WHY THIS FILE EXISTS
  The referee objection that cannot be answered by more seeds is:
  "the groups are not a random sample from a well-defined population of tasks."
  It also cannot be answered by adding a few more groups, because the choice of
  which to add is yours.

  It CAN be answered by using the whole population. The Small Groups Library
  (Besche, Eick and O'Brien; shipped with GAP and Magma) lists 15 isomorphism
  classes of order 24. This module constructs 15 pairwise non-isomorphic groups
  of that order, which matches the count.

  Note what that does and does not establish. Distinct fingerprints prove the
  15 constructed groups are pairwise non-isomorphic, so there are AT LEAST 15.
  That there are AT MOST 15 is the external classification result, not
  something proved here. Together they give the complete population, and with
  it there is no within-order group-selection discretion left to criticise.

  Consequences:
    n goes from 8 to 15.
    Quaternionic-type count goes from 3 to 5 (Z2xDic12 and Z3sdZ8 are new).
    The exact one-sided ordering test becomes 1/C(15,5) = 1/3003 = 3.3e-4.
    Seven of the fifteen are genuinely out of sample for a hypothesis formed
    on the original six, so a real preregistration is possible -- see PREREG.md.
    q-mass now takes four values (0, 1/6, 1/3, 1/2) instead of three, so it can
    be tested as a dose rather than a binary.

NAMING. The repo calls the dihedral group of order 24 "D12" and the dihedral
group of order 8 "D4" (geometer's convention, subscript = number of sides).
This module uses the order convention (D24, D8) because that is what a referee
will expect, and exports REPO_ALIAS so old result files still line up.
"""
import itertools

import numpy as np

# ----------------------------------------------------------------- builders


def Zn(n):
    return np.array([[(i + j) % n for j in range(n)] for i in range(n)],
                    dtype=np.int64)


def direct(TA, TB):
    """Direct product; element (a,b) -> index a + |A|*b."""
    nA, nB = TA.shape[0], TB.shape[0]
    T = np.zeros((nA * nB, nA * nB), dtype=np.int64)
    for a1 in range(nA):
        for b1 in range(nB):
            for a2 in range(nA):
                for b2 in range(nB):
                    T[a1 + nA * b1, a2 + nA * b2] = TA[a1, a2] + nA * TB[b1, b2]
    return T


def dihedral(m):
    """D_{2m}, order 2m. Element (a,x) -> index a + m*x."""
    T = np.zeros((2 * m, 2 * m), dtype=np.int64)
    for a in range(m):
        for x in (0, 1):
            for b in range(m):
                for y in (0, 1):
                    T[a + m * x, b + m * y] = ((a + (b if x == 0 else -b)) % m
                                               + m * (x ^ y))
    return T


def dicyclic(m):
    """Dic_{4m}: a^{2m}=e, b^2=a^m, b a b^-1 = a^-1. Order 4m."""
    n = 2 * m
    T = np.zeros((2 * n, 2 * n), dtype=np.int64)
    for i, j in itertools.product(range(n), range(2)):
        for k, l in itertools.product(range(n), range(2)):
            v = (i + (k if j == 0 else -k) + (m if j + l >= 2 else 0)) % n
            T[i + n * j, k + n * l] = v + n * ((j + l) % 2)
    return T


def quaternion8():
    units = ["e", "i", "j", "k"]
    mul = {("e", u): (1, u) for u in units}
    mul.update({(u, "e"): (1, u) for u in units})
    mul.update({("i", "i"): (-1, "e"), ("j", "j"): (-1, "e"),
                ("k", "k"): (-1, "e"), ("i", "j"): (1, "k"),
                ("j", "k"): (1, "i"), ("k", "i"): (1, "j"),
                ("j", "i"): (-1, "k"), ("k", "j"): (-1, "i"),
                ("i", "k"): (-1, "j")})

    def ix(s, u):
        return units.index(u) + 4 * (s == -1)

    T = np.zeros((8, 8), dtype=np.int64)
    for s1 in (1, -1):
        for u1 in units:
            for s2 in (1, -1):
                for u2 in units:
                    s3, u3 = mul[(u1, u2)]
                    T[ix(s1, u1), ix(s2, u2)] = ix(s1 * s2 * s3, u3)
    return T


def sym(k):
    perms = list(itertools.permutations(range(k)))
    ix = {p: i for i, p in enumerate(perms)}
    T = np.zeros((len(perms), len(perms)), dtype=np.int64)
    for i, p in enumerate(perms):
        for j, q in enumerate(perms):
            T[i, j] = ix[tuple(p[q[t]] for t in range(k))]
    return T


def alt4():
    perms = [p for p in itertools.permutations(range(4))
             if sum(1 for a in range(4) for b in range(a + 1, 4)
                    if p[a] > p[b]) % 2 == 0]
    ix = {p: i for i, p in enumerate(perms)}
    T = np.zeros((12, 12), dtype=np.int64)
    for i, p in enumerate(perms):
        for j, q in enumerate(perms):
            T[i, j] = ix[tuple(p[q[t]] for t in range(4))]
    return T


def SL23():
    els = [(a, b, c, d) for a, b, c, d in itertools.product(range(3), repeat=4)
           if (a * d - b * c) % 3 == 1]
    assert len(els) == 24
    ix = {m: i for i, m in enumerate(els)}
    T = np.zeros((24, 24), dtype=np.int64)
    for i, (a, b, c, d) in enumerate(els):
        for j, (e, f, g, h) in enumerate(els):
            T[i, j] = ix[((a * e + b * g) % 3, (a * f + b * h) % 3,
                          (c * e + d * g) % 3, (c * f + d * h) % 3)]
    return T


def semidirect_Z3(T8, sign):
    """Z3 rtimes G with |G| = 8, acting through a sign homomorphism.
    Element (g, z) -> index g + 8*z."""
    T = np.zeros((24, 24), dtype=np.int64)
    for g1 in range(8):
        for z1 in range(3):
            for g2 in range(8):
                for z2 in range(3):
                    z = (z1 + sign(g1) * z2) % 3
                    T[g1 + 8 * z1, g2 + 8 * z2] = T8[g1, g2] + 8 * z
    return T


def semidirect_cyclic(m, k, act):
    """Z_m rtimes Z_k, generator of Z_k acting by x -> act*x mod m."""
    assert pow(act, k, m) == 1 % m
    T = np.zeros((m * k, m * k), dtype=np.int64)
    for a in range(m):
        for b in range(k):
            for c in range(m):
                for d in range(k):
                    T[a + m * b, c + m * d] = ((a + pow(act, b, m) * c) % m
                                               + m * ((b + d) % k))
    return T


# The 15, in a fixed order. Names follow GAP SmallGroup(24, k) where sensible.
ORDER24 = {
    "Z24":       lambda: Zn(24),                                    # (24,2)
    "Z12xZ2":    lambda: direct(Zn(12), Zn(2)),                     # (24,9)
    "Z6xZ2xZ2":  lambda: direct(direct(Zn(6), Zn(2)), Zn(2)),       # (24,15)
    "S4":        lambda: sym(4),                                    # (24,12)
    "SL23":      SL23,                                              # (24,3)
    "Z2xA4":     lambda: direct(Zn(2), alt4()),                     # (24,13)
    "D24":       lambda: dihedral(12),                              # (24,6)
    "Dic24":     lambda: dicyclic(6),                               # (24,4)
    "D8xZ3":     lambda: direct(dihedral(4), Zn(3)),                # (24,10)
    "Q8xZ3":     lambda: direct(quaternion8(), Zn(3)),              # (24,11)
    "Z2xD12":    lambda: direct(Zn(2), dihedral(6)),                # (24,14)
    "Z2xDic12":  lambda: direct(Zn(2), dicyclic(3)),                # (24,7)
    "Z4xS3":     lambda: direct(Zn(4), sym(3)),                     # (24,5)
    "Z3sdZ8":    lambda: semidirect_cyclic(3, 8, 2),                # (24,1)
    "Z3sdD8":    lambda: semidirect_Z3(                             # (24,8)
        dihedral(4), lambda i: -1 if (i % 4) % 2 else 1),
}

REPO_ALIAS = {"D24": "D12", "D8xZ3": "D4xZ3"}   # old names in results/*.json

_T = {}


def cayley(name):
    if name not in _T:
        T = ORDER24[name]()
        verify(T, name)
        _T[name] = T
    return _T[name]


# ------------------------------------------------------------- verification

def verify(T, name=""):
    """Identity, inverses, EXHAUSTIVE associativity (all 24^3 = 13824 triples)."""
    n = T.shape[0]
    ident = [g for g in range(n)
             if np.all(T[g] == np.arange(n)) and np.all(T[:, g] == np.arange(n))]
    assert len(ident) == 1, f"{name}: identity"
    e = ident[0]
    for g in range(n):
        assert (T[g] == e).any(), f"{name}: no inverse for {g}"
    for a in range(n):
        assert np.array_equal(T[T[a]], T[a][T]), f"{name}: associativity at {a}"
    return e


def relabel(T, perm):
    """Relabel elements: new index i is old perm[i]. Isomorphic group.

    NOTE ON THE 'RANDOM RELABELLING' CONTROL. With one-hot inputs, no biases,
    and i.i.d. init, relabelling by pi is exactly a permutation of the input
    and output coordinate axes. The composite (relabel pi, split seed s)
    induces a uniformly random 403-subset of the ORIGINAL pairs, and the init
    distribution is permutation-equivariant. So relabelling is distributionally
    identical to resampling the split. It is run anyway in e2 as an empirical
    check on that argument -- if the two arms differ, something is wrong.
    """
    n = T.shape[0]
    pinv = np.empty(n, dtype=np.int64)
    pinv[perm] = np.arange(n)
    return pinv[T[np.ix_(perm, perm)]]


# --------------------------------------------------------- isotypic sectors

def isotypic_sectors(T, seed=0, tol=1e-7):
    """Real isotypic projectors, straight from the Cayley table.

    Class sums span the centre of R[G]. A random SYMMETRIC central element
    Z = sum_k r_k (M_k + M_k^T) is scalar on each real isotypic component, so
    the eigenspaces of Z are exactly those components (conjugate pairs of
    complex-type irreps merge, which is what a real network sees). Each sector
    carries two integers:
        trace  = real dimension: d^2 for real and quaternionic type, 2d^2 for
                 a merged complex-conjugate pair
        s      = (1/|G|) sum_g chi(g^2) on the block
               = +d real type, 0 complex pair, -d quaternionic type
    Everything is asserted numerically: idempotency, integer trace, integer s,
    and sum_S P_S = I. That last one is why the logit decomposition is exact
    rather than an attribution heuristic.
    """
    n = T.shape[0]
    E = np.zeros((n, n, n))
    for g in range(n):
        E[g, np.arange(n), T[:, g]] = 1.0
    e = [g for g in range(n) if np.all(T[g] == np.arange(n))][0]
    inv = np.array([np.where(T[g] == e)[0][0] for g in range(n)])
    seen, cls = set(), []
    for g in range(n):
        if g in seen:
            continue
        orb = sorted({T[T[h, g], inv[h]] for h in range(n)})
        seen |= set(orb)
        cls.append(orb)
    rng = np.random.RandomState(seed)
    Z = np.zeros((n, n))
    for c in cls:
        M = E[c].sum(axis=0)
        Z += rng.uniform(1, 2) * (M + M.T)
    w, V = np.linalg.eigh(Z)
    o = np.argsort(w)
    w, V = w[o], V[:, o]
    blocks, start = [], 0
    for i in range(1, n + 1):
        if i == n or w[i] - w[i - 1] > tol * (1 + abs(w[i])):
            blocks.append((start, i))
            start = i
    Msq = E[[T[g, g] for g in range(n)]].sum(axis=0)
    secs = []
    for a, b in blocks:
        P = V[:, a:b] @ V[:, a:b].T
        assert np.allclose(P @ P, P, atol=1e-8), "projector not idempotent"
        tr, s = P.trace(), np.trace(P @ Msq) / n
        assert abs(tr - round(tr)) < 1e-7, f"non-integer trace {tr}"
        assert abs(s - round(s)) < 1e-6, f"non-integer FS sum {s}"
        secs.append(dict(P=P, trace=int(round(tr)), s=int(round(s))))
    assert np.allclose(sum(x["P"] for x in secs), np.eye(n), atol=1e-8)
    return sorted(secs, key=lambda d: (-d["trace"], -d["s"]))


def sector_label(sec):
    t, s = sec["trace"], sec["s"]
    if s > 0:
        return f"real d={s} (tr {t})"
    if s < 0:
        return f"QUATERNIONIC d={-s} (tr {t})"
    return f"complex pair (tr {t})"


def qmass(T):
    """Fraction of the regular representation carried by quaternionic-type
    sectors. This is the paper's explanatory variable."""
    return sum(s["trace"] for s in isotypic_sectors(T) if s["s"] < 0) / T.shape[0]


# -------------------------------------------------------------- invariants

def invariants(T):
    """Every cheap Cayley-table invariant a network could plausibly key on.
    Used by e4_predictor_race.py to ask whether q-mass adds anything
    out-of-sample beyond the obvious alternatives. A referee will say the
    twins differ in far more than Frobenius-Schur type; this quantifies it."""
    n = T.shape[0]
    e = [g for g in range(n) if np.all(T[g] == np.arange(n))][0]
    inv = np.array([np.where(T[g] == e)[0][0] for g in range(n)])
    orders = []
    for g in range(n):
        x, k = g, 1
        while x != e:
            x, k = T[x, g], k + 1
        orders.append(k)
    orders = np.array(orders)
    seen, sizes = set(), []
    for g in range(n):
        if g in seen:
            continue
        orb = {T[T[h, g], inv[h]] for h in range(n)}
        seen |= orb
        sizes.append(len(orb))
    centre = [g for g in range(n) if all(T[g, h] == T[h, g] for h in range(n))]
    D = {T[T[inv[a], inv[b]], T[a, b]] for a in range(n) for b in range(n)}
    grew = True
    while grew:
        grew = False
        for a in list(D):
            for b in list(D):
                if T[a, b] not in D:
                    D.add(T[a, b])
                    grew = True
    secs = isotypic_sectors(T)
    dims = []
    for s in secs:
        if s["s"] > 0:
            dims.append(s["s"])
        elif s["s"] < 0:
            dims.append(-s["s"])
        else:
            d = int(round((s["trace"] / 2) ** 0.5))
            dims += [d, d]
    sq = np.array([T[g, g] for g in range(n)])
    return dict(
        n_classes=len(sizes),
        d_max=int(max(dims)),
        sum_d3=int(sum(d ** 3 for d in dims)),
        frac_dim_gt1=float(sum(d * d for d in dims if d > 1) / n),
        n_involutions=int((orders == 2).sum()),
        n_sqrt_e=int((sq == e).sum()),
        exponent=int(np.lcm.reduce(orders)),
        centre_size=len(centre),
        derived_size=len(D),
        abelianisation=n // len(D),
        commuting_prob=float(np.mean(T == T.T)),
        mean_order=float(orders.mean()),
        max_order=int(orders.max()),
        n_order2=int((orders == 2).sum()),
        n_order3=int((orders == 3).sum()),
        n_order4=int((orders == 4).sum()),
        n_order6=int((orders == 6).sum()),
        n_order8=int((orders == 8).sum()),
        n_order12=int((orders == 12).sum()),
        is_abelian=int(np.array_equal(T, T.T)),
        is_nilpotent=int(len(centre) > 1 and len(D) < n),
        q_mass=qmass(T),
        n_quat_sectors=sum(1 for s in secs if s["s"] < 0),
        max_quat_d=max([-s["s"] for s in secs if s["s"] < 0] + [0]),
    )


def fingerprint(T):
    """Isomorphism invariant strong enough to separate all 15 groups of order
    24: element-order profile, class-size multiset, isotypic sector types."""
    iv = invariants(T)
    n = T.shape[0]
    e = [g for g in range(n) if np.all(T[g] == np.arange(n))][0]
    orders = []
    for g in range(n):
        x, k = g, 1
        while x != e:
            x, k = T[x, g], k + 1
        orders.append(k)
    inv = np.array([np.where(T[g] == e)[0][0] for g in range(n)])
    seen, sizes = set(), []
    for g in range(n):
        if g in seen:
            continue
        orb = {T[T[h, g], inv[h]] for h in range(n)}
        seen |= orb
        sizes.append(len(orb))
    return (tuple(np.bincount(orders, minlength=25).tolist()),
            tuple(sorted(sizes)),
            tuple(sorted((s["trace"], s["s"]) for s in isotypic_sectors(T))),
            iv["centre_size"], iv["derived_size"])


# --------------------------------------------------------- character twins

def character_table(T):
    """The ORDINARY COMPLEX character table, computed properly.

    Class sums span the centre of C[G] and commute, so they are simultaneously
    diagonalisable; the eigenvalues are the central characters
    omega_i(C_k) = |C_k| chi_i(g_k) / chi_i(1), and the degrees follow from
    the orthogonality relation |G| = chi_i(1)^2 sum_k |omega_i(C_k)|^2 / |C_k|.
    This is the standard Burnside/Dixon construction.

    Returns (chars, class_sizes) with chars[i][k] = chi_i(g_k).
    """
    n = T.shape[0]
    e = [g for g in range(n) if np.all(T[g] == np.arange(n))][0]
    inv = np.array([np.where(T[g] == e)[0][0] for g in range(n)])
    seen, cls = set(), []
    for g in range(n):
        if g in seen:
            continue
        orb = sorted({T[T[h, g], inv[h]] for h in range(n)})
        seen |= set(orb)
        cls.append(orb)
    k = len(cls)
    sizes = np.array([len(c) for c in cls])
    idx = np.empty(n, dtype=np.int64)
    for i, c in enumerate(cls):
        for g in c:
            idx[g] = i
    A = np.zeros((k, k, k))
    for i, Ci in enumerate(cls):
        for j, Cj in enumerate(cls):
            cnt = np.zeros(k)
            for x in Ci:
                for y in Cj:
                    cnt[idx[T[x, y]]] += 1
            A[i, j] = cnt / sizes
    rng = np.random.RandomState(1)
    M = sum(rng.randn() * A[i] for i in range(k))
    _, V = np.linalg.eig(M)
    Vi = np.linalg.inv(V)
    om = np.array([[(Vi @ A[i] @ V)[t, t] for i in range(k)] for t in range(k)])
    chars = np.array([np.sqrt(np.real(n / np.sum(om[t] * np.conj(om[t]) / sizes)))
                      * om[t] / sizes for t in range(k)])
    return chars, sizes


def same_character_table(T1, T2, tol=1e-4):
    """Do two groups have the same ordinary complex character table, up to a
    permutation of conjugacy classes and of irreducible characters?

    An EARLIER version of this function compared only the multiset of
    irreducible degrees and the multiset of class sizes. Those are necessary,
    not sufficient, and the difference is not academic: it wrongly certified
    Z2xD12 / Z2xDic12 as a twin pair. Z2xDic12 has abelianisation C2 x C4 and
    therefore four linear characters taking the values +-i, while every linear
    character of Z2xD12 (abelianisation C2^3) is real. Their tables cannot
    agree under any permutation.

    The test used here compares the multiset of rows, each row reduced to the
    multiset of its values. That is permutation-invariant on both axes and
    strong enough to separate the cases that arise at order 24; it is a
    necessary condition rather than a decision procedure, so a positive result
    should be checked against a known classification for anything load-bearing.
    """
    C1, _ = character_table(T1)
    C2, _ = character_table(T2)
    if C1.shape != C2.shape:
        return False

    def rows(C):
        return sorted(tuple(sorted(np.round(np.abs(C[i]), 4)))
                      + tuple(sorted(np.round(C[i].real, 4)))
                      + tuple(sorted(np.round(np.abs(C[i].imag), 4)))
                      for i in range(C.shape[0]))
    return rows(C1) == rows(C2)


# Verified genuine character-table twins. D8 and Q8 have identical tables and
# the table of A x B is the Kronecker product of the tables of A and B, so
# D8xZ3 / Q8xZ3 is a twin pair; D24 / Dic24 is verified directly.
TWIN_PAIRS = [("D8xZ3", "Q8xZ3"), ("D24", "Dic24")]

# NOT a twin pair. D12 and Dic12 already have different character tables
# (abelianisations C2^2 and C4), so tensoring with Z2 cannot fix it. Retained
# as a weaker matched contrast: equal multisets of irreducible degrees and of
# conjugacy-class sizes, differing Frobenius-Schur type.
MATCHED_CONTRASTS = [("Z2xD12", "Z2xDic12")]


# ------------------------------------------------------------------- main

if __name__ == "__main__":
    print(f"{'group':11s} {'q-mass':>7s} {'k':>3s} {'dmax':>4s} {'invol':>6s} "
          f"{'exp':>4s} {'|Z|':>4s} {'|G^ab|':>7s}   sectors")
    fps, qs = {}, {}
    for name in ORDER24:
        T = cayley(name)
        f = fingerprint(T)
        assert f not in fps, f"{name} isomorphic to {fps[f]}"
        fps[f] = name
        iv = invariants(T)
        qs[name] = iv["q_mass"]
        secs = isotypic_sectors(T)
        print(f"{name:11s} {iv['q_mass']:7.4f} {iv['n_classes']:3d} "
              f"{iv['d_max']:4d} {iv['n_involutions']:6d} {iv['exponent']:4d} "
              f"{iv['centre_size']:4d} {iv['abelianisation']:7d}   "
              f"{[(s['trace'], s['s']) for s in secs]}")
    assert len(fps) == 15, f"got {len(fps)} distinct groups, need 15"
    print(f"\n{len(fps)} pairwise non-isomorphic groups of order 24 "
          f"constructed. The Small Groups Library lists 15 isomorphism "
          f"classes of order 24,\nso this is the complete population; the "
          f"upper bound is that classification result, not proved here.")
    quat = [g for g, v in qs.items() if v > 0]
    print(f"quaternionic-type ({len(quat)}): "
          + ", ".join(f"{g} (q={qs[g]:.4f})" for g in quat))
    from math import comb
    print(f"exact one-sided ordering test if they occupy the bottom "
          f"{len(quat)}: 1/C(15,{len(quat)}) = 1/{comb(15, len(quat))} "
          f"= {1/comb(15, len(quat)):.2e}")
    print("\ncharacter-table twin pairs (verified):")
    for a, b in TWIN_PAIRS:
        ok = same_character_table(cayley(a), cayley(b))
        print(f"  {a:10s} / {b:10s}  same char table: {ok}   "
              f"q-mass {qs[a]:.4f} vs {qs[b]:.4f}")
    print("degree- and class-size-matched contrasts (NOT twins):")
    for a, b in MATCHED_CONTRASTS:
        ok = same_character_table(cayley(a), cayley(b))
        print(f"  {a:10s} / {b:10s}  same char table: {ok}   "
              f"q-mass {qs[a]:.4f} vs {qs[b]:.4f}")
