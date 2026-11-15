"""Block kernel of minimize_loss (Psi^T S Psi products) against dense numpy, float64 and float32 storage."""

from types import SimpleNamespace

import numba
import numpy as np
import pytest

from salted.minimize_loss import PackedSym, _blocks_setup, _curv, _psi_s, _psi_t
from salted.psi_builder import PsiBlocks

SPECIES = ["H", "C", "O"]
LMAX = {"H": 1, "C": 2, "O": 2}
NMAX = {("H", 0): 3, ("H", 1): 2, ("C", 0): 4, ("C", 1): 1, ("C", 2): 2, ("O", 0): 2, ("O", 1): 3, ("O", 2): 1}
MCUT = {("H", 0): 5, ("H", 1): 7, ("C", 0): 3, ("C", 1): 11, ("C", 2): 6, ("O", 0): 9, ("O", 1): 4, ("O", 2): 13}
# O is absent from structure 1 and H from structure 2: empty column groups
STRUCTS = [["H", "C", "O", "H", "C"], ["C", "H", "H"], ["O", "C", "O"], ["H", "O", "C", "C", "O", "H"]]

CUML, TOTSIZE = {}, 0
for _s in SPECIES:
    for _l in range(LMAX[_s] + 1):
        for _n in range(NMAX[(_s, _l)]):
            CUML[(_s, _l, _n)] = TOTSIZE
            TOTSIZE += MCUT[(_s, _l)]


def make_psi(symbols, rng):
    """PsiBlocks laid out as PsiBuilder.psi_blocks does, and the same matrix dense."""
    keys = [(s, l) for s in SPECIES if s in symbols for l in range(LMAX[s] + 1)]
    kern = {(s, l): rng.normal(size=(symbols.count(s) * (2 * l + 1), MCUT[(s, l)])) for s, l in keys}
    base, off = {}, 0
    for key in keys:
        base[key] = off
        off += kern[key].size
    vals = np.concatenate([kern[key].ravel() for key in keys])
    ispe, tab, i = {s: 0 for s in SPECIES}, [], 0
    for s in symbols:
        for l in range(LMAX[s] + 1):
            nc = MCUT[(s, l)]
            voff = base[(s, l)] + ispe[s] * (2 * l + 1) * nc
            for n in range(NMAX[(s, l)]):
                tab.append((voff, 2 * l + 1, nc, i, CUML[(s, l, n)]))
                i += 2 * l + 1
        ispe[s] += 1
    tab = np.array(tab, dtype=np.int64)
    tab = tab[np.argsort(tab[:, 4], kind="stable")]
    return PsiBlocks(vals, tab, (i, TOTSIZE))


def dense_psi(p):
    a = np.zeros(p.shape)
    for voff, nr, nc, r0, c0 in p.tab:
        a[r0:r0 + nr, c0:c0 + nc] = p.vals[voff:voff + nr * nc].reshape(nr, nc)
    return a


def make_case(dtype, seed=0):
    rng = np.random.default_rng(seed)
    psis, ovls = [], []
    for symbols in STRUCTS:
        p = make_psi(symbols, rng)
        g = rng.normal(size=(p.shape[0], p.shape[0]))
        o = PackedSym.from_dense(g @ g.T / p.shape[0] + np.eye(p.shape[0]))
        p.vals = p.vals.astype(dtype)
        o.ap = o.ap.astype(dtype)
        psis.append(p)
        ovls.append(o)
    store = SimpleNamespace(npsi=len(psis), _overlaps=ovls, get_psi=psis.__getitem__, get_overlap=ovls.__getitem__)
    # dense reference from the stored (possibly rounded) values
    dense = [(dense_psi(SimpleNamespace(vals=p.vals.astype(np.float64), tab=p.tab, shape=p.shape)),
              PackedSym(o.ap.astype(np.float64), o.n).dense()) for p, o in zip(psis, ovls)]
    return _blocks_setup(store, TOTSIZE), dense, rng


def relerr(a, ref):
    return np.abs(a - ref).max() / np.abs(ref).max()


@pytest.mark.parametrize("dtype,tol", [(np.float64, 1e-13), (np.float32, 1e-12)])
def test_kernel_matches_dense(dtype, tol):
    kargs, dense, rng = make_case(dtype)
    assert kargs is not None
    d = rng.normal(size=TOTSIZE)
    e = rng.normal(size=sum(a.shape[0] for a, _ in dense))

    ref_curv = sum(2.0 * a.T @ (s @ (a @ d)) for a, s in dense)
    assert relerr(_curv(kargs, d), ref_curv) <= tol

    y, part = _psi_s(kargs, d, e)
    roff = np.cumsum([0] + [a.shape[0] for a, _ in dense])
    us = [a @ d + e[roff[k]:roff[k + 1]] for k, (a, _) in enumerate(dense)]
    ref_y = np.concatenate([s @ u for u, (_, s) in zip(us, dense)])
    ref_part = np.array([u @ (s @ u) for u, (_, s) in zip(us, dense)])
    assert relerr(y, ref_y) <= tol
    assert relerr(part, ref_part) <= tol

    ref_t = sum(a.T @ y[roff[k]:roff[k + 1]] for k, (a, _) in enumerate(dense))
    assert relerr(_psi_t(kargs, y), ref_t) <= tol


def test_kernel_independent_of_threads():
    kargs, _, rng = make_case(np.float64, seed=1)
    d = rng.normal(size=TOTSIZE)
    nt = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        one = _curv(kargs, d)
    finally:
        numba.set_num_threads(nt)
    assert np.array_equal(one, _curv(kargs, d))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_matrix_bytes_matches_stored(dtype):
    """PsiBuilder.matrix_bytes (used to split structures over ranks) against the arrays actually stored."""
    from salted.psi_builder import PsiBuilder

    pb = SimpleNamespace(species=SPECIES, lmax=LMAX, nmax=NMAX, ncol=MCUT, atomic_symbols=dict(enumerate(STRUCTS)))
    rng = np.random.default_rng(2)
    for k, symbols in enumerate(STRUCTS + [["H", "Na", "O"]]):  # Na: not a target species, no rows
        p = make_psi([s for s in symbols if s in SPECIES], rng)
        n = p.shape[0]
        stored = p.vals.astype(dtype).nbytes + p.tab.nbytes + np.dtype(dtype).itemsize * n * (n + 1) // 2
        pb.atomic_symbols[k] = symbols
        assert PsiBuilder.matrix_bytes(pb, k, np.dtype(dtype).itemsize) == stored
