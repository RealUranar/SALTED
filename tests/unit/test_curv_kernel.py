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

def layout(mcut):
    cuml, totsize = {}, 0
    for s in SPECIES:
        for l in range(LMAX[s] + 1):
            for n in range(NMAX[(s, l)]):
                cuml[(s, l, n)] = totsize
                totsize += mcut[(s, l)]
    return cuml, totsize


CUML, TOTSIZE = layout(MCUT)


def make_psi(symbols, rng, mcut=MCUT):
    """PsiBlocks laid out as PsiBuilder.psi_blocks does, and the same matrix dense."""
    cuml, totsize = layout(mcut)
    keys = [(s, l) for s in SPECIES if s in symbols for l in range(LMAX[s] + 1)]
    kern = {(s, l): rng.normal(size=(symbols.count(s) * (2 * l + 1), mcut[(s, l)])) for s, l in keys}
    base, off = {}, 0
    for key in keys:
        base[key] = off
        off += kern[key].size
    vals = np.concatenate([np.zeros(0)] + [kern[key].ravel() for key in keys])
    ispe, tab, i = {s: 0 for s in SPECIES}, [], 0
    for s in symbols:
        for l in range(LMAX[s] + 1):
            nc = mcut[(s, l)]
            voff = base[(s, l)] + ispe[s] * (2 * l + 1) * nc
            for n in range(NMAX[(s, l)]):
                tab.append((voff, 2 * l + 1, nc, i, cuml[(s, l, n)]))
                i += 2 * l + 1
        ispe[s] += 1
    tab = np.array(tab, dtype=np.int64).reshape(-1, 5)
    tab = tab[np.argsort(tab[:, 4], kind="stable")]
    return PsiBlocks(vals, tab, (i, totsize))


def dense_psi(p):
    a = np.zeros(p.shape)
    for voff, nr, nc, r0, c0 in p.tab:
        a[r0:r0 + nr, c0:c0 + nc] = p.vals[voff:voff + nr * nc].reshape(nr, nc)
    return a


def make_case(dtype, seed=0, structs=STRUCTS, mcut=MCUT):
    rng = np.random.default_rng(seed)
    psis, ovls = [], []
    for symbols in structs:
        p = make_psi(symbols, rng, mcut)
        g = rng.normal(size=(p.shape[0], p.shape[0]))
        o = PackedSym.from_dense(g @ g.T / max(p.shape[0], 1) + np.eye(p.shape[0]))
        p.vals = p.vals.astype(dtype)
        o.ap = o.ap.astype(dtype)
        psis.append(p)
        ovls.append(o)
    store = SimpleNamespace(npsi=len(psis), _overlaps=ovls, get_psi=psis.__getitem__, get_overlap=ovls.__getitem__)
    # dense reference from the stored (possibly rounded) values
    dense = [(dense_psi(SimpleNamespace(vals=p.vals.astype(np.float64), tab=p.tab, shape=p.shape)),
              PackedSym(o.ap.astype(np.float64), o.n).dense()) for p, o in zip(psis, ovls)]
    return _blocks_setup(store, layout(mcut)[1]), dense, rng


def relerr(a, ref):
    return np.abs(a - ref).max() / np.abs(ref).max()


@pytest.mark.parametrize("dtype,tol", [(np.float64, 1e-13), (np.float32, 1e-12)])
def test_kernel_matches_dense(dtype, tol):
    _check_kernel(*make_case(dtype), tol)


def test_kernel_empty_blocks():
    """An Mcut = 0 key and a structure without atoms of a target species."""
    mcut = {**MCUT, ("C", 1): 0}
    p = make_psi(["C", "H"], np.random.default_rng(9), mcut)
    a = dense_psi(p)
    x, y = np.random.default_rng(10).normal(size=p.shape[1]), np.random.default_rng(11).normal(size=p.shape[0])
    assert np.allclose(p.dot(x), a @ x) and np.allclose(p.rdot(y), a.T @ y)
    _check_kernel(*make_case(np.float64, structs=STRUCTS + [[]], mcut=mcut), 1e-13)


def _check_kernel(kargs, dense, rng, tol):
    assert kargs is not None
    d = rng.normal(size=dense[0][0].shape[1])
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


def _stored_case(rng):
    """Dense overlaps and float64 PsiBlocks of all structures."""
    out = []
    for symbols in STRUCTS:
        p = make_psi(symbols, rng)
        g = rng.normal(size=(p.shape[0], p.shape[0]))
        s = g @ g.T / p.shape[0] + np.eye(p.shape[0])
        s[0, -1] += 1e-9  # stored overlaps are symmetric only up to round-off
        out.append((p, s))
    return out


@pytest.mark.parametrize("mode", ["memory", "mmap"])
def test_matrix_store_float32_round_trip(mode, tmp_path, monkeypatch):
    """float32 storage: everything spilled to the pack in mmap mode, read back as the same float32 values."""
    from salted.minimize_loss import MatrixStore

    monkeypatch.setenv("SALTED_RAM_BUDGET_GB", "0")
    monkeypatch.setenv("SALTED_PSI_CACHE_DIR", str(tmp_path / "cache"))
    cases = _stored_case(np.random.default_rng(3))
    store = MatrixStore(mode, str(tmp_path), 0, packed=True, dtype=np.float32)
    for k, (p, s) in enumerate(cases):
        path = tmp_path / f"overlap_conf{k}.npy"
        np.save(path, s)
        store.add_overlap(str(path))
        store.add_psi(PsiBlocks(p.vals.astype(np.float32), p.tab, p.shape))
    store.finalize()

    nbytes = 0
    for k, (p, s) in enumerate(cases):
        ap, vals = store.get_overlap(k).ap, store.get_psi(k).vals
        assert ap.dtype == np.float32 and vals.dtype == np.float32
        assert np.array_equal(ap, PackedSym.from_dense(s).ap.astype(np.float32))
        assert np.array_equal(vals, p.vals.astype(np.float32))
        nbytes += ap.nbytes + vals.nbytes
    if mode == "mmap":
        assert store.ram_bytes == 0 and store.disk_bytes == nbytes
    else:
        assert store.disk_bytes == 0

    kargs = _blocks_setup(store, TOTSIZE)
    assert kargs is not None
    d = np.random.default_rng(4).normal(size=TOTSIZE)
    ref = sum(2.0 * dense_psi(SimpleNamespace(vals=store.get_psi(k).vals.astype(np.float64), tab=p.tab, shape=p.shape)).T
              @ (PackedSym(store.get_overlap(k).ap.astype(np.float64), p.shape[0]).dense()
                 @ (dense_psi(SimpleNamespace(vals=store.get_psi(k).vals.astype(np.float64), tab=p.tab, shape=p.shape)) @ d))
              for k, (p, _) in enumerate(cases))
    assert relerr(_curv(kargs, d), ref) <= 1e-12


def test_matrix_store_float32_needs_packed(tmp_path):
    from salted.minimize_loss import MatrixStore

    with pytest.raises(ValueError):
        MatrixStore("memory", str(tmp_path), 0, packed=False, dtype=np.float32)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_precond_from_stored_values(dtype):
    """Jacobi diagonal built at setup from the dense overlap equals diag(2 Psi^T S Psi) of the stored matrices."""
    from salted.minimize_loss import _precond_add_blocks

    for p, s in _stored_case(np.random.default_rng(5)):
        p = PsiBlocks(p.vals.astype(dtype), p.tab, p.shape)
        diag = np.zeros(TOTSIZE)
        _precond_add_blocks(diag, p, s)
        a = dense_psi(SimpleNamespace(vals=p.vals.astype(np.float64), tab=p.tab, shape=p.shape))
        stored_s = PackedSym.from_dense(s).ap.astype(dtype).astype(np.float64)
        ref = 2.0 * np.einsum("ij,ij->j", PackedSym(stored_s, p.shape[0]).dense() @ a, a)
        # float64 uses the unsymmetrised S, which differs from the stored one by round-off only
        assert relerr(diag, ref) <= 1e-13
