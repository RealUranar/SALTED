"""Kronecker block preconditioner of minimize_loss against dense matrices.

Serial under pytest.  Under MPI (e.g. srun -n 3) the same tests deal the
structures over the ranks; every rank checks the assembled M^-1 x."""

from types import SimpleNamespace

import numpy as np
import pytest

from salted.minimize_loss import PackedSym
from salted.precond_kron import assign_units, kron_precond
from test_curv_kernel import CUML, LMAX, MCUT, NMAX, SPECIES, STRUCTS, TOTSIZE, dense_psi, make_psi

BLOCKS = [(CUML[(s, l, 0)], NMAX[(s, l)], MCUT[(s, l)]) for s in SPECIES for l in range(LMAX[s] + 1)]
REGUL = 1e-3


def _comm():
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    return MPI.COMM_WORLD if MPI.COMM_WORLD.Get_size() > 1 else None


def _onsite_overlap(symbols, rng):
    """S with only its same-centre part delta_mm' R_(s,l)(n, n'), in make_psi's row order."""
    R = {}
    for s in SPECIES:
        for l in range(LMAX[s] + 1):
            q = rng.normal(size=(NMAX[(s, l)],) * 2)
            R[(s, l)] = q @ q.T + np.eye(NMAX[(s, l)])
    rows = []
    for s in symbols:
        for l in range(LMAX[s] + 1):
            rows.append(np.kron(R[(s, l)], np.eye(2 * l + 1)))
    T = sum(r.shape[0] for r in rows)
    S, i = np.zeros((T, T)), 0
    for r in rows:
        S[i:i + r.shape[0], i:i + r.shape[0]] = r
        i += r.shape[0]
    return S


def _case(kind, dtype, seed=0, regul=REGUL):
    rng = np.random.default_rng(seed)
    psis, ovls, dense = [], [], []
    for symbols in STRUCTS:
        p = make_psi(symbols, rng)
        if kind == "onsite":
            S = _onsite_overlap(symbols, np.random.default_rng(seed + 100))  # same R in every structure
        else:
            g = rng.normal(size=(p.shape[0], p.shape[0]))
            S = g @ g.T / p.shape[0] + np.eye(p.shape[0])
        o = PackedSym.from_dense(S)
        p.vals, o.ap = p.vals.astype(dtype), o.ap.astype(dtype)
        psis.append(p)
        ovls.append(o)
        dense.append((dense_psi(SimpleNamespace(vals=p.vals.astype(np.float64), tab=p.tab, shape=p.shape)),
                      PackedSym(o.ap.astype(np.float64), o.n).dense()))
    norm = 1.0 / len(STRUCTS)
    A = norm * sum(2.0 * a.T @ s @ a for a, s in dense) + 2.0 * regul * np.eye(TOTSIZE)
    return psis, ovls, dense, norm, A


def _reference(kind, dense, norm, A, regul=REGUL):
    """Dense block-diagonal M^-1 (onsite: the exact blocks of A; full: E (R x G + 2 regul)^-1 E)."""
    Minv = np.zeros_like(A)
    for c0, nmax, nc in BLOCKS:
        sl = slice(c0, c0 + nmax * nc)
        if kind == "onsite":
            Minv[sl, sl] = np.linalg.inv(A[sl, sl])
            continue
        cols = slice(c0, c0 + nc)
        G = 2.0 * norm * sum(a[:, cols].T @ a[:, cols] for a, _ in dense)
        Rs = []
        for a, s in dense:
            rows = [int(np.flatnonzero(a[:, c0 + n * nc:c0 + (n + 1) * nc].any(axis=1))[0]) for n in range(nmax)
                    if a[:, c0 + n * nc:c0 + (n + 1) * nc].any()]
            if rows:
                Rs.append(s[np.ix_(rows, rows)])
        R = sum(Rs) / max(len(Rs), 1)
        M = np.kron(R, G) + 2.0 * regul * np.eye(nmax * nc)
        e = np.sqrt(np.diag(M) / np.diag(A)[sl])
        if sum(int(a[:, cols].any(axis=1).sum()) for a, _ in dense) < nc:  # singular G: no scaling
            e[:] = 1.0
        Minv[sl, sl] = e[:, None] * np.linalg.inv(M) * e[None, :]
    return Minv


@pytest.mark.parametrize("kind,dtype,tiny", [
    ("onsite", np.float64, False), ("full", np.float64, False), ("full", np.float64, True),
    ("full", np.float32, True),
])
def test_matches_dense(kind, dtype, tiny):
    """tiny: one structure per chunk and one column per Reduce panel."""
    comm = _comm()
    rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
    psis, ovls, dense, norm, A = _case(kind, dtype)
    mine = [k for k in range(len(psis)) if k % size == rank]
    store = SimpleNamespace(npsi=len(mine), get_psi=lambda k: psis[mine[k]], get_overlap=lambda k: ovls[mine[k]])
    kw = dict(chunk_bytes=1, panel_bytes=1) if tiny else {}
    M = kron_precond(store, TOTSIZE, BLOCKS, norm, REGUL, np.diag(A).copy(), comm, **kw)
    x = np.random.default_rng(7).normal(size=TOTSIZE)
    ref = _reference(kind, dense, norm, A) @ x
    got = M(x)
    assert np.abs(got - ref).max() <= 1e-10 * np.abs(ref).max()
    assert np.array_equal(M(x), got)  # deterministic


def test_singular_gram_beats_jacobi():
    """O lam 0 has fewer kernel rows (5) than columns (9), so its G is singular;
    with a small regul the preconditioned system must still beat Jacobi."""
    comm = _comm()
    rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
    regul = 1e-8
    psis, ovls, _, norm, A = _case("full", np.float64, regul=regul)
    mine = [k for k in range(len(psis)) if k % size == rank]
    store = SimpleNamespace(npsi=len(mine), get_psi=lambda k: psis[mine[k]], get_overlap=lambda k: ovls[mine[k]])
    M = kron_precond(store, TOTSIZE, BLOCKS, norm, regul, np.diag(A).copy(), comm)
    Minv = np.column_stack([M(x) for x in np.eye(TOTSIZE)])
    L = np.linalg.cholesky(A)

    def cond(P):
        ev = np.linalg.eigvalsh(L.T @ (0.5 * (P + P.T)) @ L)
        return ev[-1] / ev[0]

    assert cond(Minv) < cond(np.diag(1.0 / np.diag(A)))


def test_assign_units_spreads_largest():
    blocks = [(0, 2, 100), (200, 3, 10), (230, 1, 50)]
    owner = assign_units(blocks, 4, order=[0, 2, 1, 3])
    assert sorted(owner) == [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0)]
    assert owner[(0, 0)] == 0 and owner[(0, 1)] == 2 and owner[(2, 0)] == 1
    assert len({owner[(1, i)] for i in range(3)} | {owner[(0, 0)], owner[(0, 1)], owner[(2, 0)]}) == 4
