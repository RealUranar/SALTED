"""Kronecker block preconditioner for the CG weight fit of minimize_loss.

A = norm sum_k 2 Psi_k^T S_k Psi_k + 2 regul I.  The columns of one species
and lam, all radial channels n, form one block.  Its preconditioner is

    M = R (x) G + 2 regul I

with R (nmax x nmax) the same-centre radial overlap of one atom, averaged
over the structures that hold the species, and G = 2 norm sum_k V_k^T V_k,
V_k the stacked kernel blocks of the species' atoms in structure k.  M is the
exact block when S keeps only its same-centre part.  With R = Q diag(l) Q^T

    M^-1 = (Q (x) I) diag_i (l_i G + 2 regul I)^-1 (Q^T (x) I),

so a block needs one Cholesky factor of size nc per eigenvalue of R.  Each
factor (block, i) is a unit owned by one rank; every rank applies its units
and one Allreduce sums M^-1 r.  M is scaled symmetrically, E M E, so that its
diagonal equals the Jacobi diagonal, unless G has fewer rows than columns.
"""
import contextlib
import ctypes
import glob
import os.path as osp
import time

import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve

CHUNK_BYTES = 512 * 2**20  # stacked local kernel blocks per product
PANEL_BYTES = 256 * 2**20  # columns of G sent per Reduce


def _openblas_threads():
    """(get, set) of the OpenBLAS that scipy.linalg calls, or None."""
    libs = glob.glob(osp.join(osp.dirname(scipy.__file__) + ".libs", "*openblas*"))
    for path in libs + [None]:
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        for pre in ("scipy_openblas_", "openblas_"):
            if hasattr(lib, pre + "set_num_threads"):
                return getattr(lib, pre + "get_num_threads"), getattr(lib, pre + "set_num_threads")
    return None


_OB = _openblas_threads()


@contextlib.contextmanager
def _one_blas_thread():
    """OpenBLAS's threaded Cholesky segfaults on large blocks in our image."""
    if _OB is None:
        yield
        return
    n = _OB[0]()
    _OB[1](1)
    try:
        yield
    finally:
        _OB[1](n)


def _packed_sub(o, rows):
    """S[rows][:, rows] from a PackedSym (row-packed upper triangle) or a dense array."""
    if isinstance(o, np.ndarray):
        return o[np.ix_(rows, rows)]
    n = o.n
    i, j = rows[:, None], rows[None, :]
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    return o.ap[lo * n - lo * (lo - 1) // 2 + (hi - lo)]


def assign_units(blocks, size, order=None):
    """Owner rank of every unit (block b, eigenvalue i): largest first, each to the
    least loaded rank, ties in the given rank order.  Load is nc^2 (factor bytes
    and apply time).  order should alternate nodes so big units spread over them."""
    order = list(range(size)) if order is None else list(order)
    units = [(b, i) for b, (_, nmax, _) in enumerate(blocks) for i in range(nmax)]
    units.sort(key=lambda u: (-blocks[u[0]][2] ** 2, u))
    load = np.zeros(size)
    owner = {}
    for u in units:
        o = order[int(np.argmin(load[order]))]
        owner[u] = o
        load[o] += blocks[u[0]][2] ** 2
    return owner


class KronPrecond:
    """r -> M^-1 r (collective)."""

    def __init__(self, units, totsize, comm):
        self.units, self.totsize, self.comm = units, totsize, comm
        self.nbytes = sum(u["F"][0].nbytes for u in units)
        self.t_apply, self.n_apply = 0.0, 0

    def __call__(self, r):
        t = time.perf_counter()
        z = np.zeros(self.totsize)
        with _one_blas_thread():
            for u in self.units:
                c0, nmax, nc, q, e = u["c0"], u["nmax"], u["nc"], u["q"], u["e"]
                x = (r[c0:c0 + nmax * nc] * e).reshape(nmax, nc)
                y = cho_solve(u["F"], q @ x, check_finite=False)
                z[c0:c0 + nmax * nc] += np.outer(q, y).ravel() * e
        if self.comm is not None:
            from mpi4py import MPI
            self.comm.Allreduce(MPI.IN_PLACE, z, op=MPI.SUM)
        self.t_apply += time.perf_counter() - t
        self.n_apply += 1
        return z


def kron_precond(matrices, totsize, blocks, norm, regul, diag, comm=None, order=None,
                 chunk_bytes=CHUNK_BYTES, panel_bytes=PANEL_BYTES):
    """Build the preconditioner from this rank's structures (PsiBlocks Psi, packed
    or dense S, any storage dtype; sums in float64).  Collective.

    blocks: [(c0, nmax, nc)] per (species, lam), channel n at columns
    c0 + n*nc.  diag: the global Jacobi diagonal incl. 2 regul.  The result is
    deterministic for a given rank layout."""
    rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
    if comm is not None:
        from mpi4py import MPI
    owner = assign_units(blocks, size, order)
    shift = 2.0 * regul

    # c0 of a column group -> its table slice, per local structure
    psis = [matrices.get_psi(k) for k in range(matrices.npsi)]
    ovls = [matrices.get_overlap(k) for k in range(matrices.npsi)]
    gidx = [{int(p.tab[g0, 4]): (g0, g1) for g0, g1 in zip(p.grp[:-1], p.grp[1:])} for p in psis]

    def nrows(k, c0):
        g0, g1 = gidx[k][c0]
        return (g1 - g0) * int(psis[k].tab[g0, 1])

    def stacked(ch, c0, nc):
        """The kernel blocks of structures ch, stacked, in float64."""
        V = np.empty((sum(nrows(k, c0) for k in ch), nc))
        i = 0
        for k in ch:
            p = psis[k]
            g0, g1 = gidx[k][c0]
            nr = int(p.tab[g0, 1])
            for o in p.tab[g0:g1, 0]:
                V[i:i + nr] = p.vals[o:o + nr * nc].reshape(nr, nc)
                i += nr
        return V

    units, t_sum, t_fac = [], 0.0, 0.0
    for b, (c0, nmax, nc) in enumerate(blocks):
        ta = time.time()
        cs = [c0 + j * nc for j in range(nmax)]
        mine = [k for k in range(len(psis)) if c0 in gidx[k]]
        mine_units = [i for i in range(nmax) if owner[(b, i)] == rank]
        owners = sorted({owner[(b, i)] for i in range(nmax)})

        # same-centre radial overlap: first atom, m = 0, of every channel
        R = np.zeros((nmax, nmax))
        for k in mine:
            r0 = np.array([psis[k].tab[gidx[k][c][0], 3] for c in cs])
            R += _packed_sub(ovls[k], r0)
        cnt = np.array([len(mine), sum(nrows(k, c0) for k in mine)], dtype=np.int64)  # structures, rows of G
        if comm is not None:
            comm.Allreduce(MPI.IN_PLACE, R, op=MPI.SUM)
            comm.Allreduce(MPI.IN_PLACE, cnt, op=MPI.SUM)
        R /= max(int(cnt[0]), 1)

        # G in column panels, each summed over local chunks of stacked V, then reduced to every owner
        chunks, cur, cb = [], [], 0
        for k in mine:
            nb = 8 * nrows(k, c0) * nc
            if cur and cb + nb > chunk_bytes:
                chunks.append(cur)
                cur, cb = [], 0
            cur.append(k)
            cb += nb
        if cur:
            chunks.append(cur)
        G = np.zeros((nc, nc)) if mine_units else None
        w = max(1, min(nc, panel_bytes // (8 * nc)))
        for a0 in range(0, nc, w):
            a1 = min(nc, a0 + w)
            loc = np.zeros((nc, a1 - a0))
            for ch in chunks:
                Vs = stacked(ch, c0, nc)
                loc += Vs.T @ Vs[:, a0:a1]
                del Vs
            if comm is None:
                G[:, a0:a1] += loc
                continue
            for o in owners:
                rb = np.empty_like(loc) if rank == o else None
                comm.Reduce(loc, rb, op=MPI.SUM, root=o)
                if rank == o:
                    G[:, a0:a1] += rb
            del loc
        t_sum += time.time() - ta
        if not mine_units:
            continue

        tb = time.time()
        G *= 2.0 * norm
        lr, Q = np.linalg.eigh(R)
        dk = (np.diag(R)[:, None] * np.diag(G)[None, :]).ravel() + shift
        dt = diag[c0:c0 + nmax * nc]
        e = np.sqrt(np.where(dt > 0.0, dk / np.where(dt > 0.0, dt, 1.0), 1.0))
        if cnt[1] < nc:
            # G is singular, and A = M = 2 regul on its null space; scaling would tilt
            # those directions into range(G) and blow up M^-1 A
            e[:] = 1.0
        for j, i in enumerate(mine_units):
            Mi = G if j == len(mine_units) - 1 else G.copy()  # the last unit reuses G's memory
            Mi *= max(lr[i], 0.0)
            Mi[np.diag_indices(nc)] += shift
            with _one_blas_thread():  # Mi.T: the same symmetric matrix in Fortran order, factored in place
                F = cho_factor(Mi.T, lower=False, overwrite_a=True, check_finite=False)
            units.append({"c0": c0, "nmax": nmax, "nc": nc, "q": np.ascontiguousarray(Q[:, i]), "e": e, "F": F})
        del G, Mi
        t_fac += time.time() - tb

    pc = KronPrecond(units, totsize, comm)
    pc.t_sum, pc.t_fac, pc.nblocks, pc.nunits = t_sum, t_fac, len(blocks), len(owner)
    pc.max_block = max(nmax * nc for _, nmax, nc in blocks)
    return pc
