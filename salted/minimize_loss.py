import ctypes
import gc
import hashlib
import os
import os.path as osp
import shutil
import time

import numpy as np
from numba import njit, prange, types
from numba.extending import get_cython_function_address, overload
from numba.typed import List
from scipy import sparse

from salted import get_averages
from salted.psi_builder import PsiBlocks, PsiBuilder
from salted.selection_utils import load_training_indices
from salted.sys_utils import (
    ParseConfig,
    check_MPI_tasks_count,
    detect_mpi,
    distribute_jobs,
    format_index_ranges,
    get_atom_idx,
    read_frames,
    read_system,
    run_or_abort,
    split_by_cost,
)

def _drop_from_page_cache(root, keep):
    """posix_fadvise(DONTNEED) on every file under root except the dir keep; no file is changed."""
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if osp.join(d, x) != keep]
        for f in files:
            try:
                fd = os.open(osp.join(d, f), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)

def _trim_heap():
    """Return heap freed during setup to the OS; glibc otherwise keeps it per rank."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):  # not glibc
        pass


def _ram_budget(nlocal):
    """Bytes of matrices this rank keeps in RAM in mmap mode; the rest is paged
    from the node-local pack file.

    SALTED_RAM_BUDGET_GB sets it per rank (0 = everything on disk).  Otherwise
    SALTED_RAM_FRACTION (default 0.5) of this node's memory, shared by the
    nlocal ranks on the node: the Slurm allocation if Slurm states one, capped
    by MemAvailable at start-up (--mem=0 leaves only the latter)."""
    gb = os.environ.get("SALTED_RAM_BUDGET_GB")
    if gb is not None:
        return int(float(gb) * 2**30)
    with open("/proc/meminfo") as f:
        node = next(int(l.split()[1]) * 1024 for l in f if l.startswith("MemAvailable:"))
    slurm = int(os.environ.get("SLURM_MEM_PER_NODE", "0")) * 2**20
    if not slurm and "SLURM_MEM_PER_CPU" in os.environ:
        slurm = int(os.environ["SLURM_MEM_PER_CPU"]) * int(os.environ.get("SLURM_CPUS_ON_NODE", "1")) * 2**20
    if slurm:
        node = min(node, slurm)
    return int(node * float(os.environ.get("SALTED_RAM_FRACTION", "0.5")) / nlocal)


def _dot(psi, x):
    """Psi @ x for a scipy matrix or a PsiBlocks (bit-identical to each other)."""
    if isinstance(psi, PsiBlocks):
        return psi.dot(x)
    return sparse.csr_matrix.dot(psi, x)


def _rdot(psi, y):
    """Psi.T @ y for a scipy matrix or a PsiBlocks."""
    if isinstance(psi, PsiBlocks):
        return psi.rdot(y)
    return sparse.csc_matrix.dot(psi.T, y)


@njit(fastmath=True)
def _spmv_rows(ap, x, zc, i0, i1):
    """Rows i0:i1 of the packed S @ x, scattered into the buffer zc."""
    n = x.shape[0]
    for i in range(i0, i1):
        # Two 1-D loops over the row vectorise; one fused loop does not.
        o = i * n - i * (i - 1) // 2
        r = ap[o:o + n - i]
        xs = x[i:]
        acc = 0.0
        for k in range(n - i):
            acc += r[k] * xs[k]
        xi = x[i]
        zs = zc[i + 1:]
        for k in range(1, n - i):
            zs[k - 1] += r[k] * xi
        zc[i] += acc


@njit(fastmath=True)
def _sum_rows(z, y, j0, j1):
    for j in range(j0, j1):
        s = 0.0
        for c in range(z.shape[0]):
            s += z[c, j]
        y[j] = s


@njit(parallel=True, fastmath=True)
def _spmv_upper(ap, chunks, x):
    """S @ x from the row-packed upper triangle ap of a symmetric S.

    Row i holds S[i, i:].  Each fixed chunk of rows scatters its lower-triangle
    half into its own buffer and the buffers are summed in chunk order, so the
    bits do not depend on the thread count."""
    n = x.shape[0]
    nc = chunks.shape[0] - 1
    z = np.zeros((nc, n))
    for c in prange(nc):
        _spmv_rows(ap, x, z[c], chunks[c], chunks[c + 1])
    y = np.empty(n)
    for j in prange(n):
        _sum_rows(z, y, j, j + 1)
    return y


# --------------------------------------------------------------------------
# Block kernel for Psi^T S Psi products over all structures of a rank.
#
# Psi_k is stored as (atom, lam) blocks B (nr = 2 lam + 1 rows, nc columns),
# one block serving all nmax radial channels n at rows r0 + n*nr and columns
# c0 + n*nc.  Each block is read once for all its channels:
#   z[r0 + n*nr + m]  = sum_c B[m, c] d[c0 + n*nc + c]   (Z = D B^T, dgemm)
#   t[c0 + n*nc + c] += sum_m B[m, c] y[r0 + n*nr + m]   (T += Y B,  dgemm)
# Psi values and packed S may be float64 or float32; sums are float64.
# --------------------------------------------------------------------------

_vp = ctypes.c_void_p
_dgemm = ctypes.CFUNCTYPE(None, *([_vp] * 13))(
    get_cython_function_address("scipy.linalg.cython_blas", "dgemm"))


def _as_f64(src, scr):
    """src as float64: src itself, or its copy in the scratch buffer scr."""


@overload(_as_f64)
def _as_f64_impl(src, scr):
    if src.dtype == types.float64:
        return lambda src, scr: src

    def impl(src, scr):
        s = scr[:src.shape[0]]
        for i in range(src.shape[0]):
            s[i] = src[i]
        return s
    return impl


@njit
def _z_blocks(vals, ub, d, z, ia, fa, ch, scr):
    """z = Psi d for one structure (ub: its unique-block table)."""
    for b in range(ub.shape[0]):
        voff, nr, nc, r0, c0, nm = ub[b, 0], ub[b, 1], ub[b, 2], ub[b, 3], ub[b, 4], ub[b, 5]
        blk = _as_f64(vals[voff:voff + nr * nc], scr)
        ia[0] = nr
        ia[1] = nm
        ia[2] = nc
        fa[0] = 1.0
        fa[1] = 0.0
        # Z^T (nr x nm) = (B^T as nc x nr)^T @ (D^T as nc x nm), column-major
        _dgemm(ch[1:].ctypes, ch[0:].ctypes, ia[0:].ctypes, ia[1:].ctypes, ia[2:].ctypes, fa[0:].ctypes,
               blk.ctypes, ia[2:].ctypes, d[c0:].ctypes, ia[2:].ctypes, fa[1:].ctypes,
               z[r0:].ctypes, ia[0:].ctypes)


@njit
def _t_blocks(vals, ub, y, t, ia, fa, ch, scr):
    """t += Psi^T y over the blocks in ub of one structure."""
    for b in range(ub.shape[0]):
        voff, nr, nc, r0, c0, nm = ub[b, 0], ub[b, 1], ub[b, 2], ub[b, 3], ub[b, 4], ub[b, 5]
        blk = _as_f64(vals[voff:voff + nr * nc], scr)
        ia[0] = nc
        ia[1] = nm
        ia[2] = nr
        fa[0] = 1.0
        fa[1] = 1.0
        # T^T (nc x nm) += (B^T as nc x nr) @ (Y^T as nr x nm), column-major
        _dgemm(ch[0:].ctypes, ch[0:].ctypes, ia[0:].ctypes, ia[1:].ctypes, ia[2:].ctypes, fa[0:].ctypes,
               blk.ctypes, ia[0:].ctypes, y[r0:].ctypes, ia[2:].ctypes, fa[1:].ctypes,
               t[c0:].ctypes, ia[0:].ctypes)


@njit
def _spmv_packed(ap, x, y):
    """y = S x from the row-packed upper triangle ap (row i holds S[i, i:])."""
    n = x.shape[0]
    y[:] = 0.0
    o = 0
    for i in range(n):
        m = n - i
        r = ap[o:o + m]
        xs = x[i:]
        # four fixed partial sums let the reduction vectorise without fastmath
        a0 = 0.0
        a1 = 0.0
        a2 = 0.0
        a3 = 0.0
        k = 0
        while k + 4 <= m:
            a0 += r[k] * xs[k]
            a1 += r[k + 1] * xs[k + 1]
            a2 += r[k + 2] * xs[k + 2]
            a3 += r[k + 3] * xs[k + 3]
            k += 4
        while k < m:
            a0 += r[k] * xs[k]
            k += 1
        xi = x[i]
        ys = y[i + 1:]
        for k in range(1, m):
            ys[k - 1] += r[k] * xi
        y[i] += (a0 + a1) + (a2 + a3)
        o += m


@njit(parallel=True)
def _stage_psi_s(vals, aps, ubs, roff, x, e, maxblk):
    """Per structure k: u = Psi_k x + e_k, y_k = S_k u and part_k = u . y_k
    (e empty: no offset).  Returns y (concatenated over k) and part."""
    npsi = len(vals)
    y = np.empty(roff[-1])
    part = np.empty(npsi)
    for kk in prange(npsi):
        k = np.int64(kk)
        a, b = roff[k], roff[k + 1]
        u = np.empty(b - a)
        ia = np.zeros(4, dtype=np.int32)
        fa = np.zeros(2)
        ch = np.array([ord("N"), ord("T")], dtype=np.uint8)
        scr = np.empty(maxblk)
        _z_blocks(vals[k], ubs[k], x, u, ia, fa, ch, scr)
        if e.shape[0] > 0:
            u += e[a:b]
        yk = y[a:b]
        _spmv_packed(aps[k], u, yk)
        s = 0.0
        for i in range(b - a):
            s += u[i] * yk[i]
        part[k] = s
    return y, part


@njit(parallel=True)
def _stage_psi_t(vals, ubs, roff, kptr, y, totsize, maxblk):
    """sum_k Psi_k^T y_k.  Threaded over (species, lam) keys, whose column
    ranges are disjoint, each summed over structures in order: the result does
    not depend on the thread count."""
    npsi = len(vals)
    t = np.zeros(totsize)
    for g in prange(kptr.shape[1] - 1):
        ia = np.zeros(4, dtype=np.int32)
        fa = np.zeros(2)
        ch = np.array([ord("N"), ord("T")], dtype=np.uint8)
        scr = np.empty(maxblk)
        for k in range(npsi):
            b0, b1 = kptr[k, g], kptr[k, g + 1]
            if b0 < b1:
                _t_blocks(vals[k], ubs[k][b0:b1], y[roff[k]:roff[k + 1]], t, ia, fa, ch, scr)
    return t


def _unique_blocks(tab):
    """(voff, nr, nc, r0, c0, nmax) per stored (atom, lam) block of a
    PsiBlocks table, sorted by first column, then by value offset."""
    t = tab[np.lexsort((tab[:, 4], tab[:, 0]))]
    first = np.flatnonzero(np.r_[True, np.diff(t[:, 0]) != 0])
    nm = np.diff(np.r_[first, len(t)])
    pos = np.arange(len(t)) - np.repeat(first, nm)
    g = np.repeat(np.arange(len(first)), nm)
    ub = np.column_stack([t[first, :5], nm]).astype(np.int64)
    if not (np.array_equal(t[:, 3], ub[g, 3] + pos * ub[g, 1])
            and np.array_equal(t[:, 4], ub[g, 4] + pos * ub[g, 2])
            and np.array_equal(t[:, 1], ub[g, 1]) and np.array_equal(t[:, 2], ub[g, 2])):
        raise ValueError("PsiBlocks channels are not laid out at r0 + n*nr, c0 + n*nc")
    return np.ascontiguousarray(ub[np.lexsort((ub[:, 0], ub[:, 4]))])


def _blocks_setup(matrices, totsize):
    """Kernel arguments (vals, aps, ubs, roff, kptr, totsize, maxblk) when
    every Psi is PsiBlocks and every S is PackedSym, else None (the
    per-structure loops then run)."""
    n = matrices.npsi
    if len(matrices._overlaps) != n:  # density-response: three Psi per S
        return None
    psis = [matrices.get_psi(k) for k in range(n)]
    ovls = [matrices.get_overlap(k) for k in range(n)]
    if not all(isinstance(p, PsiBlocks) for p in psis) or not all(isinstance(o, PackedSym) for o in ovls):
        return None

    def ro(a):  # pack arrays are read-only; numba lists need one array type
        a = a.view()
        a.flags.writeable = False
        return a

    vals, aps, ubs = List(), List(), List()
    for p, o in zip(psis, ovls):
        vals.append(ro(p.vals))
        aps.append(ro(o.ap))
        ubs.append(_unique_blocks(p.tab))
    roff = np.concatenate(([0], np.cumsum([p.shape[0] for p in psis]))).astype(np.int64)
    # kptr[k, g]:kptr[k, g+1] are structure k's blocks of key g (empty if the species is absent)
    keys = np.append(np.unique(np.concatenate([u[:, 4] for u in ubs])), totsize)
    kptr = np.array([np.searchsorted(u[:, 4], keys) for u in ubs], dtype=np.int64)
    maxblk = int(max((u[:, 1] * u[:, 2]).max(initial=0) for u in ubs))
    return vals, aps, ubs, roff, kptr, totsize, maxblk


_NO_OFFSET = np.empty(0)


def _psi_s(kargs, x, e=_NO_OFFSET):
    """S_k (Psi_k x + e_k) concatenated over structures k, and per structure
    (Psi_k x + e_k) . S_k (Psi_k x + e_k)."""
    vals, aps, ubs, roff, kptr, totsize, maxblk = kargs
    return _stage_psi_s(vals, aps, ubs, roff, np.ascontiguousarray(x, dtype=np.float64), e, maxblk)


def _psi_t(kargs, y):
    """sum_k Psi_k^T y_k for y concatenated over structures k."""
    vals, aps, ubs, roff, kptr, totsize, maxblk = kargs
    return _stage_psi_t(vals, ubs, roff, kptr, y, totsize, maxblk)


def _curv(kargs, d):
    """sum_k 2 Psi_k^T S_k Psi_k d."""
    return 2.0 * _psi_t(kargs, _psi_s(kargs, d)[0])


class PackedSym:
    """
    Overlap stored as the upper triangle of (S + S^T) / 2: half the bytes.

    S_ij = <phi_i|phi_j> is symmetric by definition; the stored matrices
    differ from S^T by round-off only (~3e-14), so this changes results in the
    last bits, not in substance.  gpr.packed_overlap: false keeps dense S.
    """

    NCHUNK = 64  # fixed, so the summation order is independent of threads

    def __init__(self, ap, n):
        self.ap, self.n = ap, n
        i = np.arange(n + 1, dtype=np.int64)
        rowptr = i * n - i * (i - 1) // 2
        self.chunks = np.searchsorted(
            rowptr, np.linspace(0, rowptr[-1], self.NCHUNK + 1)).astype(np.int64)

    @classmethod
    def from_dense(cls, s):
        n = s.shape[0]
        return cls((0.5 * (s + s.T))[np.triu_indices(n)], n)

    @property
    def shape(self):
        return (self.n, self.n)

    def dot(self, x):
        return _spmv_upper(self.ap, self.chunks, np.ascontiguousarray(x, dtype=np.float64))

    def dense(self):
        u = np.zeros((self.n, self.n))
        u[np.triu_indices(self.n)] = self.ap
        return u + np.triu(u, 1).T


PRECOND_BLK = 2048  # rows of psi^T per chunk; caps the dense temporary


def _precond_add(diag_hessian, psi, ovlp):
    """Add one structure's diag(2 psi^T S psi) to diag_hessian (scipy psi)."""
    if isinstance(ovlp, PackedSym):
        ovlp = ovlp.dense()
    totsize = diag_hessian.shape[0]
    psiT = psi.T.tocsr()

    for beg in range(0, totsize, PRECOND_BLK):
        end = min(beg + PRECOND_BLK, totsize)
        blk = psiT[beg:end]
        if blk.nnz == 0:
            del blk
            continue

        tmp = blk.dot(ovlp)
        diag_hessian[beg:end] += 2.0 * np.asarray(
            blk.multiply(tmp).sum(axis=1)
        ).ravel()

        del tmp, blk

    del psiT


def _precond_add_blocks(diag_hessian, psi, ovlp):
    """_precond_add for a PsiBlocks psi: one dense S_gg V_g product per column
    group g (species, lam, n), V_g being that group's stacked kernel blocks.
    Same sum in BLAS order: differs from _precond_add in the last bits only,
    which moves the CG path, not the gradtol it converges to.

    ovlp is the dense overlap; with Psi stored below float64 its block is
    rounded as MatrixStore stores it, so the diagonal is the one of the
    rounded problem CG solves."""
    tab = psi.tab
    for g0, g1 in zip(psi.grp[:-1], psi.grp[1:]):
        t = tab[g0:g1]
        nr, nc, c0 = int(t[0, 1]), int(t[0, 2]), int(t[0, 4])
        rows = (t[:, 3, None] + np.arange(nr)).ravel()
        v = np.concatenate([psi.vals[o:o + nr * nc] for o in t[:, 0]]).reshape(-1, nc)
        s = ovlp[np.ix_(rows, rows)]
        if v.dtype != np.float64:
            s = (0.5 * (s + s.T)).astype(v.dtype).astype(np.float64)
            v = v.astype(np.float64)
        diag_hessian[c0:c0 + nc] += 2.0 * np.einsum("ij,ij->j", s @ v, v)


def _aux_size_for_species(spe, lmax, nmax):
    """Number of auxiliary coefficients carried by one atom of species ``spe``."""
    return sum(
        nmax[(spe, l)] * (2 * l + 1)
        for l in range(lmax[spe] + 1)
    )

def _aux_indices_for_targets(symbols, target_species, lmax, nmax):
    """
    Return full-vector coefficient indices belonging to all target species.

    ``target_species`` comes directly from ``inp.system.species``.  Indices are
    collected in the original atom/auxiliary-function order, not grouped by
    species.  Therefore the selected coefficient vector has the same row order
    as a Psi built for those targets, and ``ovlp[np.ix_(idx, idx)]`` retains all
    target-target couplings, including couplings between different atoms and
    between different selected species.
    """
    target_set = set(target_species)
    pieces = []
    offset = 0

    for spe in symbols:
        naux = _aux_size_for_species(spe, lmax, nmax)
        if spe in target_set:
            pieces.append(np.arange(offset, offset + naux, dtype=np.int64))
        offset += naux

    if pieces:
        indices = np.concatenate(pieces)
    else:
        indices = np.empty(0, dtype=np.int64)

    return indices, offset


def _full_average_coefficients(symbols, lmax, nmax, av_coefs):
    """Build the average-density coefficient vector in full SALTED ordering."""
    size = sum(_aux_size_for_species(spe, lmax, nmax) for spe in symbols)
    result = np.zeros(size)
    i = 0

    for spe in symbols:
        for l in range(lmax[spe] + 1):
            for n in range(nmax[(spe, l)]):
                if l == 0:
                    result[i] = av_coefs[spe][n]
                i += 2 * l + 1

    return result

class MatrixStore:
    """
    Overlap and Psi matrices of this rank's structures.

    "memory": everything stays in RAM.

    "mmap": matrices stay in RAM until this rank's budget (_ram_budget) is
    used up; the rest is appended to one pack file per rank in the node-local
    cache directory, which finalize() maps once for the whole run.  No file is
    opened per structure or per CG step, and the mapped pages stay mapped.
    Every matrix holds the same bytes wherever it lives, so the split does not
    change the results.

    dtype (gpr.matrix_dtype) is the type packed overlaps are stored in; Psi
    arrives in its storage type from PsiBuilder.psi_blocks.
    """

    ALIGN = 4096  # pack offsets; every array starts page-aligned

    def __init__(self, mode, saltedpath, rank, nlocal=1, packed=False, dtype=np.float64):
        mode = str(mode).strip().lower()
        if mode not in ("memory", "mmap"):
            raise ValueError(
                "gpr.matrix_storage must be 'memory' or 'mmap', "
                f"got {mode!r}"
            )
        self.dtype = np.dtype(dtype)
        if self.dtype != np.float64 and not packed:
            raise ValueError("gpr.matrix_dtype other than float64 needs gpr.packed_overlap: true")

        self.mode = mode
        self.rank = rank
        self.packed = packed
        # Arrays/matrices, or (offset, dtype, shape) pack references until finalize().
        self._overlaps = []
        self._psis = []
        self.ram_bytes = 0
        self.disk_bytes = 0
        self.budget = None
        self.cache_dir = None
        self._pack = None

        if self.mode == "mmap":
            self.budget = _ram_budget(nlocal)
            cache_root = os.environ.get(
                "SALTED_PSI_CACHE_DIR",
                osp.join(saltedpath, ".psi_mmap_cache"),
            )
            self.cache_dir = osp.join(cache_root, f"rank_{rank:05d}")

            # Node-local scratch is expected to be fresh for every job.  Remove
            # only this rank's directory to avoid ever touching another rank.
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            os.makedirs(self.cache_dir, exist_ok=True)
            self._pack = open(osp.join(self.cache_dir, "pack.bin"), "wb")

    def _keep(self, arr):
        """arr itself while it fits the RAM budget, else its place in the pack."""
        if self.budget is None or self.ram_bytes + arr.nbytes <= self.budget:
            self.ram_bytes += arr.nbytes
            return arr
        arr = np.ascontiguousarray(arr)
        self._pack.write(b"\0" * (-self._pack.tell() % self.ALIGN))
        ref = (self._pack.tell(), arr.dtype, arr.shape)
        arr.tofile(self._pack)
        self.disk_bytes += arr.nbytes
        return ref

    def add_overlap(self, path, indices=None):
        """
        Add an overlap matrix, optionally restricted to a principal submatrix,
        and return it (in RAM) for immediate use.

        ``indices`` is the ordered list of auxiliary functions represented by
        the corresponding Psi rows.  Advanced indexing with ``np.ix_`` keeps
        every selected-selected coupling, including couplings between distinct
        atoms of the target species.  The source is read once, here; the CG
        loop never touches it again.
        """
        ovlp = np.load(path, mmap_mode="r", allow_pickle=False)
        if indices is None:
            ovlp = np.array(ovlp)
        else:
            ovlp = ovlp[np.ix_(np.asarray(indices, dtype=np.int64), np.asarray(indices, dtype=np.int64))]
        if self.packed:
            p = PackedSym.from_dense(ovlp)
            p.ap = self._keep(p.ap.astype(self.dtype, copy=False))
            self._overlaps.append(p)
        else:
            self._overlaps.append(self._keep(ovlp))
        return ovlp

    def add_psi(self, mat):
        if self.budget is None:
            self._psis.append(mat)
        elif isinstance(mat, PsiBlocks):
            self._psis.append(("blocks", [self._keep(mat.vals)], (mat.tab, mat.shape)))
        elif mat.getformat() == "coo":
            self._psis.append(("coo", [self._keep(a) for a in (mat.data, mat.row, mat.col)], mat.shape))
        else:
            # Converting would change the sparse summation order.
            raise TypeError(f"mmap storage keeps COO or PsiBlocks Psi only, got {mat.getformat()!r}")

    def finalize(self):
        """Map the pack once and resolve its references into arrays/matrices."""
        if self._pack is None:
            return
        self._pack.close()
        pack = (np.memmap(self._pack.name, dtype=np.uint8, mode="r")
                if osp.getsize(self._pack.name) else None)
        self._pack = None

        def get(x):
            if not isinstance(x, tuple):
                return x
            off, dtype, shape = x
            n = dtype.itemsize * int(np.prod(shape))
            return np.asarray(pack[off:off + n]).view(dtype).reshape(shape)

        self._overlaps = [get(o) for o in self._overlaps]
        for o in self._overlaps:
            if isinstance(o, PackedSym):
                o.ap = get(o.ap)
        for i, e in enumerate(self._psis):
            if not isinstance(e, tuple):
                continue
            kind, parts, meta = e
            parts = [get(p) for p in parts]
            if kind == "blocks":
                self._psis[i] = PsiBlocks(parts[0], *meta)
            else:
                self._psis[i] = sparse.coo_matrix(
                    (parts[0], (parts[1], parts[2])), shape=meta, copy=False)

    def get_overlap(self, index) -> np.ndarray | PackedSym:
        return self._overlaps[index]

    def get_psi(self, index) -> PsiBlocks:
        return self._psis[index]

    def psi_shape(self, index=0):
        return self._psis[index].shape

    @property
    def npsi(self):
        return len(self._psis)


def build():
    inp = ParseConfig().parse_input()
    # frequently used parameters
    saltedname = inp.salted.saltedname
    saltedpath = inp.salted.saltedpath
    saltedtype = inp.salted.saltedtype
    average = inp.system.average
    zeta = inp.gpr.z
    Menv = inp.gpr.Menv
    Ntrain = inp.gpr.Ntrain
    regul = inp.gpr.regul
    gradtol = inp.gpr.gradtol

    # Authoritative list of species predicted by this model.  Do not infer it
    # from Psi dimensions: inp.system.species already defines all targets.
    target_species = inp.system.species

    # The caller will add this configuration field.
    matrix_storage = str(inp.gpr.matrix_storage).strip().lower()
    if matrix_storage not in ("memory", "mmap"):
        raise ValueError(
            "gpr.matrix_storage must be either 'memory' or 'mmap', "
            f"got {matrix_storage!r}"
        )
    matrix_dtype = np.dtype(inp.gpr.matrix_dtype)

    comm, size, rank, parallel = detect_mpi()

    # gpr.fast_minimizer (default true): Jacobi-preconditioned CG with
    # buffer-based collectives. Same linear system, same gradtol, same
    # solution - different iterate path, so set it false to bit-reproduce a
    # model built before this existed.
    fast_minimizer = bool(inp.gpr.fast_minimizer)
    if parallel:
        from mpi4py import MPI

    fdir = f"rkhs-vectors_{saltedname}"
    rdir = f"regrdir_{saltedname}"

    # data_selection.py is the single authority for Ntrain selection.
    # The file stores ORIGINAL/global combined.xyz configuration indices.
    train_indices = load_training_indices(inp)
    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, atomic_coords, natoms, natmax = read_system(conf_indices=train_indices)

    atom_per_spe, natoms_per_spe = get_atom_idx(ndata, natoms, species, atomic_symbols, conf_indices=train_indices)

    # load average density coefficients if needed
    if average:
        if rank == 0:
            get_averages.build()
        if parallel:
            comm.Barrier()

        av_coefs = {}
        for spe in target_species:
            av_coefs[spe] = np.load(
                os.path.join(
                    saltedpath, "coefficients", "averages", f"averages_{spe}.npy"
                )
            )

    dirpath = os.path.join(saltedpath, rdir, f"M{Menv}_zeta{zeta}")
    if rank == 0:
        os.makedirs(dirpath, exist_ok=True)
    if parallel:
        comm.Barrier()

    if saltedtype == "density":
        psi_builder = PsiBuilder(
            rank,
            system=(
                species, lmax, nmax, lmax_max, nnmax, ndata,
                atomic_symbols, atomic_coords, natoms, natmax,
            ),
            atom_info=(atom_per_spe, natoms_per_spe),
        )

    # Distribute structures to tasks
    ntraintot = int(inp.gpr.trainfrac * Ntrain)

    if parallel:
        check_MPI_tasks_count(comm, ntraintot, "training structures")
        if saltedtype == "density":
            # The slowest rank sets the step time, and a rank's step time and
            # memory follow the bytes of its matrices, not its structure count.
            jobs = train_indices[:ntraintot]
            cut = split_by_cost([psi_builder.matrix_bytes(i, matrix_dtype.itemsize) for i in jobs], size)
            trainrange = jobs[cut[rank]:cut[rank + 1]]
        else:
            trainrange = distribute_jobs(comm, train_indices[:ntraintot])
        if inp.salted.verbose:
            print(f"Task {rank} handles the following structures: {format_index_ranges(trainrange,True)}", flush=True)
    else:
        trainrange = train_indices[:ntraintot]

    # A normal Python list is useful because it is indexed repeatedly below.
    trainrange = list(trainrange)
    ntrain = len(trainrange)

    # Prepared alongside the reduced/full reference coefficients.  Computing
    # these once avoids reconstructing the average vector in every CG step.
    average_list = []

    def loss_func(weights, matrices : MatrixStore, coef_list):
        """Compute the electron-density loss function."""

        loss = 0.0

        if kargs is not None:
            loss = _psi_s(kargs, weights, resid0)[1].sum()
        elif saltedtype == "density":
            for iconf in range(ntrain):
                ref_coefs = coef_list[iconf]

                psi = matrices.get_psi(iconf)
                ovlp = matrices.get_overlap(iconf)

                # Same sparse operation/order as the previous implementation.
                pred_coefs = _dot(psi, weights)
                if average:
                    pred_coefs += average_list[iconf]

                ref_projs = ovlp.dot(ref_coefs)
                pred_projs = ovlp.dot(pred_coefs)

                # collect gradient contributions
                loss += sparse.csc_matrix.dot(
                    pred_coefs - ref_coefs, pred_projs - ref_projs
                )

                # In mmap mode these are the only live mappings for this
                # structure.  CPython releases them immediately here.
                del psi, ovlp

        elif saltedtype == "density-response":
            itot = 0
            for iconf in range(ntrain):
                ovlp = matrices.get_overlap(iconf)

                for icart in ["x", "y", "z"]:
                    ref_coefs = np.load(
                        osp.join(
                            saltedpath,
                            f"coefficients/{icart}/",
                            f"coefficients_conf{trainrange[iconf]}.npy",
                        )
                    )

                    psi = matrices.get_psi(itot)
                    pred_coefs = _dot(psi, weights)

                    ref_projs = ovlp.dot(ref_coefs)
                    pred_projs = ovlp.dot(pred_coefs)

                    loss += sparse.csc_matrix.dot(
                        pred_coefs - ref_coefs, pred_projs - ref_projs
                    )

                    del psi
                    itot += 1

                del ovlp

        loss *= norm
        if parallel:
            loss = comm.allreduce(loss)

        # add regularization term
        loss += regul * np.dot(weights, weights)

        return loss

    def grad_func(weights, matrices : MatrixStore, coef_list):
        """Compute the gradient of the electron-density loss function."""

        gradient = np.zeros(totsize)

        if kargs is not None:
            gradient = 2.0 * _psi_t(kargs, _psi_s(kargs, weights, resid0)[0])
        elif saltedtype == "density":
            for iconf in range(ntrain):

                ref_coefs = coef_list[iconf]

                psi = matrices.get_psi(iconf)
                ovlp = matrices.get_overlap(iconf)

                pred_coefs = _dot(psi, weights)
                if average:
                    pred_coefs += average_list[iconf]

                ref_projs = ovlp.dot(ref_coefs)
                pred_projs = ovlp.dot(pred_coefs)

                gradient += 2.0 * _rdot(psi, pred_projs - ref_projs,
                )

                del psi, ovlp

        elif saltedtype == "density-response":
            itot = 0
            for iconf in range(ntrain):
                ovlp = matrices.get_overlap(iconf)

                for icart in ["x", "y", "z"]:
                    ref_coefs = np.load(
                        osp.join(
                            saltedpath,
                            "coefficients",
                            f"{icart}/coefficients_conf{trainrange[iconf]}.npy",
                        )
                    )

                    psi = matrices.get_psi(itot)
                    pred_coefs = _dot(psi, weights)

                    ref_projs = ovlp.dot(ref_coefs)
                    pred_projs = ovlp.dot(pred_coefs)

                    gradient += 2.0 * _rdot(psi, pred_projs - ref_projs,
                    )

                    del psi
                    itot += 1

                del ovlp

        if parallel:
            if fast_minimizer:
                # Allreduce needs a contiguous float64 buffer; the sparse dots
                # above can hand back a matrix type. Normalise before reducing.
                gradient = np.ascontiguousarray(gradient, dtype=np.float64)
                comm.Allreduce(MPI.IN_PLACE, gradient, op=MPI.SUM)
            else:
                gradient = comm.allreduce(gradient)
            gradient = gradient * norm + 2.0 * regul * weights
        else:
            gradient *= norm
            gradient += 2.0 * regul * weights
        return gradient

    def precond_func(matrices : MatrixStore):
        """Diagonal (Jacobi) preconditioner: diag(2 * sum psi^T S psi)."""

        # With PsiBlocks it was summed during matrix preparation, from the
        # scipy matrix, in the same structure order.
        if precond_diag is not None:
            return precond_diag

        diag_hessian = np.zeros(totsize)

        for iconf in range(ntrain):
            psi = matrices.get_psi(iconf)
            ovlp = matrices.get_overlap(iconf)
            _precond_add(diag_hessian, psi, ovlp)
            del psi, ovlp

        return diag_hessian

    def curv_func(cg_dire, matrices : MatrixStore):
        """Compute curvature on the given CG direction."""

        Ad = np.zeros(totsize)

        if kargs is not None:
            Ad = _curv(kargs, cg_dire)
        elif saltedtype == "density":
            for iconf in range(ntrain):
                psi = matrices.get_psi(iconf)
                ovlp = matrices.get_overlap(iconf)

                psi_x_dire = _dot(psi, cg_dire)
                Ad += 2.0 * _rdot(psi, ovlp.dot(psi_x_dire),
                )

                del psi_x_dire, psi, ovlp

        elif saltedtype == "density-response":
            itot = 0
            for iconf in range(ntrain):
                ovlp = matrices.get_overlap(iconf)

                for _icart in ["x", "y", "z"]:
                    psi = matrices.get_psi(itot)

                    psi_x_dire = _dot(psi, cg_dire)
                    Ad += 2.0 * _rdot(psi, ovlp.dot(psi_x_dire),
                    )

                    del psi_x_dire, psi
                    itot += 1

                del ovlp

        if parallel:
            if fast_minimizer:
                # Buffer-based Allreduce, not the pickle-based lowercase one.
                Ad = np.ascontiguousarray(Ad, dtype=np.float64)
                comm.Allreduce(MPI.IN_PLACE, Ad, op=MPI.SUM)
            else:
                Ad = comm.allreduce(Ad)
            Ad = Ad * norm + 2.0 * regul * cg_dire
        else:
            Ad *= norm
            Ad += 2.0 * regul * cg_dire

        return Ad

    # -------------------------------------------------------------------------
    # Matrix preparation
    # -------------------------------------------------------------------------

    mem_time = time.time()
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED) if parallel else None
    nlocal = node_comm.Get_size() if parallel else 1
    matrices = MatrixStore(matrix_storage, saltedpath, rank, nlocal,
                           packed=bool(inp.gpr.packed_overlap), dtype=matrix_dtype)
    coef_list = []
    precond_diag = None
    # SALTED_PSI_BLOCKS=0 restores the scipy COO Psi (same results up to rounding).
    use_blocks = saltedtype == "density" and os.environ.get("SALTED_PSI_BLOCKS", "1") != "0"
    if matrix_dtype != np.float64 and not use_blocks:
        raise ValueError("gpr.matrix_dtype other than float64 needs saltedtype 'density' with PsiBlocks "
                         "(SALTED_PSI_BLOCKS unset or 1)")

    if rank == 0:
        print(
            f"preparing matrices with gpr.matrix_storage={matrix_storage!r}, "
            f"gpr.matrix_dtype={matrix_dtype.name!r}...",
            flush=True,
        )

    # Density Psi is now always generated by PsiBuilder here.  The storage flag
    # controls only whether the generated matrices remain in RAM or are cached
    # as mmap-able files.
    if saltedtype == "density":
        frames = read_frames(inp.system.filename)
        if use_blocks and fast_minimizer:
            precond_diag = np.zeros(psi_builder.totsize)

        report_every = max(
            1,
            int(os.environ.get("SALTED_MATRIX_REPORT_EVERY", "10")),
        )

        if rank == 0:
            print(
                f"target species from inp.system.species: {target_species}",
                flush=True,
            )

        for local_i, iconf in enumerate(trainrange):
            symbols = frames[iconf].get_chemical_symbols()

            if use_blocks:
                kblocks = psi_builder.build_blocks(iconf, frames[iconf])
                psi = psi_builder.psi_blocks(iconf, *kblocks, dtype=matrix_dtype)
            else:
                psi = psi_builder.build(iconf, frames[iconf])
            full_coefs = np.load(
                osp.join(
                    saltedpath, "coefficients", f"coefficients_conf{iconf}.npy",
                ),
                allow_pickle=False,
            )

            # Select the union of all auxiliary-function blocks whose atom
            # species is listed in inp.system.species.  The order follows the
            # original structure, so non-contiguous atom blocks are allowed.
            aux_idx, expected_full_size = _aux_indices_for_targets(
                symbols, target_species, lmax, nmax,
            )

            # Validate the assumed atom/l/n/m coefficient ordering against the
            # actual full QM coefficient vector before slicing anything.
            if expected_full_size != full_coefs.shape[0]:
                raise ValueError(
                    f"Configuration {iconf}: auxiliary-basis bookkeeping gives "
                    f"{expected_full_size} coefficients, but the reference file "
                    f"contains {full_coefs.shape[0]}. Cannot safely select the "
                    "target-species overlap block."
                )

            if psi.shape[0] != aux_idx.size:
                present_targets = [
                    spe for spe in dict.fromkeys(symbols)
                    if spe in target_species
                ]
                raise ValueError(
                    f"Configuration {iconf}: Psi has {psi.shape[0]} rows, but "
                    f"inp.system.species={target_species} selects {aux_idx.size} "
                    f"auxiliary coefficients (present targets: {present_targets}). "
                    "Psi/reference ordering is therefore inconsistent."
                )

            # If every auxiliary function is targeted, avoid an unnecessary
            # full-matrix advanced-indexing copy.  Otherwise reduce both the
            # coefficients and overlap to the same target index set.
            all_aux_selected = (
                aux_idx.size == full_coefs.shape[0]
                and np.array_equal(
                    aux_idx, np.arange(full_coefs.shape[0], dtype=np.int64)
                )
            )

            if all_aux_selected:
                overlap_idx = None
                ref_coefs = full_coefs
            else:
                overlap_idx = aux_idx
                ref_coefs = full_coefs[aux_idx]

            ovlp = matrices.add_overlap(
                osp.join(
                    saltedpath, "overlaps", f"overlap_conf{iconf}.npy",
                ),
                indices=overlap_idx,
            )
            if use_blocks:
                if fast_minimizer and matrices.packed:
                    # packed_overlap already gives up the last bits; so may this.
                    _precond_add_blocks(precond_diag, psi, ovlp)
                elif fast_minimizer:
                    _precond_add(
                        precond_diag,
                        psi_builder.coo_from_blocks(iconf, *kblocks),
                        ovlp,
                    )
                del kblocks
            del ovlp
            matrices.add_psi(psi)
            coef_list.append(ref_coefs)

            if average:
                # Build averages only for atoms belonging to configured targets,
                # in the same atom order used by aux_idx/Psi.
                target_symbols = [spe for spe in symbols if spe in target_species]
                average_coeffs = _full_average_coefficients(
                    target_symbols, lmax, nmax, av_coefs,
                )

                if average_coeffs.shape[0] != ref_coefs.shape[0]:
                    raise ValueError(
                        f"Configuration {iconf}: average-density vector has "
                        f"{average_coeffs.shape[0]} entries, but the selected "
                        f"reference has {ref_coefs.shape[0]}."
                    )
                average_list.append(average_coeffs)

            # Drop the full reference immediately after taking a target slice.
            if ref_coefs is not full_coefs:
                del full_coefs

            # In mmap mode the cached copy is now authoritative; do not retain
            # the just-built sparse matrix.
            if matrix_storage == "mmap":
                del psi

            if (
                inp.salted.verbose
                and (
                    (local_i + 1) % report_every == 0
                    or local_i + 1 == ntrain
                )
            ):
                if matrix_storage == "mmap":
                    print(
                        f"[rank {rank}] cached matrices {local_i + 1}/{ntrain}: "
                        f"RAM {matrices.ram_bytes / 1024**3:.3f} GiB of "
                        f"{matrices.budget / 1024**3:.3f}, "
                        f"pack {matrices.disk_bytes / 1024**3:.3f} GiB",
                        flush=True,
                    )
                else:
                    print(
                        f"[rank {rank}] loaded matrices "
                        f"{local_i + 1}/{ntrain}",
                        flush=True,
                    )

        # PsiBuilder and the complete ASE frame list are no longer used during
        # minimisation in either storage mode.
        del psi_builder, frames

    elif saltedtype == "density-response":
        # There is no PsiBuilder path for density-response in the supplied
        # implementation.  Preserve its existing source (.npz files), but
        # optionally convert one matrix at a time into mmap-able local storage.
        for local_i, iconf in enumerate(trainrange):
            matrices.add_overlap(
                osp.join(
                    saltedpath, "overlaps", f"overlap_conf{iconf}.npy",
                )
            )

            for icart in ["x", "y", "z"]:
                psi = sparse.load_npz(
                    osp.join(
                        saltedpath, fdir, f"M{Menv}_zeta{zeta}", f"psi-nm_conf{iconf}_{icart}.npz",
                    )
                )
                matrices.add_psi(psi)
                if matrix_storage == "mmap":
                    del psi

    else:
        raise ValueError(f"Unsupported saltedtype {saltedtype!r}")

    matrices.finalize()
    if matrices.npsi == 0:
        raise RuntimeError("No Psi matrices were prepared")

    totsize = matrices.psi_shape(0)[1]
    kargs = _blocks_setup(matrices, totsize)
    # Psi w + resid0 is the residual (prediction minus reference), per structure
    resid0 = None
    if kargs is not None:
        resid0 = np.concatenate([(average_list[k] if average else 0.0) - c for k, c in enumerate(coef_list)])
        resid0 = np.ascontiguousarray(resid0, dtype=np.float64)
    norm = 1.0 / float(ntraintot)

    # These objects are only needed to construct Psi.  atomic_symbols/natoms
    # remain live because the average-density branch uses them later.
    del atomic_coords, atom_per_spe, natoms_per_spe
    gc.collect()
    _trim_heap()

    if parallel:
        comm.Barrier()
    
    # Prep inputs are dead from here on; free their page cache for the spilled packs
    if matrix_storage == "mmap" and (not parallel or node_comm.Get_rank() == 0):
        _drop_from_page_cache(saltedpath, keep=osp.dirname(matrices.cache_dir))

    if rank == 0:
        print(
            f"matrix preparation took {time.time() - mem_time:.1f} s", flush=True,
        )
        print(f"problem dimensionality: {totsize}", flush=True)
        if matrix_storage == "mmap":
            print(
                f"rank 0 matrices: RAM {matrices.ram_bytes / 1024**3:.3f} GiB "
                f"(budget {matrices.budget / 1024**3:.3f}), pack "
                f"{matrices.disk_bytes / 1024**3:.3f} GiB in {matrices.cache_dir}",
                flush=True,
            )

    start = time.time()
    step_time = start
    # -------------------------------------------------------------------------
    # Preconditioner
    # -------------------------------------------------------------------------

    if fast_minimizer:
        _tp = time.time()
        diag_hessian = precond_func(matrices)

        if parallel:
            diag_hessian = np.ascontiguousarray(
                diag_hessian,
                dtype=np.float64,
            )
            comm.Allreduce(MPI.IN_PLACE, diag_hessian, op=MPI.SUM)

        diag_hessian = diag_hessian * norm + 2.0 * regul

        bad = ~(diag_hessian > 0.0)
        if bad.any() and rank == 0:
            print(
                f"WARNING: {int(bad.sum())} of {totsize} preconditioner "
                f"diagonal entries were non-positive; using 1.0 for those.",
                flush=True,
            )

        P = np.where(
            bad,
            1.0,
            1.0 / np.where(bad, 1.0, diag_hessian),
        )

        if rank == 0:
            spread = (
                diag_hessian[~bad].max()
                / diag_hessian[~bad].min()
            )
            print(
                f"Jacobi preconditioner active (gpr.fast_minimizer): "
                f"built in {time.time() - _tp:.1f} s, diag range "
                f"[{diag_hessian[~bad].min():.3e}, "
                f"{diag_hessian[~bad].max():.3e}], "
                f"spread {spread:.1f}x",
                flush=True,
            )
        _trim_heap()
    else:
        P = np.ones(totsize)

    reg_log10_intstr = str(int(np.log10(regul)))

    # -------------------------------------------------------------------------
    # Restart / initialization
    # -------------------------------------------------------------------------

    init = True

    if inp.gpr.restart:
        wpath = osp.join(
            saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"weights_N{ntraintot}_reg{reg_log10_intstr}.npy",
        )
        dpath = osp.join(
            saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"dvector_N{ntraintot}_reg{reg_log10_intstr}.npy",
        )
        rpath = osp.join(
            saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"rvector_N{ntraintot}_reg{reg_log10_intstr}.npy",
        )

        if osp.exists(wpath) and osp.exists(dpath) and osp.exists(rpath):
            init = False
            w = np.load(wpath)
            d = np.load(dpath)
            r = np.load(rpath)
            s = np.multiply(P, r)
            delnew = np.dot(r, s)
            loss = loss_func(w, matrices, coef_list)
        else:
            print(
                "Warning: One or more required files to restart do not exist. "
                "Reverting to default initialization."
            )

    if init:
        w = np.ones(totsize) * 1e-04
        loss = loss_func(w, matrices, coef_list)
        r = -grad_func(w, matrices, coef_list)
        d = np.multiply(P, r)
        delnew = np.dot(r, d)

    # -------------------------------------------------------------------------
    # Conjugate-gradient minimisation
    # -------------------------------------------------------------------------

    if rank == 0:
        print("minimizing...")

    for i in range(100000):
        Ad = curv_func(d, matrices)
        curv = np.dot(d, Ad)
        alpha = delnew / curv
        w = w + alpha * d

        if (i + 1) % 50 == 0:
            loss_old = loss.copy()
            loss = loss_func(w, matrices, coef_list)

            if loss > loss_old:
                if rank == 0:
                    print(
                        "WARNING: loss function increased, search direction "
                        "reset as the steepest descent."
                    )

                r = -grad_func(w, matrices, coef_list)

                if rank == 0:
                    print(
                        f"step {i + 1}, gradient norm: "
                        f"{np.linalg.norm(r):.3e}, loss: {loss:.3e}",
                        flush=True,
                    )

                if np.linalg.norm(r) < gradtol:
                    break

                d = np.multiply(P, r)
                delnew = np.dot(r, d)

            else:
                r -= alpha * Ad

                if rank == 0:
                    print(
                        f"step {i + 1}, gradient norm: "
                        f"{np.linalg.norm(r):.3e}, loss: {loss:.3e}"
                        f"Time per step: {((time.time() - step_time) / 50):.1f} s",
                        flush=True,
                    )
                step_time = time.time()
                
                if np.linalg.norm(r) < gradtol:
                    break

                s = np.multiply(P, r)
                delold = delnew.copy()
                delnew = np.dot(r, s)
                beta = delnew / delold
                d = s + beta * d

        else:
            r -= alpha * Ad

            if np.linalg.norm(r) < gradtol:
                if rank == 0:
                    print(
                        f"step {i + 1}, gradient norm: "
                        f"{np.linalg.norm(r):.3e}",
                        flush=True,
                    )
                break

            s = np.multiply(P, r)
            delold = delnew.copy()
            delnew = np.dot(r, s)
            beta = delnew / delold
            d = s + beta * d

        # w, d and r of the same step, so a restart continues the run exactly
        if (i + 1) % 50 == 0 and rank == 0:
            np.save(
                osp.join(
                    saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"weights_N{ntraintot}_reg{reg_log10_intstr}.npy",
                ),
                w,
            )
            np.save(
                osp.join(
                    saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"dvector_N{ntraintot}_reg{reg_log10_intstr}.npy",
                ),
                d,
            )
            np.save(
                osp.join(
                    saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"rvector_N{ntraintot}_reg{reg_log10_intstr}.npy",
                ),
                r,
            )

    # -------------------------------------------------------------------------
    # Final save
    # -------------------------------------------------------------------------

    if rank == 0:
        np.save(
            osp.join(
                saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"weights_N{ntraintot}_reg{reg_log10_intstr}.npy",
            ),
            w,
        )
        np.save(
            osp.join(
                saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"dvector_N{ntraintot}_reg{reg_log10_intstr}.npy",
            ),
            d,
        )
        np.save(
            osp.join(
                saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"rvector_N{ntraintot}_reg{reg_log10_intstr}.npy",
            ),
            r,
        )

        print("minimization completed succesfully!")
        print(
            f"minimization time: {((time.time() - start) / 60):.2f} minutes"
        )


if __name__ == "__main__":
    run_or_abort(build)
