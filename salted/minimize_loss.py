import gc
import hashlib
import os
import os.path as osp
import shutil
import time

import numpy as np
from numba import njit, prange
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
)

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


@njit(fastmath=True)
def _spmv_upper_serial(ap, chunks, x):
    """_spmv_upper on one thread, same chunks and summation order."""
    n = x.shape[0]
    nc = chunks.shape[0] - 1
    z = np.zeros((nc, n))
    for c in range(nc):
        _spmv_rows(ap, x, z[c], chunks[c], chunks[c + 1])
    y = np.empty(n)
    _sum_rows(z, y, 0, n)
    return y


@njit
def _blocks_dot_serial(vals, tab, x, nrows):
    """psi_builder._blocks_dot on one thread (no fastmath: the same bits)."""
    y = np.zeros(nrows)
    for b in range(tab.shape[0]):
        voff, nr, nc, r0, c0 = tab[b, 0], tab[b, 1], tab[b, 2], tab[b, 3], tab[b, 4]
        for m in range(nr):
            s = 0.0
            v0 = voff + m * nc
            for c in range(nc):
                v = vals[v0 + c]
                if v != 0.0:
                    s += v * x[c0 + c]
            y[r0 + m] = s
    return y


@njit(parallel=True)
def _curv_blocks(vals, tabs, aps, chunks, roff, gptr, jobs, d, totsize):
    """sum_k 2 Psi_k^T S_k Psi_k d over all structures k, threaded over
    structures and then over column ranges instead of inside each product.

    Every element is summed exactly as the per-structure loop
    Ad += 2.0 * psi.rdot(S.dot(psi.dot(d))) sums it: structures in order, each
    one's rdot from 0.0 in block order.  Same bits, any thread count."""
    npsi = len(vals)
    w = np.empty(roff[-1])
    for kk in prange(npsi):
        k = np.int64(kk)
        z = _blocks_dot_serial(vals[k], tabs[k], d, roff[k + 1] - roff[k])
        w[roff[k]:roff[k + 1]] = _spmv_upper_serial(aps[k], chunks[k], z)
    ad = np.zeros(totsize)
    for j in prange(jobs.shape[0]):
        g, ca, cb = jobs[j, 0], jobs[j, 1], jobs[j, 2]
        tmp = np.empty(cb - ca)
        for k in range(npsi):
            b0, b1 = gptr[k, g], gptr[k, g + 1]
            if b0 == b1:
                continue  # the loop adds 2.0 * 0.0 here: no change
            tmp[:] = 0.0
            v, tab, r = vals[k], tabs[k], roff[k]
            for b in range(b0, b1):
                voff, nr, nc, r0 = tab[b, 0], tab[b, 1], tab[b, 2], tab[b, 3]
                for m in range(nr):
                    yr = w[r + r0 + m]
                    v0 = voff + m * nc
                    for c in range(ca, cb):
                        x = v[v0 + c]
                        if x != 0.0:
                            tmp[c - ca] += x * yr
            c0 = tab[b0, 4]
            for c in range(ca, cb):
                ad[c0 + c] += 2.0 * tmp[c - ca]
    return ad


CURV_COLS = 64  # columns per job in _curv_blocks


def _curv_setup(matrices, totsize):
    """_curv_blocks arguments when every Psi is PsiBlocks and every S is
    PackedSym, else None (the per-structure loop then runs)."""
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

    vals, tabs, aps, chunks = (List() for _ in range(4))
    for p, o in zip(psis, ovls):
        vals.append(ro(p.vals))
        tabs.append(ro(p.tab))
        aps.append(ro(o.ap))
        chunks.append(ro(o.chunks))
    roff = np.concatenate(([0], np.cumsum([p.shape[0] for p in psis]))).astype(np.int64)
    # Column groups (one per species, lam, n) are global; gptr[k, g]:gptr[k, g+1]
    # are structure k's blocks of group g (empty if the species is absent).
    starts = np.unique(np.concatenate([p.tab[:, 4] for p in psis]))
    bounds = np.append(starts, totsize)
    gptr = np.array([np.searchsorted(p.tab[:, 4], bounds) for p in psis], dtype=np.int64)
    ncol = {}
    for p in psis:
        ncol.update(zip(p.tab[:, 4].tolist(), p.tab[:, 2].tolist()))
    jobs = np.array([(g, a, min(a + CURV_COLS, ncol[s]))
                     for g, s in enumerate(starts.tolist()) for a in range(0, ncol[s], CURV_COLS)],
                    dtype=np.int64)
    return vals, tabs, aps, chunks, roff, gptr, jobs


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
    which moves the CG path, not the gradtol it converges to."""
    tab = psi.tab
    for g0, g1 in zip(psi.grp[:-1], psi.grp[1:]):
        t = tab[g0:g1]
        nr, nc, c0 = int(t[0, 1]), int(t[0, 2]), int(t[0, 4])
        rows = (t[:, 3, None] + np.arange(nr)).ravel()
        v = np.concatenate([psi.vals[o:o + nr * nc] for o in t[:, 0]]).reshape(-1, nc)
        diag_hessian[c0:c0 + nc] += 2.0 * np.einsum("ij,ij->j", ovlp[np.ix_(rows, rows)] @ v, v)


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
    """

    ALIGN = 4096  # pack offsets; every array starts page-aligned

    def __init__(self, mode, saltedpath, rank, nlocal=1, packed=False):
        mode = str(mode).strip().lower()
        if mode not in ("memory", "mmap"):
            raise ValueError(
                "gpr.matrix_storage must be 'memory' or 'mmap', "
                f"got {mode!r}"
            )

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
            p.ap = self._keep(p.ap)
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

    # Distribute structures to tasks
    ntraintot = int(inp.gpr.trainfrac * Ntrain)

    if parallel:
        check_MPI_tasks_count(comm, ntraintot, "training structures")
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

        if saltedtype == "density":
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

        if saltedtype == "density":
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

        if curv_args is not None:
            Ad = _curv_blocks(*curv_args, np.ascontiguousarray(cg_dire, dtype=np.float64), totsize)
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
    nlocal = comm.Split_type(MPI.COMM_TYPE_SHARED).Get_size() if parallel else 1
    matrices = MatrixStore(matrix_storage, saltedpath, rank, nlocal,
                           packed=bool(inp.gpr.packed_overlap))
    coef_list = []
    precond_diag = None
    # SALTED_PSI_BLOCKS=0 restores the scipy COO Psi; both give identical bits.
    use_blocks = saltedtype == "density" and os.environ.get("SALTED_PSI_BLOCKS", "1") != "0"

    if rank == 0:
        print(
            f"preparing matrices with gpr.matrix_storage={matrix_storage!r}...",
            flush=True,
        )

    # Density Psi is now always generated by PsiBuilder here.  The storage flag
    # controls only whether the generated matrices remain in RAM or are cached
    # as mmap-able files.
    if saltedtype == "density":
        psi_builder = PsiBuilder(
            rank,
            system=(
                species, lmax, nmax, lmax_max, nnmax, ndata,
                atomic_symbols, atomic_coords, natoms, natmax,
            ),
            atom_info=(atom_per_spe, natoms_per_spe),
        )
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
                psi = psi_builder.psi_blocks(iconf, *kblocks)
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
    curv_args = _curv_setup(matrices, totsize)
    norm = 1.0 / float(ntraintot)

    # These objects are only needed to construct Psi.  atomic_symbols/natoms
    # remain live because the average-density branch uses them later.
    del atomic_coords, atom_per_spe, natoms_per_spe
    gc.collect()

    if parallel:
        comm.Barrier()

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
                        f"{np.linalg.norm(r):.3e}, loss: {loss:.3e}",
                        flush=True,
                    )

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
