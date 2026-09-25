import gc
import os
import os.path as osp
import shutil
import time

import numpy as np
from ase.io import read
from scipy import sparse

from salted import get_averages
from salted.psi_builder import PsiBuilder
from salted.selection_utils import load_training_indices
from salted.sys_utils import (
    ParseConfig,
    check_MPI_tasks_count,
    detect_mpi,
    distribute_jobs,
    format_index_ranges,
    get_atom_idx,
    read_system,
)

def _sparse_nbytes(mat):
    """Bytes occupied by the numerical/index arrays of a scipy sparse matrix."""
    total = 0
    seen = set()
    for name in ("data", "indices", "indptr", "row", "col"):
        arr = getattr(mat, name, None)
        if arr is not None and hasattr(arr, "nbytes") and id(arr) not in seen:
            total += arr.nbytes
            seen.add(id(arr))
    return total


def _save_npy(path, arr):
    """Write one array as an uncompressed .npy suitable for mmap."""
    np.save(path, np.asarray(arr), allow_pickle=False)


def _save_sparse_for_mmap(mat, prefix):
    """
    Persist a scipy sparse matrix without changing its sparse format/order.

    Returns a small in-memory descriptor.  Only COO/CSR/CSC are supported
    intentionally: silently converting another sparse format could change
    summation order and therefore numerical reproducibility.
    """
    if mat.getformat() != "coo":
        raise TypeError(
            f"mmap matrix storage currently supports only COO/CSR/CSC Psi matrices, "
            f"got format {mat.getformat()!r}. Refusing an implicit conversion because it "
            f"could change sparse summation order."
        )
    shape = tuple(int(x) for x in mat.shape)

    paths = {
        "data": f"{prefix}_data.npy",
        "row": f"{prefix}_row.npy",
        "col": f"{prefix}_col.npy",
    }
    _save_npy(paths["data"], mat.data)
    _save_npy(paths["row"], mat.row)
    _save_npy(paths["col"], mat.col)
    return {
        "shape": shape,
        "paths": paths,
        "bytes": _sparse_nbytes(mat),
    }


def _load_sparse_mmap(desc):
    """
    Reconstruct a scipy sparse matrix backed by read-only np.memmap arrays.

    No sparse-format conversion is performed, so data/index ordering is the
    same as when the matrix was cached.
    """
    p = desc["paths"]
    data = np.load(p["data"], mmap_mode="r", allow_pickle=False)
    row = np.load(p["row"], mmap_mode="r", allow_pickle=False)
    col = np.load(p["col"], mmap_mode="r", allow_pickle=False)
    return sparse.coo_matrix((data, (row, col)), shape=desc["shape"], copy=False)

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
    Uniform access to overlap and Psi matrices for both storage strategies.

    In "memory" mode get_overlap()/get_psi() return retained objects.

    In "mmap" mode they create lightweight read-only mappings on demand.
    The caller should keep only the returned local reference for the duration
    of the current structure calculation.
    """

    def __init__(self, mode, saltedpath, rank):
        mode = str(mode).strip().lower()
        if mode not in ("memory", "mmap"):
            raise ValueError(
                "gpr.matrix_storage must be 'memory' or 'mmap', "
                f"got {mode!r}"
            )

        self.mode = mode
        self.rank = rank

        self._overlap_paths = []
        self._overlaps = []

        self._psis = []
        self._psi_desc = []

        self.psi_cache_bytes = 0
        self.overlap_cache_bytes = 0

        if self.mode == "mmap":
            cache_root = os.environ.get(
                "SALTED_PSI_CACHE_DIR",
                osp.join(saltedpath, ".psi_mmap_cache"),
            )
            self.cache_dir = osp.join(cache_root, f"rank_{rank:05d}")

            # Node-local scratch is expected to be fresh for every job.  Remove
            # only this rank's directory to avoid ever touching another rank.
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            os.makedirs(self.cache_dir, exist_ok=True)
        else:
            self.cache_dir = None

    def add_overlap(self, path, indices=None, label=None):
        """
        Add an overlap matrix, optionally restricted to a principal submatrix.

        ``indices`` is the ordered list of auxiliary functions represented by
        the corresponding Psi rows.  Advanced indexing with ``np.ix_`` keeps
        every selected-selected coupling, including couplings between distinct
        atoms of the target species.
        """
        if indices is not None:
            indices = np.asarray(indices, dtype=np.int64)

        if self.mode == "memory":
            ovlp = np.load(path, allow_pickle=False)
            if indices is not None:
                ovlp = ovlp[np.ix_(indices, indices)]
            self._overlaps.append(ovlp)
            return

        if indices is None:
            # Full-species case: map the original overlap directly on demand.
            self._overlap_paths.append(path)
            return

        if label is None:
            raise ValueError("label is required when caching a reduced overlap")

        # A non-contiguous principal-submatrix selection creates a dense copy.
        # Do it once here, then mmap only the reduced matrix during minimisation
        # instead of repeatedly selecting it from the full NFS-backed matrix.
        full_ovlp = np.load(path, mmap_mode="r", allow_pickle=False)
        reduced_ovlp = full_ovlp[np.ix_(indices, indices)]
        cache_path = osp.join(self.cache_dir, f"overlap_{label}.npy")
        _save_npy(cache_path, reduced_ovlp)
        self._overlap_paths.append(cache_path)
        self.overlap_cache_bytes += reduced_ovlp.nbytes
        del reduced_ovlp, full_ovlp

    def add_psi(self, mat, label):
        if self.mode == "memory":
            self._psis.append(mat)
            return

        prefix = osp.join(self.cache_dir, f"psi_{label}")
        desc = _save_sparse_for_mmap(mat, prefix)
        self._psi_desc.append(desc)
        self.psi_cache_bytes += desc["bytes"]

    def get_overlap(self, index):
        if self.mode == "memory":
            return self._overlaps[index]

        return np.load(
            self._overlap_paths[index],
            mmap_mode="r",
            allow_pickle=False,
        )

    def get_psi(self, index):
        if self.mode == "memory":
            return self._psis[index]

        return _load_sparse_mmap(self._psi_desc[index])

    def psi_shape(self, index=0):
        if self.mode == "memory":
            return self._psis[index].shape
        return self._psi_desc[index]["shape"]

    @property
    def npsi(self):
        if self.mode == "memory":
            return len(self._psis)
        return len(self._psi_desc)


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

    def loss_func(weights, matrices, coef_list):
        """Compute the electron-density loss function."""

        loss = 0.0

        if saltedtype == "density":
            for iconf in range(ntrain):
                ref_coefs = coef_list[iconf]

                psi = matrices.get_psi(iconf)
                ovlp = matrices.get_overlap(iconf)

                # Same sparse operation/order as the previous implementation.
                pred_coefs = sparse.csr_matrix.dot(psi, weights)
                if average:
                    pred_coefs += average_list[iconf]

                ref_projs = np.dot(ovlp, ref_coefs)
                pred_projs = np.dot(ovlp, pred_coefs)

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
                    pred_coefs = sparse.csr_matrix.dot(psi, weights)

                    ref_projs = np.dot(ovlp, ref_coefs)
                    pred_projs = np.dot(ovlp, pred_coefs)

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

    def grad_func(weights, matrices, coef_list):
        """Compute the gradient of the electron-density loss function."""

        gradient = np.zeros(totsize)

        if saltedtype == "density":
            for iconf in range(ntrain):

                ref_coefs = coef_list[iconf]

                psi = matrices.get_psi(iconf)
                ovlp = matrices.get_overlap(iconf)

                pred_coefs = sparse.csr_matrix.dot(psi, weights)
                if average:
                    pred_coefs += average_list[iconf]

                ref_projs = np.dot(ovlp, ref_coefs)
                pred_projs = np.dot(ovlp, pred_coefs)

                gradient += 2.0 * sparse.csc_matrix.dot(
                    psi.T,
                    pred_projs - ref_projs,
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
                    pred_coefs = sparse.csr_matrix.dot(psi, weights)

                    ref_projs = np.dot(ovlp, ref_coefs)
                    pred_projs = np.dot(ovlp, pred_coefs)

                    gradient += 2.0 * sparse.csc_matrix.dot(
                        psi.T,
                        pred_projs - ref_projs,
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

    PRECOND_BLK = 2048  # rows of psi^T per chunk; caps the dense temporary

    def precond_func(matrices):
        """Diagonal (Jacobi) preconditioner: diag(2 * sum psi^T S psi)."""

        diag_hessian = np.zeros(totsize)

        for iconf in range(ntrain):
            psi = matrices.get_psi(iconf)
            ovlp = matrices.get_overlap(iconf)

            # This is still the largest per-structure sparse temporary, but it
            # is released before the next structure is mapped.
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

            del psiT, psi, ovlp

        return diag_hessian

    def curv_func(cg_dire, matrices):
        """Compute curvature on the given CG direction."""

        Ad = np.zeros(totsize)

        if saltedtype == "density":
            for iconf in range(ntrain):
                psi = matrices.get_psi(iconf)
                ovlp = matrices.get_overlap(iconf)

                psi_x_dire = sparse.csr_matrix.dot(psi, cg_dire)
                Ad += 2.0 * sparse.csc_matrix.dot(
                    psi.T,
                    np.dot(ovlp, psi_x_dire),
                )

                del psi_x_dire, psi, ovlp

        elif saltedtype == "density-response":
            itot = 0
            for iconf in range(ntrain):
                ovlp = matrices.get_overlap(iconf)

                for _icart in ["x", "y", "z"]:
                    psi = matrices.get_psi(itot)

                    psi_x_dire = sparse.csr_matrix.dot(psi, cg_dire)
                    Ad += 2.0 * sparse.csc_matrix.dot(
                        psi.T,
                        np.dot(ovlp, psi_x_dire),
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
    matrices = MatrixStore(matrix_storage, saltedpath, rank)
    coef_list = []

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
        frames = read(inp.system.filename, ":")

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
            label = f"{local_i:06d}_conf{iconf}"
            symbols = atomic_symbols[iconf]

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

            matrices.add_overlap(
                osp.join(
                    saltedpath, "overlaps", f"overlap_conf{iconf}.npy",
                ),
                indices=overlap_idx,
                label=label,
            )
            matrices.add_psi(psi, label=label)
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
                        f"Psi {matrices.psi_cache_bytes / 1024**3:.3f} GiB, "
                        f"reduced overlaps "
                        f"{matrices.overlap_cache_bytes / 1024**3:.3f} GiB",
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
                matrices.add_psi(
                    psi,
                    label=f"{local_i:06d}_conf{iconf}_{icart}",
                )
                if matrix_storage == "mmap":
                    del psi

    else:
        raise ValueError(f"Unsupported saltedtype {saltedtype!r}")

    if matrices.npsi == 0:
        raise RuntimeError("No Psi matrices were prepared")

    totsize = matrices.psi_shape(0)[1]
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
                f"Psi mmap cache: {matrices.cache_dir}", flush=True,
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
    build()
