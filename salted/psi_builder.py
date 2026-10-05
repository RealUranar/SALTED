import os.path as osp

import numpy as np
from numba import njit, prange
from scipy import sparse

from salted import sph_utils
from salted.sys_utils import ParseConfig, build_featomic_hyper_params, get_atom_idx, get_feats_projs, read_system


class arraylist:
    def __init__(self):
        self.data = np.zeros((100000,))
        self.capacity = 100000
        self.size = 0

    def update(self, row):
        n = row.shape[0]
        self.add(row, n)

    def add(self, x, n):
        if self.size + n >= self.capacity:
            self.capacity *= 2
            newdata = np.zeros((self.capacity,))
            newdata[: self.size] = self.data[: self.size]
            self.data = newdata

        self.data[self.size : self.size + n] = x
        self.size += n

    def finalize(self):
        return self.data[: self.size]


class PsiBuilder:
    """Per-structure RKHS feature vectors.

    Holds everything that does not depend on the structure, so the caller pays for
    it once instead of once per structure. `build` reproduces the body of the
    rkhs_vector loop exactly; both rkhs_vector and train_fused go through here so
    the two cannot drift apart.
    """

    def __init__(self, rank: int = 0, system=None, atom_info=None):
        inp = ParseConfig().parse_input()

        self.saltedname, self.saltedpath, self.saltedtype = inp.salted.saltedname, inp.salted.saltedpath, inp.salted.saltedtype
        self.filename = inp.system.filename
        self.rep1, self.rep2 = inp.descriptor.rep1, inp.descriptor.rep2
        self.nrad1, self.nrad2 = inp.descriptor.rep1.nrad, inp.descriptor.rep2.nrad
        self.nang1, self.nang2 = inp.descriptor.rep1.nang, inp.descriptor.rep2.nang
        self.zeta = inp.gpr.z
        self.neighspe1, self.neighspe2 = inp.descriptor.rep1.neighspe, inp.descriptor.rep2.neighspe
        self.ncut = inp.descriptor.sparsify.ncut
        self.sparsify = self.ncut > 0
        self.nspe1 = len(inp.descriptor.rep1.neighspe)
        self.nspe2 = len(inp.descriptor.rep2.neighspe)
        
        self.HP1 = build_featomic_hyper_params(inp.descriptor.rep1)
        self.HP2 = build_featomic_hyper_params(inp.descriptor.rep2)
        

        if self.saltedtype != "density":
            raise NotImplementedError(f"PsiBuilder supports saltedtype='density', got {self.saltedtype!r}")

        self.rank = rank
        (self.species, self.lmax, self.nmax, self.lmax_max, nnmax, self.ndata,
         self.atomic_symbols, atomic_coords, self.natoms, natmax) = system if system else read_system()
        if atom_info is not None:
            self.atom_idx, self.natom_dict = atom_info
        else:
            self.atom_idx, self.natom_dict = get_atom_idx(
                self.ndata, self.natoms, self.species, self.atomic_symbols
            )

        if self.sparsify:
            self.vfps = {
                lam: np.load(osp.join(self.saltedpath, f"equirepr_{self.saltedname}", f"fps{self.ncut}-{lam}.npy"))
                for lam in range(self.lmax_max + 1)
            }

        self.Vmat, self.Mspe, self.power_env_sparse = get_feats_projs(self.species, self.lmax)

        self.cuml_Mcut = {}
        self.totsize = 0
        for spe in self.species:
            for lam in range(self.lmax[spe] + 1):
                for n in range(self.nmax[(spe, lam)]):
                    self.cuml_Mcut[(spe, lam, n)] = self.totsize
                    self.totsize += self.Vmat[(lam, spe)].shape[1]

        # For zeta == 1 get_feats_projs has already folded Vmat into
        # power_env_sparse and build() never reads it again, so holding it costs
        # 216 MB per rank for nothing. Every rank pays that, and minimize_loss
        # runs 45 of them on 186 GB nodes.
        if self.zeta == 1:
            self.Vmat = None

        # lam-dependent only: hoisted out of the per-structure loop, where the
        # wigner loadtxt was costing one shared-filesystem open per structure per lam.
        self.per_lam = {}
        for lam in range(self.lmax_max + 1):
            llmax, llvec = sph_utils.get_angular_indexes_symmetric(lam, self.nang1, self.nang2)
            wigner3j = np.loadtxt(osp.join(
                self.saltedpath, "wigners",
                f"wigner_lam-{lam}_lmax1-{self.nang1}_lmax2-{self.nang2}.dat",
            ))
            c2r = sph_utils.complex_to_real_transformation([2 * lam + 1])[0]
            self.per_lam[lam] = (llmax, llvec, wigner3j, c2r)

        self.reps_equivalent = sph_utils.reps_equivalent(
            self.rep1, self.neighspe1, self.HP1, self.rep2, self.neighspe2, self.HP2
        )

    def build(self, iconf: int, structure) -> sparse.coo_matrix:
        return self.coo_from_blocks(iconf, *self.build_blocks(iconf, structure))

    def build_blocks(self, iconf: int, structure):
        """The per-(species, lam) kernel blocks Psi and the row count Tsize.

        Every (atom, lam) block is repeated nmax[(spe, lam)] times in the COO
        matrix, once per radial channel n, only shifted in rows and columns.
        PsiBlocks stores it once."""
        natoms = self.natoms[iconf]

        omega1 = sph_utils.get_representation_coeffs(
            structure, self.rep1.type, self.HP1, self.rank, self.neighspe1,
            self.species, self.nang1, self.nrad1, natoms)
        if self.reps_equivalent:
            omega2 = omega1
        else:
            omega2 = sph_utils.get_representation_coeffs(
                structure, self.rep2.type, self.HP2, self.rank, self.neighspe2,
                self.species, self.nang2, self.nrad2, natoms)

        v1 = np.transpose(omega1, (1, 3, 0, 2)).copy()
        v2 = np.transpose(omega2, (1, 3, 0, 2)).copy()

        power = {}
        for lam in range(self.lmax_max + 1):
            llmax, llvec, wigner3j, c2r = self.per_lam[lam]

            if self.sparsify:
                featsize = self.nspe1 * self.nspe2 * self.nrad1 * self.nrad2 * llmax
                nfps = len(self.vfps[lam])
                p = sph_utils.equicombsparse_numba(
                    natoms, self.nang1, self.nang2, self.nspe1 * self.nrad1,
                    self.nspe2 * self.nrad2, v1, v2, wigner3j, llmax, llvec, lam, c2r,
                    featsize, nfps, self.vfps[lam])
                featsize = self.ncut
            else:
                featsize = self.nspe1 * self.nspe2 * self.nrad1 * self.nrad2 * llmax
                p = sph_utils.equicomb_numba(
                    natoms, self.nang1, self.nang2, self.nspe1 * self.nrad1,
                    self.nspe2 * self.nrad2, v1, v2, wigner3j, llmax, llvec, lam, c2r,
                    featsize)

            if lam == 0:
                power[lam] = p.reshape(natoms, featsize)
            else:
                power[lam] = p.reshape(natoms, 2 * lam + 1, featsize)

        Psi = {}
        ispe = {}
        Tsize = 0
        for spe in self.species:
            ispe[spe] = 0
            nat_spe = self.natom_dict[(iconf, spe)]

            if self.zeta == 1:
                kernel0_nm = np.dot(
                    power[0][self.atom_idx[(iconf, spe)]], self.power_env_sparse[(0, spe)].T)
                Psi[(spe, 0)] = kernel0_nm
            else:
                kernel0_nm = np.dot(
                    power[0][self.atom_idx[(iconf, spe)]], self.power_env_sparse[(0, spe)].T)
                kernel_nm = kernel0_nm**self.zeta
                Psi[(spe, 0)] = np.real(np.dot(kernel_nm, self.Vmat[(0, spe)]))

            Tsize += nat_spe * self.nmax[(spe, 0)]

            for lam in range(1, self.lmax[spe] + 1):
                if self.zeta == 1:
                    Psi[(spe, lam)] = np.dot(
                        power[lam][self.atom_idx[(iconf, spe)]].reshape(
                            nat_spe * (2 * lam + 1), power[lam].shape[-1]),
                        self.power_env_sparse[(lam, spe)].T)
                else:
                    kernel_nm = np.dot(
                        power[lam][self.atom_idx[(iconf, spe)]].reshape(
                            nat_spe * (2 * lam + 1), power[lam].shape[-1]),
                        self.power_env_sparse[(lam, spe)].T)
                    kernel_nm_blocks = kernel_nm.reshape(
                        nat_spe, 2 * lam + 1, self.Mspe[spe], 2 * lam + 1)
                    kernel_nm_blocks *= kernel0_nm[:, np.newaxis, :, np.newaxis] ** (self.zeta - 1)
                    kernel_nm = kernel_nm_blocks.reshape(
                        nat_spe * (2 * lam + 1), self.Mspe[spe] * (2 * lam + 1))
                    Psi[(spe, lam)] = np.real(np.dot(kernel_nm, self.Vmat[(lam, spe)]))

                Tsize += nat_spe * self.nmax[(spe, lam)] * (2 * lam + 1)

        return Psi, Tsize

    def coo_from_blocks(self, iconf, Psi, Tsize) -> sparse.coo_matrix:
        natoms = self.natoms[iconf]
        ispe = {spe: 0 for spe in self.species}
        srows = arraylist()
        scols = arraylist()
        psi_nonzero = arraylist()

        i = 0
        for iat in range(natoms):
            spe = self.atomic_symbols[iconf][iat]
            for l in range(self.lmax[spe] + 1):
                i1 = ispe[spe] * (2 * l + 1)
                i2 = ispe[spe] * (2 * l + 1) + 2 * l + 1
                x = Psi[(spe, l)][i1:i2]
                nz = np.nonzero(x)
                vals = x[nz]
                for n in range(self.nmax[(spe, l)]):
                    psi_nonzero.update(vals)
                    srows.update(nz[0] + i)
                    scols.update(nz[1] + self.cuml_Mcut[(spe, l, n)])
                    i += 2 * l + 1
            ispe[spe] += 1

        ij = np.vstack((srows.finalize(), scols.finalize()))
        return sparse.coo_matrix(
            (psi_nonzero.finalize(), ij), shape=(Tsize, self.totsize))

    def psi_blocks(self, iconf, Psi, Tsize):
        """The same matrix as coo_from_blocks, each (atom, lam) block stored once."""
        natoms = self.natoms[iconf]
        base, off = {}, 0
        for key, arr in Psi.items():
            base[key] = off
            off += arr.size
        vals = np.concatenate([np.ascontiguousarray(a, dtype=np.float64).ravel() for a in Psi.values()])
        ispe = {spe: 0 for spe in self.species}
        tab = []
        i = 0
        for iat in range(natoms):
            spe = self.atomic_symbols[iconf][iat]
            for l in range(self.lmax[spe] + 1):
                nc = Psi[(spe, l)].shape[1]
                voff = base[(spe, l)] + ispe[spe] * (2 * l + 1) * nc
                for n in range(self.nmax[(spe, l)]):
                    tab.append((voff, 2 * l + 1, nc, i, self.cuml_Mcut[(spe, l, n)]))
                    i += 2 * l + 1
            ispe[spe] += 1
        tab = np.array(tab, dtype=np.int64).reshape(-1, 5)
        # Group by column range (one per (spe, lam, n)); the stable sort keeps the
        # atom order inside a group, which is the COO order rdot must replay.
        tab = tab[np.argsort(tab[:, 4], kind="stable")]
        return PsiBlocks(vals, tab, (Tsize, self.totsize))


class PsiBlocks:
    """Psi as (atom, lam) kernel blocks, each stored once instead of nmax times.

    dot/rdot replay scipy's coo_matvec on the coo_from_blocks matrix exactly:
    zeros skipped as np.nonzero skips them, every output element summed from
    0.0 in COO entry order, and no FMA (numba contracts only under fastmath).
    The results are therefore bit-identical, not merely close.

    Both run threaded (NUMBA_NUM_THREADS) without changing any summation:
    every row of Psi @ x comes from one block, every column of Psi.T @ y from
    one column group, and a group is walked serially in atom order."""

    def __init__(self, vals, tab, shape):
        self.vals, self.tab, self.shape = vals, tab, shape
        # tab is sorted by first column; grp[g]:grp[g+1] are the blocks of group g.
        self.grp = np.concatenate(([0], np.flatnonzero(np.diff(tab[:, 4])) + 1, [len(tab)]))

    @property
    def nbytes(self):
        return self.vals.nbytes + self.tab.nbytes

    def dot(self, x):
        return _blocks_dot(self.vals, self.tab, np.ascontiguousarray(x, dtype=np.float64), self.shape[0])

    def rdot(self, y):
        """Psi.T @ y."""
        return _blocks_rdot(self.vals, self.tab, self.grp, np.ascontiguousarray(y, dtype=np.float64), self.shape[1])


@njit(parallel=True, cache=False)
def _blocks_dot(vals, tab, x, nrows):
    y = np.zeros(nrows)
    for b in prange(tab.shape[0]):
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


@njit(parallel=True, cache=False)
def _blocks_rdot(vals, tab, grp, y, ncols):
    t = np.zeros(ncols)
    for g in prange(grp.shape[0] - 1):
        for b in range(grp[g], grp[g + 1]):
            voff, nr, nc, r0, c0 = tab[b, 0], tab[b, 1], tab[b, 2], tab[b, 3], tab[b, 4]
            for m in range(nr):
                yr = y[r0 + m]
                v0 = voff + m * nc
                for c in range(nc):
                    v = vals[v0 + c]
                    if v != 0.0:
                        t[c0 + c] += v * yr
    return t
