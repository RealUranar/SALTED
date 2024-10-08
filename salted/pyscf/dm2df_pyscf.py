import argparse
import os
import sys
import os.path as osp
from typing import List, Tuple, Union, Dict
import time
import re, glob

import numpy as np
from pyscf import gto, df, lib
from pyscf.gto import basis
from ase.io import read
from scipy import special

from salted.basis_client import BasisClient, SpeciesBasisData
from salted.sys_utils import ParseConfig, parse_index_str, ARGHELP_INDEX_STR, Irreps, PLACEHOLDER
from salted.pyscf.get_basis_info import get_aux_basis_name, read_basis

__doc__ = """
PySCF orders all angular momentum components for L>1 as -L,...,0,...,+L,
and as +1,-1,0 for L=1, corresponding to X,Y,Z in Cartesian-SPH.
For details please check https://pyscf.org/user/gto.html#ordering-of-basis-functions
Make sure to provide the density matrix following this convention!
"""


def cal_df_coeffs_old(
    atoms: List,
    qmbasis: str,
    ribasis: str,
    dm: np.ndarray,
    lmax: Dict[str, int],
    nmax: Dict[Tuple[str, int], int],
):
    pyscf_time = time.time()
    mol = gto.M(atom=atoms, basis=qmbasis, unit='angstrom')
    auxmol = gto.M(atom=atoms, basis=ribasis, unit='angstrom')
    pmol = mol + auxmol
    assert dm.shape[0] == mol.nao_nr(), f"{dm.shape=}, {mol.nao_nr()=}"

    overlap = auxmol.intor("int1e_ovlp_sph")  # AO overlap matrix
    eri2c = auxmol.intor('int2c2e_sph')  # 2-centers 2-electrons integral
    eri3c = pmol.intor(  # 3-centers 2-electrons integral
        'int3c2e_sph',
        shls_slice=(0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + auxmol.nbas)
    )
    eri3c = eri3c.reshape(mol.nao_nr(), mol.nao_nr(), -1)
    rho = np.einsum('ijp,ij->p', eri3c, dm)
    rho = np.linalg.solve(eri2c, rho)
    pyscf_time = time.time() - pyscf_time

    # Reorder L=1 components following the -1,0,+1 convention
    reorder_time = time.time()
    Coef = np.zeros(len(rho))
    Over = np.zeros((len(rho), len(rho)))

    symb = [at[0] for at in atoms]
    import salted.cython.dm2df_fast_reorder as dm2df
    Coef, Over = dm2df.reorder(rho, overlap, symb, lmax, nmax)
    # i1 = 0
    # for iat in range(len(atoms)):
    #     spe1 = atoms[iat][0]
    #     for l1 in range(lmax[spe1]+1):
    #         for n1 in range(nmax[(spe1,l1)]):
    #             for im1 in range(2*l1+1):
    #                 if l1==1 and im1!=2:
    #                     Coef[i1] = rho[i1+1]
    #                 elif l1==1 and im1==2:
    #                     Coef[i1] = rho[i1-2]
    #                 else:
    #                     Coef[i1] = rho[i1]
    #                 i2 = 0
    #                 for jat in range(len(atoms)):
    #                     spe2 = atoms[jat][0]
    #                     for l2 in range(lmax[spe2]+1):
    #                         for n2 in range(nmax[(spe2,l2)]):
    #                             for im2 in range(2*l2+1):
    #                                 if l1==1 and im1!=2 and l2!=1:
    #                                     Over[i1,i2] = overlap[i1+1,i2]
    #                                 elif l1==1 and im1==2 and l2!=1:
    #                                     Over[i1,i2] = overlap[i1-2,i2]
    #                                 elif l2==1 and im2!=2 and l1!=1:
    #                                     Over[i1,i2] = overlap[i1,i2+1]
    #                                 elif l2==1 and im2==2 and l1!=1:
    #                                     Over[i1,i2] = overlap[i1,i2-2]
    #                                 elif l1==1 and im1!=2 and l2==1 and im2!=2:
    #                                     Over[i1,i2] = overlap[i1+1,i2+1]
    #                                 elif l1==1 and im1!=2 and l2==1 and im2==2:
    #                                     Over[i1,i2] = overlap[i1+1,i2-2]
    #                                 elif l1==1 and im1==2 and l2==1 and im2!=2:
    #                                     Over[i1,i2] = overlap[i1-2,i2+1]
    #                                 elif l1==1 and im1==2 and l2==1 and im2==2:
    #                                     Over[i1,i2] = overlap[i1-2,i2-2]
    #                                 else:
    #                                     Over[i1,i2] = overlap[i1,i2]
    #                                 i2 += 1
    #                 i1 += 1

    # Compute density projections on auxiliary functions
    Proj = np.dot(Over,Coef)
    reorder_time = time.time() - reorder_time

    return {
        "coef": Coef,
        "proj": Proj,
        "over": Over,
        "pyscf_time": pyscf_time,
        "reorder_time": reorder_time,
    }

def cal_df_coeffs(
    atoms: List,
    qmbasis: str|dict,
    ribasis: str|dict,
    dm: np.ndarray,
    irreps: Irreps,
):
    pyscf_time = time.time()
    mol = gto.M(atom=atoms, basis=qmbasis, unit='angstrom')
    auxmol = gto.M(atom=atoms, basis=ribasis, unit='angstrom')
    pmol = mol + auxmol
    assert dm.shape[0] == mol.nao_nr(), f"{dm.shape=}, {mol.nao_nr()=}"

    overlap = auxmol.intor("int1e_ovlp_sph")  # AO overlap matrix
    eri2c = auxmol.intor('int2c2e_sph')  # 2-centers 2-electrons integral
    # https://github.com/pyscf/pyscf/blob/master/examples/df/42-overwrite_get_jk.py
    # https://github.com/pyscf/pyscf/issues/1729
    eri3c = pmol.intor(  # 3-centers 2-electrons integral
        'int3c2e_sph',
        shls_slice=(0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + auxmol.nbas)
    )
    eri3c = eri3c.reshape(mol.nao_nr(), mol.nao_nr(), -1)
    rho = np.einsum('ijp,ij->p', eri3c, dm)
    rho = np.linalg.solve(eri2c, rho)
    pyscf_time = time.time() - pyscf_time

    """ Reorder L=1 components from +1,-1,0 to -1,0,+1 """
    reorder_time = time.time()
    l1_slices = irreps.slices_l(1)
    l1_indexes = np.array([sl.start for sl in l1_slices]).reshape(-1, 1) + np.array([0, 1, 2]).reshape(1, -1)  # shape (n_l1, 3)
    """
    +1,-1,0 reordered to -1,0,+1
    from: 0,1,2; to: 2,0,1 (receiver)
    """
    l1_indexes_from = l1_indexes.reshape(-1)  # shape (3*n_l1,)
    l1_indexes_to = l1_indexes[:, [2,0,1]].reshape(-1)  # shape (3*n_l1,)

    coef_reordered = rho
    coef_reordered[l1_indexes_to] = rho[l1_indexes_from]
    overlap_reordered = overlap
    overlap_reordered[l1_indexes_to] = overlap[l1_indexes_from]
    overlap_reordered[:, l1_indexes_to] = overlap[:, l1_indexes_from]

    # Compute density projections on auxiliary functions
    proj_reordered = np.dot(overlap_reordered, coef_reordered)
    reorder_time = time.time() - reorder_time

    return {
        "coef": coef_reordered,
        "proj": proj_reordered,
        "over": overlap_reordered,
        "pyscf_time": pyscf_time,
        "reorder_time": reorder_time,
    }


#print "Computing ab-initio energies.."
#
## Hartree energy
#J = np.einsum('Q,mnQ->mn', rho, eri3c)
#e_h = np.einsum('ij,ji', J, dm) * 0.5
#f = open("hartree_energy.dat", 'a')
#print >> f, e_h
#f.close()
#
## Nuclear-electron energy
#h = mol.intor_symmetric('int1e_nuc')
#e_Ne = np.einsum('ij,ji', h, dm)
#f = open("external_energy.dat", 'a')
#print >> f, e_Ne
#f.close()

def main(geom_indexes: Union[List[int], None], num_threads: int = None):
    # global reorder_time, pyscf_time
    inp = ParseConfig().parse_input()
    geoms_all = read(inp.system.filename, ":")
    if geom_indexes is None:
        geom_indexes = list(range(len(geoms_all)))
    elif len(geom_indexes) == 1:
        geom_indexes = np.loadtxt(os.path.join(inp.salted.saltedpath,"missing_indexes.txt"),dtype=int)[geom_indexes]
        if type(geom_indexes) != np.ndarray and type(geom_indexes) != list:
            geom_indexes = [geom_indexes]
    else:
        geom_indexes = [i for i in geom_indexes if i < len(geoms_all)]  # indexes start from 0
    geoms = [geoms_all[i] for i in geom_indexes]

    """ prepare directories to store data """
    for data_dname in ("coefficients", "projections", "overlaps"):
        if not osp.exists(dpath := osp.join(inp.qm.path2qm, data_dname)):
            os.mkdir(dpath)

    """See if any structures already exist, if they do, do not compute again."""
    alreadyCalculated = np.array([re.findall(r'\d+',s) for s in glob.glob(f"{dpath}/*")], dtype=int).flatten()
    if len(alreadyCalculated) > 0:
        print("Found existing calculations, resuming from bevore", file=sys.stdout, flush=True)
        geom_indexes = np.setdiff1d(np.array(geom_indexes), alreadyCalculated)
    print(f"Calculate density fitting coefficients for these structures: {geom_indexes}", file=sys.stdout, flush=True)

    """ set pyscf.lib.num_threads """
    if num_threads is not None:
        lib.num_threads(num_threads)

    # lmax, nmax = BasisClient().read_as_old_format(get_aux_basis_name(inp.qm.qmbasis))
    ribasis = get_aux_basis_name(inp.qm.qmbasis)  # RI basis name in pyscf
    #check if ribasis is in pyscf
    
    if inp.qm.qmbasis != PLACEHOLDER:
        ribasis = read_basis(inp.qm.dfbasis)
        qmbasis = read_basis(inp.qm.qmbasis)
        
    basis_data = BasisClient().read(get_aux_basis_name(inp.qm.qmbasis))
    df_irreps_by_spe = {
        spe: Irreps(tuple((cnt, lam) for lam, cnt in enumerate(spe_basis_data["nmax"])))
        for spe, spe_basis_data in basis_data.items()
    }

    """ do density fitting """
    pyscf_time, reorder_time = 0.0, 0.0
    start_time = time.time()
    for cal_idx, (geom_idx, geom) in enumerate(zip(geom_indexes, geoms)):
        print(f"calculate {geom_idx=}, progress: {cal_idx}/{len(geom_indexes)}", file=sys.stdout, flush=True)
        symb = geom.get_chemical_symbols()
        coords = geom.get_positions()
        atoms = [(s, c) for s, c in zip(symb, coords)]
        irreps = sum([df_irreps_by_spe[spe] for spe in symb], Irreps([]))
        dm = np.load(osp.join(inp.qm.path2qm, "density_matrices", f"dm_conf{geom_idx}.npy"))
        reordered_data = cal_df_coeffs(atoms, qmbasis, ribasis, dm, irreps)
        
        """Uncomment if testing consistency with old code is needed."""
        # reordered_data_old = cal_df_coeffs_old(atoms, inp.qm.qmbasis, ribasis, dm, lmax, nmax)  # for checking consistency
        # assert np.allclose(reordered_data_old["coef"], reordered_data["coef"])  # for checking consistency
        # assert np.allclose(reordered_data_old["over"], reordered_data["over"])
        # assert np.allclose(reordered_data_old["proj"], reordered_data["proj"])
        np.save(osp.join(inp.salted.saltedpath, inp.qm.path2qm, "coefficients", f"coefficients_conf{geom_idx}.npy"), reordered_data["coef"])
        np.save(osp.join(inp.salted.saltedpath, inp.qm.path2qm, "projections", f"projections_conf{geom_idx}.npy"), reordered_data["proj"])
        np.save(osp.join(inp.salted.saltedpath, inp.qm.path2qm, "overlaps", f"overlap_conf{geom_idx}.npy"), reordered_data["over"])
        pyscf_time += reordered_data["pyscf_time"]
        reorder_time += reordered_data["reorder_time"]
    end_time = time.time()
    print(f"Calculation finished, time cost on density fitting: {end_time - start_time:.2f}s")
    print(f"Time cost on PySCF: {pyscf_time:.2f}s, on reordering: {reorder_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # create a parser obj, which accepts the indexes to calculate, start from 0
    # formats: 1,2,3 or 1-3 or None (all structures)
    parser.add_argument(
        "-i", "--idx", type=str, default="all",
        help=ARGHELP_INDEX_STR,
    )
    parser.add_argument(
        "-c", "--cpu", type=int, default=None,
        help="Number of CPU cores to use. Default is None (for do nothing)."
    )
    args = parser.parse_args()

    """ pyscf.lib.num_threads is more important than numpy OMP"""
    # # https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy
    # if args.cpu is not None:
    #     assert isinstance(args.cpu, int) and args.cpu > 0, f"{args.cpu=}"
    #     for env_var in (
    #         "OMP_NUM_THREADS",
    #         "OPENBLAS_NUM_THREADS",
    #         "MKL_NUM_THREADS",
    #         "VECLIB_MAXIMUM_THREADS",
    #         "NUMEXPR_NUM_THREADS",
    #     ):
    #         os.environ[env_var] = str(args.cpu)

    main(parse_index_str(args.idx), args.cpu)
