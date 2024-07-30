import argparse
import os
import sys
from typing import List, Tuple, Union
import time

import re, glob

import numpy as np
from ase.io import read
from pyscf import gto, dft, lib, df

from salted.sys_utils import ParseConfig, parse_index_str, ARGHELP_INDEX_STR
from salted.pyscf.get_basis_info import read_basis


def run_pyscf(
    atoms: List,
    basis: str|dict,
    dfBasis: str|dict,
    xc: str,
    spin = 0,
    charge = 0,
    solvent_eps: float = 1,
    verbose: int = 0,
):
    # Get PySCF objects for wave-function and density-fitted basis
    try:
        mol = gto.M(atom=atoms, unit='angstrom', max_memory=5000, spin=spin, charge=charge, verbose=verbose, basis=basis)
    except lib.exceptions.BasisNotFoundError:
        print("Using self defined basis", file=sys.stdout, flush=True)
        #Had to turn off verbosity for the creation as it prints out the whole basis set
        mol = gto.M(atom=atoms, unit='angstrom', max_memory=5000, spin=spin, charge=charge, verbose=0, basis=read_basis(basis, species=set([a[0] for a in atoms])))
   

    #Choose solvent model, if any
    if solvent_eps != 1:
        from pyscf.solvent import PCM
        m = PCM(mol.RKS())
        m.with_solvent.method = "COSMO"
        m.with_solvent.eps = solvent_eps
        print(f"Solvent eps = {solvent_eps}", file=sys.stdout, flush=True)
    else:
        m = dft.rks.RKS(mol)
    
    m.verbose = verbose
    m.max_cycle = 400
    
    #If single atom, use atom guess
    if len(atoms) == 1:
        m.init_guess = "atom"
    
    # #Setup Functional
    if "r2scan" in xc.lower():
        m._numint.libxc = dft.xcfun
    m.xc = xc
    
    # #Setup Grids
    m.grids.radi_method = dft.gauss_chebyshev
    m.grids.level = 0
    
    #Setup density fitting
    m = m.density_fit()
    try:
        m.with_df.auxbasis = df.addons.DEFAULT_AUXBASIS[gto.basis._format_basis_name(basis)][0]
    except KeyError:
        m.with_df.auxbasis = read_basis(basis, species=set([a[0] for a in atoms]))
    
    try:
        m.kernel()
    except ValueError:
        print(f"\nError in configuration", file = sys.stdout, flush=True)
        return
    return m.make_rdm1()


def main(geom_indexes: Union[List[int], None], num_threads: int = None):
    inp = ParseConfig().parse_input()
    geoms_all = read(inp.system.filename, ":")
    if geom_indexes is None:
        geom_indexes = list(range(len(geoms_all)))
    elif len(geom_indexes) == 1:
        geom_indexes = np.loadtxt(os.path.join(inp.salted.saltedpath,"missing_indexes.txt"),dtype=int, ndmin=1)[geom_indexes]
        if type(geom_indexes) != np.ndarray and type(geom_indexes) != list:
            geom_indexes = [geom_indexes]
    else:
        geom_indexes = [i for i in geom_indexes if i < len(geoms_all)]  # indexes start from 0
    geoms = [geoms_all[i] for i in geom_indexes]

    """ prepare the output directory """
    dirpath = os.path.join(inp.qm.path2qm, "density_matrices")
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    
    """See if any structures already exist, if they do, do not compute again."""
    alreadyCalculated = np.array([re.findall(r'\d+',s) for s in glob.glob(f"{dirpath}/*.npy")], dtype=int).flatten()
    if len(alreadyCalculated) > 0:
        print("Found existing calculations, resuming from bevore", file=sys.stdout, flush=True)
        geom_indexes = np.setdiff1d(np.array(geom_indexes), alreadyCalculated)
    print(f"Calculating density matrix for configurations: {geom_indexes}", file=sys.stdout, flush=True)

    """ set pyscf.lib.num_threads """
    if num_threads is not None:
        lib.num_threads(num_threads)

    """ do DFT calculation """
    verbose = 4 if len(geom_indexes) == 1 else 4
    start_time = time.time()
    for cal_idx, (geom_idx, geom) in enumerate(zip(geom_indexes, geoms)):
        print(f"calcualte {geom_idx=}, progress: {cal_idx}/{len(geom_indexes)}", file = sys.stdout, flush=True)
        geom.translate(-geom.get_center_of_mass())
        symb = geom.get_chemical_symbols()
        coords = geom.get_positions()
        atoms = [(s, c) for s, c in zip(symb, coords)]
        
        dm = run_pyscf(atoms = atoms, basis = inp.qm.qmbasis, dfBasis=inp.qm.dfbasis, xc= inp.qm.functional, solvent_eps = inp.qm.solvent_eps, verbose=verbose)
        np.save(os.path.join(dirpath, f"dm_conf{geom_idx}.npy"), dm)
    end_time = time.time()
    print(f"Calculation finished, time cost on DFT: {end_time - start_time:.2f}s")


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

    main(parse_index_str(args.idx), args.cpu)
