import argparse
import os
import sys
import glob
import re
# from multiprocessing import Pool, cpu_count
import tqdm
import numpy as np
from ase.io import read
from pyscf import gto
from pyscf import scf,dft, df
from pyscf import lib
from pyscf.gto import basis
from multiprocessing import Pool, current_process
sys.path.insert(0, './')
import inp

# Initialize geometry
geoms = read(inp.filename,":")
dirpath = os.path.join(inp.path2qm, "density_matrices")

def add_command_line_arguments(parsetext):
    parser = argparse.ArgumentParser(description=parsetext,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-iconf", "--confidx",  type=int, default=-1, help="Structure index")
    args = parser.parse_args()
    return args

def set_variable_values(args):
    iconf = args.confidx
    return iconf

def doSCF(i):
    geom = geoms[i]
    symb = geom.get_chemical_symbols()
    coords = geom.get_positions()
    natoms = len(coords)
    atoms = []
    for j in range(natoms):
        coord = coords[j]
        atoms.append([symb[j],(coord[0],coord[1],coord[2])])

    # Get PySCF objects for wave-function and density-fitted basis
    mol = gto.M(atom=atoms,basis=inp.qmbasis)
    mol.verbose = 0
    m = dft.rks.RKS(mol)
    
    if "r2scan" in inp.functional.lower():
        m._numint.libxc = dft.xcfun
    m.grids.radi_method = dft.gauss_chebyshev
    m.grids.level = 0
    m = m.density_fit()
    m.with_df.auxbasis = df.addons.DEFAULT_AUXBASIS[basis._format_basis_name(inp.qmbasis)][0]
    m.xc = inp.functional
    
    # print(f"Process: {current_process().name.split('-')[-1]} - Calculating density matrix for configuration {i+1}", file = sys.stdout.flush(), flush=True)
    m.kernel()
    # #Read checkpoint from a preliminary run
    # m.chkfile = 'start_checkpoint'
    # m.init_guess = "chkfile"
    # if i % 100 == 0:
    #     m.kernel()
    # else:
    #     m.kernel(dump_chk = False)

    dm = m.make_rdm1()
    np.save(os.path.join(dirpath, f"dm_conf{i+1}.npy"), dm)
    
if __name__ == "__main__":
    args = add_command_line_arguments("")
    iconf = set_variable_values(args)

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    if iconf != -1:
        print("Calculating density matrix for configuration", iconf)
        iconf -= 1 # 0-based indexing
        conf_list = [iconf]
    else:
        conf_list = range(len(geoms))

    #See if any structures already exist, if they do, do not compute again.
    alreadyCalculated = np.array([re.findall(r'\d+',s) for s in glob.glob(f"{dirpath}/*")], dtype=int).flatten()-1
    if len(alreadyCalculated) > 0:
        print("Found existing calculations, resuming from bevore")
        conf_list = list(conf_list)
        for i in alreadyCalculated:
            conf_list.remove(i)

    # coresPerThread = 2
    # print(f"Running {len(conf_list)} PySCF Calculations with {lib.num_threads() // coresPerThread} threads and {coresPerThread} cores per thread.")
    # cpus = lib.num_threads() // coresPerThread
    # with lib.with_omp_threads(coresPerThread):
    #     with Pool(processes=cpus) as p:
    #         with tqdm.tqdm(total=len(conf_list)) as pbar:
    #             async_results = [p.apply_async(doSCF, args=(i,), callback=lambda: pbar.update()) for i in conf_list]
    #             results = [async_result.get() for async_result in async_results]
    for i in tqdm.tqdm(conf_list,total=len(geoms), initial=len(geoms)-len(conf_list)):
        doSCF(i)
