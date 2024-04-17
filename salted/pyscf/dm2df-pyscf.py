import argparse
import os
import sys
import os.path as osp

import numpy as np
from pyscf import gto
from ase.io import read
from scipy import special
import tqdm

import glob, re
import salted.cython.dm2df_fast_reorder as dm2df
from salted.pyscf.get_basis_info import get_aux_basis_name

debug = False
from salted import basis

sys.path.insert(0, './')
import inp

def add_command_line_arguments(parsetext):
    parser = argparse.ArgumentParser(description=parsetext,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-iconf", "--confidx",  type=int, default=-1, help="Structure index")
    args = parser.parse_args()
    return args

def set_variable_values(args):
    iconf = args.confidx
    return iconf

args = add_command_line_arguments("")
iconf = set_variable_values(args)

xyzfile = read(inp.filename,":")
ndata = len(xyzfile)

if iconf != -1:
    print("Calculating density matrix for configuration", iconf)
    iconf -= 1 # 0-based indexing
    conf_list = [iconf]
else:
    conf_list = range(ndata)


#See if any structures already exist, if they do, do not compute again.
alreadyCalculated = np.array([re.findall(r'\d+',s) for s in glob.glob(f"{osp.join(inp.path2qm, 'coefficients')}/*")], dtype=int).flatten()
if len(alreadyCalculated) > 0:
    print("Found existing calculations, resuming from before")
    conf_list = list(conf_list)
    for i in alreadyCalculated:
        conf_list.remove(i)

# read basis
[lmax,nmax] = basis.basiset(get_aux_basis_name(inp.qmbasis))
    
dirpath = os.path.join(inp.path2qm, "coefficients")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(inp.path2qm, "projections")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(inp.path2qm, "overlaps")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

print("PySCF orders all angular momentum components for L>1 as -L,...,0,...,+L,") 
print("and as +1,-1,0 for L=1, corresponding to X,Y,Z in Cartesian-SPH.")
print("Make sure to provide the density matrix following this convention!")
print("---------------------------------------------------------------------------------")
print("Reading geometry and basis sets...")

sys.path.insert(0, "/usr/local/lib/python3.10/site-packages/salted/pyscf")
for iconf in tqdm.tqdm(conf_list):

    # Initialize geometry
    geom = xyzfile[iconf]
    symb = geom.get_chemical_symbols()
    coords = geom.get_positions()
    natoms = len(coords)
    atoms = []
    for i in range(natoms):
        coord = coords[i]
        atoms.append([symb[i],(coord[0],coord[1],coord[2])])
    
    # Get PySCF objects for wave-function and density-fitted basis
    mol = gto.M(atom=atoms,basis=inp.qmbasis)
    ribasis = inp.qmbasis+" jkfit"
    auxmol = gto.M(atom=atoms,basis=ribasis)
    pmol = mol + auxmol
    
    if debug: print("Computing overlap matrix...")
    
    # Get and save overlap matrix
    overlap = auxmol.intor('int1e_ovlp_sph')
    
    if debug: print("Computing density-fitted coefficients...")
    
    # Number of atomic orbitals
    nao = mol.nao_nr()
    # Number of auxiliary atomic orbitals
    naux = auxmol.nao_nr()
    # 2-centers 2-electrons integral
    eri2c = auxmol.intor('int2c2e_sph')
    # 3-centers 2-electrons integral
    eri3c = pmol.intor('int3c2e_sph', shls_slice=(0,mol.nbas,0,mol.nbas,mol.nbas,mol.nbas+auxmol.nbas))
    eri3c = eri3c.reshape(mol.nao_nr(), mol.nao_nr(), -1)
    # Load 1-electron reduced density-matrix
    dm=np.load(osp.join(
        inp.path2qm, "density_matrices", f"dm_conf{iconf+1}.npy"
    ))
    # Compute density fitted coefficients
    rho = np.einsum('ijp,ij->p', eri3c, dm)
    rho = np.linalg.solve(eri2c, rho)
    
    if debug: print("Reordering...")
    
    Coef, Over = dm2df.reorder(rho, overlap, symb, lmax, nmax)

    # Reorder L=1 components following the -1,0,+1 convention
    # Coef = rho.copy()
    # Over = overlap.copy()
    # i1 = 0
    # i1_over = 0

    # for spe1 in symb:
    #     for l1 in range(lmax[spe1]+1):
    #         if l1 == 1:
    #             for n1 in range(nmax[(spe1,l1)]):
    #                 for im1 in range(2*l1+1):
    #                     if im1!=2:
    #                         Coef[i1] = rho[i1+1]
    #                     elif im1==2:
    #                         Coef[i1] = rho[i1-2] 
    #                     i1 += 1
    #         else:
    #             i1 += nmax[(spe1, l1)] * (2 * l1 + 1)
    #         for n1 in range(nmax[(spe1, l1)]):
    #             for im1 in range(2 * l1 + 1):
    #                 i2 = 0
    #                 for spe2 in symb:
    #                     for l2 in range(lmax[spe2]+1):
    #                         if l2 != 1 and l1 != 1:
    #                             i2 += nmax[(spe2, l2)] * (2 * l2 + 1)
    #                             continue
    #                         for n2 in range(nmax[(spe2,l2)]):
    #                             for im2 in range(2*l2+1):
    #                                 if l1==1 and im1!=2 and l2!=1:
    #                                     Over[i1_over,i2] = overlap[i1_over+1,i2]
    #                                 elif l1==1 and im1==2 and l2!=1:
    #                                     Over[i1_over,i2] = overlap[i1_over-2,i2]
    #                                 elif l2==1 and im2!=2 and l1!=1:
    #                                     Over[i1_over,i2] = overlap[i1_over,i2+1]
    #                                 elif l2==1 and im2==2 and l1!=1:
    #                                     Over[i1_over,i2] = overlap[i1_over,i2-2]
    #                                 elif l1==1 and im1!=2 and l2==1 and im2!=2:
    #                                     Over[i1_over,i2] = overlap[i1_over+1,i2+1]
    #                                 elif l1==1 and im1!=2 and l2==1 and im2==2:
    #                                     Over[i1_over,i2] = overlap[i1_over+1,i2-2]
    #                                 elif l1==1 and im1==2 and l2==1 and im2!=2:
    #                                     Over[i1_over,i2] = overlap[i1_over-2,i2+1]
    #                                 elif l1==1 and im1==2 and l2==1 and im2==2:
    #                                     Over[i1_over,i2] = overlap[i1_over-2,i2-2]
    #                                 i2 += 1
    #                 i1_over += 1
    
    # Compute density projections on auxiliary functions
    Proj = np.dot(Over,Coef)
    
    # Save projections and overlaps

    np.save(osp.join(inp.path2qm, "coefficients", f"coefficients_conf{iconf}.npy"), Coef)
    np.save(osp.join(inp.path2qm, "projections", f"projections_conf{iconf}.npy"), Proj)
    np.save(osp.join(inp.path2qm, "overlaps", f"overlap_conf{iconf}.npy"), Over)
    
    # --------------------------------------------------
    
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
