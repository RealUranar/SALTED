import sys
import os
import os.path as osp

import numpy as np
from ase.io import read
from pyscf import gto
from pyscf import scf,dft
from pyscf import lib
from salted import basis  # WARNING: relative import
from salted.pyscf.get_basis_info import get_aux_basis_name

from salted.sys_utils import ParseConfig, parse_index_str, ARGHELP_INDEX_STR

# Initialize geometry
inp = ParseConfig().parse_input()
geoms_all = read(inp.system.filename, ":")
conf_list = range(len(geoms_all))

# read basis
[lmax,nmax] = basis.basiset(get_aux_basis_name(inp.qm.qmbasis))


for iconf in conf_list:
    geom = geoms_all[iconf]
    symb = geom.get_chemical_symbols()
    coords = geom.get_positions()
    natoms = len(coords)
    atoms = []
    for i in range(natoms):
        coord = coords[i]
        atoms.append([symb[i],(coord[0],coord[1],coord[2])])

    # Get PySCF objects for wave-function and density-fitted basis
    mol = gto.M(atom=atoms,basis=inp.qm.qmbasis)
    mol.verbose = 2
    mol.max_memory = 10_000
    m = dft.RKS(mol)
    if "r2scan" in inp.qm.functional.lower():
        m._numint.libxc = dft.xcfun
    m.grids.radi_method = dft.gauss_chebyshev
    m.grids.level = 0
    m = m.density_fit()
    m.with_df.auxbasis = get_aux_basis_name(inp.qm.qmbasis)
    m.xc = inp.qm.functional
    # Save density matrix
    m.kernel()

    dm = m.make_rdm1()

    ribasis = inp.qmbasis+" jkfit"
    auxmol = gto.M(atom=atoms,basis=ribasis)
    pmol = mol + auxmol
    
    print("Computing density-fitted coefficients...")
    
    # Number of atomic orbitals
    nao = mol.nao_nr()
    # 2-centers 2-electrons integral
    eri2c = auxmol.intor('int2c2e_sph')
    # 3-centers 2-electrons integral
    eri3c = pmol.intor('int3c2e_sph', shls_slice=(0,mol.nbas,0,mol.nbas,mol.nbas,mol.nbas+auxmol.nbas))
    eri3c = eri3c.reshape(mol.nao_nr(), mol.nao_nr(), -1)
  
    # Compute density fitted coefficients
    rho = np.einsum('ijp,ij->p', eri3c, dm)
    rho = np.linalg.solve(eri2c, rho)
    
    print("Reordering...")
    
    # Reorder L=1 components following the -1,0,+1 convention
    Coef = np.zeros(len(rho),float)
    i1 = 0
    for iat in range(natoms):
        spe1 = symb[iat]
        for l1 in range(lmax[spe1]+1):
            for n1 in range(nmax[(spe1,l1)]):
                for im1 in range(2*l1+1):
                    if l1==1 and im1!=2:
                        Coef[i1] = rho[i1+1]
                    elif l1==1 and im1==2:
                        Coef[i1] = rho[i1-2]
                    else:
                        Coef[i1] = rho[i1]
                    i1 += 1
    
    if not osp.exists("reference"):
        os.mkdir("reference")
    # Save Coefficents
    np.save(osp.join("reference/", f'{inp.predict_filename.removesuffix(".xyz")}_{iconf}.npy'), Coef)
   
    
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
