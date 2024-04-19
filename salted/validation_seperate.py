import os
import sys
import time
import os.path as osp

import h5py
import numpy as np
from scipy import special
from ase.data import atomic_numbers
from ase.io import read

from rascaline import SphericalExpansion
from rascaline import LodeSphericalExpansion
from metatensor import Labels

from salted.lib import equicomb
from salted.lib import equicombfield

from salted import sph_utils
from salted import basis
from salted import efield
from salted.sys_utils import read_system, get_atom_idx,get_conf_range


def build():

    sys.path.insert(0, './')
    import inp

    if inp.parallel:
        from mpi4py import MPI
        # MPI information
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    #    print('This is task',rank+1,'of',size)
    else:
        rank = 0
        size = 1

    validation_xyz_file = "validation.xyz"
    saltedname = inp.saltedname
    predname = inp.predname
    rep1 = inp.rep1
    rcut1 = inp.rcut1
    sig1 = inp.sig1
    nrad1 = inp.nrad1
    nang1 = inp.nang1
    neighspe1 = inp.neighspe1
    rep2 = inp.rep2
    rcut2 = inp.rcut2
    sig2 = inp.sig2
    nrad2 = inp.nrad2
    nang2 = inp.nang2
    neighspe2 = inp.neighspe2
    ncut = inp.ncut

    valPAth = osp.join(inp.saltedpath, "seperate_validation")

    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system(osp.join(valPAth, validation_xyz_file))
    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

    bohr2angs = 0.529177210670

    # Kernel parameters
    M = inp.Menv
    zeta = inp.z
    reg = inp.regul

    # Distribute structures to tasks
    ndata_true = ndata
    if inp.parallel:
        conf_range = get_conf_range(rank,size,ndata,list(range(ndata)))
        conf_range = comm.scatter(conf_range,root=0)
        ndata = len(conf_range)
        print(f"Task {rank+1} handles the following structures: {conf_range}", flush=True)
    else:
        conf_range = list(range(ndata))
    natoms_total = sum(natoms[conf_range])

    loadstart = time.time()

    # Load training feature vectors and RKHS projection matrix
    Mspe = {}
    power_env_sparse = {}
    if inp.field: power_env_sparse_field = {}
    Vmat = {}
    vfps = {}
    if inp.field: vfps_field = {}
    for lam in range(lmax_max+1):
        # Load sparsification details
        if ncut > 0:
            vfps[lam] = np.load(osp.join(
                inp.saltedpath, f"equirepr_{saltedname}", f"fps{ncut}-{lam}.npy"
            ))
        if ncut > 0 and inp.field:
            vfps_field[lam] = np.load(osp.join(
                inp.saltedpath, f"equirepr_{saltedname}", f"fps{ncut}-{lam}_field.npy"
            ))
        for spe in species:
            # load sparse equivariant descriptors
            power_env_sparse[(lam,spe)] = h5py.File(osp.join(
                inp.saltedpath, f"equirepr_{saltedname}", f"FEAT-{lam}-M-{M}.h5"
            ), 'r')[spe][:]
            if inp.field:
                power_env_sparse_field[(lam,spe)] = h5py.File(osp.join(
                    inp.saltedpath, f"equirepr_{saltedname}", f"FEAT-{lam}-M-{M}_field.h5"
                ), 'r')[spe][:]
            if lam == 0:
                Mspe[spe] = power_env_sparse[(lam,spe)].shape[0]
            # load RKHS projectors
            Vmat[(lam,spe)] = np.load(osp.join(
                inp.saltedpath,
                f"kernels_{saltedname}_field" if inp.field else f"kernels_{saltedname}",
                f"spe{spe}_l{lam}",
                f"M{M}_zeta{zeta}",
                f"projector.npy",
            ))
            # precompute projection on RKHS if linear model
            if zeta==1:
                power_env_sparse[(lam,spe)] = np.dot(
                    Vmat[(lam,spe)].T, power_env_sparse[(lam,spe)]
                )
                if inp.field:
                    power_env_sparse_field[(lam,spe)] = np.dot(
                        Vmat[(lam,spe)].T, power_env_sparse_field[(lam,spe)]
                    )

    reg_log10_intstr = str(int(np.log10(reg)))  # for consistency

    # load regression weights
    ntrain = int(inp.Ntrain*inp.trainfrac)
    weights = np.load(osp.join(
        inp.saltedpath,
        f"regrdir_{saltedname}_field" if inp.field else f"regrdir_{saltedname}",
        f"M{M}_zeta{zeta}",
        f"weights_N{ntrain}_reg{reg_log10_intstr}.npy"
    ))

    if rank == 0:  print(f"load time: {(time.time()-loadstart):.2f} s")

    start = time.time()

    HYPER_PARAMETERS_DENSITY = {
        "cutoff": rcut1,
        "max_radial": nrad1,
        "max_angular": nang1,
        "atomic_gaussian_width": sig1,
        "center_atom_weight": 1.0,
        "radial_basis": {"Gto": {"spline_accuracy": 1e-6}},
        "cutoff_function": {"ShiftedCosine": {"width": 0.1}},
    }

    HYPER_PARAMETERS_POTENTIAL = {
        "potential_exponent": 1,
        "cutoff": rcut2,
        "max_radial": nrad2,
        "max_angular": nang2,
        "atomic_gaussian_width": sig2,
        "center_atom_weight": 1.0,
        "radial_basis": {"Gto": {"spline_accuracy": 1e-6}}
    }

    frames = read(osp.join(valPAth, validation_xyz_file),":")
    frames = [frames[i] for i in conf_range]

    if rank == 0: print(f"The dataset contains {ndata_true} frames.")

    if rep1=="rho":
        # get SPH expansion for atomic density
        calculator = SphericalExpansion(**HYPER_PARAMETERS_DENSITY)

    elif rep1=="V":
        # get SPH expansion for atomic potential
        calculator = LodeSphericalExpansion(**HYPER_PARAMETERS_POTENTIAL)

    else:
        if rank == 0:
            raise ValueError(f"Unknown representation {rep1=}, expected 'rho' or 'V'")
        else:
            exit()

    descstart = time.time()

    nspe1 = len(neighspe1)
    keys_array = np.zeros(((nang1+1)*len(species)*nspe1,3),int)
    i = 0
    for l in range(nang1+1):
        for specen in species:
            for speneigh in neighspe1:
                keys_array[i] = np.array([l,atomic_numbers[specen],atomic_numbers[speneigh]],int)
                i += 1

    keys_selection = Labels(
        names=["spherical_harmonics_l","species_center","species_neighbor"],
        values=keys_array
    )

    spx = calculator.compute(frames, selected_keys=keys_selection)
    spx = spx.keys_to_properties("species_neighbor")
    spx = spx.keys_to_samples("species_center")

    # Get 1st set of coefficients as a complex numpy array
    omega1 = np.zeros((nang1+1,natoms_total,2*nang1+1,nspe1*nrad1),complex)
    for l in range(nang1+1):
        c2r = sph_utils.complex_to_real_transformation([2*l+1])[0]
        omega1[l,:,:2*l+1,:] = np.einsum('cr,ard->acd',np.conj(c2r.T),spx.block(spherical_harmonics_l=l).values)

    if rep2=="rho":
        # get SPH expansion for atomic density
        calculator = SphericalExpansion(**HYPER_PARAMETERS_DENSITY)

    elif rep2=="V":
        # get SPH expansion for atomic potential
        calculator = LodeSphericalExpansion(**HYPER_PARAMETERS_POTENTIAL)

    else:
        if rank == 0:
            raise ValueError(f"Unknown representation {rep2=}, expected 'rho' or 'V'")
        else:
            exit()

    nspe2 = len(neighspe2)
    keys_array = np.zeros(((nang2+1)*len(species)*nspe2,3),int)
    i = 0
    for l in range(nang2+1):
        for specen in species:
            for speneigh in neighspe2:
                keys_array[i] = np.array([l,atomic_numbers[specen],atomic_numbers[speneigh]],int)
                i += 1

    keys_selection = Labels(
        names=["spherical_harmonics_l","species_center","species_neighbor"],
        values=keys_array
    )

    spx_pot = calculator.compute(frames, selected_keys=keys_selection)
    spx_pot = spx_pot.keys_to_properties("species_neighbor")
    spx_pot = spx_pot.keys_to_samples("species_center")

    # Get 2nd set of coefficients as a complex numpy array
    omega2 = np.zeros((nang2+1,natoms_total,2*nang2+1,nspe2*nrad2),complex)
    for l in range(nang2+1):
        c2r = sph_utils.complex_to_real_transformation([2*l+1])[0]
        omega2[l,:,:2*l+1,:] = np.einsum('cr,ard->acd',np.conj(c2r.T),spx_pot.block(spherical_harmonics_l=l).values)

    if inp.field:
    # get SPH expansion for a uniform and constant external field aligned along Z
        omega_field = np.zeros((natoms_total,nrad2),complex)
        for iat in range(natoms_total):
            omega_field[iat] = efield.get_efield_sph(nrad2,rcut2)

    if rank == 0: print(f"coefficients time: {(time.time()-descstart):.2f} s\n")

    # base directory path for this prediction
    dirpath = osp.join(
        inp.saltedpath,
        f"predictions_{saltedname}_field_{predname}" \
            if inp.field else f"predictions_{saltedname}_{predname}",
        f"M{M}_zeta{zeta}",
        f"N{ntrain}_reg{reg_log10_intstr}",
    )

    # Create directory for predictions
    if rank == 0:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
    if size > 1:  comm.Barrier()

    # Compute equivariant descriptors for each lambda value entering the SPH expansion of the electron density
    psi_nm = {}
    for lam in range(lmax_max+1):

        if rank == 0: print(f"lambda = {lam}")

        equistart = time.time()

        # Select relevant angular components for equivariant descriptor calculation
        llmax = 0
        lvalues = {}
        for l1 in range(nang1+1):
            for l2 in range(nang2+1):
                # keep only even combination to enforce inversion symmetry
                if (lam+l1+l2)%2==0 :
                    if abs(l2-lam) <= l1 and l1 <= (l2+lam) :
                        lvalues[llmax] = [l1,l2]
                        llmax+=1
        # Fill dense array from dictionary
        llvec = np.zeros((llmax,2),int)
        for il in range(llmax):
            llvec[il,0] = lvalues[il][0]
            llvec[il,1] = lvalues[il][1]

        # Load the relevant Wigner-3J symbols associated with the given triplet (lam, lmax1, lmax2)
        wigner3j = np.loadtxt(osp.join(
            inp.saltedpath, "wigners", f"wigner_lam-{lam}_lmax1-{nang1}_lmax2-{nang2}.dat"
        ))
        wigdim = wigner3j.size

        # Reshape arrays of expansion coefficients for optimal Fortran indexing
        v1 = np.transpose(omega1,(2,0,3,1))
        v2 = np.transpose(omega2,(2,0,3,1))

        # Compute complex to real transformation matrix for the given lambda value
        c2r = sph_utils.complex_to_real_transformation([2*lam+1])[0]

        # Perform symmetry-adapted combination following Eq.S19 of Grisafi et al., PRL 120, 036002 (2018)
        p = equicomb.equicomb(natoms_total,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r)

        # Define feature space and reshape equivariant descriptor
        featsize = nspe1*nspe2*nrad1*nrad2*llmax
        p = np.transpose(p,(4,0,1,2,3)).reshape(natoms_total,2*lam+1,featsize)

        if rank == 0: print(f"equivariant time: {(time.time()-equistart):.2f} s")

        normstart = time.time()

        # Normalize equivariant descriptor
        inner = np.einsum('ab,ab->a', p.reshape(natoms_total,(2*lam+1)*featsize),p.reshape(natoms_total,(2*lam+1)*featsize))
        p = np.einsum('abc,a->abc', p,1.0/np.sqrt(inner))

        if rank == 0: print(f"norm time: {(time.time()-normstart):.2f} s")

        sparsestart = time.time()

        if ncut > 0:
            p = p.reshape(natoms_total*(2*lam+1),featsize)
            p = p.T[vfps[lam]].T
            featsize = inp.ncut

        if rank == 0: print(f"sparse time: {(time.time()-sparsestart):.2f} s")

        fillstart = time.time()

        # Fill vector of equivariant descriptor
        if lam==0:
            p = p.reshape(natoms_total,featsize)
            pvec = np.zeros((ndata,natmax,featsize))
        else:
            p = p.reshape(natoms_total,2*lam+1,featsize)
            pvec = np.zeros((ndata,natmax,2*lam+1,featsize))

        j = 0
        for i,iconf in enumerate(conf_range):
            for iat in range(natoms[iconf]):
                pvec[i,iat] = p[j]
                j += 1

        if rank == 0: print(f"fill vector time: {(time.time()-fillstart):.2f} s")

        if inp.field:
            #########################################################
            #                 START E-FIELD HERE
            #########################################################

            # Select relevant angular components for equivariant descriptor calculation
            llmax = 0
            lvalues = {}
            for l1 in range(nang1+1):
                # keep only even combination to enforce inversion symmetry
                if (lam+l1+1)%2==0 :
                    if abs(1-lam) <= l1 and l1 <= (1+lam) :
                        lvalues[llmax] = [l1,1]
                        llmax+=1
            # Fill dense array from dictionary
            llvec = np.zeros((llmax,2),int)
            for il in range(llmax):
                llvec[il,0] = lvalues[il][0]
                llvec[il,1] = lvalues[il][1]

            # Load the relevant Wigner-3J symbols associated with the given triplet (lam, lmax1, lmax2)
            wigner3j = np.loadtxt(osp.join(
                inp.saltedpath, "wigners", f"wigner_lam-{lam}_lmax1-{nang1}_field.dat"
            ))
            wigdim = wigner3j.size

            # Perform symmetry-adapted combination following Eq.S19 of Grisafi et al., PRL 120, 036002 (2018)
            v2 = omega_field.T
            p = equicombfield.equicombfield(natoms_total,nang1,nspe1*nrad1,nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r)

            # Define feature space and reshape equivariant descriptor
            featsizefield = nspe1*nrad1*nrad2*llmax
            p = np.transpose(p,(4,0,1,2,3)).reshape(natoms_total,2*lam+1,featsizefield)

            if rank == 0: print(f"field equivariant time: {(time.time()-equistart):.2f} s")

            normstart = time.time()

            # Normalize equivariant descriptor
            inner = np.einsum('ab,ab->a', p.reshape(natoms_total,(2*lam+1)*featsizefield),p.reshape(natoms_total,(2*lam+1)*featsizefield))
            p = np.einsum('abc,a->abc', p,1.0/np.sqrt(inner))

            if rank == 0: print(f"field norm time: {(time.time()-normstart):.2f} s")

            if ncut > -1:
                p = p.reshape(natoms_total*(2*lam+1),featsizefield)
                p = p.T[vfps_field[lam]].T
                featsizefield = inp.ncut

            fillstart = time.time()

            # Fill vector of equivariant descriptor
            if lam==0:
                p = p.reshape(natoms_total,featsizefield)
                pvec_field = np.zeros((ndata,natmax,featsizefield))
            else:
                p = p.reshape(natoms_total,2*lam+1,featsizefield)
                pvec_field = np.zeros((ndata,natmax,2*lam+1,featsizefield))

            j = 0
            for i,iconf in enumerate(conf_range):
                for iat in range(natoms[iconf]):
                    pvec_field[i,iat] = p[j]
                    j += 1

        rkhsstart = time.time()

        if lam==0:

            if zeta==1:
                 # Compute scalar kernels
                 kernel0_nm = {}
                 for i,iconf in enumerate(conf_range):
                     for spe in species:
                         if inp.field:
                             psi_nm[(iconf,spe,lam)] = np.dot(pvec_field[i,atom_idx[(iconf,spe)]],power_env_sparse_field[(lam,spe)].T)
                         else:
                             psi_nm[(iconf,spe,lam)] = np.dot(pvec[i,atom_idx[(iconf,spe)]],power_env_sparse[(lam,spe)].T)
            else:
                 # Compute scalar kernels
                 kernel0_nm = {}
                 for i,iconf in enumerate(conf_range):
                     for spe in species:
                         kernel0_nm[(iconf,spe)] = np.dot(pvec[i,atom_idx[(iconf,spe)]],power_env_sparse[(lam,spe)].T)
                         if inp.field:
                             kernel_nm = np.dot(pvec_field[i,atom_idx[(iconf,spe)]],power_env_sparse_field[(lam,spe)].T) * kernel0_nm[(iconf,spe)]**(zeta-1)
                         else:
                             kernel_nm = kernel0_nm[(iconf,spe)]**zeta
                         # Project on RKHS
                         psi_nm[(iconf,spe,lam)] = np.dot(kernel_nm,Vmat[(lam,spe)])

        else:

            if zeta==1:
                # Compute covariant kernels
                for i,iconf in enumerate(conf_range):
                    for spe in species:
                        if inp.field:
                            psi_nm[(iconf,spe,lam)] = np.dot(pvec_field[i,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*lam+1),pvec_field.shape[-1]),power_env_sparse_field[(lam,spe)].T)
                        else:
                            psi_nm[(iconf,spe,lam)] = np.dot(pvec[i,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*lam+1),featsize),power_env_sparse[(lam,spe)].T)
            else:
                # Compute covariant kernels
                for i,iconf in enumerate(conf_range):
                    for spe in species:
                        if inp.field:
                            kernel_nm = np.dot(pvec_field[i,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*lam+1),pvec_field.shape[-1]),power_env_sparse_field[(lam,spe)].T)
                        else:
                            kernel_nm = np.dot(pvec[i,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*lam+1),featsize),power_env_sparse[(lam,spe)].T)
                        for i1 in range(natom_dict[(iconf,spe)]):
                            for i2 in range(Mspe[spe]):
                                kernel_nm[i1*(2*lam+1):i1*(2*lam+1)+2*lam+1][:,i2*(2*lam+1):i2*(2*lam+1)+2*lam+1] *= kernel0_nm[(iconf,spe)][i1,i2]**(zeta-1)
                        # Project on RKHS
                        psi_nm[(iconf,spe,lam)] = np.dot(kernel_nm,Vmat[(lam,spe)])

        if rank == 0: print("rkhs time:", time.time()-rkhsstart,flush=True)

    predstart = time.time()
    comm.Barrier()
    # Load spherical averages if required
    if inp.average:
        av_coefs = {}
        for spe in species:
            av_coefs[spe] = np.load(f"averages_{spe}.npy")

    with open("errors.dat","w"):
        pass
    errorfile = open("errors.dat","a")
    error_density = 0
    variance = 0
    # Perform equivariant predictions
    for iconf in conf_range:

        Tsize = 0
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            for l in range(lmax[spe]+1):
                for n in range(nmax[(spe,l)]):
                    Tsize += 2*l+1

        # compute predictions per channel
        C = {}
        ispe = {}
        isize = 0
        for spe in species:
            ispe[spe] = 0
            for l in range(lmax[spe]+1):
                for n in range(nmax[(spe,l)]):
                    Mcut = psi_nm[(iconf,spe,l)].shape[1]
                    C[(spe,l,n)] = np.dot(psi_nm[(iconf,spe,l)],weights[isize:isize+Mcut])
                    isize += Mcut

        # init averages array if asked
        if inp.average:
            Av_coeffs = np.zeros(Tsize)

        # fill vector of predictions
        i = 0
        pred_coefs = np.zeros(Tsize)
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            for l in range(lmax[spe]+1):
                for n in range(nmax[(spe,l)]):
                    pred_coefs[i:i+2*l+1] = C[(spe,l,n)][ispe[spe]*(2*l+1):ispe[spe]*(2*l+1)+2*l+1]
                    if inp.average and l==0:
                        Av_coeffs[i] = av_coefs[spe][n]
                    i += 2*l+1
            ispe[spe] += 1

        # add back spherical averages if required
        if inp.average:
            pred_coefs += Av_coeffs

        # load reference
        ref_coefs = np.load(osp.join(
            valPAth, "coefficients", f"coefficients_conf{iconf}.npy"
        ))
        overl = np.load(osp.join(
            valPAth, "overlaps", f"overlap_conf{iconf}.npy"
        ))
        ref_projs = np.dot(overl,ref_coefs)
        # compute predicted density projections <phi|rho>
        pred_projs = np.dot(overl,pred_coefs)

        # compute error
        error = np.dot(pred_coefs-ref_coefs,pred_projs-ref_projs)
        error_density += error
        if inp.average:
            ref_projs -= np.dot(overl,Av_coeffs)
            ref_coefs -= Av_coeffs
        var = np.dot(ref_coefs,ref_projs)
        variance += var
        print(f"{iconf+1:d} {(np.sqrt(error/var)*100):.3e}", file=errorfile)
        print(f"{iconf+1}: {(np.sqrt(error/var)*100):<3.3f} % RMSE", flush=True)
    errorfile.close()
    if inp.parallel:
        error_density = comm.allreduce(error_density)
        variance = comm.allreduce(variance)
        if rank == 0:
            errs = np.loadtxt("errors.dat")
            np.savetxt("errors.dat", errs[errs[:,0].argsort()])
    if rank == 0:
        print(f"\n % RMSE: {(100*np.sqrt(error_density/variance)):<3.3f}", flush=True)
        with open("errors.dat","a") as f:
            print(f"Total RMSE: {(100*np.sqrt(error_density/variance)):<3.3f}", file=f)



if __name__ == "__main__":
    build()
