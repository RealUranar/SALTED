import os
import random
import sys
import time
import os.path as osp

import numpy as np
import h5py
from ase.data import atomic_numbers
from ase.io import read

from salted import sph_utils
from salted import basis

from salted.sph_utils import equicombfps
from salted.sys_utils import ParseConfig, build_featomic_hyper_params, do_fps, get_atom_idx, read_system
from salted.selection_utils import (
    DEFAULT_SELECTION_SEED,
    parse_model_setup,
    select_configurations,
)

def _load_training_indices(inp):
    path = osp.join(
        inp.salted.saltedpath,
        f"regrdir_{inp.salted.saltedname}",
        f"training_set_N{inp.gpr.Ntrain}.txt",
    )
    if not osp.isfile(path):
        raise FileNotFoundError(
            f"Training-set file not found: {path}. Run 'python -m salted.data_selection' first."
        )
    return np.atleast_1d(np.loadtxt(path, dtype=int)).astype(int).tolist()

def build():

    inp = ParseConfig().parse_input()
    # frequently used parameters
    saltedpath = inp.salted.saltedpath
    rep1, rep2 = inp.descriptor.rep1.type, inp.descriptor.rep2.type
    nrad1, nrad2 = inp.descriptor.rep1.nrad, inp.descriptor.rep2.nrad
    nang1, nang2 = inp.descriptor.rep1.nang, inp.descriptor.rep2.nang
    neighspe1, neighspe2 = inp.descriptor.rep1.neighspe, inp.descriptor.rep2.neighspe
    nsamples = inp.descriptor.sparsify.nsamples
    Ntrain = inp.gpr.Ntrain
    nspe1 = len(inp.descriptor.rep1.neighspe)
    nspe2 = len(inp.descriptor.rep2.neighspe)
    ncut = inp.descriptor.sparsify.ncut
    sparsify = ncut > 0
    HP1 = build_featomic_hyper_params(inp.descriptor.rep1)
    HP2 = build_featomic_hyper_params(inp.descriptor.rep2)

    # Generate directories for saving descriptors
    sdir = osp.join(saltedpath, f"equirepr_{inp.salted.saltedname}")
    if not osp.exists(sdir):
        os.mkdir(sdir)

    if not sparsify:
        print(
            "ERROR: inp parameter sparsify=False. "
            "Make sure to include a sparsify section with ncut>0 if you want to sparsify the descriptor\n",
            file=sys.stderr
        )
        sys.exit(1)

    train_indices = _load_training_indices(inp)
    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, atomic_coords, natoms, natmax = read_system(conf_indices=train_indices)
    if len(train_indices) != Ntrain:
        raise ValueError(
            f"Training-set file contains {len(train_indices)} indices, expected Ntrain={Ntrain}."
        )
    if nsamples > Ntrain:
        raise ValueError(f"nsamples ({nsamples}) cannot exceed Ntrain ({Ntrain}).")

    start = time.time()

    all_frames = read(inp.system.filename, ":", parallel=False)
    groups = parse_model_setup(osp.join(os.getcwd(), "model_setup"))
    samplesel = inp.descriptor.sparsify.samplesel

    print("=== nsamples selection ===")
    print(f"Sample selection mode: {samplesel}")
    print(f"Available training configurations: {ndata}")
    conf_range = select_configurations(
        train_indices,
        groups,
        nsamples,
        samplesel,
        seed=DEFAULT_SELECTION_SEED,
        frames=all_frames if samplesel == "equal_rmsd_fps" else None,
        selection_label="nsamples",
    )

    np.savetxt(
        osp.join(sdir, f"sample_set_N{nsamples}.txt"),
        np.asarray(conf_range, dtype=int),
        fmt="%i",
    )
    print(f"Selected sample configurations: {len(conf_range)}")
    print(f"Selection seed: {DEFAULT_SELECTION_SEED}")

    frames = [all_frames[i] for i in conf_range]
    natoms_selected = [natoms[i] for i in conf_range]
    natoms_total = sum(natoms_selected)

    omega1 = sph_utils.get_representation_coeffs(frames, rep1, HP1, 0, neighspe1, species, nang1, nrad1, natoms_total)
    if sph_utils.reps_equivalent(rep1, neighspe1, HP1, rep2, neighspe2, HP2):
        omega2 = omega1
    else:
        omega2 = sph_utils.get_representation_coeffs(frames, rep2, HP2, 0, neighspe2, species, nang2, nrad2, natoms_total)

    # Reshape arrays of expansion coefficients for optimal Fortran indexing
    v1 = np.transpose(omega1,(1,3,0,2)).copy()
    v2 = np.transpose(omega2,(1,3,0,2)).copy()
    del omega1, omega2

    # Compute equivariant descriptors for each lambda value entering the SPH expansion of the electron density
    for lam in range(lmax_max + 1):
        llmax, llvec = sph_utils.get_angular_indexes_symmetric(lam, nang1, nang2)
        
        # Load the relevant Wigner-3J symbols associated with the given triplet (lam, lmax1, lmax2)
        wigner3j = np.loadtxt(osp.join(saltedpath, "wigners", f"wigner_lam-{lam}_lmax1-{nang1}_lmax2-{nang2}.dat"))
        
        # Compute complex to real transformation matrix for the given lambda value
        c2r = sph_utils.complex_to_real_transformation([2 * lam + 1])[0]

        # compute normalized equivariant descriptor
        featsize = nspe1 * nspe2 * nrad1 * nrad2 * llmax
        print(f"lambda = {lam}, feature space size = {featsize}")

        if ncut >= featsize:
            print(
                f"ERROR: requested number of sparse features larger than total feature space size: "
                f"{ncut} > {featsize}. Please remove the inp.descriptor.sparsify section or reduce ncut value."
            )
            sys.exit(1)

        pvec = equicombfps(natoms_total, nang1, nang2, nspe1 * nrad1, nspe2 * nrad2, v1, v2, wigner3j, llmax, llvec, lam, c2r, featsize,)
        vfps = do_fps(pvec, ncut, verbose=inp.salted.verbose)
        np.save(osp.join(sdir, f"fps{ncut}-{lam}.npy"), vfps)

    print(f"Feature sparsification finished in {time.time() - start:.1f} s")

if __name__ == "__main__":
    build()
