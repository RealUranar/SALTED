# Input Configuration

The file `inp.yaml` in the working directory is used to configure the input data for the model.
This file will be parsed by the `sys_utils.ParseConfig` as a dictionary.

## Input controlling file structure

This lists all possible inputs which can be listed in `inp.yaml`.
Some options may be omitted, in which case default values are chosen.
The `type` columns follow the Python typing conventions.
The `default` column lists if the variable is required or not (then it lists the default value).

For all the path-related variables, the path can be either a relative path or an absolute path.

### SALTED definition `inp.salted`

| var name | type | default | usage |
| -:| :-: | :-: | :- |
| `saltedname` | `str` | **Required** | A label to identify a particular SALTED setup. |
| `saltedpath` | `str` | **Required** | Location of all files produced by SALTED. Either relative to the working directory or an absolute path. |
| `saltedtype` | `str` | `density` | Option for selecting the type of SALTED target (`density` or `density-response`) |
| `seed` | `int` (≥ 0) | `42` | Random seed. Used only by `gpr.sparse_selection: hybrid` (its subsample draws). |
| `verbose` | `bool` | `False` | Output verbosity. |

### System difinition `inp.system`

| var name | type | default | usage |
| -:| :-: | :-: | :- |
| `filename` | `str` | **Required** | An extended-XYZ file consisting of input structures. |
| `species` | `list[str]` | **Required** | List of element species considered in the electron density expansion. |
| `average` | `bool` | `True` | Whether we use averaged coefficients to set a baseline for the density. Normally this should be true, unless a density difference is learned. |

### Information about QM training set generation `inp.qm`

| var name | type | default | usage |
| -:| :-: | :-: | :- |
| `path2qm` | `str` | **Required** | Location of the quantum-mechanical training data. |
| `qmcode` | `Literal["aims"] \| Literal["cp2k"] \| Literal["pyscf"]` | **Required** | Ab initio software used to generate training data. |
| `dfbasis` | `str` | **Required** | A label for the auxiliary basis set used to expand the density. |
| `dfbasis_file` | `str` | Optional | Path to an external basis dataset yaml file holding `dfbasis`. If unset, the `basis_data.yaml` inside to the installed salted package is used. Mainly intended for testing purposes. |
| `qmbasis` | `str` | Required if `qmcode=pyscf` | Wavefunction basis set to use when generating the training data (only for PySCF). |
| `functional` | `str` | Required if `qmcode=pyscf` | DFT functional to use when generating the training data (only for PySCF). |
| `dfmetric` | `Literal["identity"] \| Literal["coulomb"]` | Required if `qmcode=cp2k` | Metric used to perform the density fitting (only for CP2K). `coulomb` enables the calculation of the Hartree energy in `salted.cp2k.density_fitting` and `salted.validation`. |
| `periodic` | `bool` | Required if `qmcode=cp2k` | Periodic boundary conditions (only for CP2K). |

### Rascaline atomic environment parameters `inp.descriptor.rep[n]`

| var name | type | default | usage |
| -:| :-: | :-: | :- |
| `type` | `Literal["rho"] \| Literal["V"]` | **Required** | Representation type, `"rho"` for atomic density and `"V"` for atomic potential. |
| `rcut` | `float` | **Required** | Radial cutoff (Angstrom) of the structural representation. |
| `nrad` | `int` | **Required** | Number of radial functions to be used for the structural representation. |
| `nang` | `int` | **Required** | Maximum angular momentum to be used for the structural representation. |
| `sig` | `float` | **Required** | Gaussian function width (Angstrom) for atomic density and/or derived atomic potential. |
| `neighspe` | `list[str]` | **Required** | List of atomic species to be used for structural representation. |

### Feature sparsification parameters `inp.descriptor.sparsify`

| var name | type | default | usage |
| -:| :-: | :-: | :- |
| `nsamples` | `int` | `100` | Number of structures to use for feature sparsification. |
| `ncut` | `int` | `0` | Sets maximum number of sparse (by FPS) descriptor features to retain. `0` for no sparsification. |

### Descriptor normalisation `inp.descriptor.normalisation`

| var name | type | default | usage |
| -:| :-: | :-: | :- |
| `normalisation` | `Literal["unit"] \| Literal["rel_power"]` | `"unit"` | Normalisation of each atom's lambda-SOAP vectors. `unit`: every lambda block is divided by its own length. `rel_power`: lambda = 0 is divided by its own length and every lambda > 0 block by the length of the same atom's lambda = 0 vector, so lambda > 0 features that vanish by site symmetry stay small near a symmetric site. With `rel_power` the lambda > 0 kernels are no longer of unit scale, so `gpr.eigcut` and `gpr.regul` act on them on a different scale. Training and prediction must use the same value. |

### Prediction variabls `inp.predict`

Remember to set `inp.predict` if one wants to predict densities.

| var name | type | default | usage |
| -:| :-: | :-: | :- |
| `filename` | `str` | Required if predict | An extended-XYZ file consisting of structures whose densities we wish to predict. |
| `predname` | `str` | Required if predict | A label to identify a particular set of predictions. |
| `predict_data` | `str` | Required if predict and `qmcode=aims` | Path to ab initio output for prediction, relative to path2qm. |
| `alpha_only` | `bool` | `False` | Whether to limit predictions to the subset of coefficients associated with the calculation of the polarizability tensor. Only usable in combintation with `saltedtype=density-response` and `qmcode=cp2k`. |

### ML (GPR) variables `inp.gpr`

| var name | type | default | usage |
| -:| :-: | :-: | :- |
| `z` | `float` | `2.0` | Kernel exponent $\zeta$. |
| `Menv` | `int` | **Required** | Number of reference environments. |
| `sparse_selection` | `Literal["fps"] \| Literal["hybrid"]` | `"fps"` | Rule that picks the `Menv` reference environments in `sparse_selection`, on the $\lambda=0$ rows of `FEAT-0.h5`, with one budget of `Menv` over all species. `"fps"`: farthest point sampling over all atoms pooled (squared Euclidean distance), the original selection. `"hybrid"`: kernel novelty, with the prediction kernel $k(a,b)=(\mathbf{x}_a\cdot\mathbf{x}_b)^\zeta$ within each species and the novelty $1 - k_{aR}K_{RR}^{-1}k_{Ra}/k(a,a)$ of an atom w.r.t. the references $R$ of its species (incremental Cholesky, see `salted/novelty_selection.py`). The first `sparse_hybrid_fraction` of the references (rounded half up) each go to the atom that removes the most novelty summed over the atoms of its species (estimated from 256 candidates and 4096 evaluation atoms per species and step, drawn with `salted.seed`), the rest to the most novel atom; each step goes to the species with the best candidate, so a species can end with very few references (a warning is printed below 2). |
| `sparse_hybrid_fraction` | `float` | `0.6` | `hybrid` only: share $f \in [0,1]$ of the references picked by summed novelty before the most-novel-atom tail. |
| `Ntrain` | `int` | **Required** | Number of training structures. |
| `trainfrac` | `float` | `1.0` | Training dataset fraction. Training dataset size is `Ntrain * trainfrac`. |
| `regul` | `float` | `1e-6` | Regularization parameter $\eta$. |
| `eigcut` | `float` | `1e-10` | Eigenvalues cutoff for RKHS projection. |
| `gradtol` | `float` | `1e-5` | Minimum gradient norm tolerance for CG minimization. |
| `restart` | `bool` | `False` | Whether to restart from the weights of the previous minimization checkpoint (CG restarts from their gradient). |
| `trainsel` | `Literal["sequential"] \| Literal["random"]` | `"random"` | Select the training set at random or sequentially from the entire dataset. |
| `matrix_dtype` | `Literal["float64"] \| Literal["float32"]` | `"float64"` | Storage type of the Psi values and the packed overlaps during `minimize_loss`, in RAM and in the spill file. `"float32"` halves their memory; all sums, the coefficients and the weights stay float64, and the loss, gradient and preconditioner are computed from the rounded matrices, so the fit solves the problem those matrices define. Needs `saltedtype: density` and `packed_overlap: true`. |
| `preconditioner` | `Literal["kronc"] \| Literal["jacobi"]` | `"kronc"` | Preconditioner of the CG weight fit in `minimize_loss` (with `fast_minimizer: true`). `"kronc"` couples all radial channels of one species and $\lambda$: per block the same-centre radial overlap $\otimes$ the kernel Gram matrix, plus $2\eta$, one Cholesky factor per radial eigenvalue, spread over the MPI ranks. It needs far fewer CG steps than `"jacobi"` (the diagonal) and reaches a lower loss at the same `gradtol`; its factors take $8 \sum n_\mathrm{max} n_c^2$ bytes in total. It needs `saltedtype: density` with the block Psi (`SALTED_PSI_BLOCKS` unset or 1); otherwise the fit falls back to `"jacobi"`. |
| `sparse_algorithm` | `Literal["dense"] \| Literal["numba"] \| Literal["omp_sparse"]` | `"numba"` | The algorithm to compute Hessian matrices using RKHS vector sparsity. `"numba"` uses JIT-compiled kernels (default). `"omp_sparse"` uses OpenMP Fortran extensions (requires `make`; falls back to `"numba"` if unavailable). `"dense"` disables sparse operations. |
| `Mcut` | `Literal["fixed"] \| Literal["gaussian"] \| list[int]` | `"fixed"` | Per-`lam` truncation of the sparsified atomic environments used as the RKHS basis. `"fixed"` keeps all `Menv` environments at every `lam`. `"gaussian"` uses `round(Menv * exp(-0.05 * lam**2))`. `list[int]` of length `>= max(lmax)+1` gives explicit per-`lam` counts. NB: all modes share the same on-disk storage cache layout; switching modes overwrites previously generated files. |

!!! note "Tuning the Numba cache budget for Hessian computation"
    When `sparse_algorithm: numba` is used, the L3 cache budget can be controlled via the environment variable `SALTED_NUMBA_CACHE_BYTES` (default: 32 MiB). For MPI runs, set it to approximately `total_L3 / ranks_per_node / 2`:
    ```bash
    export SALTED_NUMBA_CACHE_BYTES=$(( 512*1024*1024 / 16 / 2 ))  # 512 MiB L3, 16 ranks/node
    ```

---

## API

For details please check the source code.

::: salted.sys_utils.ParseConfig

<br> <!-- larger space -->

---
