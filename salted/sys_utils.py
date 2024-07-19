# ruff: noqa: E501
import os
import re
import sys
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import yaml
from ase.io import read

from salted import basis


def read_system(filename:str=None, spelist:List[str]=None, dfbasis:str=None):
    """read a geometry file and return the formatted information

    Args:
        filename (str, optional): geometry file. Defaults to None.
        spelist (List[str], optional): list of species. Defaults to None.
        dfbasis (str, optional): density fitting basis. Defaults to None.

    Notes:
        By default (all parameters are None), it reads the geometry file for training dataset.
        If one wants to read other files, please specify all the parameters (filename, spelist, dfbasis).

    Returns:
        speclist (List[str]): list of species
        lmax (Dict[str, int]): maximum l for each species
        nmax (Dict[Tuple[str, int], int]): maximum n for each species and l
        llmax (int): maximum l in the system
        nnmax (int): maximum n in the system
        ndata (int): number of configurations
        atomic_symbols (List[List[str]]): list of atomic symbols for each configuration
        natoms (numpy.ndarray): number of atoms for each configuration, shape (ndata,)
        natmax (int): maximum number of atoms in the system
    """
    if (filename is None) and (spelist is None) and (dfbasis is None):
        inp = ParseConfig().parse_input()
        
        filename = inp.system.filename
        spelist = inp.system.species
        dfbasis = inp.qm.dfbasis
        if inp.qm.qmcode=="pyscf":
            from salted.pyscf.get_basis_info import get_aux_basis_name
            dfbasis = get_aux_basis_name(dfbasis)
    elif (filename is not None) and (spelist is not None) and (dfbasis is not None):
        pass
    else:
        raise ValueError(
            "Invalid parameters, should be either all None or all not None, "
            "please check the docstring for more details."
        )

    # read basis
    [lmax,nmax] = basis.basiset(dfbasis)
    llist = []
    nlist = []
    for spe in spelist:
        llist.append(lmax[spe])
        for l in range(lmax[spe]+1):
            nlist.append(nmax[(spe,l)])
    nnmax = max(nlist)
    llmax = max(llist)

    # read system
    xyzfile = read(filename, ":", parallel=False)
    ndata = len(xyzfile)

    # Define system excluding atoms that belong to species not listed in SALTED input
    atomic_symbols = []
    natoms = np.zeros(ndata,int)
    for iconf in range(len(xyzfile)):
        atomic_symbols.append(xyzfile[iconf].get_chemical_symbols())
        natoms_total = len(atomic_symbols[iconf])
        excluded_species = []
        for iat in range(natoms_total):
            spe = atomic_symbols[iconf][iat]
            if spe not in spelist:
                excluded_species.append(spe)
        excluded_species = set(excluded_species)
        for spe in excluded_species:
            atomic_symbols[iconf] = list(filter(lambda a: a != spe, atomic_symbols[iconf]))
        natoms[iconf] = int(len(atomic_symbols[iconf]))

    # Define maximum number of atoms
    natmax = max(natoms)

    return spelist, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax

def get_atom_idx(ndata,natoms,spelist,atomic_symbols):
    # initialize useful arrays
    atom_idx = {}
    natom_dict = {}
    for iconf in range(ndata):
        for spe in spelist:
            atom_idx[(iconf,spe)] = []
            natom_dict[(iconf,spe)] = 0
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            if spe in spelist:
               atom_idx[(iconf,spe)].append(iat)
               natom_dict[(iconf,spe)] += 1

    return atom_idx,natom_dict

def get_conf_range(rank,size,ntest,testrangetot):
    if rank == 0:
        testrange = [[] for _ in range(size)]
        blocksize = int(ntest/float(size))
#       print(ntest,blocksize)
        if isinstance(testrangetot, list):
            pass
        elif isinstance(testrangetot, np.ndarray):
            testrangetot = testrangetot.tolist()
        else:
            raise ValueError(
                f"Invalid type for testrangetot, "
                f"should be list or numpy.ndarray, but got {type(testrangetot)}"
            )
        for i in range(size):
            if i == (size-1):
                rem = ntest - (i+1)*blocksize
#               print(i,(i+1)*blocksize,rem)
                if rem < 0:
                    testrange[i] = testrangetot[i*blocksize:ntest]
                else:
                    testrange[i] = testrangetot[i*blocksize:(i+1)*blocksize]
                    for j in range(rem):
                        testrange[j].append(testrangetot[(i+1)*blocksize+j])
            else:
                testrange[i] = testrangetot[i*blocksize:(i+1)*blocksize]
#           print(i,len(testrange[i]))
    else:
        testrange = None

    return testrange


ARGHELP_INDEX_STR = """Indexes to calculate, start from 0. Format: 1,3-5,7-10. \
Default is "all", which means all structures."""

def parse_index_str(index_str:Union[str, Literal["all"]]) -> Union[None, Tuple]:
    """Parse index string, e.g. "1,3-5,7-10" -> (1,3,4,5,7,8,9,10)

    If index_str is "all", return None. (indicating all structures)
    If index_str is "1,3-5,7-10", return (1,3,4,5,7,8,9,10)
    """

    if index_str == "all":
        return None
    else:
        assert isinstance(index_str, str)
        indexes = []
        for s in index_str.split(","):  # e.g. ["1", "3-5", "7-10"]
            assert all([c.isdigit() or c == "-" for c in s]), f"Invalid index format: {s}"
            if "-" in s:
                assert s.count("-") == 1, f"Invalid index format: {s}"
                start, end = s.split("-")
                assert start.isdigit() and end.isdigit, f"Invalid index format: {s}"
                indexes.extend(range(int(start), int(end) + 1))
            elif s.isdigit():
                indexes.append(int(s))
            else:
                raise ValueError(
                    f"Invalid index format: {s}, "
                    f"should be digits or ranges joined by comma, e.g. 1,3-5,7-10"
                )
        return tuple(indexes)


def sort_grid_data(data:np.ndarray) -> np.ndarray:
    """Sort real space grid data
    The grid data is 2D array with 4 columns (x,y,z,value).
    Sort the grid data in the order of x, y, z.

    Args:
        data (np.ndarray): grid data, shape (n,4)

    Returns:
        np.ndarray: sorted grid data, shape (n,4)
    """
    assert data.ndim == 2
    assert data.shape[1] == 4
    data = data[np.lexsort((data[:,2], data[:,1], data[:,0]))]  # last key is primary
    return data

def get_feats_projs(species,lmax):
    import h5py
    import os.path as osp
    inp = ParseConfig().parse_input()
    Vmat = {}
    Mspe = {}
    power_env_sparse = {}
    sdir = osp.join(inp.salted.saltedpath, f"equirepr_{inp.salted.saltedname}")
    features = h5py.File(osp.join(sdir,f"FEAT_M-{inp.gpr.Menv}.h5"),'r')
    projectors = h5py.File(osp.join(sdir,f"projector_M{inp.gpr.Menv}_zeta{inp.gpr.z}.h5"),'r')
    for spe in species:
        for lam in range(lmax[spe]+1):
             # load RKHS projectors
             Vmat[(lam,spe)] = projectors["projectors"][spe][str(lam)][:]
             # load sparse equivariant descriptors
             power_env_sparse[(lam,spe)] = features["sparse_descriptors"][spe][str(lam)][:]
             if lam == 0:
                 Mspe[spe] = power_env_sparse[(lam,spe)].shape[0]
             # precompute projection on RKHS if linear model
             if inp.gpr.z==1:
                 power_env_sparse[(lam,spe)] = np.dot(
                     Vmat[(lam,spe)].T, power_env_sparse[(lam,spe)]
                 )
    features.close()
    projectors.close()

    return Vmat,Mspe,power_env_sparse

class AttrDict:
    """Access dict keys as attributes

    The attr trick only works for nested dicts.
    One just needs to wrap the dict in an AttrDict object.

    Example:
    ```python
    d = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    d = AttrDict(d)
    print(d.a)  # 1
    print(d.b.c)  # 2
    print(d.b.d.e)  # 3
    print(d.to_dict()["b"]["d"]["e"])  # 3
    ```
    """
    def __init__(self, d: dict):
        self._mydict = d

    def __repr__(self):
        def rec_repr(d: dict, offset=''):
            return '\n'.join([
                offset + f"{k}: {repr(v)}"
                if not isinstance(v, dict)
                else offset + f"{k}:\n{rec_repr(v, offset+'  ')}"
                for k, v in d.items()
            ])
        return rec_repr(self._mydict, offset='')

    def __getattr__(self, name):
        assert name in self._mydict.keys(), f"{name} not in {self._mydict.keys()}"
        value = self._mydict[name]
        if isinstance(value, dict):
            return AttrDict(value)
        else:
            return value

    def to_dict(self,):
        """get the original dict"""
        return self._mydict


PLACEHOLDER = "__PLACEHOLDER__"  # a unique placeholder for optional keys

class ParseConfig:
    """Input configuration file parser

    To use it, make sure an `inp.yaml` file exists in the current working directory,
    and simply run `ParseConfig().parse_input()`.

    In our context, "input file" equals to "confiuration file", refers to the SALTED input file named `inp.yaml`.
    """

    def __init__(self, _dev_inp_fpath: Optional[str]=None):
        """Initialize configuration parser

        Args:
            _dev_inp_fpath (Optional[str], optional): Path to the input file. Defaults to None.
                Don't use this argument, it's for testing only!!!
        """
        if _dev_inp_fpath is None:
            self.inp_fpath = os.path.join(os.getcwd(), 'inp.yaml')
        else:
            self.inp_fpath = _dev_inp_fpath
        assert os.path.exists(self.inp_fpath), (
            f"Missing compulsory input file. "
            f"Expected input file path: {self.inp_fpath}"
        )

    def parse_input(self) -> AttrDict:
        """Parse input file
        Procedure:
        - get loader (for constructors and resolvers)
        - load yaml

        Returns:
            AttrDict: Parsed input file
        """
        with open(self.inp_fpath) as file:
            inp = yaml.load(file, Loader=self.get_loader())
        if inp is None:
            raise ValueError(f"Input file is empty, please check the input file at path {self.inp_fpath}")
        inp = self.check_input(inp)
        return AttrDict(inp)

    def get_all_params(self) -> Tuple:
        """return all parameters with a tuple

        About `sparsify` in the return tuple:
            - If ncut <=0, sparsify = False.
            - If ncut > 0, sparsify = True.

        Please copy & paste:
        ```python
        (saltedname, saltedpath,
         filename, species, average, field, parallel,
         path2qm, qmcode, qmbasis, dfbasis,
         filename_pred, predname, predict_data,
         rep1, rcut1, sig1, nrad1, nang1, neighspe1,
         rep2, rcut2, sig2, nrad2, nang2, neighspe2,
         sparsify, nsamples, ncut,
         zeta, Menv, Ntrain, trainfrac, regul, eigcut,
         gradtol, restart, blocksize, trainsel) = ParseConfig().get_all_params()
        ```
        """
        inp = self.parse_input()
        if inp.qm.qmcode=="pyscf":
            from salted.pyscf.get_basis_info import get_aux_basis_name
            inp.qm.df_basis = get_aux_basis_name(inp.qm.qmbasis)
        sparsify = False if inp.descriptor.sparsify.ncut <= 0 else True  # determine if sparsify by ncut
        return (
            inp.salted.saltedname, inp.salted.saltedpath,
            inp.system.filename, inp.system.species, inp.system.average, inp.system.field, inp.system.parallel,
            inp.qm.path2qm, inp.qm.qmcode, inp.qm.qmbasis, inp.qm.dfbasis,
            inp.prediction.filename, inp.prediction.predname, inp.prediction.predict_data,
            inp.descriptor.rep1.type, inp.descriptor.rep1.rcut, inp.descriptor.rep1.sig,
            inp.descriptor.rep1.nrad, inp.descriptor.rep1.nang, inp.descriptor.rep1.neighspe,
            inp.descriptor.rep2.type, inp.descriptor.rep2.rcut, inp.descriptor.rep2.sig,
            inp.descriptor.rep2.nrad, inp.descriptor.rep2.nang, inp.descriptor.rep2.neighspe,
            sparsify, inp.descriptor.sparsify.nsamples, inp.descriptor.sparsify.ncut,
            inp.gpr.z, inp.gpr.Menv, inp.gpr.Ntrain, inp.gpr.trainfrac, inp.gpr.regul, inp.gpr.eigcut,
            inp.gpr.gradtol, inp.gpr.restart, inp.gpr.blocksize, inp.gpr.trainsel
        )

    def get_all_params_simple1(self) -> Tuple:
        """return all parameters with a tuple

        Please copy & paste:
        ```python
        (
            filename, species, average, field, parallel,
            rep1, rcut1, sig1, nrad1, nang1, neighspe1,
            rep2, rcut2, sig2, nrad2, nang2, neighspe2,
            sparsify, nsamples, ncut,
            z, Menv, Ntrain, trainfrac, regul, eigcut,
            gradtol, restart, blocksize, trainsel
        ) = ParseConfig().get_all_params_simple1()
        ```
        """
        inp = self.parse_input()
        sparsify = False if inp.descriptor.sparsify.ncut == 0 else True
        return (
            inp.system.filename, inp.system.species, inp.system.average, inp.system.field, inp.system.parallel,
            inp.descriptor.rep1.type, inp.descriptor.rep1.rcut, inp.descriptor.rep1.sig,
            inp.descriptor.rep1.nrad, inp.descriptor.rep1.nang, inp.descriptor.rep1.neighspe,
            inp.descriptor.rep2.type, inp.descriptor.rep2.rcut, inp.descriptor.rep2.sig,
            inp.descriptor.rep2.nrad, inp.descriptor.rep2.nang, inp.descriptor.rep2.neighspe,
            sparsify, inp.descriptor.sparsify.nsamples, inp.descriptor.sparsify.ncut,
            inp.gpr.z, inp.gpr.Menv, inp.gpr.Ntrain, inp.gpr.trainfrac, inp.gpr.regul, inp.gpr.eigcut,
            inp.gpr.gradtol, inp.gpr.restart, inp.gpr.blocksize, inp.gpr.trainsel
        )

    def get_sparsify_params(self) -> Tuple:
        """return sparsify parameters with a tuple

        Please copy & paste:
        ```python
        (
            nsamples, ncut
        ) = ParseConfig().get_sparsify_params()
        ```
        """
        inp = self.parse_input()
        return (
            inp.descriptor.sparsify.nsamples, inp.descriptor.sparsify.ncut, inp.descriptor.sparsify.forced_indices
        )

    def check_input(self, inp:Dict):
        """Check keys (required, optional, not allowed), and value types and ranges


        Format: (required, default value, value type, value extra check)

        About required:
            - True -> required
            - False -> optional, will fill in default value if not found
            - False + PLACEHOLDER -> optional in some cases, but required in others cases
            - (if the default value is $PLACEHOLDER, it means the key is optional for some cases, but required for others)

        About PLACEHOLDER:
            - If a key is optional in some cases, but required in others, the default value is set to PLACEHOLDER.
            - The extra value checking should consider the PLACEHOLDER value!

        About sparsify:
            - The config doesn't explicitly require the sparsify section, and ncut is 0 by default (don't sparsify).
        """

        rep_template = {
            "type": (True, None, str, lambda inp, val: val in ('rho', 'V')),  # descriptor, rho -> SOAP, V -> LODE
            "rcut": (True, None, float, lambda inp, val: val > 0),  # cutoff radius
            "sig": (True, None, float, lambda inp, val: val > 0),  # Gaussian width
            "nrad": (True, None, int, lambda inp, val: val > 0),  # number of radial basis functions
            "nang": (True, None, int, lambda inp, val: val > 0),  # number of angular basis functions
            "neighspe": (True, None, list, lambda inp, val: (
                all(isinstance(i, str) for i in val)
                # and
                # all(i in inp["system"]["species"] for i in val)  # species might be a subset of neighspe in Andrea's application
            )),  # list of neighbor species
        }
        inp_template = {  # NOQA
            "salted": {
                "saltedname": (True, None, str, None),  # salted workflow identifier
                "saltedpath": (True, None, str, lambda inp, val: check_path_exists(val)),  # path to SALTED outputs / working directory
            },
            "system": {
                "filename": (True, None, str, lambda inp, val: check_path_exists(val)),  # path to geometry file (training set)
                "species": (True, None, list, lambda inp, val: all(isinstance(i, str) for i in val)),  # list of species in all geometries
                "average": (False, True, bool, None),  # if bias the GPR by the average of predictions
                "field": (False, False, bool, None),  # if predict the field response
                "parallel": (False, False, bool, None),  # if use mpi4py
                "seed": (False, 42, int, None),  # random seed
            },
            "qm": {
                "path2qm": (True, None, str, lambda inp, val: check_path_exists(val)),  # path to the QM calculation outputs
                "qmcode": (True, None, str, lambda inp, val: val.lower() in ('aims', 'pyscf', 'cp2k')),  # quantum mechanical code
                "dfbasis": (False, PLACEHOLDER, str, None),  # density fitting basis, Only not required for PySCF
                #### below are optional, but required for some qmcode ####
                "qmbasis": (False, PLACEHOLDER, str, lambda inp, val: check_with_qmcode(inp, val, "pyscf")),  # quantum mechanical basis, only for PySCF
                "functional": (False, PLACEHOLDER, str, lambda inp, val: check_with_qmcode(inp, val, "pyscf")),  # quantum mechanical functional, only for PySCF
                "solvent_eps" : (False, 1.0, float, lambda inp, val: solvant_to_eps(val) > 0),  # solvent dielectric constant, only for PySCF
                "pseudocharge": (False, PLACEHOLDER, float, lambda inp, val: check_with_qmcode(inp, val, "cp2k")),  # pseudo nuclear charge, only for CP2K
                "coeffile": (False, PLACEHOLDER, str, lambda inp, val: check_with_qmcode(inp, val, "cp2k")),
                "ovlpfile": (False, PLACEHOLDER, str, lambda inp, val: check_with_qmcode(inp, val, "cp2k")),
                "periodic": (False, PLACEHOLDER, str, lambda inp, val: check_with_qmcode(inp, val, "cp2k")),  # periodic boundary conditions, only for CP2K
            },
            "prediction": {
                "filename": (False, PLACEHOLDER, str, None),  # path to the prediction file
                "predname": (False, PLACEHOLDER, str, None),  # SALTED prediction identifier
                #### below are optional, but required for some qmcode ####
                "predict_data": (False, PLACEHOLDER, str, lambda inp, val: check_with_qmcode(inp, val, "aims")),  # path to the prediction data by QM code, only for AIMS
            },
            "descriptor": {
                "rep1": rep_template,  # descriptor 1
                "rep2": rep_template,  # descriptor 2
                "sparsify": {
                    "nsamples": (False, 100, int, lambda inp, val: val > 0),  # number of samples for sparsifying feature channel
                    "ncut": (False, 0, int, lambda inp, val: (val == 0) or (val > 0)),  # number of features to keep
                    "forced_indices": (False, [], list, None), # indices to force inclusion in the sparsified set
                }
            },
            "gpr": {
                "z": (False, 2.0, float, lambda inp, val: val > 0),  # kernel exponent
                "Menv": (True, None, int, lambda inp, val: val > 0),  # number of sparsified atomic environments
                "Ntrain": (True, None, int, lambda inp, val: val > 0),  # size of the training set (rest is validation set)
                "trainfrac": (False, 1.0, float, lambda inp, val: (val > 0) and (val <= 1.0)),  # fraction of the training set used for training
                "regul": (False, 1e-6, float, lambda inp, val: val > 0),  # regularisation parameter
                "eigcut": (False, 1e-10, float, lambda inp, val: val > 0),  # eigenvalues cutoff
                "gradtol": (False, 1e-5, float, lambda inp, val: val > 0),  # min gradient as stopping criterion for CG minimization
                "restart": (False, False, bool, lambda inp, val: isinstance(val, bool)),  # if restart the minimization
                "blocksize": (False, 0, int, lambda inp, val: val >= 0),  # block size for matrix inversion
                "trainsel": (False, 'random', str, lambda inp, val: val in ('random', 'sequential')),  # if shuffle the training set
            }
        }

        def rec_apply_default_vals(_inp:Dict, _inp_template:Dict[str, Union[Dict, Tuple]], _prev_key:str):
            """apply default values if optional parameters are not found"""

            """check if the keys in inp exist in inp_template"""
            for key, val in _inp.items():
                if key not in _inp_template.keys():
                    raise ValueError(f"Invalid input key: {_prev_key+key}. Please remove it from the input file.")
            """apply default values"""
            for key, val in _inp_template.items():
                if isinstance(val, dict):
                    """we can ignore a section if it's not required"""
                    if key not in _inp.keys():
                        _inp[key] = dict()  # make it an empty dict
                    _inp[key] = rec_apply_default_vals(_inp[key], _inp_template[key], _prev_key+key+".")  # and apply default values
                elif isinstance(val, tuple):
                    (required, val_default, val_type, extra_check_func) = val
                    if key not in _inp.keys():
                        if required:
                            raise ValueError(f"Missing compulsory input key: {_prev_key+key} in the input file.")
                        else:
                            _inp[key] = val_default
                else:
                    raise ValueError(f"Invalid input template: {val}. Did you changed the template for parsing?")
            return _inp

        def rec_check_vals(_inp:Dict, _inp_template:Dict[str, Union[Dict, Tuple]], _prev_key:str):
            """check values' type and range"""
            for key, template in _inp_template.items():
                if isinstance(template, dict):
                    rec_check_vals(_inp[key], _inp_template[key], _prev_key+key+".")
                elif isinstance(template, tuple):
                    val = _inp[key]
                    (required, val_default, val_type, extra_check_func) = _inp_template[key]
                    """
                    There are cases that a value is required for certain conditions,
                    so we always need to run extra_check_func
                    """
                    if (not isinstance(val, val_type)) and (val != PLACEHOLDER):  # if is PLACEHOLDER, then don't check the type
                        if not (key == "solvent_eps" and isinstance(val, str)):
                            raise ValueError(
                                f"Incorrect input value type: key={_prev_key+key}, value={val}, "
                                f"current value type is {type(val)}, but expected {val_type}"
                            )
                    if extra_check_func is not None:  # always run extra_check_func if not None
                        if not extra_check_func(inp, val):
                            if hasattr(extra_check_func, "parse_error_msg"):
                                parse_error_msg = extra_check_func.parse_error_msg
                            else:
                                parse_error_msg = ""
                            raise ValueError(
                                f"Input value failed its check: key={_prev_key+key}, value={val}. "
                                f"{parse_error_msg}"
                                f"Please check the required conditions."
                            )
                else:
                    raise ValueError(f"Invalid input template type: {template} of type {type(template)}")
                
                #Bad hack because i hate this framework
                if key == "solvent_eps":
                    inp["qm"]["solvent_eps"] = solvant_to_eps(inp["qm"]["solvent_eps"])

        inp = rec_apply_default_vals(inp, inp_template, "")  # now inp has all the keys as in inp_template in all levels
        rec_check_vals(inp, inp_template, "") # check the values' type and range

        return inp

    def get_loader(self) -> yaml.SafeLoader:
        """Add constructors to the yaml.SafeLoader
        For details, see: https://pyyaml.org/wiki/PyYAMLDocumentation
        """
        loader = yaml.SafeLoader

        """for path concatenation, like !join_path [a, b, c] -> a/b/c"""
        def join_path(loader: yaml.SafeLoader, node: yaml.Node) -> str:
            seq = loader.construct_sequence(node)
            return os.path.join(*seq)
        loader.add_constructor('!join_path', join_path)

        """for scientific notation, like 1e-4 -> float(1e-4)"""
        pattern = re.compile(r'^-?[0-9]+(\.[0-9]*)?[eEdD][-+]?[0-9]+$')
        loader.add_implicit_resolver('!float_sci', pattern, list(u'-+0123456789'))  # NOQA
        def float_sci(loader, node):
            value = loader.construct_scalar(node)
            return float(value)
        loader.add_constructor('!float_sci', float_sci)
        return loader


def solvant_to_eps(solvant:str|float) -> float:
    if isinstance(solvant, float):
        if solvant <= 0.0:
            raise ValueError(f"Invalid dielectric constant: {solvant}, should be > 0")
        return solvant
    solvant_eps = {'water': 78.3553,
 'acetonitrile': 35.688,
 'methanol': 32.613,
 'ethanol': 24.852,
 'isoquinoline': 11.0,
 'quinoline': 9.16,
 'chloroform': 4.7113,
 'diethylether': 4.24,
 'dichloromethane': 8.93,
 'dichloroethane': 10.125,
 'carbontetrachloride': 2.228,
 'benzene': 2.2706,
 'toluene': 2.3741,
 'chlorobenzene': 5.6968,
 'nitromethane': 36.562,
 'heptane': 1.9113,
 'cyclohexane': 2.0165,
 'aniline': 6.8882,
 'acetone': 20.493,
 'tetrahydrofuran': 7.4257,
 'dimethylsulfoxide': 46.826,
 'argon': 1.43,
 'krypton': 1.519,
 'xenon': 1.706,
 'n-octanol': 9.8629,
 '1,1,1-trichloroethane': 7.0826,
 '1,1,2-trichloroethane': 7.1937,
 '1,2,4-trimethylbenzene': 2.3653,
 '1,2-dibromoethane': 4.9313,
 '1,2-ethanediol': 40.245,
 '1,4-dioxane': 2.2099,
 '1-bromo-2-methylpropane': 7.7792,
 '1-bromooctane': 5.0244,
 '1-bromopentane': 6.269,
 '1-bromopropane': 8.0496,
 '1-butanol': 17.332,
 '1-chlorohexane': 5.9491,
 '1-chloropentane': 6.5022,
 '1-chloropropane': 8.3548,
 '1-decanol': 7.5305,
 '1-fluorooctane': 3.89,
 '1-heptanol': 11.321,
 '1-hexanol': 12.51,
 '1-hexene': 2.0717,
 '1-hexyne': 2.615,
 '1-iodobutane': 6.173,
 '1-iodohexadecane': 3.5338,
 '1-iodopentane': 5.6973,
 '1-iodopropane': 6.9626,
 '1-nitropropane': 23.73,
 '1-nonanol': 8.5991,
 '1-pentanol': 15.13,
 '1-pentene': 1.9905,
 '1-propanol': 20.524,
 '2,2,2-trifluoroethanol': 26.726,
 '2,2,4-trimethylpentane': 1.9358,
 '2,4-dimethylpentane': 1.8939,
 '2,4-dimethylpyridine': 9.4176,
 '2,6-dimethylpyridine': 7.1735,
 '2-bromopropane': 9.361,
 '2-butanol': 15.944,
 '2-chlorobutane': 8.393,
 '2-heptanone': 11.658,
 '2-hexanone': 14.136,
 '2-methoxyethanol': 17.2,
 '2-methyl-1-propanol': 16.777,
 '2-methyl-2-propanol': 12.47,
 '2-methylpentane': 1.89,
 '2-methylpyridine': 9.9533,
 '2-nitropropane': 25.654,
 '2-octanone': 9.4678,
 '2-pentanone': 15.2,
 '2-propanol': 19.264,
 '2-propen-1-ol': 19.011,
 '3-methylpyridine': 11.645,
 '3-pentanone': 16.78,
 '4-heptanone': 12.257,
 '4-methyl-2-pentanone': 12.887,
 '4-methylpyridine': 11.957,
 '5-nonanone': 10.6,
 'aceticacid': 6.2528,
 'acetophenone': 17.44,
 'a-chlorotoluene': 6.7175,
 'anisole': 4.2247,
 'benzaldehyde': 18.22,
 'benzonitrile': 25.592,
 'benzylalcohol': 12.457,
 'bromobenzene': 5.3954,
 'bromoethane': 9.01,
 'bromoform': 4.2488,
 'butanal': 13.45,
 'butanoicacid': 2.9931,
 'butanone': 18.246,
 'butanonitrile': 24.291,
 'butylamine': 4.6178,
 'butylethanoate': 4.9941,
 'carbondisulfide': 2.6105,
 'cis-1,2-dimethylcyclohexane': 2.06,
 'cis-decalin': 2.2139,
 'cyclohexanone': 15.619,
 'cyclopentane': 1.9608,
 'cyclopentanol': 16.989,
 'cyclopentanone': 13.58,
 'decalin-mixture': 2.196,
 'dibromomethane': 7.2273,
 'dibutylether': 3.0473,
 'diethylamine': 3.5766,
 'diethylsulfide': 5.723,
 'diiodomethane': 5.32,
 'diisopropylether': 3.38,
 'dimethyldisulfide': 9.6,
 'diphenylether': 3.73,
 'dipropylamine': 2.9112,
 'e-1,2-dichloroethene': 2.14,
 'e-2-pentene': 2.051,
 'ethanethiol': 6.667,
 'ethylbenzene': 2.4339,
 'ethylethanoate': 5.9867,
 'ethylmethanoate': 8.331,
 'ethylphenylether': 4.1797,
 'fluorobenzene': 5.42,
 'formamide': 108.94,
 'formicacid': 51.1,
 'hexanoicacid': 2.6,
 'iodobenzene': 4.547,
 'iodoethane': 7.6177,
 'iodomethane': 6.865,
 'isopropylbenzene': 2.3712,
 'm-cresol': 12.44,
 'mesitylene': 2.265,
 'methylbenzoate': 6.7367,
 'methylbutanoate': 5.5607,
 'methylcyclohexane': 2.024,
 'methylethanoate': 6.8615,
 'methylmethanoate': 8.8377,
 'methylpropanoate': 6.0777,
 'm-xylene': 2.3478,
 'n-butylbenzene': 2.36,
 'n-decane': 1.9846,
 'n-dodecane': 2.006,
 'n-hexadecane': 2.0402,
 'n-hexane': 1.8819,
 'nitrobenzene': 34.809,
 'nitroethane': 28.29,
 'n-methylaniline': 5.96,
 'n-methylformamide-mixture': 181.56,
 'n,n-dimethylacetamide': 37.781,
 'n,n-dimethylformamide': 37.219,
 'n-nonane': 1.9605,
 'n-octane': 1.9406,
 'n-pentadecane': 2.0333,
 'n-pentane': 1.8371,
 'n-undecane': 1.991,
 'o-chlorotoluene': 4.6331,
 'o-cresol': 6.76,
 'o-dichlorobenzene': 9.9949,
 'o-nitrotoluene': 25.669,
 'o-xylene': 2.5454,
 'pentanal': 10.0,
 'pentanoicacid': 2.6924,
 'pentylamine': 4.201,
 'pentylethanoate': 4.7297,
 'perfluorobenzene': 2.029,
 'p-isopropyltoluene': 2.2322,
 'propanal': 18.5,
 'propanoicacid': 3.44,
 'propanonitrile': 29.324,
 'propylamine': 4.9912,
 'propylethanoate': 5.5205,
 'p-xylene': 2.2705,
 'pyridine': 12.978,
 'sec-butylbenzene': 2.3446,
 'tert-butylbenzene': 2.3447,
 'tetrachloroethene': 2.268,
 'tetrahydrothiophene-s,s-dioxide': 43.962,
 'tetralin': 2.771,
 'thiophene': 2.727,
 'thiophenol': 4.2728,
 'trans-decalin': 2.1781,
 'tributylphosphate': 8.1781,
 'trichloroethene': 3.422,
 'triethylamine': 2.3832,
 'xylene-mixture': 2.3879,
 'z-1,2-dichloroethene': 9.2,
 'vacuum': 1.0}    
    
    solvant = solvant.lower()
    if solvant not in solvant_eps:
        print(f"Invalid solvant: {solvant}. Did you mean:")
        for possible_solvent in solvant_eps:
            if possible_solvent.find(solvant) != -1:
                print(possible_solvent)
        raise ValueError(f"Invalid solvant: {solvant}. Please check the solvant name.")
    
    return solvant_eps[solvant.lower()]

def check_with_qmcode(inp, val, qmcode:Union[str, List[str]]) -> bool:
    """This means the entry is required IF and ONLY IF when using a / some specific qmcode(s)"""
    if isinstance(qmcode, str):
        qmcode = [qmcode,]
    return (
        ((inp["qm"]["qmcode"].lower() not in qmcode) and (val == PLACEHOLDER))  # if not using this qmcode, do not specify it
        or
        ((inp["qm"]["qmcode"].lower() in qmcode) and (val != PLACEHOLDER))  # if using this qmcode, do specify it
    )
check_with_qmcode.parse_error_msg = "Value imcompatible with the qm.qmcode."

def check_path_exists(path:str) -> bool:
    """Check if the path exists, the path should be either absolute or relative to the current working directory
    If the path is a placeholder, return True
    """
    if path == PLACEHOLDER:
        return True
    else:
        return os.path.exists(path)
check_path_exists.parse_error_msg = "Path (value) does not exist."

def test_inp():
    """test your `inp.yaml` file by trying to parse it"""
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    print(f"Try to parse the input file: {os.path.join(cwd, 'inp.yaml')}")
    inp = ParseConfig().parse_input()
    print(inp)
    print("Input file parsed successfully.")


class Irreps(tuple):
    """Handle irreducible representation arrays, like slices, multiplicities, etc."""

    def __new__(cls, irreps:Union[str, List[int], Tuple[int]]) -> 'Irreps':
        """Create an Irreps object

        Args:
            irreps (Union[str, List[int], Tuple[int]]): irreps info
                - str, e.g. `1x0+2x1+3x2+3x3+2x4+1x5`
                    - multiplicities and l values joined by `x`
                - Tuple[Tuple[int]], e.g. ((1, 0), (2, 1), (3, 2), (3, 3), (2, 4), (1, 5),)
                    - each tuple is (multiplicity, l)
                - Tuple[int], e.g. (0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5,)
                    - list of l values, the multiplicities are calculated automatically

        Notes:
            The internal representation is a tuple of tuples, each tuple is (multiplicity, l).
            e.g. ((1, 0), (2, 1), (3, 2), (3, 3), (2, 4), (1, 5),)

        Returns:
            Irreps object
        """
        if isinstance(irreps, str):
            irreps_info_split = tuple(sec.strip() for sec in irreps.split("+") if len(sec) > 0)  # ("1x0", "2x1", ...)
            mul_l_tuple = tuple(  # ((1, 0), (2, 1), ...)
                tuple(int(i.strip()) for i in sec.split("x"))
                for sec in irreps_info_split
            )
            return super().__new__(cls, mul_l_tuple)
        elif isinstance(irreps, list) or isinstance(irreps, tuple):
            if len(irreps) == 0:
                return super().__new__(cls, ())
            elif isinstance(irreps[0], tuple) or isinstance(irreps[0], list):
                assert all(
                    all(isinstance(i, int) for i in mul_l) and len(mul_l) == 2 and mul_l[0] >= 0 and mul_l[1] >= 0
                    for mul_l in irreps
                ), ValueError(f"Invalid irreps_info: {irreps}")
                return super().__new__(cls, tuple(tuple(mul_l) for mul_l in irreps))
            elif isinstance(irreps[0], int):
                assert all(isinstance(i, int) and i >= 0 for i in irreps), ValueError(f"Invalid irreps format: {irreps}")
                this_l_cnt, this_l = 1, irreps[0]
                mul_l_list:List[Tuple[int]] = []
                for l in irreps[1:]:
                    if l == this_l:
                        this_l_cnt += 1
                    else:
                        mul_l_list.append((this_l_cnt, this_l))
                        this_l_cnt, this_l = 1, l
                mul_l_list.append((this_l_cnt, this_l))
                print(mul_l_list)
                return super().__new__(cls, tuple(mul_l_list))
            else:
                raise ValueError(f"Invalid irreps format: {irreps}")
        else:
            raise ValueError(f"Invalid irreps format: {irreps}")

    @property
    def dim(self):
        """total dimension / length by magnetic quantum number"""
        return sum(mul * (2*l + 1) for mul, l in self)

    @property
    def num_irreps(self):
        """number of irreps, the sum of multiplicities of each l"""
        return sum(mul for mul, _ in self)

    @property
    def ls(self) -> List[int]:
        """list of l values in the irreps"""
        return tuple(l for mul, l in self for _ in range(mul))

    @property
    def lmax(self) -> int:
        """maximum l in the irreps"""
        return max(tuple(l for _, l in self))

    def __repr__(self):
        return "+".join(f"{mul}x{l}" for mul, l in self)

    def __add__(self, other: 'Irreps') -> 'Irreps':
        return Irreps(super().__add__(other))

    def slices(self) -> List[slice]:
        """return all the slices for each l"""
        if hasattr(self, "_slices"):
            return self._slices
        else:
            self._slices = []
            ls = self.ls
            l_m_nums = tuple(2*l + 1 for l in ls)
            pointer = 0
            for m_num in l_m_nums:
                self._slices.append(slice(pointer, pointer+m_num))
                pointer += m_num
            assert pointer == self.dim
            self._slices = tuple(self._slices)
        return self._slices

    def slices_l(self, l:int) -> List[slice]:
        """return all the slices for a specific l"""
        return tuple(sl for _l, sl in zip(self.ls, self.slices()) if l == _l)

    def simplify(self) -> 'Irreps':
        """sort by l, and combine the same l"""
        uniq_ls = tuple(set(self.ls))
        mul_ls = tuple((self.ls.count(l), l) for l in uniq_ls)
        return Irreps(mul_ls)

    def sort(self) -> 'Irreps':
        """sort by l, return the sorted Irreps and the permutation"""
        raise NotImplementedError
