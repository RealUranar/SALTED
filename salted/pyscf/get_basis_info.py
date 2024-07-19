"""Translate basis info from PySCF calculation to SALTED basis info"""

import os, sys, glob
from typing import Dict, List, Tuple

import yaml
from ase.io import read

from pyscf import df
from pyscf.gto import basis, BasisNotFoundError

from salted.basis_client import (
    BasisClient,
    SpeciesBasisData,
)
from salted.get_basis_info import get_parser
from salted.sys_utils import ParseConfig


def build(dryrun: bool = False, force_overwrite: bool = False):
    """Scheme: load density fitting basis from pyscf module,
    update the basis_data dict,
    and write to the database when all species are recorded.
    """
    inp = ParseConfig().parse_input()
    assert inp.qm.qmcode.lower() == "pyscf", f"{inp.qm.qmcode=}, but expected 'pyscf'"

    spe_set = set(inp.system.species)  # remove duplicates
    qmbasis = inp.qm.qmbasis

    """load density fitting basis from pyscf module"""
    try:
        dfbasis = df.addons.DEFAULT_AUXBASIS[basis._format_basis_name(qmbasis)][0]  # get the proper DF basis name in PySCF
        print(f"{spe_set=}, {qmbasis=}, and the parsed {dfbasis=}")
        basis_data: Dict[str, SpeciesBasisData] = load_from_pyscf(list(spe_set), dfbasis)
    except KeyError:
        print("Selected qmbasis is not available in PySCF. Guessing this is self defined.")
        basis_data: Dict[str, SpeciesBasisData] = load_from_file(list(spe_set), qmbasis)
    

    """write to the database"""
    if dryrun:
        print("Dryrun mode, not writing to the database")
        print(f"{basis_data=}")
    else:
        BasisClient().write(get_aux_basis_name(qmbasis), basis_data, force_overwrite)




def load_from_pyscf(species_list: List[str], dfbasis: str) -> Dict[str, "SpeciesBasisData"]:
    """
    Load the xxx-jkfit density fitting basis from PySCF.

    Args:
        species_list: List of species, e.g., ['H', 'O'].
        dfbasis: Quantum chemistry basis set name, e.g., 'cc-pvdz'.

    Returns:
        Dict[str, SpeciesBasisData]: Species and basis data.
    """
    spe_dfbasis_info = {spe: basis.load(dfbasis, spe) for spe in species_list}
    
    # Extract the l numbers and compose the Dict[str, SpeciesBasisData] (species and basis data)
    basis_data = {spe: collect_l_nums(ribasis_info) for spe, ribasis_info in spe_dfbasis_info.items()}
    return basis_data

def find_basis_file(qmbasis: str, fit: bool = True) -> Tuple[str, str]:
    """
    Return the file path of the basis set file and the name of the basis set.

    Args:
        qmbasis: Quantum chemistry basis set name.
        fit: Boolean to select fit or non-fit basis set file.

    Returns:
        Tuple[str, str]: File path of the basis set file and the name of the basis set.
    """
    basis_files = glob.glob(os.path.join('/basis_data', '*'))
    basis_files = [os.path.basename(f) for f in basis_files if qmbasis in f]
    basis_file = [file_name for file_name in basis_files if ("fit" in file_name) == fit][0]
    return os.path.join('/basis_data', basis_file), basis_file.split(".")[0]

def read_basis(qmbasis: str, species: List[str] = None, fit = True) -> Dict[str, List]:
    """
    Read basis data for given species from a file.

    Args:
        qmbasis: Quantum chemistry basis set name.
        species: List of species, e.g., ['H', 'O']. If None, defaults to a predefined list.

    Returns:
        Dict[str, List]: Species and basis data.
    """
    if species is None:
        species = [
            "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",
            "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce",
            "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir",
            "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
            "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc",
            "Lv", "Ts", "Og"
        ]
    
    basis_data = {}
    file_dest, dfbasis = find_basis_file(qmbasis, fit)
    with open(file_dest, "r") as f:
        file_cont = f.read()
    
    for spe in species:
        try:
            basis_data[spe] = basis.parse(file_cont, spe)
        except BasisNotFoundError:
            break
    return basis_data

def load_from_file(species_list: List[str], qmbasis: str) -> Dict[str, "SpeciesBasisData"]:
    """
    Load basis data from a file (NWChem preferred, CP2K?, Gaussian?).

    Args:
        species_list: List of species, e.g., ['H', 'O'].
        qmbasis: Quantum chemistry basis set name, e.g., 'cc-pvdz'.

    Returns:
        Dict[str, SpeciesBasisData]: Species and basis data.
    """
    spe_dfbasis_info = read_basis(qmbasis, species_list)
    
    # Extract the l numbers and compose the Dict[str, SpeciesBasisData] (species and basis data)
    basis_data = {spe: collect_l_nums(dfbasis_info) for spe, dfbasis_info in spe_dfbasis_info.items()}
    return basis_data


# def collect_l_nums(data:List[int, List[float]]) -> SpeciesBasisData:
# use Annotated
def collect_l_nums(data: List) -> SpeciesBasisData:
    """collect l numbers for each species based on the data from PySCF
    input: above dict value,
        e.g.
        [
            [
                0,
                [883.9992943, 0.33024477],
                [286.8428015, 0.80999791],
            ],
            [0, [48.12711454, 1.0]],
            [1, [102.99176249, 1.0]],
            [2, [10.59406836, 1.0]],
            ...
        ]
        there might be multiple [exponents, coefficients] for one atomic orbital
    output: max l number, and a list of counts of each l number
    """
    l_nums = [d for d, *_ in data]  # [0, 0, 0, 0, 1, 1, 1, 2, 2, ...]
    l_max = max(l_nums)
    l_cnt = [0 for _ in range(l_max + 1)]  # [0, 0, 0, ...] to the max l number
    for l_num in l_nums:
        l_cnt[l_num] += 1
    return {
        "lmax": max(l_nums),
        "nmax": l_cnt,
    }

def get_aux_basis_name(qmbasis: str) -> str:
    """get the auxiliary basis name for density fitting from PySCF
    Args:
        qmbasis: quantum chemistry basis set name, e.g. cc-pvdz
    Returns:
        str: auxiliary basis name
    """
    try:
        ribasis = df.addons.DEFAULT_AUXBASIS[basis._format_basis_name(qmbasis)][0]
    except KeyError:
        import glob
        print("Selected qmbasis is not available in PySCF. Guessing this is self defined.")
        ribasis_path, ribasis = find_basis_file(qmbasis)
    return ribasis

if __name__ == "__main__":
    print("Please call `python -m salted.get_basis_info` instead of this file")

    parser = get_parser()
    args = parser.parse_args()

    build(dryrun=args.dryrun, force_overwrite=args.force_overwrite)

