"""Translate density fitting basis info from FHI-aims / CP2K to SALTED density fitting basis info.
This is just an entry point for the actual implementation.
in salted/aims/get_basis_info.py and salted/cp2k/get_basis_info.py.
"""

import argparse
import sys

from salted.sys_utils import ParseConfig


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="run without writing to files, and print the result",
    )
    parser.add_argument(
        "--force_overwrite",
        action="store_true",
        help="force overwrite the existing basis data",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    inp = ParseConfig().parse_input()
    if inp.qm.qmcode.lower() == "aims":
        from salted.aims.get_basis_info import build
    elif inp.qm.qmcode.lower() == "cp2k":
        from salted.cp2k.get_basis_info import build
    elif inp.qm.qmcode.lower() == "pyscf":
        from salted.pyscf.get_basis_info import build
    else:
        raise ValueError(f"Unknown qmcode: {inp.qm.qmcode}")

    build(dryrun=args.dryrun, force_overwrite=args.force_overwrite)
