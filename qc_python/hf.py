#!/usr/bin/env python
# coding: utf-8

"""
Author: Yudong Qiu
Solve the Schrodinger's Equation with Hartree-Fock method.
Read in atomic coordinates and basis set
Build Overlap matrix and Hamiltonian matrix
Build two electron integrals tensor G = <pq|rs>
Use core Hamiltonian as initial guess for Fock matrix
Solve FC = ESC for C and E
Use C to build density matrix
Use density matrix to build new Fock matrix
Iterate until the energy and density converge.
Get the HF result.
"""

import numpy as np
import math
import collections
import os
import time

from qc_python.common import read_xyz
from qc_python.basis_integrals import load_basis
from qc_python.rhf import solve_restricted_hartree_fock
from qc_python.uhf import solve_unrestricted_hartree_fock

this_file_folder = os.path.dirname(os.path.realpath(__file__))

basis_choices = [f[:-6] for f in os.listdir(os.path.join(this_file_folder, 'basis')) if f.endswith('.basis')]

def main():
    import argparse
    parser = argparse.ArgumentParser("Solve the Schrodinger's Equation for one-electron system.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('infile', help='Input xyz file containing the coordinates.')
    parser.add_argument('-b', '--basis', default='sto-3g', choices=basis_choices, help='basis set to use')
    parser.add_argument('-c', '--charge', default=0, type=int, help='charge of molecule')
    parser.add_argument('-s', '--spinmult', default=1, type=int, help='Spin multiplicity of molecule')
    parser.add_argument('-n', "--no_diis", action='store_true', default=False, help='Enable DIIS.')
    parser.add_argument('-u', "--unrestricted", action='store_true', default=False, help='Do unrestricted HF')
    #parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Verbose printing')
    args = parser.parse_args()

    # load the basis set
    basis_set = load_basis(args.basis)
    # read the xyz coordinates
    elems, coords = read_xyz(args.infile)
    # store starting time
    t0 = time.time()
    # solve the HF equations
    if not args.unrestricted:
        solve_restricted_hartree_fock(elems, coords, basis_set, charge=args.charge, enable_DIIS=(not args.no_diis), verbose=True)
    else:
        solve_unrestricted_hartree_fock(elems, coords, basis_set, charge=args.charge, spinmult=args.spinmult, enable_DIIS=(not args.no_diis), verbose=True)
    # print time used
    t1 = time.time()
    print("--- %.3f seconds ---" % (t1-t0))


if __name__ == "__main__":
    main()
