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
import basis_integrals

chemical_elements = ["", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]

def read_xyz(filename):
    """ Read an xyz file and return an elems list and a coords list """
    lines = open(filename).readlines()
    noa = int(lines[0])
    elems = []
    coords = []
    for line in lines[2:2+noa]:
        ls = line.split()
        elems.append(ls[0].lower().capitalize())
        coords.append(ls[1:4])
    coords = np.array(coords, dtype=float)
    # convert to atomic unit
    if 'bohr' not in lines[1].lower():
        ANGS_TO_BOHR = 1.8897261349925714
        coords *= ANGS_TO_BOHR
    return elems, coords

def write_xyz(elems, coords, filename):
    """ Write an xyz file."""
    BOHR_TO_ANGS = 0.529177208
    coords_in_angs = coords * BOHR_TO_ANGS
    with open(filename, 'w') as outfile:
        outfile.write('%d\n\n' % len(elems))
        for e, c in zip(elems, coords):
            x, y, z = c
            outfile.write('%-2s %13.7f %13.7f %13.7f\n' % (e, x, y, z))

def calc_nuclear_repulsion(elems, coords):
    nuc_charge_coords = [(chemical_elements.index(e), R) for e,R in zip(elems, coords)]
    noa = len(nuc_charge_coords)
    result = 0.0
    for i in range(noa):
        z_i, R_i = nuc_charge_coords[i]
        for j in range(i+1, noa):
            z_j, R_j = nuc_charge_coords[j]
            result += z_i * z_j / np.sqrt(np.sum((R_i - R_j)**2))
    return result

def solve_restricted_hartree_fock(elems, coords, basis_set, charge, verbose=True):
    """ Restricted Hartree-Fock """
    # compute nuclear repulsion energy
    E_nuc = calc_nuclear_repulsion(elems, coords)
    # compute one-electron integral matrices
    Smat, Tmat, Vmat = basis_integrals.build_one_e_matrices(elems, coords, basis_set)
    Hmat = Tmat + Vmat
    # build two-electron integral tensor g = (pq|rs)
    G_ao = basis_integrals.build_two_electron_tensor(elems, coords, basis_set)
    if verbose:
        print("One Electron Integrals Calculated:")
        print("\nOverlap matrix S")
        print(Smat)
        print("\nKinetic energy matrix T")
        print(Tmat)
        print("\nNuclear attraction matrix V")
        print(Vmat)
        print("\nCore Hamiltonian matrix H = T + V")
        print(Hmat)
        print("\nTwo-electron Integrals G")
        print(G_ao)

    # Compute the number of electrons in the system
    n_electron = sum(chemical_elements.index(e) for e in elems) - charge
    if n_electron%2 != 0:
        raise RuntimeError("RHF can not work with odd number of electrons!")
    # number of doubly occupied orbitals for RHF
    ndo = int(n_electron/2)

    # Diagonalize overlap matrix and form S^(-1/2) matrix
    Seigval, Seigvec = np.linalg.eigh(Smat)
    Shalf = np.diag(Seigval**-0.5)
    Shalf = np.dot(Seigvec, np.dot(Shalf, Seigvec.T))
    # Solve the one electron problem
    Hp = np.dot(Shalf, np.dot(Hmat, Shalf))
    e, Cp = np.linalg.eigh(Hp)
    Cmat = np.dot(Shalf, Cp)
    # sort the orbitals from low to high energy
    idx = e.argsort()
    e = e[idx]
    Cmat = Cmat[:, idx]
    # Build the initial Density Matrix
    C_do = Cmat[:, :ndo]
    Dmat = np.dot(C_do, C_do.T) * 2
    print Dmat
    # Build the initial Fock Matrix
    Fmat = Hmat + np.einsum('rs,pqrs->pq',Dmat,G_ao) - 0.5*np.einsum('rs,prqs->pq',Dmat,G_ao)
    Ehf = np.einsum('pq,pq', Dmat, Hmat+Fmat) * 0.5
    print "Initial Electronic Energy : %13.7f Eh" % Ehf
    print "Nuclear Repulsion         : %13.7f Eh" % E_nuc
    print "Initial Total Energy      : %13.7f Eh" % (Ehf + E_nuc)

def main():
    import argparse
    parser = argparse.ArgumentParser("Solve the Schrodinger's Equation for one-electron system.")
    parser.add_argument('infile', help='Input xyz file containing the coordinates.')
    parser.add_argument('-b', '--basis', default='sto-3g', help='basis set to use')
    parser.add_argument('-c', '--charge', default=0, type=int, help='charge of molecule')
    args = parser.parse_args()

    # load the basis set
    basis_set = basis_integrals.load_basis(args.basis)
    # read the xyz coordinates
    elems, coords = read_xyz(args.infile)
    solve_restricted_hartree_fock(elems, coords, basis_set, args.charge, verbose=True)


if __name__ == "__main__":
    main()
