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

def solve_restricted_hartree_fock(elems, coords, basis_set, charge, maxiter=150, enable_DIIS=True, verbose=True):
    """ Restricted Hartree-Fock """
    # Compute the number of electrons in the system
    n_electron = sum(chemical_elements.index(e) for e in elems) - charge
    if n_electron%2 != 0:
        raise RuntimeError("RHF can not work with odd number of electrons!")
    # number of doubly occupied orbitals for RHF
    ndo = int(n_electron/2)
    # compute nuclear repulsion energy
    E_nuc = calc_nuclear_repulsion(elems, coords)
    # compute one-electron integral matrices
    Smat, Tmat, Vmat = basis_integrals.build_one_e_matrices(elems, coords, basis_set)
    Hmat = Tmat + Vmat
    # check if we have enough basis functions to hold all the electrons
    if ndo > len(Smat):
        raise RuntimeError("Number of basis functions is smaller than number of doubly occupied orbitals")
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
    # Solve the FC = ESC equation by converting it to Ft C' = E C'
    # Diagonalize overlap matrix and form S^(-1/2) matrix
    Seigval, Seigvec = np.linalg.eig(Smat)
    Shalf = np.diag(Seigval**-0.5)
    Shalf = np.dot(Seigvec, np.dot(Shalf, Seigvec.T))
    # intial guess density
    Dmat = np.zeros_like(Smat)
    E_hf = 0
    converged = False
    # DIIS
    if enable_DIIS is True:
        n_err_mat = 6
        diis_start_n = 4
        diis_err_mats = []# np.zeros([n_err_mat, nbf, nbf], dypte=np.float64)
        diis_fmats = []# np.zeros([n_err_mat, nbf, nbf], dypte=np.float64)
        if verbose:
            print(" *** DIIS Enabled ***")
    if verbose:
        print("\n *** SCF Iterations *** ")
        print("Iter         HF Energy         delta E         RMS |D|")
        print("-------------------------------------------------------")
    for i in range(maxiter):
        Fmat = Hmat + np.einsum("rs,pqrs->pq",Dmat,G_ao) * 2 - np.einsum("rs,prqs->pq",Dmat,G_ao)
        if enable_DIIS and i > 0:
            FDS = np.einsum("pi,ij,jq->pq",Fmat,Dmat,Smat)
            SDF = np.einsum("pi,ij,jq->pq",Smat,Dmat,Fmat)
            diis_err_mats.append(FDS-SDF)
            diis_err_mats = diis_err_mats[-n_err_mat:]
            diis_fmats.append(Fmat.copy())
            diis_fmats = diis_fmats[-n_err_mat:]
            # compute Bmat_ij = Err_i . Err_j
            n_diis = len(diis_err_mats)
            if n_diis >= diis_start_n:
                Bmat = -np.ones([n_diis+1, n_diis+1])
                for di in range(n_diis):
                    for dj in range(di, n_diis):
                        Bmat[di,dj] = Bmat[dj,di] = np.dot(diis_err_mats[di].ravel(), diis_err_mats[dj].ravel())
                Bmat[-1,-1] = 0
                # Solve the equation Bmat * C = [0,0,..,-1]
                right_vec = np.zeros(n_diis+1)
                right_vec[-1] = -1
                C_array = np.linalg.solve(Bmat, right_vec)
                # Form the new guess Fmat
                new_Fmat = np.zeros_like(Fmat)
                for di in range(n_diis):
                    new_Fmat += C_array[di] * diis_fmats[di]
                Fmat = new_Fmat
        Ft = np.einsum("pi,ij,jq->pq",Shalf,Fmat,Shalf)
        Feigval, Feigvec = np.linalg.eigh(Ft)
        idx = Feigval.argsort()
        Feigval = Feigval[idx]
        Feigvec = Feigvec[:,idx]
        Cmat = np.dot(Shalf, Feigvec)
        C_do = Cmat[:, :ndo]
        Dmat_new = np.dot(C_do, C_do.T)
        E_hf_new = np.einsum("pq,pq", Dmat, Hmat+Fmat)
        dE = E_hf_new - E_hf
        D_rms = np.sqrt(np.mean((Dmat_new-Dmat)**2))
        # update E_hf and Dmat
        E_hf = E_hf_new
        Dmat = Dmat_new
        # print iteration information
        if verbose is True:
            print(" %-4d %17.10f  %14.4e  %14.4e" %(i, E_hf, dE, D_rms))
        # check convergence
        if abs(dE) < 1.0E-10 and abs(D_rms) < 1.0E-8:
            converged = True
            break
    if converged == False:
        print("SCF didn't converge in %d iterations!" % maxiter)
        raise RuntimeError
    E_total = E_nuc + E_hf
    if verbose:
        print("\nSCF converged!\n")
        print("Orbital Energies (Eh) and coefficients")
        print('E:  '+''.join(["%17.7f"%e for e in Feigval]))
        print('-' * (17 * len(Feigval) + 4))
        for i,row in enumerate(Cmat):
            print('c%-3d'%i + ''.join(["%17.7f"%c for c in row]))
        print("\nNuclear Repulsion Energy =   %17.10f   Eh" % E_nuc)
        print("Total Electronic Energy  =   %17.10f   Eh" % E_hf)
        print("Final Total Energy       =   %17.10f   Eh" % E_total)
    return {"E_nuc":E_nuc, "E_hf": E_hf, "E_total":E_total, "E_orbs": Feigval, "Cmat": Cmat, "Dmat": Dmat}

def main():
    import argparse
    parser = argparse.ArgumentParser("Solve the Schrodinger's Equation for one-electron system.")
    parser.add_argument('infile', help='Input xyz file containing the coordinates.')
    parser.add_argument('-b', '--basis', default='sto-3g', help='basis set to use')
    parser.add_argument('-c', '--charge', default=0, type=int, help='charge of molecule')
    parser.add_argument('-n', "--no_diis", action='store_true', default=False, help='Enable DIIS.')
    args = parser.parse_args()

    # load the basis set
    basis_set = basis_integrals.load_basis(args.basis)
    # read the xyz coordinates
    elems, coords = read_xyz(args.infile)
    solve_restricted_hartree_fock(elems, coords, basis_set, args.charge, enable_DIIS=(not args.no_diis), verbose=True)


if __name__ == "__main__":
    main()
