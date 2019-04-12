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
import scipy.linalg
import math
import collections
import os

chemical_elements = ["", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]
ContractedBasisFunction = collections.namedtuple('ContractedBasisFunction', ['ftype', 'coefs'])

def load_basis(basis_name):
    """ Read a basis set file in the NWCHEM format
    The files can be directely downloaded from https://bse.pnl.gov/bse/portal
    And save as NAME.basis

    Input
    ------
    basis_name: name of the basis set, e.g. "sto-3g", "6-31g"

    Output
    ------
    basis_set: dict() object contains the basis set information
        element as key, value is a list of namedtuples, each representing a contracted Gaussian function.
        Example: (from 3-21G)
        basis_set['H'] = [
            (ftype: 'S', coefs:[[5.44, 0.156], [0.824, 0.904]]),
            (ftype: 'S', coefs:[[0.183, 1.000]])
        ]
        Usage:
        basis_set['H'][0].type  --->  'S'
        basis_set['H'][1].coefs --->  [[0.183, 1.000]]
    """
    basis_set = dict()
    reading = False
    current_function = None
    basis_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','basis')
    for line in open(os.path.join(basis_folder,basis_name.lower()+'.basis')):
        if line.startswith("#"):
            continue
        if line.startswith("BASIS"):
            reading = True
        if reading is True:
            if line.startswith("END"):
                reading = False
                continue
            ls = line.split()
            if len(ls) == 2:
                if ls[1].isalpha(): # if title line like "H   S"
                    elem, spdf = ls
                    if elem not in basis_set:
                        basis_set[elem] = []
                    current_function = ContractedBasisFunction(ftype=spdf, coefs=[])
                    basis_set[elem].append(current_function)
                else: # if value line like "      5.4471780              0.1562850"
                    a, c = float(ls[0]), float(ls[1])
                    # normalize Gaussian (2a/pi)^3/4 * exp(-a*r^2)
                    norm = (2.0 * a / np.pi)**0.75
                    current_function.coefs.append((a, c*norm))
    return basis_set

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

def build_one_e_matrices(elems, coords, basis_set):
    """ Build 3 one-electron matrices: S, T, V """
    # build a list of contracted functions with the center coordinates R
    cf_R_list = []
    for e, R in zip(elems, coords):
        for cf in basis_set[e]:
            cf_R_list.append((cf, R))
    nbf = len(cf_R_list)
    # list of nuclear charges
    nuc_charge_coords = [(chemical_elements.index(e), R) for e,R in zip(elems, coords)]
    # build the overlap matrix
    overlap_mat = np.zeros(nbf*nbf, dtype=float).reshape(nbf, nbf)
    kinetic_mat = np.zeros(nbf*nbf, dtype=float).reshape(nbf, nbf)
    nuclear_attraction_mat = np.zeros(nbf*nbf, dtype=float).reshape(nbf, nbf)
    for i in range(nbf):
        cf_i, R_i = cf_R_list[i]
        for j in range(i, nbf):
            cf_j, R_j = cf_R_list[j]
            s, t, v = integrate_cf_i_j(cf_i, R_i, cf_j, R_j, nuc_charge_coords)
            overlap_mat[i,j] = overlap_mat[j,i] = s
            kinetic_mat[i,j] = kinetic_mat[j,i] = t
            nuclear_attraction_mat[i,j] = nuclear_attraction_mat[j,i] = v
    return overlap_mat, kinetic_mat, nuclear_attraction_mat

def integrate_cf_i_j(cf_i, R_i, cf_j, R_j, nuc_charge_coords):
    """ Integrate two contracted Gaussian functions """
    if cf_i.ftype == 'S' and cf_j.ftype == 'S':
        # loop over each primitive Gaussian function in the contracted function and sum up
        s, t, v = sum(integrate_S_S(pf_i, R_i, pf_j, R_j, nuc_charge_coords) for pf_i in cf_i.coefs for pf_j in cf_j.coefs)
    else:
        raise NotImplementedError("Havn't implemented integration between %s and %s functions yet." % (cf_i.ftype, cf_j.ftype))
    return s, t, v

def integrate_S_S(pf_i, R_i, pf_j, R_j, nuc_charge_coords):
    """ Integrate two primitive Gaussian functions both of S type
    Reference: Modern Quantum Chemistry Page 411-412 """
    a_i, c_i = pf_i
    a_j, c_j = pf_j
    aij = a_i*a_j/(a_i+a_j)
    # proportionality constant
    K = np.exp( -aij * np.sum((R_i-R_j)**2) )
    # new joint center
    Rp = (a_i * R_i + a_j * R_j) / (a_i + a_j)
    # overlap integral (Eq. A.9)
    s = (np.pi/(a_i + a_j))**1.5 * K * c_i * c_j
    # kinetic integral (Eq. A.11)
    t = s * aij * ( 3 - 2 * aij * np.sum((R_i - R_j)**2) )
    # nuclear attraction integral (Eq. A.33)
    v = 0
    for z, R in nuc_charge_coords:
        x = (a_i + a_j) * np.sum((Rp - R)**2)
        if abs(x) < 1e-15:
            F0 = 1
        else:
            F0 = 0.5 * (np.pi / x)**0.5 * math.erf(x**0.5)
        v += z * F0
    v *= -2 * np.pi / (a_i+a_j) * K * c_i * c_j
    return np.array([s, t, v])

def build_two_electron_tensor(elems, coords, basis_set):
    # build a list of contracted functions with the center coordinates R
    cf_R_list = []
    for e, R in zip(elems, coords):
        for cf in basis_set[e]:
            cf_R_list.append((cf, R))
    nbf = len(cf_R_list)
    # build the tei tensor G
    G_ao = np.zeros(nbf**4, dtype=float).reshape(nbf, nbf, nbf, nbf)
    for p in range(nbf):
        cf_p, R_p = cf_R_list[p]
        for q in range(p, nbf):
            cf_q, R_q = cf_R_list[q]
            for r in range(nbf):
                cf_r, R_r = cf_R_list[r]
                for s in range(r, nbf):
                    cf_s, R_s = cf_R_list[s]
                    g = integrate_cf_pqrs(cf_p, R_p, cf_q, R_q, cf_r, R_r, cf_s, R_s)
                    G_ao[p,q,r,s] = G_ao[q,p,r,s] = G_ao[p,q,s,r] = G_ao[q,p,s,r] = g
    return G_ao

def integrate_cf_pqrs(cf_p, R_p, cf_q, R_q, cf_r, R_r, cf_s, R_s):
    if cf_p.ftype == 'S' and cf_q.ftype == 'S' and cf_r.ftype == 'S' and cf_s.ftype == 'S':
        # loop over each primitive Gaussian function in the contracted function and sum up
        g = sum(integrate_SSSS(pf_p, R_p, pf_q, R_q, pf_r, R_r, pf_s, R_s) for pf_p in cf_p.coefs
                for pf_q in cf_q.coefs for pf_r in cf_r.coefs for pf_s in cf_s.coefs)
    else:
        raise NotImplementedError("Havn't implemented two electron intergral of type (%s%s|%s%s)" % (cf_p.ftype, cf_q.ftype,
                                                                                                     cf_r.ftype, cf_s.ftype))
    return g

def integrate_SSSS(pf_p, R_p, pf_q, R_q, pf_r, R_r, pf_s, R_s):
    """ Primitive Gaussian integrals (SS|SS)
    Reference: Modern Quantum Chemistry Page 415-416 """
    a_p, c_p = pf_p
    a_q, c_q = pf_q
    a_r, c_r = pf_r
    a_s, c_s = pf_s
    # step 1: combine R_p and R_q into R_pq, R_r and R_s into R_rs
    R_pq = (a_p * R_p + a_q * R_q) / (a_p + a_q)
    R_rs = (a_r * R_r + a_s * R_s) / (a_r + a_s)
    # step 2: Eq. A41 first term
    first_term = 2 * np.pi**2.5 / ((a_p+a_q)*(a_r+a_s)*(a_p+a_q+a_r+a_s)**0.5)
    # step 3: Eq. A41 second term
    K_pq = np.exp( -a_p*a_q/(a_p+a_q) * np.sum((R_p-R_q)**2) )
    K_rs = np.exp( -a_r*a_s/(a_r+a_s) * np.sum((R_r-R_s)**2) )
    second_term = K_pq * K_rs
    # step 4: Eq. A41 third term
    x = (a_p+a_q)*(a_r+a_s)/(a_p+a_q+a_r+a_s) * np.sum((R_pq-R_rs)**2)
    if abs(x) < 1e-15:
        F0 = 1
    else:
        F0 = 0.5 * (np.pi / x)**0.5 * math.erf(x**0.5)
    g = first_term * second_term * F0 * c_p * c_q * c_r * c_s
    return g

def solve_restricted_hartree_fock(elems, coords, basis_set, charge, maxiter=150, verbose=True):
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
    Smat, Tmat, Vmat = build_one_e_matrices(elems, coords, basis_set)
    Hmat = Tmat + Vmat
    # build two-electron integral tensor g = (pq|rs)
    G_ao = build_two_electron_tensor(elems, coords, basis_set)
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
    # Solve the FC = ESC equation

    # intial guess density
    Dmat = np.zeros_like(Smat)
    E_hf = 0
    converged = False
    if verbose:
        print("\n *** SCF Iterations *** ")
        print("Iter         HF Energy         delta E         RMS |D|")
        print("-------------------------------------------------------")
    for i in range(maxiter):
        Fmat = Hmat + np.einsum("rs,pqrs->pq",Dmat,G_ao*2) - np.einsum("rs,prqs->pq",Dmat,G_ao)
        Feigval, Cmat = scipy.linalg.eig(Fmat, b=Smat)
        Feigval = Feigval.real
        print Feigval
        print Cmat
        #idx = Feigval.argsort()
        #Feigval = Feigval[idx]
        #Feigvec = Feigvec[:,idx]
        #Cmat = np.dot(Shalf, Feigvec)
        Dmat_new = np.dot(Cmat.T[:ndo].T, Cmat.T[:ndo])
        print Dmat_new
        E_hf_new = np.einsum("pq,pq", Dmat_new, Hmat+Fmat)
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
    args = parser.parse_args()

    # load the basis set
    basis_set = load_basis(args.basis)
    # read the xyz coordinates
    elems, coords = read_xyz(args.infile)
    solve_restricted_hartree_fock(elems, coords, basis_set, args.charge, verbose=True)


if __name__ == "__main__":
    main()
