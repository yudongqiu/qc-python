#!/usr/bin/env python
# coding: utf-8

"""
Author: Yudong Qiu
Solve the Schrodinger's Equation for one-electron system (H, He)
Read in atomic coordinates and basis set
Build Overlap matrix and Hamiltonian matrix
Solve HC = ESC for C and E
"""

import numpy as np
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
                    current_function.coefs.append(map(float, ls))
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
            overlap_mat[i,j] = overlap_mat[j,i] = s if abs(s) > 1e-15 else 0
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
    #print pf_i
    a_i, c_i = pf_i
    a_j, c_j = pf_j
    # proportionality constant
    K = np.exp( -a_i*a_j/(a_i+a_j) * np.sum((R_i-R_j)**2) )
    # new joint center
    Rp = (a_i * R_i + a_j * R_j) / (a_i + a_j)
    # overlap integral (Eq. A.9)
    s = (2 * (a_i * a_j)**0.5 / (a_i + a_j))**1.5 * K * c_i * c_j
    # kinetic integral (Eq. A.11)
    t = s * (a_i * a_j)/(a_i + a_j) * ( 3 - 2 * a_i * a_j / (a_i + a_j) * np.sum((R_i - R_j)**2) )
    # nuclear attraction integral (Eq. A.33)
    v = 0
    for z, R in nuc_charge_coords:
        x = (a_i + a_j) * np.sum((Rp - R)**2)
        if abs(x) < 1e-15:
            F0 = 1
        else:
            F0 = 0.5 * (np.pi / x)**0.5 * math.erf(x**0.5)
        v += - 2 * np.pi / (a_i+a_j) * z * K * c_i * c_j * (4*a_i*a_j/np.pi**2)**0.75 * F0
    return np.array([s, t, v])

def solve_one_e(elems, coords, basis_set, verbose=True):
    # compute nuclear repulsion energy
    E_nuc = calc_nuclear_repulsion(elems, coords)
    # compute one-electron integral matrices
    s_mat, t_mat, v_mat = build_one_e_matrices(elems, coords, basis_set)
    h_mat = t_mat + v_mat
    if verbose:
        print("One Electron Integrals Calculated:")
        print("\nOverlap matrix S")
        print(s_mat)
        print("\nKinetic energy matrix T")
        print(t_mat)
        print("\nNuclear attraction matrix V")
        print(v_mat)
        print("\nCore Hamiltonian matrix H = T + V")
        print(h_mat)
    # Solve the HC = ESC equation by converting it to Ft C' = E C'
    # Diagonalize overlap matrix and form S^(-1/2) matrix
    s_eigval, s_eigvec = np.linalg.eig(s_mat)
    s_half = np.diag(s_eigval**-0.5)
    s_half = np.dot(s_eigvec, np.dot(s_half, s_eigvec.T))
    # Form Ft matrix
    Ft_mat = np.dot(s_half, np.dot(h_mat, s_half))
    # Diagonalize Ft matrix
    f_eigval, f_eigvec = np.linalg.eig(Ft_mat)
    # sort the eigenvalues and eigenvectors from low to high
    idx = f_eigval.argsort()
    f_eigval = f_eigval[idx]
    f_eigvec = f_eigvec[:,idx]
    # Form Coefficient Matrix C
    c_mat = np.dot(s_half, f_eigvec)
    # print results
    if verbose:
        print("\nOne-electron problem solved!")
        print("Orbital Energies (Eh) and coefficients")
        print('E:  '+''.join(["%17.7f"%e for e in f_eigval]))
        print('-' * (17 * len(f_eigval) + 4))
        for i,row in enumerate(c_mat):
            print('c%-3d'%i + ''.join(["%17.7f"%c for c in row]))
    E_elec = f_eigval[0]
    E_total = E_nuc + E_elec
    if verbose:
        print("\nNuclear Repulsion Energy =   %17.7f   Eh" % E_nuc)
        print("Total Electronic Energy  =   %17.7f   Eh" % E_elec)
        print("Final Total Energy       =   %17.7f   Eh" % E_total)
    return E_total

def main():
    import argparse
    parser = argparse.ArgumentParser("Solve the Schrodinger's Equation for one-electron system.")
    parser.add_argument('infile', help='Input xyz file containing the coordinates.')
    parser.add_argument('-b', '--basis', default='sto-3g', help='basis set to use')
    args = parser.parse_args()

    # load the basis set
    basis_set = load_basis(args.basis)
    # read the xyz coordinates
    elems, coords = read_xyz(args.infile)
    solve_one_e(elems, coords, basis_set, verbose=True)



if __name__ == "__main__":
    main()
