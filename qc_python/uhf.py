
"""
Author: Yudong Qiu
Functions for solving unrestricted Hartree-Fock
"""

import numpy as np

from qc_python import basis_integrals
from qc_python.common import chemical_elements, calc_nuclear_repulsion

def solve_unrestricted_hartree_fock(elems, coords, basis_set, charge=0, spinmult=1, maxiter=150, enable_DIIS=True, verbose=False):
    """ Unrestricted Hartree-Fock """
    # Compute the number of electrons in the system
    n_electron = sum(chemical_elements.index(e) for e in elems) - charge
    if verbose:
        print("This system has a total of %d electrons" % n_electron)
    n_single_e = spinmult - 1
    if (n_electron+n_single_e) % 2 != 0:
        raise RuntimeError("The specified charge %d and spinmult %d is impossible!" % (charge, spinmult))
    # number of alpha and beta orbitals
    n_a = int((n_electron + n_single_e) / 2)
    n_b = n_electron - n_a
    # compute nuclear repulsion energy
    E_nuc = calc_nuclear_repulsion(elems, coords)
    # compute one-electron integral matrices
    Smat, Tmat, Vmat = basis_integrals.build_one_e_matrices(elems, coords, basis_set)
    Hmat = Tmat + Vmat
    # check if we have enough basis functions to hold all the electrons
    if n_a > len(Smat):
        raise RuntimeError("Number of basis functions is smaller than number of alpha orbitals")
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
    Dmat_a = np.zeros_like(Smat)
    Dmat_b = np.zeros_like(Smat)
    E_hf = 0
    converged = False
    # DIIS
    if enable_DIIS is True:
        n_err_mat = 6
        diis_start_n = 4
        diis_err_mats = []
        #diis_err_mats_a = []
        #diis_err_mats_b = []
        diis_fmats_a = []
        diis_fmats_b = []
        if verbose:
            print(" *** DIIS Enabled ***")
    if verbose:
        print("\n *** SCF Iterations *** ")
        print("Iter         HF Energy         delta E         RMS |D|")
        print("-------------------------------------------------------")
    for i in range(maxiter):
        Fmat = Hmat + np.einsum("rs,pqrs->pq",(Dmat_a+Dmat_b),G_ao)
        Fmat_a = Fmat - np.einsum("rs,prqs->pq",Dmat_a,G_ao)
        Fmat_b = Fmat - np.einsum("rs,prqs->pq",Dmat_b,G_ao)
        if enable_DIIS and i > 0:
            ## DIIS for alpha spin
            #FDS = np.einsum("pi,ij,jq->pq",Fmat_a,Dmat_a,Smat)
            #SDF = np.einsum("pi,ij,jq->pq",Smat,Dmat_a,Fmat_a)
            #diis_err_mats_a.append(FDS-SDF)
            #diis_err_mats_a = diis_err_mats_a[-n_err_mat:]
            diis_fmats_a.append(Fmat_a)
            diis_fmats_a = diis_fmats_a[-n_err_mat:]
            ## DIIS for beta spin
            #FDS = np.einsum("pi,ij,jq->pq",Fmat_b,Dmat_b,Smat)
            #SDF = np.einsum("pi,ij,jq->pq",Smat,Dmat_b,Fmat_b)
            #diis_err_mats_b.append(FDS-SDF)
            #diis_err_mats_b = diis_err_mats_b[-n_err_mat:]
            diis_fmats_b.append(Fmat_b)
            diis_fmats_b = diis_fmats_b[-n_err_mat:]
            FDS_a = np.einsum("pi,ij,jq->pq",Fmat_a,Dmat_a,Smat)
            SDF_a = np.einsum("pi,ij,jq->pq",Smat,Dmat_a,Fmat_a)
            FDS_b = np.einsum("pi,ij,jq->pq",Fmat_b,Dmat_b,Smat)
            SDF_b = np.einsum("pi,ij,jq->pq",Smat,Dmat_b,Fmat_b)
            diis_err_mats.append(FDS_a-SDF_a+FDS_b-SDF_b)
            diis_err_mats = diis_err_mats[-n_err_mat:]
            # compute Bmat_ij = Err_i . Err_j
            n_diis = len(diis_err_mats)
            if n_diis >= diis_start_n:
                Fmat_a = DIIS_extrapolate_F(diis_err_mats, diis_fmats_a)
                Fmat_b = DIIS_extrapolate_F(diis_err_mats, diis_fmats_b)
        # solve the alpha HF equation F_a C_a = e_a S C_a
        Feigval_a, Cmat_a = solve_FCeSC(Fmat_a, Shalf)
        C_occ = Cmat_a[:, :n_a]
        Dmat_a_new = np.dot(C_occ, C_occ.T)
        # solve the beta HF equation F_b C_b = e_b S C_b
        Feigval_b, Cmat_b = solve_FCeSC(Fmat_b, Shalf)
        C_occ = Cmat_b[:, :n_b]
        Dmat_b_new = np.dot(C_occ, C_occ.T)

        E_hf_new = 0.5 * (np.einsum("pq,pq", Dmat_a, Hmat+Fmat_a) + np.einsum("pq,pq", Dmat_b, Hmat+Fmat_b))
        dE = E_hf_new - E_hf
        D_rms = np.sqrt(np.mean((Dmat_a_new-Dmat_a)**2)) + np.sqrt(np.mean((Dmat_b_new-Dmat_b)**2))
        # update E_hf and Dmat
        E_hf = E_hf_new
        Dmat_a = Dmat_a_new
        Dmat_b = Dmat_b_new
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
        print("\nSCF converged!")
        print("\nOrbital Energies (Eh) and coefficients for Alpha electrons")
        print('E:  '+''.join(["%17.7f"%e for e in Feigval_a]))
        print('-' * (17 * len(Feigval_a) + 4))
        for i,row in enumerate(Cmat_a):
            print('c%-3d'%i + ''.join(["%17.7f"%c for c in row]))
        print("\nOrbital Energies (Eh) and coefficients for Beta electrons")
        print('E:  '+''.join(["%17.7f"%e for e in Feigval_b]))
        print('-' * (17 * len(Feigval_b) + 4))
        for i,row in enumerate(Cmat_b):
            print('c%-3d'%i + ''.join(["%17.7f"%c for c in row]))
        print("\nNuclear Repulsion Energy =   %17.10f   Eh" % E_nuc)
        print("Total Electronic Energy  =   %17.10f   Eh" % E_hf)
        print("Final Total Energy       =   %17.10f   Eh" % E_total)
    return {"E_nuc":E_nuc, "E_hf": E_hf, "E_total":E_total, "E_orbs_a": Feigval_a, "Cmat_a": Cmat_a, "Dmat_a": Dmat_a,
                                                            "E_orbs_b": Feigval_b, "Cmat_b": Cmat_b, "Dmat_b": Dmat_b}

def solve_FCeSC(Fmat, Shalf):
    Ft = np.einsum("pi,ij,jq->pq",Shalf,Fmat,Shalf)
    Feigval, Feigvec = np.linalg.eigh(Ft)
    idx = Feigval.argsort()
    Feigval = Feigval[idx]
    Feigvec = Feigvec[:,idx]
    Cmat = np.dot(Shalf, Feigvec)
    return Feigval, Cmat

def DIIS_extrapolate_F(diis_err_mats, diis_fmats):
    n_diis = len(diis_err_mats)
    assert n_diis == len(diis_fmats), 'Number of Fock matrices should equal to number of error matrices'
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
    new_Fmat = np.zeros_like(diis_fmats[-1])
    for di in range(n_diis):
        new_Fmat += C_array[di] * diis_fmats[di]
    return new_Fmat