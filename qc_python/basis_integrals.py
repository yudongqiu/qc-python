#!/usr/bin/env python
# coding: utf-8

"""
Author: Yudong Qiu
Basis set parser and integrals for CARTESIAN Gaussian functions
Eg. sto-3g, and Pople Basis with SPD functions. (F function becomes spherical)
"""

import numpy as np
import math
import collections
import os
import functools
import numba
from scipy.special import factorial2

#---------------------------
# Decorator
#---------------------------

def memo(f):
    """Decorator that caches the return value for each call to f(args).
    Then when called again with same args, we can just look it up."""
    cache = {}
    @functools.wraps(f)
    def _f(*args):
        try:
            return cache[args]
        except KeyError:
            cache[args] = result = f(*args)
            return result
        except TypeError:
            # some element of args refuses to be a dict key
            return f(args)
    _f.cache = cache
    return _f

#---------------------------
# Some Physical Constants
#---------------------------

chemical_elements = ["", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]
ContractedBasisFunction = collections.namedtuple('ContractedBasisFunction', ['ftype', 'coefs'])
AxAyAz_SPDF = {'S':((0,0,0),),
              # 'P':((1,0,0),(0,1,0),(0,0,1)),
               'P':((0,0,1),(1,0,0),(0,1,0)),
               'D':((2,0,0),(0,2,0),(0,0,2),(1,1,0),(1,0,1),(0,1,1)),
               'F':((3,0,0),(0,3,0),(0,0,3),(2,1,0),(2,0,1),(1,2,0),(1,0,2),(0,2,1),(0,1,2),(1,1,1))
              }
n_SPDF = dict([ (k,len(v)) for k,v in AxAyAz_SPDF.items() ])


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
    basis_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),'basis')
    for line in open(os.path.join(basis_folder,basis_name.lower()+'.basis')):
        if line.startswith("#"):
            continue
        if line.startswith("BASIS"):
            reading = True
            continue
        if reading is True:
            if line.startswith("END"):
                reading = False
                continue
            ls = line.split()
            if len(ls) == 2 and ls[1].isalpha(): # if title line like "H   S"
                elem, spdf = ls
                if elem not in basis_set:
                    basis_set[elem] = []
                if spdf == 'SP':
                    current_function = ContractedBasisFunction(ftype='S', coefs=[])
                    extra_function = ContractedBasisFunction(ftype='P', coefs=[])
                    basis_set[elem].append(current_function)
                    basis_set[elem].append(extra_function)
                else:
                    current_function = ContractedBasisFunction(ftype=spdf, coefs=[])
                    basis_set[elem].append(current_function)
            elif len(ls) >= 2: # if value line like "      5.4471780              0.1562850"
                vs = [float(v.replace('D','e')) for v in ls]
                a = vs[0]
                if spdf == 'SP':
                    assert len(ls) == 3, 'SP type basis should have 3 columns'
                    c1, c2 = vs[1:3]
                    current_function.coefs.append([a, c1])
                    extra_function.coefs.append([a, c2])
                else:
                    current_function.coefs.append([a] + vs[1:])
    return basis_set

@memo
def norm_factor(a, A_a):
    ax, ay, az = A_a
    result = (2*a/np.pi)**0.75 * (4*a)**((ax+ay+az)*0.5)
    result /= (factorial2(2*ax-1) * factorial2(2*ay-1) * factorial2(2*az-1))**0.5
    return result

def build_one_e_matrices(elems, coords, basis_set):
    """ Build one-electron matrices: S, T, V """
    # build a list of contracted functions with the center coordinates R
    cf_R_list = []
    for e, R in zip(elems, np.array(coords)):
        for cf in basis_set[e]:
            coefs = np.array(cf.coefs)
            if len(coefs[0]) == 2: # if this is a single contracted Gaussian
                cf_R_list.append((cf, R))
            else: # if there are more than 1 pair of coefs with the same a
                for j in range(1, cf.coefs.shape[1]):
                    # split the c and a into single cf
                    new_cf = ContractedBasisFunction(ftype=cf.ftype, coefs=coefs[:, (0,j)])
                    cf_R_list.append((new_cf, R))
    # build a list of contracted functions with Angular momentum and the center coordinates R
    cf_R_A_list = []
    for cf, R in cf_R_list:
        coefs = np.array(cf.coefs)
        for A_i in AxAyAz_SPDF[cf.ftype]:
            # normalize this cf
            new_coefs = coefs.copy()
            for p in range(len(coefs)):
                new_coefs[p,1] *= norm_factor(coefs[p,0], A_i)
            new_cf = ContractedBasisFunction(ftype=cf.ftype, coefs=new_coefs)
            cf_R_A_list.append((new_cf, np.array(A_i), R))
    nbf = len(cf_R_A_list)
    # list of nuclear charges
    nuc_charge_coords = [(chemical_elements.index(e), R) for e,R in zip(elems, coords)]
    # build the overlap matrix
    overlap_mat = np.zeros(nbf*nbf, dtype=float).reshape(nbf, nbf)
    kinetic_mat = np.zeros(nbf*nbf, dtype=float).reshape(nbf, nbf)
    nuclear_attraction_mat = np.zeros(nbf*nbf, dtype=float).reshape(nbf, nbf)
    for i in range(nbf):
        cf_i, A_i, R_i = cf_R_A_list[i]
        for j in range(i,nbf):
            cf_j, A_j, R_j = cf_R_A_list[j]
            s, t, v = integrate_contracted_cartesian_gaussians(cf_i, cf_j, R_i, R_j, A_i, A_j, nuc_charge_coords)
            overlap_mat[i,j] = overlap_mat[j,i] = s if abs(s) > 1e-15 else 0
            kinetic_mat[i,j] = kinetic_mat[j,i] = t if abs(t) > 1e-15 else 0
            nuclear_attraction_mat[i,j] = nuclear_attraction_mat[j,i] = v if abs(v) > 1e-15 else 0
    return overlap_mat, kinetic_mat, nuclear_attraction_mat

def integrate_contracted_cartesian_gaussians(cf_i, cf_j, R_i, R_j, A_i, A_j, nuc_charge_coords):
    """ Integrate two contracted Gaussian functions """
    s, t, v = sum(integrate_primitive_cartesian_gaussians(pf_i, pf_j, R_i, R_j, A_i, A_j, nuc_charge_coords) for pf_i in cf_i.coefs for pf_j in cf_j.coefs)
    return s, t, v

def integrate_primitive_cartesian_gaussians(pf_i, pf_j, R_i, R_j, A_a, A_b, nuc_charge_coords):
    """ Integrate two primitive Gaussian functions """
    # Overlap
    # Reference: purple book Chapter 9.3.1
    a_i, c_i = pf_i
    a_j, c_j = pf_j
    mu = a_i*a_j/(a_i+a_j)
    p = a_i + a_j
    Rp = (a_i * R_i + a_j * R_j) / (a_i + a_j)
    Rpa = Rp - R_i
    Rpa2 = np.dot(Rpa, Rpa)
    Rab = R_i - R_j
    # proportionality constant # Eq. 9.3.10
    EAB = np.exp(-mu * np.dot(R_i-R_j, R_i-R_j))
    s00 = (np.pi/p)**1.5 * EAB * c_i * c_j

    lab = np.sum([A_a, A_b], axis=0)
    lb_max = np.max(A_b)
    # recurrence matrix for S
    s_recurrent = np.zeros([3,lab.max()+3, lb_max+2]) # some extra for T
    s_recurrent[:,0,0] = 1.0
    overlap = s00
    for xyz in range(3):
        la, lb = A_a[xyz], A_b[xyz]
        for i in range(la+lb+2):
            s_recurrent[xyz,i+1,0] = Rpa[xyz] * s_recurrent[xyz,i,0] + 0.5/p*(i*s_recurrent[xyz,i-1,0])
            for j in range(min(i+1, lb+1)):
                s_recurrent[xyz,i-j,j+1] = s_recurrent[xyz,i-j+1, j] + Rab[xyz] * s_recurrent[xyz,i-j,j]
        # finish this xyz
        overlap *= s_recurrent[xyz, la, lb]
    # kinetic energy reference http://www.mathematica-journal.com/2013/01/evaluation-of-gaussian-molecular-integrals-2/
    k_recurrent = [0,0,0]
    for xyz in range(3):
        la, lb = A_a[xyz], A_b[xyz]
        if la == 0 and lb == 0:
            k_recurrent[xyz] = 2 * a_i * a_j * s_recurrent[xyz, 1, 1]
        elif la > 0 and lb == 0:
            k_recurrent[xyz] = - la * a_j * s_recurrent[xyz, la-1, 1] + 2 * a_i * a_j * s_recurrent[xyz, la+1, 1]
        elif la ==0 and lb > 0:
            k_recurrent[xyz] = - lb * a_i * s_recurrent[xyz, 1, lb-1] + 2 * a_i * a_j * s_recurrent[xyz, 1, lb+1]
        else:
            k_recurrent[xyz] = (la * lb * s_recurrent[xyz, la-1, lb-1]
                         - 2 * la * a_j * s_recurrent[xyz, la-1, lb+1]
                         - 2 * lb * a_i * s_recurrent[xyz, la+1, lb-1]
                         + 4 * a_i * a_j * s_recurrent[xyz, la+1, lb+1]) * 0.5
    kinetic  = k_recurrent[0] * s_recurrent[1][A_a[1], A_b[1]] * s_recurrent[2][A_a[2], A_b[2]] \
             + k_recurrent[1] * s_recurrent[0][A_a[0], A_b[0]] * s_recurrent[2][A_a[2], A_b[2]] \
             + k_recurrent[2] * s_recurrent[0][A_a[0], A_b[0]] * s_recurrent[1][A_a[1], A_b[1]]
    kinetic *= s00
    # nuclear-electron attraction
    # Reference Purple Book Chapter 9.10.1
    nuc_attract = 0
    N_max = sum(A_a) + sum(A_b) + 1
    for Zc, Rc in nuc_charge_coords:
        Rpc = Rp - Rc
        x = p * np.dot(Rpc, Rpc)
        # Boys function
        Boys = np.zeros(N_max)
        Boys[0] = 0.5 * (np.pi / x)**0.5 * math.erf(x**0.5) if abs(x) > 1e-15 else 1.0
        if N_max == 1:
            nuc_attract += Zc * Boys[0]
            continue
        for n in range(N_max-1):
            Boys[n+1] = ((2*n+1) * Boys[n] - np.exp(-x)) / x * 0.5 if abs(x) > 1e-15 else (1.0/(2*n+3)) # Eq. 9.8.13
        # Auxiliary Integrals
        lab = np.sum([A_a, A_b], axis=0)
        lb_max = np.max(A_b)
        Aux = np.zeros([N_max, lab.max()+1, lb_max+1])
        Aux[:,0,0] = Boys
        N = N_max - 1
        for xyz in range(3):
            la, lb = A_a[xyz], A_b[xyz]
            for i in range(1, la+lb+1):
                N -= 1
                for dn in range(N+1):
                    Aux[N-dn, i, 0] = Rpa[xyz]*Aux[N-dn,i-1,0] - Rpc[xyz]*Aux[N+1-dn,i-1,0] + 0.5/p*(i-1)*(Aux[N-dn,i-2,0] - Aux[N+1-dn,i-2,0])
                for j in range(1, min(i+1, lb+1)):
                    for dn in range(N+1):
                        Aux[N-dn,i-j,j] = Aux[N-dn,i-j+1,j-1] + Rab[xyz] * Aux[N-dn,i-j,j-1]
            # finish up the current xyz
            tmp = Aux[:,la,lb].copy()
            Aux.fill(0)
            Aux[:,0,0] = tmp
        assert N == 0
        nuc_attract += Zc * Aux[0,0,0]
    nuc_attract *= -2.0 * np.pi / p * EAB * c_i * c_j
    return np.array([overlap, kinetic, nuc_attract])

def build_two_electron_tensor(elems, coords, basis_set):
    # build a list of contracted functions with the center coordinates R
    cf_R_list = []
    for e, R in zip(elems, np.array(coords)):
        for cf in basis_set[e]:
            coefs = np.array(cf.coefs)
            if len(coefs[0]) == 2: # if this is a single contracted Gaussian
                cf_R_list.append((cf, R))
            else: # if there are more than 1 pair of coefs with the same a
                for j in range(1, cf.coefs.shape[1]):
                    # split the c and a into single cf
                    new_cf = ContractedBasisFunction(ftype=cf.ftype, coefs=coefs[:, (0,j)])
                    cf_R_list.append((new_cf, R))
    # build a list of contracted functions with Angular momentum and the center coordinates R
    cf_R_A_list = []
    for cf, R in cf_R_list:
        coefs = np.array(cf.coefs)
        for A_i in AxAyAz_SPDF[cf.ftype]:
            # normalize this cf
            new_coefs = coefs.copy()
            for p in range(len(coefs)):
                new_coefs[p,1] *= norm_factor(coefs[p,0], A_i)
            new_cf = ContractedBasisFunction(ftype=cf.ftype, coefs=new_coefs)
            cf_R_A_list.append((new_cf, np.array(A_i), R))
    nbf = len(cf_R_A_list)
    # build the tei tensor G
    G_ao = np.zeros(nbf**4, dtype=float).reshape(nbf, nbf, nbf, nbf)
    for p in range(nbf):
        cf_p, A_p, R_p = cf_R_A_list[p]
        for q in range(p, nbf):
            cf_q, A_q, R_q = cf_R_A_list[q]
            for r in range(nbf):
                cf_r, A_r, R_r = cf_R_A_list[r]
                for s in range(r, nbf):
                    cf_s, A_s, R_s = cf_R_A_list[s]
                    g = integrate_cf_cartesian_pqrs(cf_p.coefs, cf_q.coefs, cf_r.coefs, cf_s.coefs, R_p, R_q, R_r, R_s, A_p, A_q, A_r, A_s)
                    G_ao[p,q,r,s] = G_ao[q,p,r,s] = G_ao[p,q,s,r] = G_ao[q,p,s,r] = g
    return G_ao

@numba.jit(nopython=True, nogil=True, cache=True)
def integrate_cf_cartesian_pqrs(cf_p, cf_q, cf_r, cf_s, R_p, R_q, R_r, R_s, A_p, A_q, A_r, A_s):
    # loop over each primitive Gaussian function in the contracted function and sum up
    result = 0
    #for a_p,c_p in cf_p.coefs:
    for ip in range(len(cf_p)):
        a_p = cf_p[ip,0]
        c_p = cf_p[ip,1]
        for iq in range(len(cf_q)):
            a_q = cf_q[iq,0]
            c_q = cf_q[iq,1]
            for ir in range(len(cf_r)):
                a_r = cf_r[ir,0]
                c_r = cf_r[ir,1]
                for i_s in range(len(cf_s)):
                    a_s = cf_s[i_s,0]
                    c_s = cf_s[i_s,1]
                    g = integrate_pf_cartesian_pqrs(a_p, a_q, a_r, a_s, R_p, R_q, R_r, R_s, A_p, A_q, A_r, A_s) * c_p*c_q*c_r*c_s
                    result += g
    return result

@numba.jit(nopython=True, nogil=True, cache=True)
def integrate_pf_cartesian_pqrs(alpha_p, alpha_q, alpha_r, alpha_s, R_p, R_q, R_r, R_s, A_p, A_q, A_r, A_s):
    """ Primitive Gaussian integrals (pq|rs)
    Reference: Purple book Chapter 9.10.2 """
    # a,b,c,d from book --> p,q,r,s here
    # p and q in Eq. 9.10.11 (renamed to u and v here)
    u = alpha_p + alpha_q
    v = alpha_r + alpha_s
    R_u = (alpha_p * R_p + alpha_q * R_q) / u
    R_v = (alpha_r * R_r + alpha_s * R_s) / v
    Rpq = R_p - R_q
    Rrs = R_r - R_s
    Ruv = R_u - R_v
    Rup = R_u - R_p
    # Eq. 9.10.12 coefficients
    K_pq = np.exp( -alpha_p * alpha_q / u * np.dot(Rpq, Rpq) )
    K_rs = np.exp( -alpha_r * alpha_s / v * np.dot(Rrs, Rrs) )
    coef = 2 * np.pi**2.5 / (u*v*(u+v)**0.5) * K_pq * K_rs

    N_max = np.sum(A_p + A_q + A_r + A_s) + 1
    # Boys[n] : Fn(alpha*Rpq^2) in the book
    w = u * v / (u+v) # alpha in the book
    x = w * np.dot(Ruv, Ruv)
    Boys = np.zeros(N_max)
    Boys[0] = 0.5 * (np.pi / x)**0.5 * math.erf(x**0.5) if abs(x) > 1e-15 else 1.0
    if N_max == 1:
        g = Boys[0] * coef
        return g
    for n in range(N_max-1):
        Boys[n+1] = ((2*n+1) * Boys[n] - np.exp(-x)) / x * 0.5 if abs(x) > 1e-15 else (1.0/(2*n+3)) # Eq. 9.8.13

    # Auxiliary Integrals
    # total angular momentums in xyz directions, e.g. (4,0,0) from (1,0,0) * 4
    lpqrs = A_p + A_q + A_r + A_s
    lq_max = np.max(A_q)
    lrs = A_r + A_s
    lr_max = np.max(A_r)
    ls_max = np.max(A_s)
    dim1 = lpqrs.max() + 1
    dim2 = lq_max + 1
    dim3 = lrs.max() + 1
    dim4 = ls_max + 1
    Aux = np.zeros(N_max*dim1*dim2*dim3*dim4, dtype=np.float64).reshape(N_max,dim1,dim2,dim3,dim4)
    Aux[:,0,0,0,0] = Boys
    N = N_max - 1
    k_coef = -(alpha_q * Rpq + alpha_s * Rrs)
    for xyz in range(3):
        lp, lq, lr, ls = A_p[xyz], A_q[xyz], A_r[xyz], A_s[xyz]
        # increase i first based on Eq. 9.10.26
        l_total = lpqrs[xyz] # lp+lq+lr+ls
        for i in range(l_total):
            N -= 1
            for dn in range(N+1):
                Aux[N-dn,i+1,0,0,0] = Rup[xyz]*Aux[N-dn,i,0,0,0] - w/u * Ruv[xyz] * Aux[N+1-dn,i,0,0,0] + i/(2*u)*(Aux[N-dn,i-1,0,0,0] - w/u*Aux[N+1-dn,i-1,0,0,0])
            # increase k. Eq. 9.10.27, replace i by i-k-1, p by u, q by v
            for k in range(-1, min(i+1, lr+ls)): # range from -1 to allow j to increase with lr=0 and ls=0
                ki = i - k
                if k >= 0:
                    for dn in range(N+1):
                        Aux[N-dn,ki,0,k+1,0] = 1/v * (k_coef[xyz] * Aux[N-dn,ki,0,k,0] + ki/2*Aux[N-dn,ki-1,0,k,0] + k/2*Aux[N-dn,ki,0,k-1,0] - u*Aux[N-dn,ki+1,0,k,0])
                # increase j. Eq. 9.10.28
                for j in range(-1, min(ki,lq)):
                    if j >= 0:
                        for dn in range(N+1):
                            Aux[N-dn,ki-j-1,j+1,k+1,0] = Aux[N-dn,ki-j,j,k+1,0] + Rpq[xyz]*Aux[N-dn,ki-j-1,j,k+1,0]
                    # increase l Eq. 9.10.29
                    for l in range(min(k+1,ls)):
                        for dn in range(N+1):
                            Aux[N-dn,ki-j-1,j+1,k-l,l+1] = Aux[N-dn,ki-j-1,j+1,k-l+1,l] + Rrs[xyz]*Aux[N-dn,ki-j-1,j+1,k-l,l]
        # finish current xyz
        tmp = np.zeros(N_max, dtype=np.float64)
        for dn in range(N_max):
            tmp[dn] = Aux[dn,lp,lq,lr,ls]
        Aux[:] = 0
        Aux[:, 0, 0, 0, 0] = tmp
    assert N == 0
    g = Aux[0, 0, 0, 0, 0] * coef
    return g

def test():
    basis_set = load_basis('sto-3g')
    elems = ['H','H']
    coords = [(0,0,0), (0,0,1.4)]
    Smat, Tmat, Vmat = build_one_e_matrices(elems, coords, basis_set)
    print(Smat)


if __name__ == '__main__':
    test()
