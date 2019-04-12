#!/usr/bin/env python
# coding: utf-8

import numpy as np
import one_e_qyd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def num_grad(elems, coords, basis_set, dr=0.001):
    grad = np.zeros_like(coords)
    for i_atom in range(len(grad)):
        for i_xyz in range(3):
            coords[i_atom, i_xyz] += dr
            e_p = one_e_qyd.solve_one_e(elems, coords, basis_set, verbose=False)
            coords[i_atom, i_xyz] -= dr * 2
            e_m = one_e_qyd.solve_one_e(elems, coords, basis_set, verbose=False)
            coords[i_atom, i_xyz] += dr
            grad[i_atom, i_xyz] = (e_p - e_m) / dr / 2
    return grad

def optimize_coords(elems, coords, basis_set, max_iter=100, filename=None):
    dr = 0.001
    step_length = 0.1
    print("--== Optimization start! ==--")
    last_e = 0
    for i in xrange(max_iter):
        print("\n*** Iteration %d ***\n" % i)
        e = one_e_qyd.solve_one_e(elems, coords, basis_set, verbose=False)
        print("Energy    =  %13.7f Eh" % e)
        grad = num_grad(elems, coords, basis_set)
        g_norm = np.sqrt(np.mean(grad**2))
        print("Grad norm =  %13.7f" % g_norm)
        coords -= grad / g_norm * step_length
        print("New Coordinates")
        print(coords)
        if g_norm < 1e-7:
            print("Optimization Finished!")
            if filename is not None:
                one_e_qyd.write_xyz(elems, coords, filename)
            return
        if e > last_e:
            step_length /= 2
        last_e = e
    print("Optimization not finished in %d steps." % max_iter)

def main():
    import argparse
    parser = argparse.ArgumentParser("Optimize the geometry.")
    parser.add_argument('infile', help='Input xyz file containing the coordinates.')
    parser.add_argument('-o', '--outfile', help='Output xyz file for writing the optimized coordinates.')
    parser.add_argument('-b', '--basis', default='sto-3g', help='basis set to use')
    parser.add_argument('-m', '--maxiter', default=100, type=int, help='max number of iterations')
    args = parser.parse_args()

    # load the basis set
    basis_set = one_e_qyd.load_basis(args.basis)
    # read the xyz coordinates
    elems, coords = one_e_qyd.read_xyz(args.infile)
    # run the optimization
    optimize_coords(elems, coords, basis_set, max_iter=args.maxiter, filename=args.outfile)



if __name__ == "__main__":
    main()
