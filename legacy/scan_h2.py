#!/usr/bin/env python
# coding: utf-8

import numpy as np
import one_e_qyd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

elems = ['H', 'H']
coords = np.array([[0,0,0], [0,0,1.4]])

for basis in ['sto-3g', '3-21g', '6-311g']:
    basis_set = one_e_qyd.load_basis(basis)

    d_range = np.linspace(0.5, 10.0, num=96)
    results = []
    for d in d_range:
        #print("Dist = %.3f Bohr" % d)
        coords[1][-1] = d
        energy = one_e_qyd.solve_one_e(elems, coords, basis_set, verbose=False)
        #print("Total Energy = %.7f "%energy)
        results.append(energy)
    plt.plot(d_range, results, lw=1.5, label=basis)

plt.legend()
plt.xlabel("H-H Dist (Bohr)")
plt.ylabel("Total Energy (Eh)")
plt.title("Scan of H2+ PES")
plt.savefig("scan_H2.pdf")
plt.close()

print results
