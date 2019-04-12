#!/usr/bin/env python
# coding: utf-8

import numpy as np
import one_e_qyd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

elems = ['H', 'H', 'H']
coords = np.array([[0,0,0], [0,0,1.4],[0,0,2.8]], dtype=float)

low = 1.0
high = 50.0
num = 50
for basis in ['sto-3g', '3-21g', '6-311g']:
    basis_set = one_e_qyd.load_basis(basis)
    d_range = np.linspace(low, high, num=num)
    results = np.empty(num**2).reshape(num,num)
    for i,d1 in enumerate(d_range):
        for j in range(i, num):
            d2 = d_range[j]
            coords[1][-1] = d1
            coords[2][-1] = d1 + d2
            energy = one_e_qyd.solve_one_e(elems, coords, basis_set, verbose=False)
            results[i,j] = results[j,i] = energy
    plt.figure()
    plt.imshow(results, label=basis, extent=[low,high,low,high], origin='lower')
    plt.xlabel("H1-H2 Dist")
    plt.ylabel("H2-H3 Dist")
    plt.colorbar()
    plt.gca().grid(None)
    plt.savefig("h3_map_%s.pdf"%basis)
    plt.close()

