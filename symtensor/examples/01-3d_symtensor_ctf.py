#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
'''
Simple tensor contraction with 3D symmetry
usage: mpirun -np 2 python 01-3d_symtensor_ctf.py
'''
import numpy as np
from symtensor.sym_ctf import tensor, einsum
import ctf

def make_kpts(lattice, nmp):
    ks_each_axis = []
    for n in nmp:
        ks = np.arange(n, dtype=float) / n
        ks_each_axis.append(ks)
    arrays = [np.asarray(x) for x in ks_each_axis]
    nd = len(arrays)
    dims = [nd] + [len(x) for x in arrays]
    out = np.ndarray(dims)
    shape = [-1] + [1] * nd
    for i, arr in enumerate(arrays):
        out[i] = arr.reshape(shape[:nd-i])
    scaled_kpts = out.reshape(nd,-1).T
    gvec = get_reciprocal_vectors(lattice)
    kpts = np.dot(scaled_kpts, gvec)
    return kpts

def get_reciprocal_vectors(lattice):
    b = np.linalg.inv(lattice.T)
    return 2*np.pi * b

lattice = np.eye(3)*5
kpts = make_kpts(lattice, [2,2,1])
gvec = get_reciprocal_vectors(lattice)
nkpts, nmo = len(kpts), 10

sym = ['++--', [kpts,]*4, None, gvec]

Aarray = ctf.random.random([nkpts,nkpts,nkpts,nmo,nmo,nmo,nmo])
Barray = ctf.random.random([nkpts,nkpts,nkpts,nmo,nmo,nmo,nmo])
A = tensor(Aarray, sym, verbose=1)
B = tensor(Barray, sym, verbose=1)
out = einsum('ijab,abcd->ijcd', A, B)