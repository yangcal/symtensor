import numpy as np
from symtensor.backend.ctf_funclib import rank, size
import ctf
import time

def test_sym_ctf(nmo,sym):
    from symtensor.sym_ctf import tensor, einsum
    nk = len(sym[1][0])
    warray = ctf.random.random([nk,nk,nk,nmo,nmo,nmo,nmo])
    w = tensor(warray, sym)
    t0 = time.time()
    out = einsum('ijab,abcd->ijcd',w,w)
    t1 = time.time()
    return t1-t0

def test_ctf_sparse(nmo,sym):
    from symtensor.sym_ctf import random
    w = random([nmo,]*4, sym=sym)
    w = w.make_sparse()
    t0 = time.time()
    out = ctf.einsum('IJABijab,ABCDabcd->IJCDijcd',w,w)
    t1 = time.time()
    return t1-t0

def get_reciprocal_vectors(lattice):
    b = np.linalg.inv(lattice.T)
    return 2*np.pi * b

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

lattice = np.eye(3) * 5
gvec = get_reciprocal_vectors(lattice)

# fix number of bond dimension to be 6 and change the size of symmetry sector
nmo = 8
t_nk = []
for nmp in [1,2,3,4,5]:
    kpts = make_kpts(lattice, [nmp,nmp,1])
    nk = len(kpts)
    sym = ['++--', [kpts,]*4, None, gvec]
    te = ctf.timer_epoch("iteration")
    te.begin()
    t_sym_ctf = test_sym_ctf(nmo,sym)
    te.end()
    t_nk.append([nk, t_sym_ctf])
    if rank==0:
        print("nkpts=%i, ctf_sym=%.6f"%(nk, t_sym_ctf))

# fix number of symmetry sectors to be 9 and change the size of bond dimension
nmp=3
kpts = make_kpts(lattice, [nmp,nmp,1])
nk = len(kpts)
sym = ['++--', [kpts,]*4, None, gvec]
t_nmo = []
for nmo in range(10,20,4):
    te = ctf.timer_epoch("iteration")
    te.begin()
    t_sym_ctf = test_sym_ctf(nmo,sym)
    t_nmo.append([nmo, t_sym_ctf])
    te.end()
    if rank==0:
        print("nmo=%i, ctf_sym=%.6f"%(nmo,t_sym_ctf))
