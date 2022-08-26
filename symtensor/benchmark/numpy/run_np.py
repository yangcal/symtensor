#import mkl
import sys
# Setting the number of threads, default being 1
if len(sys.argv)>1:
    nthreads=int(sys.argv[1])
else:
    nthreads=1
#mkl.set_num_threads(nthreads)

import numpy as np
import time
import itertools

def get_kconserv(lattice, kpts):
    nkpts = kpts.shape[0]
    a = lattice / (2*np.pi)
    kconserv = np.zeros((nkpts,nkpts,nkpts), dtype=int)
    kvKLM = kpts[:,None,None,:] - kpts[:,None,:] + kpts
    for N, kvN in enumerate(kpts):
        kvKLMN = np.einsum('wx,klmx->wklm', a, kvKLM - kvN, optimize=True)
        kvKLMN_int = np.rint(kvKLMN)
        mask = np.einsum('wklm->klm', abs(kvKLMN - kvKLMN_int)) < 1e-9
        kconserv[mask] = N
    return kconserv

def test_numpy_block(nmo,kconserv):
    nk = len(kconserv)
    w = np.random.random([nk,nk,nk,nmo,nmo,nmo,nmo])
    out = np.zeros(w.shape, dtype=w.dtype)
    t0 = time.time()
    for ki,kj,ka in itertools.product(range(nk),repeat=3):
        kb = kconserv[ki,ka,kj]
        out[ki,kj] += np.einsum('ijab,Xabcd->Xijcd',w[ki,kj,ka], w[ka,kb])#,optimize=True)
    t1 = time.time()
    return t1 - t0

def test_sym_numpy(nmo,sym):
    from symtensor.sym import random,einsum
    w = random([nmo,]*4, sym=sym)
    t0 = time.time()
    out = einsum('ijab,abcd->ijcd',w,w)
    t1 = time.time()
    return t1-t0

def test_numpy_sparse(nmo,sym):
    from symtensor.sym import random
    w = random([nmo,]*4, sym=sym)
    w = w.make_sparse()
    t0 = time.time()
    out = np.einsum('IJABijab,ABCDabcd->IJCDijcd',w,w, optimize=True)
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
fn = 'pauling/nmo_%i_mkl%i.npy'%(nmo,nthreads)
for nmp in [1,2,3]:
    kpts = make_kpts(lattice, [nmp,nmp,1])
    kconserv = get_kconserv(lattice, kpts)
    nk = len(kpts)
    sym = ['++--', [kpts,]*4, None, gvec]
    t_numpy_bk = test_numpy_block(nmo, kconserv)
    t_sym_numpy = test_sym_numpy(nmo,sym)
    t_numpy_sp = 0. #test_numpy_sparse(nmo,sym)
    t_nk.append([nk,t_numpy_bk, t_sym_numpy, t_numpy_sp])
    print("nmo=%i, nkpts=%i, numpy_block=%.6f, numpy_sym=%.6f, numpy_sp=%.6f"%(nmo, nk, t_numpy_bk, t_sym_numpy, t_numpy_sp))

# fix number of symmetry sectors to be 9 and change the size of bond dimension
nmp=3
kpts = make_kpts(lattice, [nmp,nmp,1])
kconserv = get_kconserv(lattice, kpts)
nk = len(kpts)
sym = ['++--', [kpts,]*4, None, gvec]
t_nmo = []
fn = 'pauling/nk_%i_mkl%i.npy'%(nk,nthreads)
for nmo in [4,6,8,10,12,14,16,18,20]:
    t_numpy_bk = test_numpy_block(nmo, kconserv)
    t_sym_numpy = test_sym_numpy(nmo,sym)
    t_numpy_sp = 0. #test_numpy_sparse(nmo,sym)
    t_nmo.append([nmo,t_numpy_bk, t_sym_numpy,t_numpy_sp])
    print("nkpts=%i, nmo=%i, numpy_block=%.6f, numpy_sym=%.6f, numpy_sp=%.6f"%(nk, nmo,t_numpy_bk, t_sym_numpy, t_numpy_sp))
