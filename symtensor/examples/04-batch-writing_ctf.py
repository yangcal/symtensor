import numpy as np
import ctf
import tensorbackends as tbs
import symtensor as st

tc = tbs.get("ctf")

import itertools
nk = 3
nao = 20

def gen_fake_eri(ki,kj,kk):
    off = ki*nk**2+kj*nk+kk
    np.random.seed(ki*nk**2)
    eri_ijk = np.random.random([nao,nao,nao,nao])
    ind = off * eri_ijk.size + np.arange(eri_ijk.size)
    return ind.ravel(), eri_ijk.ravel()

all_tasks = [[ki,kj,kk] for ki,kj,kk in itertools.product(range(nk), repeat=3)]

sym = ["++--", [np.arange(nk)]*4, None, nk]

shape = (nao,nao,nao,nao)

eri = st.frombatchfunc(gen_fake_eri, shape, all_tasks, sym=sym)

reference = np.zeros([nk,nk,nk,nao,nao,nao,nao])
for ki,kj,kk in itertools.product(range(nk), repeat=3):
    ind, val = gen_fake_eri(ki,kj,kk)
    reference.put(ind, val)
print(np.linalg.norm(eri.array.numpy() - reference))
