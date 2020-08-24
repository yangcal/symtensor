import numpy
from symtensor.ctf import frombatchfunc
import itertools
nk = 3
nao = 20

def gen_fake_eri(ki,kj,kk):
    off = ki*nk**2+kj*nk+kk
    numpy.random.seed(ki*nk**2)
    eri_ijk = numpy.random.random([nao,nao,nao,nao])
    ind = off * eri_ijk.size + numpy.arange(eri_ijk.size)
    return ind.ravel(), eri_ijk.ravel()

all_tasks = [[ki,kj,kk] for ki,kj,kk in itertools.product(range(nk), repeat=3)]

sym = ["++--", [numpy.arange(nk)]*4, None, nk]

shape = (nao,nao,nao,nao)

eri = frombatchfunc(gen_fake_eri, shape, all_tasks, sym=sym)

reference = numpy.zeros([nk,nk,nk,nao,nao,nao,nao])
for ki,kj,kk in itertools.product(range(nk), repeat=3):
    ind, val = gen_fake_eri(ki,kj,kk)
    reference.put(ind, val)
print(numpy.linalg.norm(eri.array.to_nparray() - reference))
