#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
"""CTF backend for Symtensor"""

import sys
import ctf
import numpy

FUNCTIONS = ["einsum", "zeros", "diag", "norm", "random", "asarray", "hstack", "vstack",\
             "argsort", "eye", "put", "array", "to_nparray", "nonzero", "dot"]

asarray = ctf.astensor
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

nonzero = lambda arr: arr.read_all_nnz()[0]
def put(a, ind, fill):
    a.write(ind, fill)
    return a

def argsort(array, nroots=10):
    if array.ndim>1:
        raise ValueError("CTF argsort only support 1D array")
    ind, val = array.read_local()
    nvals = nroots * size
    args = numpy.argsort(val)[:nvals]
    ind = ind[args]
    vals = val[args]
    out = numpy.vstack([ind, vals])
    tmp = numpy.hstack(comm.allgather(out))
    ind, val = tmp
    args = numpy.argsort(val)[:nroots]
    return ind[args]

thismodule = sys.modules[__name__]
for func in FUNCTIONS:
    if getattr(thismodule, func, None) is None:
        try:
            setattr(thismodule, func, getattr(ctf, func))
        except:
            raise ValueError("func %s not found in CTF, customization needed"%func)

__all__ = FUNCTIONS
