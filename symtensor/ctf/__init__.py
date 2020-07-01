from __future__ import absolute_import
from functools import wraps
from symtensor.sym import (array, einsum, zeros, \
                          zeros_like, diag, tensor, __all__)
from . import random
import numpy

__all__.extend(["random", "fromfunction"])
def backend_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = 'ctf'
        return func(*args, **kwargs)
    return wrapper

zeros = backend_wrapper(zeros)
diag = backend_wrapper(diag)
array = backend_wrapper(array)
tensor = backend_wrapper(tensor)


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def static_partition(tasks):
    segsize = (len(tasks)+size-1) // size
    start = rank * segsize
    stop = min(len(tasks), start+segsize)
    ntasks = max(comm.allgather(stop-start))
    return tasks[start:stop], ntasks

def fromfunction(func, shape, **kwargs):
    sym = kwargs.pop('sym', None)
    dtype = kwargs.pop('dtype', float)
    out = zeros(shape, sym, dtype=dtype)
    nsym = out.nsym
    if nsym==0:
        itask, ntasks = static_partition(numpy.arange(out.size))
        index = numpy.unravel_index(itask, shape)
        trunk_data = func(*index, **kwargs)
        out.put(itask, trunk_data.ravel())
    else:
        sym_shape = list(out.array.shape[:nsym-1])
        tasks, ntasks = static_partition(numpy.arange(numpy.prod(sym_shape)))
        trunk_size = numpy.prod(shape)
        for i in range(ntasks):
            if i >= len(tasks):
                out.put([], [])
                continue
            itask = tasks[i]
            idx = numpy.unravel_index(itask, sym_shape)
            trunk_data = func(*idx, **kwargs)
            trunk_idx = itask * trunk_size + numpy.arange(trunk_size)
            out.put(trunk_idx, trunk_data.ravel())
    return out
