from __future__ import absolute_import
from functools import wraps
from symtensor.sym import (array, einsum, zeros, \
                          zeros_like, diag, tensor, __all__)
from . import random
import numpy
import ctf

__all__.extend(["random", "fromfunction", "frombatchfunc"])
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

def frombatchfunc(func, shape, all_tasks, **kwargs):
    nout = kwargs.pop("nout", 1)
    sym = kwargs.pop("sym", None)
    dtype = kwargs.pop("dtype", float)
    if isinstance(shape[0], list) or isinstance(shape[0], tuple):
        shape_list = shape
        nout = len(shape)
    else:
        shape_list = [shape,] * nout

    if sym is None:
        sym_list = [sym,] *nout
    elif isinstance(sym[0], str):
        sym_list = [sym,]*nout
    else:
        sym_list = sym

    out = kwargs.pop('out', None)
    if out is None:
        if nout==1:
            out = zeros(shape, sym, dtype=dtype)
        else:
            out = [zeros(shape_list[i], sym_list[i], dtype=dtype) for i in range(nout)]
    tasks, ntasks = static_partition(all_tasks)
    for itask in range(ntasks):
        if itask>= len(tasks):
            if nout ==1:
                out.put([], [])
            else:
                for i in range(nout):
                    out[i].put([], [])
            continue
        inds, vals = func(*tasks[itask], **kwargs)
        if nout ==1:
            out.put(inds.ravel(), vals.ravel())
        else:
            for i in range(nout):
                out[i].put(inds[i].ravel(), vals[i].ravel())
    return out
