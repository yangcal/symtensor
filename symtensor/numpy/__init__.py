from __future__ import absolute_import
from functools import wraps
from symtensor.sym import (array, einsum, zeros, get_full_shape, \
                          zeros_like, diag, tensor, __all__)
from . import random
import numpy

__all__.extend(["random", "fromfunction", "frombatchfunc"])
def backend_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = 'numpy'
        return func(*args, **kwargs)
    return wrapper

zeros = backend_wrapper(zeros)
diag = backend_wrapper(diag)
array = backend_wrapper(array)
tensor = backend_wrapper(tensor)

def fromfunction(func, shape, **kwargs):
    sym = kwargs.pop('sym', None)
    dtype = kwargs.get('dtype', float)
    out = zeros(shape, sym)
    nsym = out.nsym
    if out.nsym==0:
        out.array = numpy.fromfunction(func, shape, **kwargs)
    else:
        kwargs.pop('dtype', None)
        sym_shape = list(out.array.shape[:nsym-1])
        ntasks = numpy.prod(sym_shape)
        trunk_size = numpy.prod(shape)
        for i in range(ntasks):
            idx = numpy.unravel_index(i, sym_shape)
            trunk_data = func(*idx, **kwargs)
            trunk_idx = i * trunk_size + numpy.arange(trunk_size)
            out.put(trunk_idx, trunk_data.ravel())
    return out

def frombatchfunc(func, shape, all_tasks, **kwargs):
    nout = kwargs.pop("nout", 1)
    sym = kwargs.pop("sym", None)
    dtype = kwargs.pop("dtype", float)
    if isinstance(shape[0], list) or isinstance(shape[0], tuple):
        shape_list = shape
    else:
        shape_list = [shape,] * nout

    if sym is None:
        sym_list = [sym,]*nout
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

    for itask in all_tasks:
        inds, vals = func(*itask, **kwargs)
        if nout ==1:
            out.put(inds.ravel(), vals.ravel())
        else:
            for i in range(nout):
                out[i].put(inds[i].ravel(), vals[i].ravel())
        inds = vals = None

    return out
