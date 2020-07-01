from __future__ import absolute_import
from functools import wraps
from symtensor.sym import (array, einsum, zeros, get_full_shape, \
                          zeros_like, diag, tensor, __all__)
from . import random
import numpy

__all__.extend(["random", "fromfunction"])
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
