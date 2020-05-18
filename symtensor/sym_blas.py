#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
'''
Symtensor with ctf as backend
'''
from functools import wraps
from symtensor import sym
from symtensor.settings import load_lib

BACKEND='blas'
backend = load_lib(BACKEND)
def blas_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = BACKEND
        return func(*args, **kwargs)
    return wrapper

zeros = blas_wrapper(sym.zeros)
random = blas_wrapper(sym.random)
zeros_like = sym.zeros_like
einsum = sym.symeinsum

class SYMtensor_blas(sym.SYMtensor):
    __init__ = blas_wrapper(sym.SYMtensor.__init__)

tensor = SYMtensor_blas
core_einsum = backend.einsum
