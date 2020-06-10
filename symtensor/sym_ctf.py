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

BACKEND='ctf'
backend = load_lib(BACKEND)
import ctf
def ctf_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = BACKEND
        return func(*args, **kwargs)
    return wrapper

zeros = ctf_wrapper(sym.zeros)
random = ctf_wrapper(sym.random)
zeros_like = sym.zeros_like
einsum = sym.einsum

class SYMtensor_ctf(sym.SYMtensor):
    __init__ = ctf_wrapper(sym.SYMtensor.__init__)

tensor = SYMtensor_ctf
core_einsum = ctf.einsum
