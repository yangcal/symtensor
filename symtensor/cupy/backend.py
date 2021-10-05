#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
"""Cupy backend for Symtensor"""

import sys
import cupy

FUNCTIONS = ["einsum", "zeros", "diag", "norm", "random", "asarray", "hstack", "vstack",\
             "argsort", "eye", "put", "array", "to_nparray", "nonzero", "dot"]

tensor = cupy.array
norm = cupy.linalg.norm
qr = cupy.linalg.qr
to_nparray = lambda arr: cupy.asnumpy(arr)

def put(a, ind, fill):
    a.put(ind, fill)
    return a

thismodule = sys.modules[__name__]
for func in FUNCTIONS:
    if getattr(thismodule, func, None) is None:
        try:
            setattr(thismodule, func, getattr(cupy, func))
        except:
            raise ValueError("func %s not found in cupy, customization needed"%func)

__all__ = FUNCTIONS
