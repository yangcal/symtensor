#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
"""Numpy backend for Symtensor"""

import sys
import numpy

FUNCTIONS = ["einsum", "zeros", "diag", "norm", "random", "eye", "write", "array", "to_nparray", "nonzero"]

astensor = numpy.asarray
tensor = numpy.array
norm = numpy.linalg.norm
qr = numpy.linalg.qr
to_nparray = lambda arr: arr

def write(a, ind, fill):
    a.put(ind, fill)
    return a

thismodule = sys.modules[__name__]
for func in FUNCTIONS:
    if getattr(thismodule, func, None) is None:
        try:
            setattr(thismodule, func, getattr(numpy, func))
        except:
            raise ValueError("func %s not found in NumPy, customization needed"%func)

__all__ = FUNCTIONS
