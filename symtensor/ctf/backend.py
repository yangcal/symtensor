#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
"""CTF backend for Symtensor"""

import sys
import ctf

FUNCTIONS = ["einsum", "zeros", "diag", "norm", "random", "eye", "put", "array", "to_nparray", "nonzero"]

nonzero = lambda arr: arr.read_all_nnz()[0]
def put(a, ind, fill):
    a.write(ind, fill)
    return a

thismodule = sys.modules[__name__]
for func in FUNCTIONS:
    if getattr(thismodule, func, None) is None:
        try:
            setattr(thismodule, func, getattr(ctf, func))
        except:
            raise ValueError("func %s not found in CTF, customization needed"%func)

__all__ = FUNCTIONS
