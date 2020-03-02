#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
from functools import wraps

def load_lib(libname):
    if libname == 'numpy':
        import symtensor.backend.numpy_funclib as lib
    elif libname == 'ctf':
        import symtensor.backend.ctf_funclib as lib
    else:
        raise ValueError("Library %s not recognized" %libname)
    return lib
