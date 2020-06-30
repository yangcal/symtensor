#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
import importlib

def load_lib(BACKEND):
    module_path = 'symtensor.%s.backend'%(BACKEND)
    try:
        return importlib.import_module(module_path)
    except:
        raise ValueError("backend %s not found, check symtensor.backend"%BACKEND)
