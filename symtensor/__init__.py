from __future__ import absolute_import
import importlib

from . import ctf
from .sym import *

#module_path = 'symtensor.%s'%(BACKEND)
#module_handle = importlib.import_module(module_path)
#__all__ = module_handle.__all__
#funcs = [x for x in module_handle.__dict__ if not x.startswith("_")]
#globals().update({k: getattr(module_handle, k) for k in funcs})
