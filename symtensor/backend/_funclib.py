
from symtensor.settings import load_lib

def zeros(shape, dtype=float, sym=None, backend='numpy'):
    lib = load_lib(backend)
    array = lib.zeros(shape, dtype=dtype)
    return zeros
    
