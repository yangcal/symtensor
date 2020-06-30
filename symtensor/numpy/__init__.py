from __future__ import absolute_import
from functools import wraps
from symtensor.sym import (array, einsum, zeros, \
                          zeros_like, diag, tensor, __all__)
from . import random

__all__.append("random")
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
