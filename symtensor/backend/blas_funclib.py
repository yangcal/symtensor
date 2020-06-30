#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
"""Numpy backend for Symtensor"""

from mkl_interface import einsum_batched_matmul
from symtensor.backend.numpy_funclib import *, __all__
import numpy as np

def einsum(sub, *operands, **kwargs):
    try:
        out = einsum_batched_matmul(sub, *operands)
    except:
        out = np.einsum(sub, *operands, **kwargs)
        print(sub, "BLAS failed")
    return out
