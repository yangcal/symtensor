#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
"""Numpy backend for Symtensor"""
from mkl_interface import einsum_batched_matmul
import numpy as np
BACKEND = 'blas'

einsum = np.einsum
def einsum(sub, *operands, **kwargs):
    try:
        out = einsum_batched_matmul(sub, *operands)
    except:
        out = np.einsum(sub, *operands, **kwargs)
        print(sub, "failed")
    return out
#einsum = einsum_batched_matmul
astensor = np.asarray
zeros = np.zeros
empty = np.empty
ones = np.ones
rint = np.rint
random = np.random.random
norm = np.linalg.norm
qr = np.linalg.qr
dot = np.dot
diag = np.diag
eye = np.eye
hstack = np.hstack
vstack = np.vstack
def non_zeros(a):
    idx = np.where(a.ravel()!=0)
    return idx[0]
def copy(a):
    return a.copy()

def write_all(a, ind, fill):
    a.put(ind, fill)
    return a

write = write_single = write_all

def to_nparray(a):
    return a

def find_less(a, threshold):
    idx = np.where(a.ravel()<threshold)[0]
    return idx
