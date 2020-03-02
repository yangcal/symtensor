#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
"""CTF backend for Symtensor"""
from mpi4py import MPI
import ctf

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

NAME='ctf'
einsum = ctf.einsum
astensor = ctf.astensor
zeros = ctf.zeros
empty = ctf.empty
ones = ctf.ones
rint = ctf.rint
random = ctf.random.random
hstack = ctf.hstack
vstack = ctf.vstack
dot = ctf.dot
qr = ctf.qr
diag = ctf.diag
eye = ctf.eye

def non_zeros(a):
    return a.read_all_nnz()[0]

def copy(a):
    return a.copy()

def write_all(a, ind, fill):
    a.write(ind, fill)
    return a

def write_single(a, ind, fill):
    if rank==0:
        a.write(ind, fill)
    else:
        a.write([],[])
    return a
write = write_all
def to_nparray(a):
    return a.to_nparray()

def norm(a):
    return a.norm2()

def find_less(a, threshold):
    c = a.sparsify(threshold)
    idx, vals = a.read_all_nnz()
    return idx
