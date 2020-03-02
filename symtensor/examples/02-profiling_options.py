#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
'''
MPI profiling options for CTF
Usage: mpirun -np 4 python 02-profiling_options.py
'''
import numpy as np
from symtensor.sym_ctf import random, einsum
import sys
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# runtime from rank 0
if rank != 0:
    sys.stdout = open(os.devnull, 'w')

ni = np.arange(0,3)
nj = np.arange(0,4)
nk = np.arange(1,5)
nl = np.arange(0,1)
nm = np.arange(0,5)

nbond = 50
sym1 = ['++-', [ni,nj,nk], None, None]
sym2 = ['++-', [nk,nl,nm], None, None]
ijk = random([nbond,]*3, sym1, verbose=2)
klm = random([nbond,]*3, sym2, verbose=2)
ijlm = einsum('ijk,klm->ijlm',ijk,klm)



# write runtime to file for each process
sys.stdout = open("output_rank%i.dat"%rank, "w")
nbond = 30
sym1 = ['++-', [ni,nj,nk], None, None]
sym2 = ['++-', [nk,nl,nm], None, None]
ijk = random([nbond,]*3, sym1, verbose=2)
klm = random([nbond,]*3, sym2, verbose=2)
ijlm = einsum('ijk,klm->ijlm',ijk,klm)
