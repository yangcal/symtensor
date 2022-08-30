#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
'''
Simple st.tensor contraction with 1D symmetry
usage: mpirun -np 4 python 00-1d_symtensor_ctf.py
'''

import numpy as np
import symtensor as st
import ctf
import tensorbackends as tbs

tc = tbs.get("ctf")

ni = np.arange(0,3)
nj = np.arange(0,4)
nk = np.arange(1,5)
nl = np.arange(0,1)
nm = np.arange(0,5)

nbond = 20
sym_ijk = ['++-', [ni,nj,nk], None, None] # I + J - K = 0
sym_klm = ['++-', [nk,nl,nm], None, None] # K + L - M = 0

ijk_array = tc.random.random([len(ni),len(nj),nbond,nbond,nbond])
klm_array = tc.random.random([len(nk),len(nl),nbond,nbond,nbond])

ijk = st.array(ijk_array, sym_ijk)
klm = st.array(klm_array, sym_klm)
ijlm = st.einsum('ijk,klm->ijlm',ijk,klm)
