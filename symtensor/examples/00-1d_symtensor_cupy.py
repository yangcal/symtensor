#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
'''
Simple tensor contraction with 1D symmetry
'''

import numpy as np
import symtensor as st
import ctf
import tensorbackends as tbs

tc = tbs.get("cupy")

ni = np.arange(0,3)
nj = np.arange(0,4)
nk = np.arange(1,5)
nl = np.arange(0,1)
nm = np.arange(0,5)

nbond = 20
sym_ijk = ['++-', [ni,nj,nk], None, None] # I + J - K = 0
sym_klm = ['++-', [nk,nl,nm], None, None] # K + L - M = 0

ijk_array = np.random.random([len(ni),len(nj),nbond,nbond,nbond])
klm_array = np.random.random([len(nk),len(nl),nbond,nbond,nbond])

ijk = array(ijk_array, sym_ijk, backend=tc)
klm = array(klm_array, sym_klm, backend=tc)
ijlm = einsum('ijk,klm->ijlm',ijk,klm)
