#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

import unittest
import numpy as np
try:
    import itertools
except:
    from functools import itertools
from symtensor.sym import random, einsum

thresh = 1e-8
ran =  [0,1]
no, nv = 5, 8
nmode = 3


class DMRGnpTEST(unittest.TestCase):

    def test_outprd(self):
        A = random([no,no,nv,nmode,nv], sym=["++-0-", [ran]*4, None, None])
        B = random([nv,nv,nv,nmode,nv], sym=["++-0-", [ran]*4, None, None])
        C = einsum('ijcxd,abcyd->ijxyab', A, B)

        out = np.zeros(C.array.shape)
        for ki,kj,kc,ka in itertools.product(ran, repeat=4):
            kd = ki+kj-kc
            kb = kc+kd-ka
            if kd not in ran or kb not in ran: continue
            out[ki,kj,ka] += np.einsum('ijcxd,abcyd->ijxyab', A.array[ki,kj,kc], B.array[ka,kb,kc])
        self.assertTrue(np.amax(C.array-out)<thresh)

    def test_prd(self):
        A = random([no,no,nv,nmode,nv], sym=["++-0-", [ran]*4, None, None])
        B = random([nv,nv,nv,nmode,nv], sym=["++-0-", [ran]*4, None, None])
        C = einsum('ijcxd,abcxd->ijab', A, B)

        out = np.zeros(C.array.shape)
        for ki,kj,kc,ka in itertools.product(ran, repeat=4):
            kd = ki+kj-kc
            kb = kc+kd-ka
            if kd not in ran or kb not in ran: continue
            out[ki,kj,ka] += np.einsum('ijcxd,abcxd->ijab', A.array[ki,kj,kc], B.array[ka,kb,kc])
        self.assertTrue(np.amax(C.array-out)<thresh)

    def test_hprd(self):
        A = random([no,no,nv,nv,nmode], sym=["++--0", [ran]*4, None, None])
        B = random([nv,nv,nv,nmode,nv], sym=["++-0-", [ran]*4, None, None])
        C = einsum('ijcdx,abcxd->ijabx', A, B)

        out = np.zeros(C.array.shape)
        for ki,kj,kc,ka in itertools.product(ran, repeat=4):
            kd = ki+kj-kc
            kb = kc+kd-ka
            if kd not in ran or kb not in ran: continue
            out[ki,kj,ka] += np.einsum('ijcdx,abcxd->ijabx', A.array[ki,kj,kc], B.array[ka,kb,kc])
        self.assertTrue(np.amax(C.array-out)<thresh)




if __name__ == '__main__':
    unittest.main()
